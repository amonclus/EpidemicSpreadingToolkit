import torch
import torch.nn as nn

NUM_CLASSES    = 10
EMBEDDING_DIM  = 64   # shared across all architectures


class _Heads(nn.Module):
    """Regression and classification heads."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.reg_head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.cls_head = nn.Linear(in_dim, NUM_CLASSES)

    def forward(self, emb):
        return self.reg_head(emb).squeeze(-1), self.cls_head(emb)


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

class CNNEmbedder(nn.Module):
    def __init__(self, embedding_dim: int = 64, n_features: int = 0):
        super().__init__()
        self.n_features = n_features
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.heads = _Heads(embedding_dim + n_features)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x.unsqueeze(1))

    def forward(self, x: torch.Tensor, feat: torch.Tensor = None):
        emb = self.get_embedding(x)
        if feat is not None and self.n_features > 0:
            emb = torch.cat([emb, feat], dim=-1)
        return self.heads(emb)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMEmbedder(nn.Module):
    def __init__(self, hidden_dim: int = 64, embedding_dim: int = 64,
                 n_features: int = 0):
        super().__init__()
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, dropout=0.1,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.heads = _Heads(embedding_dim + n_features)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x.unsqueeze(-1))
        return self.proj(h[-1])

    def forward(self, x: torch.Tensor, feat: torch.Tensor = None):
        emb = self.get_embedding(x)
        if feat is not None and self.n_features > 0:
            emb = torch.cat([emb, feat], dim=-1)
        return self.heads(emb)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class TransformerEmbedder(nn.Module):
    MAX_T_OBS = 300

    def __init__(self, d_model: int = 64, embedding_dim: int = 64,
                 n_features: int = 0):
        super().__init__()
        self.d_model    = d_model
        self.n_features = n_features
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = nn.Embedding(self.MAX_T_OBS, d_model)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.proj = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.heads = _Heads(embedding_dim + n_features)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        tokens = self.input_proj(x.unsqueeze(-1))
        tokens = tokens + self.pos_enc(torch.arange(t, device=x.device)).unsqueeze(0)
        tokens = torch.cat([self.cls_token.expand(b, -1, -1), tokens], dim=1)
        return self.proj(self.transformer(tokens)[:, 0])

    def forward(self, x: torch.Tensor, feat: torch.Tensor = None):
        emb = self.get_embedding(x)
        if feat is not None and self.n_features > 0:
            emb = torch.cat([emb, feat], dim=-1)
        return self.heads(emb)

    def get_attention_weights(self, x: torch.Tensor):
        b, t = x.shape
        tokens = self.input_proj(x.unsqueeze(-1))
        tokens = tokens + self.pos_enc(torch.arange(t, device=x.device)).unsqueeze(0)
        tokens = torch.cat([self.cls_token.expand(b, -1, -1), tokens], dim=1)
        _, attn = self.transformer.layers[0].self_attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=False
        )
        return attn  # (batch, nhead, seq, seq)


# ---------------------------------------------------------------------------
# Specialist regression head (two-stage pipeline)
# ---------------------------------------------------------------------------

class SpecialistRegHead(nn.Module):
    """
    Standalone regression head for the two-stage specialist regressor.
    Receives a pre-computed embedding (backbone_output ++ hand-crafted features)
    and predicts rho_final for one specific epidemic model class.
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb).squeeze(-1)
