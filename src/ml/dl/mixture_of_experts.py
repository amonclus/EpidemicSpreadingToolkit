import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


class MixtureOfExperts:
    def __init__(self, gating_model, specialists: dict = None):
        self.gating_model = gating_model
        self.specialists  = specialists or {}

    def _extract_embeddings(self, x_tensor: torch.Tensor,
                             feat_tensor: torch.Tensor, device: str) -> np.ndarray:
        """Pure backbone embedding (no feature injection) — used for specialists."""
        self.gating_model.eval()
        all_emb = []
        bs = 512
        with torch.no_grad():
            for i in range(0, len(x_tensor), bs):
                x = x_tensor[i : i + bs].to(device)
                all_emb.append(self.gating_model.get_embedding(x).cpu().numpy())
        return np.concatenate(all_emb, axis=0)

    def _forward_batched(self, x_tensor: torch.Tensor,
                          feat_tensor: torch.Tensor, device: str):
        """Returns (gate_probs, embeddings) using the full forward pass."""
        self.gating_model.eval()
        all_probs, all_emb = [], []
        bs = 512
        with torch.no_grad():
            for i in range(0, len(x_tensor), bs):
                x    = x_tensor[i : i + bs].to(device)
                feat = feat_tensor[i : i + bs].to(device)
                _, logits = self.gating_model(x, feat)
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                all_emb.append(self.gating_model.get_embedding(x).cpu().numpy())
        return np.concatenate(all_probs), np.concatenate(all_emb)

    def fit_specialists(
        self,
        x_tensor: torch.Tensor,
        feat_tensor: torch.Tensor,
        y_rho: np.ndarray,
        y_labels: np.ndarray,
        specialist_type: str = "rf",
        device: str = "cpu",
    ):
        embeddings = self._extract_embeddings(x_tensor, feat_tensor, device)

        for model_id in range(10):
            mask = y_labels == model_id
            if mask.sum() == 0:
                continue
            if specialist_type == "rf":
                reg = RandomForestRegressor(n_estimators=200, random_state=42)
            elif specialist_type == "hgb":
                reg = HistGradientBoostingRegressor(random_state=42)
            elif specialist_type == "ridge":
                reg = Ridge(alpha=1.0)
            else:
                raise ValueError(f"Unknown specialist_type: {specialist_type}")
            reg.fit(embeddings[mask], y_rho[mask])
            self.specialists[model_id] = reg
            print(f"  Fitted specialist {model_id} on {mask.sum()} samples.")

    def predict(self, x_tensor: torch.Tensor, feat_tensor: torch.Tensor,
                mode: str = "soft", device: str = "cpu") -> np.ndarray:
        gate_probs, embeddings = self._forward_batched(x_tensor, feat_tensor, device)

        spec_preds = np.zeros((len(embeddings), 10), dtype=np.float32)
        for mid, reg in self.specialists.items():
            spec_preds[:, mid] = reg.predict(embeddings).astype(np.float32)

        if mode == "soft":
            return (gate_probs * spec_preds).sum(axis=1)
        elif mode == "hard":
            best = gate_probs.argmax(axis=1)
            return spec_preds[np.arange(len(best)), best]
        else:
            raise ValueError(f"mode must be 'soft' or 'hard', got '{mode}'")

    def evaluate(self, x_tensor: torch.Tensor, feat_tensor: torch.Tensor,
                 y_true: np.ndarray, mode: str = "soft", device: str = "cpu"):
        preds = self.predict(x_tensor, feat_tensor, mode=mode, device=device)
        mae   = float(np.abs(preds - y_true).mean())
        ss_res = float(((y_true - preds) ** 2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        gate_probs, _ = self._forward_batched(x_tensor, feat_tensor, device)
        labels = gate_probs.argmax(axis=1)
        per_model_mae = {
            mid: float(np.abs(preds[labels == mid] - y_true[labels == mid]).mean())
            for mid in range(10) if (labels == mid).any()
        }
        return {"MAE": mae, "R2": r2, "per_model_MAE": per_model_mae}
