"""
Sklearn-compatible wrappers around trained PyTorch epidemic models.
Stored as .pkl via joblib so the app can load them the same way as ML models.

Usage:
    reg = joblib.load("ml_data/dl_regressor.pkl")
    clf = joblib.load("ml_data/dl_classifier.pkl")
    rho  = reg.predict(I_series_matrix)                         # (n,)
    cls  = clf.predict(I_series_matrix)                         # (n,) int labels
    prob = clf.predict_proba(I_series_matrix)                   # (n, n_classes)

    # Optionally pass graph features (n, n_graph_feats); zeros used as fallback.
    rho  = reg.predict(I_series_matrix, graph_features=gf)

Two-stage regressor (drop-in replacement for DLRegressor):
    reg = joblib.load("ml_data/dl_regressor.pkl")   # may be DLTwoStageRegressor
    rho = reg.predict(I_series_matrix)               # same interface

Dynamic wrappers (select best-matching t_obs at inference):
    reg = joblib.load("ml_data/dl_regressor.pkl")   # DLDynamicRegressor
    clf = joblib.load("ml_data/dl_classifier.pkl")  # DLDynamicClassifier
    # Model with t_obs closest to (but not exceeding) input length is used.
    # Falls back to smallest available t_obs if input is shorter than all windows.
"""

import numpy as np
import torch

from .features import extract_features_batch

N_TS_FEATURES = 22  # time-series hand-crafted features


class _DLWrapper:
    def __init__(self, state_dict, arch_name, n_features, t_obs,
                 norm_max, feat_mean, feat_std):
        self._state_dict = state_dict
        self._arch_name  = arch_name
        self._n_features = n_features
        self._t_obs      = t_obs
        self._norm_max   = float(norm_max)
        self._feat_mean  = np.asarray(feat_mean, dtype=np.float32)
        self._feat_std   = np.asarray(feat_std,  dtype=np.float32)
        self._model      = None  # lazy — reconstructed on first call

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None  # nn.Module reconstructed from state_dict on load
        return state

    def _get_model(self):
        if self._model is None:
            from .models import CNNEmbedder, LSTMEmbedder, TransformerEmbedder
            registry = {
                "CNN":         CNNEmbedder,
                "LSTM":        LSTMEmbedder,
                "Transformer": TransformerEmbedder,
            }
            m = registry[self._arch_name](n_features=self._n_features)
            m.load_state_dict(self._state_dict)
            m.eval()
            self._model = m
        return self._model

    def _prepare(self, I_series_matrix, graph_features=None):
        """Return (x_tensor, feat_tensor) ready for model forward pass."""
        I = np.asarray(I_series_matrix, dtype=np.float32)
        n = len(I)

        # Pad or truncate to t_obs so the model always sees a fixed-length input
        n_t = I.shape[1]
        if n_t < self._t_obs:
            I = np.concatenate(
                [I, np.zeros((len(I), self._t_obs - n_t), dtype=np.float32)], axis=1
            )
        series   = np.clip(I[:, :self._t_obs] / self._norm_max, 0.0, 1.0)
        x_tensor = torch.tensor(series, dtype=torch.float32)

        eff_t_obs = min(n_t, self._t_obs)
        ts_feats = extract_features_batch(I, eff_t_obs)             # (n, 22)

        n_graph = self._n_features - N_TS_FEATURES
        if n_graph > 0:
            if graph_features is not None:
                gf = np.asarray(graph_features, dtype=np.float32)
            else:
                gf = np.zeros((n, n_graph), dtype=np.float32)
            all_feats = np.concatenate([ts_feats, gf], axis=1)
        else:
            all_feats = ts_feats

        all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=0.0, neginf=0.0)
        all_feats = (all_feats - self._feat_mean) / self._feat_std
        feat_tensor = torch.tensor(all_feats, dtype=torch.float32)

        return x_tensor, feat_tensor


class DLRegressor(_DLWrapper):
    """Predicts rho_final (epidemic final size) from raw I(t)/N series."""

    def predict(self, I_series_matrix, graph_features=None):
        """
        Parameters
        ----------
        I_series_matrix : array-like, shape (n, T)
        graph_features  : array-like, shape (n, n_graph_feats), optional

        Returns
        -------
        rho_pred : np.ndarray, shape (n,)
        """
        model = self._get_model()
        x, feat = self._prepare(I_series_matrix, graph_features)
        with torch.no_grad():
            rho_pred, _ = model(x, feat)
        return rho_pred.numpy()


class DLClassifier(_DLWrapper):
    """Predicts spreading model class from raw I(t)/N series."""

    def predict(self, I_series_matrix, graph_features=None):
        """Returns integer class labels, shape (n,)."""
        model = self._get_model()
        x, feat = self._prepare(I_series_matrix, graph_features)
        with torch.no_grad():
            _, logits = model(x, feat)
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, I_series_matrix, graph_features=None):
        """Returns softmax probabilities, shape (n, n_classes)."""
        model = self._get_model()
        x, feat = self._prepare(I_series_matrix, graph_features)
        with torch.no_grad():
            _, logits = model(x, feat)
        return torch.softmax(logits, dim=1).numpy()


class DLTwoStageRegressor(_DLWrapper):
    """
    Two-stage regressor (drop-in replacement for DLRegressor):

      Stage 1 — CNN classifier identifies the epidemic model type (0–9)
      Stage 2 — specialist regression head predicts rho_final for that type

    Each specialist head is trained only on data from its own model class,
    so it learns the specific rho curve shape without cross-model interference.
    The backbone (feature extractor) is shared and frozen from the Stage-1 CNN.

    Falls back to the generalist regression head for any class whose specialist
    could not be trained (too few samples).
    """

    def __init__(self, state_dict, arch_name, n_features, t_obs,
                 norm_max, feat_mean, feat_std, specialist_heads: dict):
        """
        Parameters
        ----------
        specialist_heads : dict {model_id (int): state_dict or None}
            Trained SpecialistRegHead state dicts, one per epidemic class.
            None means fall back to the generalist regression head for that class.
        """
        super().__init__(state_dict, arch_name, n_features, t_obs,
                         norm_max, feat_mean, feat_std)
        self._specialist_heads = specialist_heads
        self._spec_cache: dict = {}

    def __getstate__(self):
        state = super().__getstate__()
        state["_spec_cache"] = {}
        return state

    def _get_specialist(self, k: int):
        if k not in self._spec_cache:
            from .models import SpecialistRegHead, EMBEDDING_DIM
            sd = self._specialist_heads.get(k)
            if sd is None:
                self._spec_cache[k] = None
            else:
                head = SpecialistRegHead(EMBEDDING_DIM + self._n_features)
                head.load_state_dict(sd)
                head.eval()
                self._spec_cache[k] = head
        return self._spec_cache[k]

    def predict(self, I_series_matrix, graph_features=None):
        """
        Parameters
        ----------
        I_series_matrix : array-like, shape (n, T)
        graph_features  : array-like, shape (n, n_graph_feats), optional

        Returns
        -------
        rho_pred : np.ndarray, shape (n,)
        """
        model = self._get_model()
        x, feat = self._prepare(I_series_matrix, graph_features)

        with torch.no_grad():
            # Stage 1: classify to get routing labels
            generalist_rho, logits = model(x, feat)
            model_ids = logits.argmax(dim=1).numpy()  # (n,)

            # Shared backbone embedding for all specialists
            raw_emb  = model.get_embedding(x)                          # (n, 64)
            full_emb = torch.cat([raw_emb, feat], dim=-1) if self._n_features > 0 else raw_emb

            # Stage 2: route each sample to its specialist
            rho_pred = generalist_rho.numpy().copy()  # fallback values
            for k in range(10):
                mask = model_ids == k
                if not mask.any():
                    continue
                specialist = self._get_specialist(k)
                if specialist is not None:
                    mask_t = torch.from_numpy(mask)
                    rho_pred[mask] = specialist(full_emb[mask_t]).numpy()

        return rho_pred


# ── Dynamic wrappers ──────────────────────────────────────────────────────────

class _DLDynamicBase:
    """
    Holds one inner model per t_obs and routes each call to the model whose
    window best matches the actual input length:
      - largest available t_obs that does not exceed the input length, OR
      - smallest available t_obs if the input is shorter than all windows.
    """

    def __init__(self, models: dict):
        """
        Parameters
        ----------
        models : dict {t_obs (int): DLRegressor / DLClassifier / DLTwoStageRegressor}
        """
        if not models:
            raise ValueError("models dict must not be empty")
        self._models = models
        self._t_obs_sorted = sorted(models.keys())

    def _select(self, n_steps: int) -> int:
        candidates = [t for t in self._t_obs_sorted if t <= n_steps]
        return max(candidates) if candidates else self._t_obs_sorted[0]

    @property
    def t_obs_available(self):
        return list(self._t_obs_sorted)


class DLDynamicRegressor(_DLDynamicBase):
    """Dynamic regressor — selects the best-matching t_obs model per call."""

    def predict(self, I_series_matrix, graph_features=None):
        n_steps = np.asarray(I_series_matrix).shape[1]
        t_obs   = self._select(n_steps)
        return self._models[t_obs].predict(I_series_matrix, graph_features)


class DLDynamicClassifier(_DLDynamicBase):
    """Dynamic classifier — selects the best-matching t_obs model per call."""

    def predict(self, I_series_matrix, graph_features=None):
        n_steps = np.asarray(I_series_matrix).shape[1]
        t_obs   = self._select(n_steps)
        return self._models[t_obs].predict(I_series_matrix, graph_features)

    def predict_proba(self, I_series_matrix, graph_features=None):
        n_steps = np.asarray(I_series_matrix).shape[1]
        t_obs   = self._select(n_steps)
        return self._models[t_obs].predict_proba(I_series_matrix, graph_features)
