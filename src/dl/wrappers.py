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
