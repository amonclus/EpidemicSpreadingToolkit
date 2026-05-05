import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from .features import extract_features_batch
from analysis.graph_features import GRAPH_FEATURE_NAMES

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "ml_data")
_CSV_PATH = os.path.join(_DATA_DIR, "ml_dataset.csv")
_NPY_PATH = os.path.join(_DATA_DIR, "ml_I_series.npy")


class EpidemicDataset(Dataset):
    """
    Returns (x, rho, model_id, features) where:
      x        — normalised I(t)/N series, shape (t_obs,)
      rho      — scalar rho_final
      model_id — int class label
      features — normalised handcrafted features, shape (N_FEATURES,)

    Stratified split is computed once on construction and shared across splits
    so the same samples always land in the same partition regardless of t_obs.
    """

    def __init__(
        self,
        t_obs: int = 30,
        split: str = "train",
        train_indices=None,
        val_indices=None,
        test_indices=None,
        norm_max: float = None,
        feat_mean: np.ndarray = None,
        feat_std: np.ndarray = None,
        seed: int = 42,
    ):
        if not os.path.exists(_CSV_PATH) or not os.path.exists(_NPY_PATH):
            raise FileNotFoundError(
                f"Dataset files not found.\n"
                f"  Expected: {_CSV_PATH}\n"
                f"           {_NPY_PATH}\n"
                "Run ml_scripts/generate_ml_dataset.py first."
            )

        self.t_obs = t_obs
        self.split = split

        df       = pd.read_csv(_CSV_PATH)
        I_series = np.load(_NPY_PATH).astype(np.float32)

        self.rho_final = df["rho_final"].values.astype(np.float32)
        self.model_ids = df["model_id"].values.astype(np.int64)
        self.I_series  = I_series  # (n, 300)

        n = len(df)

        if train_indices is None:
            train_indices, val_indices, test_indices = _stratified_split(
                n, self.model_ids, seed=seed
            )

        self.train_indices = train_indices
        self.val_indices   = val_indices
        self.test_indices  = test_indices

        split_map = {"train": train_indices, "val": val_indices, "test": test_indices}
        if split not in split_map:
            raise ValueError(f"split must be one of {list(split_map)}, got '{split}'")
        self.indices = split_map[split]

        # ── I-series normalisation (fit on train only) ────────────────────────
        if norm_max is None:
            train_series = self.I_series[train_indices, :t_obs]
            norm_max = float(train_series.max()) if train_series.max() > 0 else 1.0
        self.norm_max = norm_max

        # ── Handcrafted + graph features (fit mean/std on train only) ────────
        ts_feats    = extract_features_batch(self.I_series, t_obs)  # (n, 22)
        graph_cols  = [c for c in GRAPH_FEATURE_NAMES if c in df.columns]
        graph_feats = df[graph_cols].values.astype(np.float32)       # (n, ≤10)
        all_feats   = np.concatenate([ts_feats, graph_feats], axis=1)
        all_feats   = np.nan_to_num(all_feats, nan=0.0, posinf=0.0, neginf=0.0)

        if feat_mean is None:
            feat_mean = all_feats[train_indices].mean(axis=0)
            feat_std  = all_feats[train_indices].std(axis=0)
            feat_std[feat_std < 1e-8] = 1.0

        self.feat_mean = feat_mean
        self.feat_std  = feat_std
        self.features  = (all_feats - feat_mean) / feat_std  # (n, 22+|graph|)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        series = np.clip(self.I_series[i, : self.t_obs] / self.norm_max, 0.0, 1.0)
        x      = torch.tensor(series, dtype=torch.float32)
        rho    = torch.tensor(self.rho_final[i], dtype=torch.float32)
        mid    = torch.tensor(self.model_ids[i], dtype=torch.long)
        feat   = torch.tensor(self.features[i],  dtype=torch.float32)

        return x, rho, mid, feat


def _stratified_split(n, labels, seed=42):
    indices = np.arange(n)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, labels))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_idx, labels[temp_idx]))

    return train_idx, temp_idx[val_idx], temp_idx[test_idx]


def get_dataloaders(t_obs: int = 30, batch_size: int = 256,
                    num_workers: int = 0, seed: int = 42):
    train_ds = EpidemicDataset(t_obs=t_obs, split="train", seed=seed)
    shared = dict(
        t_obs=t_obs,
        train_indices=train_ds.train_indices,
        val_indices=train_ds.val_indices,
        test_indices=train_ds.test_indices,
        norm_max=train_ds.norm_max,
        feat_mean=train_ds.feat_mean,
        feat_std=train_ds.feat_std,
        seed=seed,
    )
    val_ds  = EpidemicDataset(split="val",  **shared)
    test_ds = EpidemicDataset(split="test", **shared)

    pin = torch.cuda.is_available()
    kw  = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )
