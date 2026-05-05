#!/usr/bin/env python3
"""
Full ML analysis on epidemic spreading simulation data.

Loads ml_data/ml_dataset.csv and ml_data/ml_I_series.npy, extracts features,
trains models, evaluates, and saves 8 plots + a summary to results/ml/.

Usage:  python ml_analysis.py
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

from analysis.graph_features import GRAPH_FEATURE_NAMES

# ── Constants ──────────────────────────────────────────────────────────────────
T_OBS_VALUES  = [10, 20, 30, 50, 75, 100]
T_OBS_DEFAULT = 30
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
N_ESTIMATORS  = 200

MODEL_COLORS = {
    "SIR": "#2196F3", "SIS": "#00BCD4",
    "BP":  "#4CAF50", "WTM": "#FF9800",
    "H1":  "#F44336", "H2":  "#E91E63",
    "H3":  "#9C27B0", "H4":  "#673AB7",
    "H5":  "#3F51B5", "H6":  "#009688",
}

MODEL_ID_MAP = {
    "SIR": 0, "SIS": 1, "BP": 2, "WTM": 3,
    "H1":  4, "H2":  5, "H3": 6, "H4":  7, "H5": 8, "H6": 9,
}
ID_TO_MODEL  = {v: k for k, v in MODEL_ID_MAP.items()}
MODEL_NAMES  = list(MODEL_COLORS.keys())    # fixed order SIR … H6
NETWORKS     = ["ER", "RGG", "Lattice", "Facebook", "GitHub"]

FEATURE_CATEGORIES = {
    # original features
    "early_growth_rate":        "growth",
    "log_amplification":        "growth",
    "doubling_time":            "growth",
    "curvature":                "shape",
    "already_peaked":           "shape",
    "peak_in_window":           "shape",
    "t_peak_in_window":         "shape",
    "peak_sharpness":           "shape",
    "I_at_t_obs":               "level",
    "I_mean_window":            "level",
    "I_total_change":           "level",
    "fraction_above_001":       "level",
    "I_std_window":             "variance",
    "max_single_step_increase": "variance",
    # new mechanistic features
    "tail_mean":                "level",
    "tail_std":                 "variance",
    "endemic_level":            "level",
    "decay_rate_after_peak":    "growth",
    "fraction_decreasing":      "shape",
    "phase_switch_score":       "shape",
    "fwhm":                     "shape",
    "autocorr_lag1":            "variance",
    # graph-structural features
    **{f: "graph" for f in GRAPH_FEATURE_NAMES},
}

CATEGORY_COLORS = {
    "growth":   "#E53935",
    "shape":    "#8E24AA",
    "level":    "#1E88E5",
    "variance": "#43A047",
    "graph":    "#FF6F00",
}

DATA_DIR    = Path("ml/ml_data")
RESULTS_DIR = Path("results/ml")

FS_TITLE  = 14
FS_LABEL  = 12
FS_TICK   = 10
FS_LEGEND = 10


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(I_series: np.ndarray, t_obs: int) -> dict:
    """Return 14 scalar features extracted from I[0:t_obs] only."""
    window = np.asarray(I_series[:t_obs], dtype=float)
    n      = len(window)
    eps    = 1e-8

    # ── Growth ────────────────────────────────────────────────────────────────
    log_I = np.log(window + eps)
    early_growth_rate = float(np.polyfit(np.arange(n), log_I, 1)[0]) if n >= 2 else 0.0
    log_amplification = float(np.log((window[-1] + eps) / (window[0] + eps)))

    doubled = np.where(window >= 2.0 * (window[0] + eps))[0]
    doubling_time = float(doubled[0]) if len(doubled) else float("nan")

    # ── Shape ─────────────────────────────────────────────────────────────────
    curvature  = float(np.mean(np.diff(window, 2))) if n >= 3 else 0.0
    peak_val   = float(window.max())
    t_peak     = int(window.argmax())
    already_peaked = 1.0 if window[-1] < peak_val else 0.0
    peak_sharpness = peak_val / (t_peak + 1)

    # ── Level ─────────────────────────────────────────────────────────────────
    I_at_t_obs         = float(window[-1])
    I_mean_window      = float(window.mean())
    I_total_change     = float(window[-1] - window[0])
    fraction_above_001 = float(np.mean(window > 0.01))

    # ── Variance ──────────────────────────────────────────────────────────────
    I_std_window = float(window.std())
    diffs        = np.diff(window)
    max_single_step_increase = float(max(0.0, diffs.max())) if len(diffs) else 0.0

    # ── Tail behaviour (SIR decays to 0; SIS/endemic stays positive) ─────────
    tail_start = max(1, int(n * 0.8))
    tail       = window[tail_start:]
    tail_mean  = float(tail.mean()) if len(tail) else float(window[-1])
    tail_std   = float(tail.std())  if len(tail) else 0.0
    # ratio of final I to peak — near 0 for SIR, >0 for SIS endemic
    endemic_level = float(window[-1] / (peak_val + eps))

    # ── Decay rate after peak ─────────────────────────────────────────────────
    post_peak = window[t_peak:]
    if len(post_peak) >= 2:
        decay_rate_after_peak = float(
            np.polyfit(np.arange(len(post_peak)), post_peak, 1)[0]
        )
    else:
        decay_rate_after_peak = 0.0
    fraction_decreasing = float(np.mean(diffs < 0)) if len(diffs) else 0.0

    # ── Phase-switch score (H2/H5 kink in second derivative) ─────────────────
    if n >= 5:
        d2 = np.diff(window, 2)
        phase_switch_score = float(np.max(np.abs(d2))) if len(d2) else 0.0
    else:
        phase_switch_score = 0.0

    # ── Full width at half maximum (pulse sharpness vs breadth) ──────────────
    half_max    = peak_val / 2.0
    above_half  = np.where(window >= half_max)[0]
    fwhm        = float(above_half[-1] - above_half[0] + 1) if len(above_half) >= 2 else 0.0

    # ── Autocorrelation lag-1 (smoothness / stochasticity) ───────────────────
    if n >= 3 and window.std() > eps:
        autocorr_lag1 = float(np.corrcoef(window[:-1], window[1:])[0, 1])
    else:
        autocorr_lag1 = 1.0

    return {
        "early_growth_rate":        early_growth_rate,
        "log_amplification":        log_amplification,
        "doubling_time":            doubling_time,
        "curvature":                curvature,
        "already_peaked":           already_peaked,
        "peak_in_window":           peak_val,
        "t_peak_in_window":         float(t_peak),
        "peak_sharpness":           float(peak_sharpness),
        "I_at_t_obs":               I_at_t_obs,
        "I_mean_window":            I_mean_window,
        "I_total_change":           I_total_change,
        "fraction_above_001":       fraction_above_001,
        "I_std_window":             I_std_window,
        "max_single_step_increase": max_single_step_increase,
        "tail_mean":                tail_mean,
        "tail_std":                 tail_std,
        "endemic_level":            endemic_level,
        "decay_rate_after_peak":    decay_rate_after_peak,
        "fraction_decreasing":      fraction_decreasing,
        "phase_switch_score":       phase_switch_score,
        "fwhm":                     fwhm,
        "autocorr_lag1":            autocorr_lag1,
    }


def build_feature_matrix(
    I_series_matrix: np.ndarray,
    t_obs: int,
    graph_features_df: pd.DataFrame | None = None,
):
    """Apply extract_features to every row, optionally appending graph features.

    Returns (X ndarray, feature_names list).
    """
    rows = [extract_features(I_series_matrix[i], t_obs)
            for i in range(len(I_series_matrix))]
    df_f = pd.DataFrame(rows)
    if graph_features_df is not None:
        df_g = graph_features_df.reset_index(drop=True)
        df_f = pd.concat([df_f, df_g], axis=1)
    return df_f.values.astype(float), list(df_f.columns)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    if not (DATA_DIR / "ml_dataset.csv").exists():
        raise FileNotFoundError(
            f"{DATA_DIR}/ml_dataset.csv not found. "
            "Run generate_ml_dataset.py first."
        )
    df    = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    I_all = np.load(DATA_DIR / "ml_I_series.npy")

    valid = df["rho_final"].notna() & ~np.any(np.isnan(I_all), axis=1)
    n_dropped = int((~valid).sum())
    df    = df[valid].reset_index(drop=True)
    I_all = I_all[valid]

    graph_feat_cols = [c for c in df.columns if c in set(GRAPH_FEATURE_NAMES)]
    print(f"Loaded {len(df):,} valid samples  ({n_dropped} failed runs dropped)")
    if graph_feat_cols:
        print(f"  Graph features found: {', '.join(graph_feat_cols)}")
    else:
        print("  No graph features in dataset — running without graph conditioning")
    return df, I_all, graph_feat_cols


# ── Pipeline factories ─────────────────────────────────────────────────────────

def rf_reg_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf",  RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)),
    ])

def ridge_pipe():
    return Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])

def rf_cls_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf",  RandomForestClassifier(
            n_estimators=N_ESTIMATORS, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])

def lr_pipe():
    return Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")),
    ])

def hgb_reg_pipe():
    # Handles NaN natively — no imputer needed
    return HistGradientBoostingRegressor(
        max_iter=300, random_state=RANDOM_STATE,
    )

def hgb_cls_pipe():
    return HistGradientBoostingClassifier(
        max_iter=300, random_state=RANDOM_STATE, class_weight="balanced",
    )


# ── Plot helper ────────────────────────────────────────────────────────────────

def save_fig(fig, name: str):
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def run_regression(df: pd.DataFrame, I_all: np.ndarray,
                   graph_df: pd.DataFrame | None = None):
    print("\n── PART 2: REGRESSION ──────────────────────────────")

    y_reg = df["rho_final"].values
    results_reg = {}

    # cached values at T_OBS_DEFAULT
    def_hgb        = None
    def_X          = None
    def_y_te       = None
    def_y_pred     = None
    def_idx_te     = None
    def_idx_tr     = None
    def_feat_names = None
    ablation_reg   = None

    for t_obs in T_OBS_VALUES:
        X, feat_names = build_feature_matrix(I_all, t_obs)

        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X, y_reg, np.arange(len(df)),
            test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=df["model_id"].values,
        )

        rf = rf_reg_pipe()
        rf.fit(X_tr, y_tr)
        y_pred_rf = rf.predict(X_te)

        hgb = hgb_reg_pipe()
        hgb.fit(X_tr, y_tr)
        y_pred_hgb = hgb.predict(X_te)

        rg = ridge_pipe()
        rg.fit(X_tr, y_tr)
        y_pred_rg = rg.predict(X_te)

        results_reg[t_obs] = {
            "rf": {
                "mae":    mean_absolute_error(y_te, y_pred_rf),
                "rmse":   float(np.sqrt(mean_squared_error(y_te, y_pred_rf))),
                "r2":     r2_score(y_te, y_pred_rf),
            },
            "hgb": {
                "mae":    mean_absolute_error(y_te, y_pred_hgb),
                "rmse":   float(np.sqrt(mean_squared_error(y_te, y_pred_hgb))),
                "r2":     r2_score(y_te, y_pred_hgb),
                "y_te":   y_te,
                "y_pred": y_pred_hgb,
                "idx_te": idx_te,
            },
            "ridge": {
                "mae":  mean_absolute_error(y_te, y_pred_rg),
                "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred_rg))),
                "r2":   r2_score(y_te, y_pred_rg),
            },
        }
        print(f"  t_obs={t_obs:3d} | HGB MAE={results_reg[t_obs]['hgb']['mae']:.4f}"
              f"  R²={results_reg[t_obs]['hgb']['r2']:.4f}"
              f"  | RF  MAE={results_reg[t_obs]['rf']['mae']:.4f}"
              f"  R²={results_reg[t_obs]['rf']['r2']:.4f}"
              f"  | Ridge MAE={results_reg[t_obs]['ridge']['mae']:.4f}")

        if t_obs == T_OBS_DEFAULT:
            def_hgb        = hgb
            def_X          = X
            def_y_te       = y_te
            def_y_pred     = y_pred_hgb
            def_idx_te     = idx_te
            def_idx_tr     = idx_tr
            def_feat_names = feat_names

    # ── Ablation: with vs without graph features at T_OBS_DEFAULT ─────────────
    if graph_df is not None:
        X_g, _ = build_feature_matrix(I_all, T_OBS_DEFAULT, graph_df)
        hgb_g  = hgb_reg_pipe()
        hgb_g.fit(X_g[def_idx_tr], y_reg[def_idx_tr])
        y_pred_g = hgb_g.predict(X_g[def_idx_te])
        ablation_reg = {
            "without_graph": {
                "mae": mean_absolute_error(def_y_te, def_y_pred),
                "r2":  r2_score(def_y_te, def_y_pred),
            },
            "with_graph": {
                "mae": mean_absolute_error(def_y_te, y_pred_g),
                "r2":  r2_score(def_y_te, y_pred_g),
            },
        }
        print(f"  ablation | w/o graph: MAE={ablation_reg['without_graph']['mae']:.4f}"
              f"  R²={ablation_reg['without_graph']['r2']:.4f}"
              f"  | with graph: MAE={ablation_reg['with_graph']['mae']:.4f}"
              f"  R²={ablation_reg['with_graph']['r2']:.4f}")

    # ── Per-model difficulty at T_OBS_DEFAULT ─────────────────────────────────
    te_model_names = df.iloc[def_idx_te]["model_name"].values
    per_model_stats = {}
    for name in MODEL_NAMES:
        mask = te_model_names == name
        if mask.sum() < 2:
            continue
        per_model_stats[name] = {
            "mae": mean_absolute_error(def_y_te[mask], def_y_pred[mask]),
            "r2":  r2_score(def_y_te[mask], def_y_pred[mask]),
        }

    # ── Feature importance via permutation (works with any estimator) ────────
    X_te_def = def_X[def_idx_te]
    perm = permutation_importance(
        def_hgb, X_te_def, def_y_te,
        n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1,
    )
    feat_imp = dict(zip(def_feat_names, perm.importances_mean))

    # ── Plot R1: MAE vs observation window ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T_OBS_VALUES,
            [results_reg[t]["hgb"]["mae"]   for t in T_OBS_VALUES],
            "^-",  color="#2E7D32", lw=2, label="HGB (best)")
    ax.plot(T_OBS_VALUES,
            [results_reg[t]["rf"]["mae"]    for t in T_OBS_VALUES],
            "o-",  color="#1565C0", lw=2, label="Random Forest")
    ax.plot(T_OBS_VALUES,
            [results_reg[t]["ridge"]["mae"] for t in T_OBS_VALUES],
            "s--", color="#B71C1C", lw=2, label="Ridge (baseline)")
    ax.set_xlabel("Observation window t_obs", fontsize=FS_LABEL)
    ax.set_ylabel("MAE",                       fontsize=FS_LABEL)
    ax.set_title("Regression Performance vs Observation Window", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "R1_mae_vs_window.png")

    # ── Plot R2: Predicted vs True (t_obs = T_OBS_DEFAULT) ───────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in MODEL_NAMES:
        mask = te_model_names == name
        if mask.sum():
            ax.scatter(def_y_te[mask], def_y_pred[mask],
                       c=MODEL_COLORS[name], label=name,
                       s=8, alpha=0.6, linewidths=0)
    lo = min(def_y_te.min(), def_y_pred.min()) - 0.01
    hi = max(def_y_te.max(), def_y_pred.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.text(0.05, 0.90,
            f"MAE = {results_reg[T_OBS_DEFAULT]['hgb']['mae']:.4f}\n"
            f"R²  = {results_reg[T_OBS_DEFAULT]['hgb']['r2']:.4f}",
            transform=ax.transAxes, fontsize=FS_LEGEND,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.set_xlabel("True ρ_final",      fontsize=FS_LABEL)
    ax.set_ylabel("Predicted ρ_final", fontsize=FS_LABEL)
    ax.set_title(f"Predicted vs True Final Epidemic Size (t_obs={T_OBS_DEFAULT}, HGB)",
                 fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND - 1, markerscale=2,
              loc="lower right", ncol=2, framealpha=0.85)
    fig.tight_layout()
    save_fig(fig, "R2_pred_vs_true.png")

    # ── Plot R3: Per-model MAE bar chart ──────────────────────────────────────
    sorted_models = sorted(per_model_stats.items(),
                           key=lambda x: x[1]["mae"], reverse=True)
    m_names  = [m[0] for m in sorted_models]
    m_maes   = [m[1]["mae"] for m in sorted_models]
    m_colors = [MODEL_COLORS[m] for m in m_names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(m_names, m_maes, color=m_colors)
    for bar, val in zip(bars, m_maes):
        ax.text(bar.get_width() + max(m_maes) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=FS_TICK)
    ax.set_xlabel("MAE",   fontsize=FS_LABEL)
    ax.set_title("Prediction Difficulty by Model", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(0, max(m_maes) * 1.20)
    fig.tight_layout()
    save_fig(fig, "R3_per_model_mae.png")

    # ── Plot R4: Feature importance (top 12) ──────────────────────────────────
    imp_sorted = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:12]
    fi_names  = [x[0] for x in imp_sorted]
    fi_vals   = [x[1] for x in imp_sorted]
    fi_colors = [CATEGORY_COLORS[FEATURE_CATEGORIES.get(f, "graph")] for f in fi_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi_names[::-1], fi_vals[::-1], color=fi_colors[::-1])
    present_cats = {FEATURE_CATEGORIES.get(f, "graph") for f in fi_names}
    handles = [Patch(color=CATEGORY_COLORS[c], label=c.capitalize())
               for c in CATEGORY_COLORS if c in present_cats]
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="lower right")
    ax.set_xlabel("Importance", fontsize=FS_LABEL)
    ax.set_title("Feature Importance for Epidemic Size Prediction", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    fig.tight_layout()
    save_fig(fig, "R4_feature_importance.png")

    # ── Plot R5: Ablation — with vs without graph features ────────────────────
    if ablation_reg is not None:
        metrics   = ["MAE (lower=better)", "R² (higher=better)"]
        wo_vals   = [ablation_reg["without_graph"]["mae"],
                     ablation_reg["without_graph"]["r2"]]
        wg_vals   = [ablation_reg["with_graph"]["mae"],
                     ablation_reg["with_graph"]["r2"]]
        x         = np.arange(len(metrics))
        width     = 0.35
        fig, ax   = plt.subplots(figsize=(7, 4))
        bars1 = ax.bar(x - width / 2, wo_vals, width, label="Time-series only",
                       color="#1565C0")
        bars2 = ax.bar(x + width / 2, wg_vals, width, label="+ Graph features",
                       color="#FF6F00")
        for bars in (bars1, bars2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{bar.get_height():.4f}",
                        ha="center", va="bottom", fontsize=FS_TICK)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=FS_LABEL)
        ax.set_title(f"Ablation: Graph Feature Contribution (t_obs={T_OBS_DEFAULT}, HGB)",
                     fontsize=FS_TITLE)
        ax.tick_params(labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save_fig(fig, "R5_ablation_graph_features.png")

    print("  Regression complete.")
    return results_reg, per_model_stats, feat_imp, ablation_reg


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def run_classification(df: pd.DataFrame, I_all: np.ndarray,
                       graph_df: pd.DataFrame | None = None):
    print("\n── PART 3: CLASSIFICATION ──────────────────────────")

    y_cls = df["model_id"].values
    results_cls = {}

    # cached at T_OBS_DEFAULT
    def_X          = None
    def_y_te       = None
    def_y_pred     = None
    def_idx_tr     = None
    def_idx_te     = None
    def_y_tr       = None
    ablation_cls   = None

    for t_obs in T_OBS_VALUES:
        X, _ = build_feature_matrix(I_all, t_obs)

        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X, y_cls, np.arange(len(df)),
            test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_cls,
        )

        hgbc = hgb_cls_pipe()
        hgbc.fit(X_tr, y_tr)
        y_pred_hgbc = hgbc.predict(X_te)

        rfc = rf_cls_pipe()
        rfc.fit(X_tr, y_tr)
        y_pred_rfc = rfc.predict(X_te)

        lr = lr_pipe()
        lr.fit(X_tr, y_tr)
        y_pred_lr = lr.predict(X_te)

        results_cls[t_obs] = {
            "hgb": {
                "acc": accuracy_score(y_te, y_pred_hgbc),
                "f1":  f1_score(y_te, y_pred_hgbc, average="macro"),
            },
            "rf": {
                "acc": accuracy_score(y_te, y_pred_rfc),
                "f1":  f1_score(y_te, y_pred_rfc, average="macro"),
            },
            "lr": {
                "acc": accuracy_score(y_te, y_pred_lr),
                "f1":  f1_score(y_te, y_pred_lr, average="macro"),
            },
        }
        print(f"  t_obs={t_obs:3d} | HGB acc={results_cls[t_obs]['hgb']['acc']:.4f}"
              f"  F1={results_cls[t_obs]['hgb']['f1']:.4f}"
              f"  | RF  acc={results_cls[t_obs]['rf']['acc']:.4f}"
              f"  | LR  acc={results_cls[t_obs]['lr']['acc']:.4f}")

        if t_obs == T_OBS_DEFAULT:
            def_X      = X
            def_y_te   = y_te
            def_y_pred = y_pred_hgbc
            def_idx_tr = idx_tr
            def_idx_te = idx_te
            def_y_tr   = y_tr

    # ── Ablation: with vs without graph features at T_OBS_DEFAULT ─────────────
    if graph_df is not None:
        X_g, _ = build_feature_matrix(I_all, T_OBS_DEFAULT, graph_df)
        hgbc_g = hgb_cls_pipe()
        hgbc_g.fit(X_g[def_idx_tr], def_y_tr)
        y_pred_g = hgbc_g.predict(X_g[def_idx_te])
        ablation_cls = {
            "without_graph": {
                "acc": accuracy_score(def_y_te, def_y_pred),
                "f1":  f1_score(def_y_te, def_y_pred, average="macro"),
            },
            "with_graph": {
                "acc": accuracy_score(def_y_te, y_pred_g),
                "f1":  f1_score(def_y_te, y_pred_g, average="macro"),
            },
        }
        print(f"  ablation | w/o graph: acc={ablation_cls['without_graph']['acc']:.4f}"
              f"  F1={ablation_cls['without_graph']['f1']:.4f}"
              f"  | with graph: acc={ablation_cls['with_graph']['acc']:.4f}"
              f"  F1={ablation_cls['with_graph']['f1']:.4f}")

    n_cls  = len(MODEL_NAMES)
    labels = [ID_TO_MODEL[i] for i in range(n_cls)]

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(def_y_te, def_y_pred, normalize="true")

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    report = classification_report(def_y_te, def_y_pred, output_dict=True)
    per_class_f1 = {
        ID_TO_MODEL[int(k)]: v["f1-score"]
        for k, v in report.items() if k.isdigit()
    }

    # ── 5 most confused pairs ─────────────────────────────────────────────────
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0.0)
    flat    = cm_off.ravel()
    top5    = np.argsort(flat)[::-1][:5]
    confused_pairs = [
        (ID_TO_MODEL[i // n_cls], ID_TO_MODEL[i % n_cls], float(flat[i]))
        for i in top5
    ]

    # ── Cross-network generalisation ──────────────────────────────────────────
    cross_net_accs = {}
    for held_out in NETWORKS:
        tr_mask  = (df["network_type"] != held_out).values
        te_mask  = (df["network_type"] == held_out).values
        hgbc_cn  = hgb_cls_pipe()
        hgbc_cn.fit(def_X[tr_mask], y_cls[tr_mask])
        cross_net_accs[held_out] = accuracy_score(
            y_cls[te_mask], hgbc_cn.predict(def_X[te_mask])
        )
        print(f"  cross-network (held-out={held_out:8s}): "
              f"acc={cross_net_accs[held_out]:.4f}")

    overall_acc = results_cls[T_OBS_DEFAULT]["hgb"]["acc"]

    # ── Plot C1: Accuracy vs window ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T_OBS_VALUES,
            [results_cls[t]["hgb"]["acc"] for t in T_OBS_VALUES],
            "^-",  color="#2E7D32", lw=2, label="HGB (best)")
    ax.plot(T_OBS_VALUES,
            [results_cls[t]["rf"]["acc"] for t in T_OBS_VALUES],
            "o-",  color="#1565C0", lw=2, label="Random Forest")
    ax.plot(T_OBS_VALUES,
            [results_cls[t]["lr"]["acc"] for t in T_OBS_VALUES],
            "s--", color="#B71C1C", lw=2, label="Logistic Regression")
    ax.axhline(0.1, color="gray", linestyle=":", lw=1.5, label="Chance (0.1)")
    ax.set_xlabel("Observation window t_obs", fontsize=FS_LABEL)
    ax.set_ylabel("Accuracy",                  fontsize=FS_LABEL)
    ax.set_title("Classification Accuracy vs Observation Window", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "C1_accuracy_vs_window.png")

    # ── Plot C2: Confusion matrix heatmap ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.4, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=FS_LABEL)
    ax.set_ylabel("True",      fontsize=FS_LABEL)
    ax.set_title(f"Model Identification Confusion Matrix (t_obs={T_OBS_DEFAULT}, HGB)",
                 fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    fig.tight_layout()
    save_fig(fig, "C2_confusion_matrix.png")

    # ── Plot C3: Cross-network generalisation ─────────────────────────────────
    net_colors = {
        "ER": "#1565C0", "RGG": "#2E7D32", "Lattice": "#E65100",
        "Facebook": "#6A1B9A", "GitHub": "#00838F",
    }
    net_labels = list(cross_net_accs.keys())
    net_accs   = [cross_net_accs[n] for n in net_labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(net_labels, net_accs,
                  color=[net_colors[n] for n in net_labels], width=0.5)
    ax.axhline(overall_acc, color="gray", linestyle="--", lw=1.5,
               label=f"t_obs=15 accuracy ({overall_acc:.3f})")
    for bar, val in zip(bars, net_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", fontsize=FS_TICK)
    ax.set_ylabel("Accuracy", fontsize=FS_LABEL)
    ax.set_title("Classification Generalisation Across Network Types", fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.set_ylim(0, min(1.2, max(net_accs) * 1.25))
    fig.tight_layout()
    save_fig(fig, "C3_cross_network.png")

    # ── Plot C4: Per-class F1 ─────────────────────────────────────────────────
    f1_names  = [n for n in MODEL_NAMES if n in per_class_f1]
    f1_vals   = [per_class_f1[n] for n in f1_names]
    f1_colors = [MODEL_COLORS[n] for n in f1_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(f1_names, f1_vals, color=f1_colors)
    for i, val in enumerate(f1_vals):
        ax.text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=FS_TICK - 1)
    ax.set_ylabel("F1 Score", fontsize=FS_LABEL)
    ax.set_title(f"Per-Model Classification F1 Score (t_obs={T_OBS_DEFAULT}, HGB)",
                 fontsize=FS_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(0, 1.18)
    fig.tight_layout()
    save_fig(fig, "C4_per_class_f1.png")

    print("  Classification complete.")
    return results_cls, per_class_f1, cross_net_accs, confused_pairs, ablation_cls


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(
    results_reg, per_model_stats, feat_imp, ablation_reg,
    results_cls, per_class_f1, cross_net_accs, confused_pairs, ablation_cls,
):
    print("\n── PART 4: SUMMARY ─────────────────────────────────")

    best_mae_t = min(T_OBS_VALUES, key=lambda t: results_reg[t]["hgb"]["mae"])
    best_mae   = results_reg[best_mae_t]["hgb"]["mae"]
    best_r2_t  = max(T_OBS_VALUES, key=lambda t: results_reg[t]["hgb"]["r2"])
    best_r2    = results_reg[best_r2_t]["hgb"]["r2"]
    hardest    = max(per_model_stats.items(), key=lambda x: x[1]["mae"])
    easiest    = min(per_model_stats.items(), key=lambda x: x[1]["mae"])
    top3_feats = [f[0] for f in
                  sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:3]]

    best_cls_t   = max(T_OBS_VALUES, key=lambda t: results_cls[t]["hgb"]["acc"])
    best_cls_acc = results_cls[best_cls_t]["hgb"]["acc"]
    best_net     = max(cross_net_accs.items(), key=lambda x: x[1])
    worst_net    = min(cross_net_accs.items(), key=lambda x: x[1])
    top_pair     = confused_pairs[0]

    lines = [
        "=" * 60,
        "REGRESSION SUMMARY:",
        f"  Best MAE: {best_mae:.3f} achieved at t_obs={best_mae_t}",
        f"  Best R²:  {best_r2:.3f} achieved at t_obs={best_r2_t}",
        f"  Hardest model to predict: {hardest[0]} (MAE={hardest[1]['mae']:.3f})",
        f"  Easiest model to predict: {easiest[0]} (MAE={easiest[1]['mae']:.3f})",
        f"  Top 3 most important features: {', '.join(top3_feats)}",
        "",
        "CLASSIFICATION SUMMARY:",
        f"  Best accuracy: {best_cls_acc * 100:.1f}% achieved at t_obs={best_cls_t}",
        f"  Chance level: 10.0%",
        f"  Most confused pair: {top_pair[0]} mistaken for {top_pair[1]}"
        f" ({top_pair[2] * 100:.1f}% of the time)",
        f"  Best generalising network:  {best_net[0]}  (accuracy={best_net[1]*100:.1f}%)",
        f"  Worst generalising network: {worst_net[0]}  (accuracy={worst_net[1]*100:.1f}%)",
    ]

    if ablation_reg is not None and ablation_cls is not None:
        mae_delta = (ablation_reg["without_graph"]["mae"]
                     - ablation_reg["with_graph"]["mae"])
        acc_delta = (ablation_cls["with_graph"]["acc"]
                     - ablation_cls["without_graph"]["acc"])
        lines += [
            "",
            "GRAPH FEATURE ABLATION (t_obs=30):",
            f"  Regression  — w/o graph: MAE={ablation_reg['without_graph']['mae']:.4f}"
            f"  R²={ablation_reg['without_graph']['r2']:.4f}",
            f"  Regression  — w/  graph: MAE={ablation_reg['with_graph']['mae']:.4f}"
            f"  R²={ablation_reg['with_graph']['r2']:.4f}"
            f"  (ΔMAE={mae_delta:+.4f})",
            f"  Classification — w/o graph: acc={ablation_cls['without_graph']['acc']:.4f}"
            f"  F1={ablation_cls['without_graph']['f1']:.4f}",
            f"  Classification — w/  graph: acc={ablation_cls['with_graph']['acc']:.4f}"
            f"  F1={ablation_cls['with_graph']['f1']:.4f}"
            f"  (Δacc={acc_delta:+.4f})",
        ]

    lines.append("=" * 60)
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "summary.txt").write_text(text)
    print(f"  Saved {RESULTS_DIR / 'summary.txt'}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — PERSIST MODELS FOR DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════

def save_models(df: pd.DataFrame, I_all: np.ndarray,
                graph_df: pd.DataFrame | None = None):
    """
    Retrain the best regressor and classifier on the FULL dataset at
    T_OBS_DEFAULT, then serialise artefacts to ml_data/:

      rf_regressor.pkl        — HGB regressor
      rf_classifier.pkl       — HGB classifier
      label_encoder.pkl       — dict {model_id: model_name}
      feature_names.pkl       — ordered list of feature names
      graph_feature_means.pkl — mean graph feature values (fallback for live app)
    """
    print("\n── PART 5: SAVING MODELS ───────────────────────────")

    X, feat_names = build_feature_matrix(I_all, T_OBS_DEFAULT, graph_df)
    y_reg = df["rho_final"].values
    y_cls = df["model_id"].values

    reg = hgb_reg_pipe()
    reg.fit(X, y_reg)

    clf = hgb_cls_pipe()
    clf.fit(X, y_cls)

    label_encoder = {int(v): k for k, v in MODEL_ID_MAP.items()}

    joblib.dump(reg,           DATA_DIR / "rf_regressor.pkl")
    joblib.dump(clf,           DATA_DIR / "rf_classifier.pkl")
    joblib.dump(label_encoder, DATA_DIR / "label_encoder.pkl")
    joblib.dump(feat_names,    DATA_DIR / "feature_names.pkl")

    print(f"  rf_regressor.pkl  — HGB regressor  trained on {len(X):,} samples")
    print(f"  rf_classifier.pkl — HGB classifier trained on {len(X):,} samples")
    print(f"  label_encoder.pkl — {label_encoder}")
    print(f"  feature_names.pkl — {len(feat_names)} features: {', '.join(feat_names)}")

    if graph_df is not None:
        graph_feat_means = {col: float(graph_df[col].mean()) for col in graph_df.columns}
        joblib.dump(graph_feat_means, DATA_DIR / "graph_feature_means.pkl")
        print(f"  graph_feature_means.pkl — means for {len(graph_feat_means)} graph features")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df, I_all, graph_feat_cols = load_data()
    graph_df = df[graph_feat_cols].copy() if graph_feat_cols else None

    print("\n── PART 1: FEATURE EXTRACTION ──────────────────────")
    sample = extract_features(I_all[0], T_OBS_DEFAULT)
    print(f"  {len(sample)} time-series features: {', '.join(sample.keys())}")

    results_reg, per_model_stats, feat_imp, ablation_reg = run_regression(
        df, I_all, graph_df
    )
    results_cls, per_class_f1, cross_net_accs, confused_pairs, ablation_cls = (
        run_classification(df, I_all, graph_df)
    )
    write_summary(
        results_reg, per_model_stats, feat_imp, ablation_reg,
        results_cls, per_class_f1, cross_net_accs, confused_pairs, ablation_cls,
    )
    save_models(df, I_all, graph_df)

    print(f"\nTotal runtime: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
