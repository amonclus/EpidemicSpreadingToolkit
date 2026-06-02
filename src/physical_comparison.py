#!/usr/bin/env python3
"""
Compare physical model fits (SIR, SIS) against DL/ML pipeline predictions.

BP and WTM are excluded: they are complex-contagion / percolation-type processes
for which no mean-field ODE approximation is principled. Fitting a sigmoid proxy
to a threshold cascade produces a meaningless effective parameter with no
correspondence to the true network-level threshold.

Run from src/:
    python physical_comparison.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.integrate import odeint
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR     = Path(__file__).parent
DATA_DIR    = SRC_DIR / "ml" / "ml_data"
RESULTS_DIR = SRC_DIR / "results"

# Ensure the src/ directory is importable (needed for joblib-loaded DL wrappers)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID_MAP = {
    "SIR": 0, "SIS": 1, "BP": 2, "WTM": 3,
    "H1":  4, "H2":  5, "H3": 6, "H4":  7, "H5": 8, "H6": 9,
}
ID_TO_MODEL    = {v: k for k, v in MODEL_ID_MAP.items()}
PHYSICAL_MODELS = ["SIR", "SIS"]
K_VALUES        = [10, 20, 30]

# Long integration grid used when running fitted models to steady state
_T_LONG = np.linspace(0, 500, 5001)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — DATA LOADING & SPLIT REPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    df       = pd.read_csv(DATA_DIR / "ml_dataset.csv")
    I_series = np.load(DATA_DIR / "ml_I_series.npy").astype(np.float32)
    return df, I_series


def get_split_indices(model_ids, seed=42):
    """Replicate EpidemicDataset._stratified_split exactly (70 / 15 / 15)."""
    n       = len(model_ids)
    indices = np.arange(n)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, model_ids))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_idx, model_ids[temp_idx]))

    return train_idx, temp_idx[val_idx], temp_idx[test_idx]


# ══════════════════════════════════════════════════════════════════════════════
# 2 — HANDCRAFTED FEATURES (mirrors ml/dl/features.py)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_one(series, t_obs):
    """Return a (22,) float32 feature vector from the first t_obs timesteps."""
    window = np.asarray(series[:t_obs], dtype=float)
    n      = len(window)
    eps    = 1e-8

    log_I             = np.log(window + eps)
    early_growth_rate = float(np.polyfit(np.arange(n), log_I, 1)[0]) if n >= 2 else 0.0
    log_amplification = float(np.log((window[-1] + eps) / (window[0] + eps)))
    doubled           = np.where(window >= 2.0 * (window[0] + eps))[0]
    doubling_time     = float(doubled[0]) if len(doubled) else np.nan
    curvature         = float(np.mean(np.diff(window, 2))) if n >= 3 else 0.0
    peak_val          = float(window.max())
    t_peak            = int(window.argmax())
    already_peaked    = 1.0 if window[-1] < peak_val else 0.0
    peak_sharpness    = peak_val / (t_peak + 1)
    I_at_t_obs        = float(window[-1])
    I_mean_window     = float(window.mean())
    I_total_change    = float(window[-1] - window[0])
    fraction_above    = float(np.mean(window > 0.01))
    I_std_window      = float(window.std())
    diffs             = np.diff(window)
    max_single_step   = float(max(0.0, diffs.max())) if len(diffs) else 0.0
    tail_start        = max(1, int(n * 0.8))
    tail              = window[tail_start:]
    tail_mean         = float(tail.mean()) if len(tail) else float(window[-1])
    tail_std          = float(tail.std())  if len(tail) else 0.0
    endemic_level     = float(window[-1] / (peak_val + eps))
    post_peak         = window[t_peak:]
    decay_rate        = (
        float(np.polyfit(np.arange(len(post_peak)), post_peak, 1)[0])
        if len(post_peak) >= 2 else 0.0
    )
    fraction_dec      = float(np.mean(diffs < 0)) if len(diffs) else 0.0
    phase_switch      = float(np.max(np.abs(np.diff(window, 2)))) if n >= 5 else 0.0
    half_max          = peak_val / 2.0
    above_half        = np.where(window >= half_max)[0]
    fwhm              = float(above_half[-1] - above_half[0] + 1) if len(above_half) >= 2 else 0.0
    autocorr          = (
        float(np.corrcoef(window[:-1], window[1:])[0, 1])
        if n >= 3 and window.std() > eps else 1.0
    )

    return np.array([
        early_growth_rate, log_amplification, doubling_time, curvature,
        already_peaked, peak_val, float(t_peak), peak_sharpness,
        I_at_t_obs, I_mean_window, I_total_change, fraction_above,
        I_std_window, max_single_step, tail_mean, tail_std, endemic_level,
        decay_rate, fraction_dec, phase_switch, fwhm, autocorr,
    ], dtype=np.float32)


def build_feature_matrix(I_series, t_obs):
    X = np.stack([_extract_one(I_series[i], t_obs) for i in range(len(I_series))])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3 — HGB BASELINE (re-fitted on the canonical train split)
# ══════════════════════════════════════════════════════════════════════════════

def fit_hgb(I_train, y_train, t_obs):
    X = build_feature_matrix(I_train, t_obs)
    hgb = HistGradientBoostingRegressor(max_iter=300, random_state=42)
    hgb.fit(X, y_train)
    return hgb


def predict_hgb(hgb, I_batch, t_obs):
    return hgb.predict(build_feature_matrix(I_batch, t_obs))


# ══════════════════════════════════════════════════════════════════════════════
# 4 — DL / CNN INFERENCE (via DLDynamicRegressor pkl)
# ══════════════════════════════════════════════════════════════════════════════

def load_dl_regressor():
    pkl = DATA_DIR / "dl_regressor.pkl"
    if not pkl.exists():
        return None
    # The pkl was saved from inside src/ml/, so its classes are recorded as
    # 'dl.wrappers.*' rather than 'ml.dl.wrappers.*'.  Adding src/ml/ to
    # sys.path makes 'import dl' resolve to the same package.
    ml_dir = str(SRC_DIR / "ml")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    try:
        reg = joblib.load(pkl)
        print(f"  Loaded DL regressor  (t_obs available: {reg.t_obs_available})")
        return reg
    except Exception as exc:
        print(f"  WARNING: could not load dl_regressor.pkl — {exc}")
        return None


def predict_dl(dl_reg, I_batch, k):
    """Pass the first k timesteps; DLDynamicRegressor selects the matching model."""
    return dl_reg.predict(I_batch[:, :k])


# ══════════════════════════════════════════════════════════════════════════════
# 5 — PHYSICAL MODEL ODESS AND FITTERS
# ══════════════════════════════════════════════════════════════════════════════

# ── ODE right-hand sides ─────────────────────────────────────────────────────

def _sir_rhs(y, t, beta, gamma):
    S, I, R = y
    inf = beta * S * I
    rec = gamma * I
    return [-inf, inf - rec, rec]


def _sis_rhs(y, t, beta, gamma):
    S, I = y
    inf = beta * S * I
    rec = gamma * I
    return [-inf + rec, inf - rec]


# ── Long-horizon integrations to get rho_final ───────────────────────────────

def _rho_sir(beta, gamma, I0):
    S0  = max(0.0, 1.0 - I0)
    sol = odeint(_sir_rhs, [S0, I0, 0.0], _T_LONG, args=(beta, gamma))
    return float(sol[-1, 2])                 # R(∞)


def _rho_sis(beta, gamma, I0):
    S0  = max(0.0, 1.0 - I0)
    sol = odeint(_sis_rhs, [S0, I0], _T_LONG, args=(beta, gamma))
    return float(sol[:, 1].max())            # peak I


# ── Per-model curve fitters ───────────────────────────────────────────────────

def fit_sir(obs, k):
    t_grid = np.arange(k, dtype=float)
    I0     = float(max(obs[0], 1e-6))

    def _curve(t, beta, gamma):
        S0  = max(0.0, 1.0 - I0)
        sol = odeint(_sir_rhs, [S0, I0, 0.0], t_grid, args=(beta, gamma))
        return np.clip(sol[:, 1], 0.0, 1.0)

    try:
        popt, _ = curve_fit(
            _curve, t_grid, obs,
            p0=[0.3, 0.1], bounds=([0.01, 0.01], [5.0, 5.0]),
            maxfev=3000, method="trf",
        )
        rho = _rho_sir(popt[0], popt[1], I0)
        return (None, True) if not (0.0 <= rho <= 1.0) else (rho, False)
    except Exception:
        return None, True


def fit_sis(obs, k):
    t_grid = np.arange(k, dtype=float)
    I0     = float(max(obs[0], 1e-6))

    def _curve(t, beta, gamma):
        S0  = max(0.0, 1.0 - I0)
        sol = odeint(_sis_rhs, [S0, I0], t_grid, args=(beta, gamma))
        return np.clip(sol[:, 1], 0.0, 1.0)

    try:
        popt, _ = curve_fit(
            _curve, t_grid, obs,
            p0=[0.3, 0.1], bounds=([0.01, 0.01], [5.0, 5.0]),
            maxfev=3000, method="trf",
        )
        rho = _rho_sis(popt[0], popt[1], I0)
        return (None, True) if not (0.0 <= rho <= 1.0) else (rho, False)
    except Exception:
        return None, True


_FIT_FUNC = {"SIR": fit_sir, "SIS": fit_sis}


# ══════════════════════════════════════════════════════════════════════════════
# 6 — RESULTS TABLE PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def _mae_safe(y_true, y_pred):
    """MAE ignoring NaN pairs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    return mean_absolute_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else np.nan


def print_table(results_df, k):
    kdf    = results_df[results_df["k"] == k]
    has_dl = kdf["rho_dl"].notna().any()
    has_ml = kdf["rho_ml"].notna().any()

    dl_hdr = "DL (CNN)" if has_dl else "DL (N/A)"
    ml_hdr = "Classical ML" if has_ml else "Classical ML(N/A)"

    print(f"\nPhysical model fit vs ML/DL pipeline — MAE at k = {k} observations")
    print("=" * 71)
    print(f"{'Model':<8} | {'Physical fit':>14} | {dl_hdr:>10} | {ml_hdr:>14} | {'N series':>8}")
    print("-" * 71)

    fail_rates = {}
    for mname in PHYSICAL_MODELS:
        mdf      = kdf[kdf["model_label"] == mname]
        n_total  = len(mdf)
        n_failed = int(mdf["fit_failed"].sum())
        fail_rates[mname] = n_failed / n_total if n_total > 0 else 0.0

        valid_phys = mdf[~mdf["fit_failed"]]
        phys_mae = _mae_safe(valid_phys["rho_true"], valid_phys["rho_phys"])
        dl_mae   = _mae_safe(mdf["rho_true"], mdf["rho_dl"]) if has_dl else np.nan
        ml_mae   = _mae_safe(mdf["rho_true"], mdf["rho_ml"]) if has_ml else np.nan

        dl_str = f"{dl_mae:.3f}" if not np.isnan(dl_mae) else "  N/A  "
        ml_str = f"{ml_mae:.3f}" if not np.isnan(ml_mae) else "    N/A    "
        print(f"{mname:<8} | {phys_mae:>14.3f} | {dl_str:>10} | {ml_str:>14} | {n_total:>8}")

    pool_valid = kdf[~kdf["fit_failed"]]
    pool_phys  = _mae_safe(pool_valid["rho_true"], pool_valid["rho_phys"])
    pool_dl    = _mae_safe(kdf["rho_true"], kdf["rho_dl"]) if has_dl else np.nan
    pool_ml    = _mae_safe(kdf["rho_true"], kdf["rho_ml"]) if has_ml else np.nan

    dl_str = f"{pool_dl:.3f}" if not np.isnan(pool_dl) else "  N/A  "
    ml_str = f"{pool_ml:.3f}" if not np.isnan(pool_ml) else "    N/A    "
    print("-" * 71)
    print(f"{'Pooled':<8} | {pool_phys:>14.3f} | {dl_str:>10} | {ml_str:>14} | {len(kdf):>8}")
    print("=" * 71)
    fail_line = "  ".join(f"{m} {fail_rates[m]*100:.1f}%" for m in PHYSICAL_MODELS)
    print(f"Fit failure rate: {fail_line}")


# ══════════════════════════════════════════════════════════════════════════════
# 7 — FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def make_figure(results_df):
    has_dl = results_df["rho_dl"].notna().any()
    has_ml = results_df["rho_ml"].notna().any()

    if not has_dl and not has_ml:
        print("\nWARNING: DL and ML predictions unavailable — figure shows physical fits only.")

    COLORS = {
        "Physical fit": "#E53935",
        "DL (CNN)":     "#43A047",
        "Classical ML": "#1E88E5",
    }

    fig   = plt.figure(figsize=(22, 7))
    gs    = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.38)
    x_labels = PHYSICAL_MODELS + ["Pooled"]
    x_pos    = np.arange(len(x_labels))
    bar_w    = 0.22

    # Determine how many bar groups are active so we can centre them
    active  = ["Physical fit"] + (["DL (CNN)"] if has_dl else []) + (["Classical ML"] if has_ml else [])
    n_bars  = len(active)
    offsets = {name: (i - (n_bars - 1) / 2.0) * bar_w for i, name in enumerate(active)}

    for col, k in enumerate(K_VALUES):
        ax  = fig.add_subplot(gs[col])
        kdf = results_df[results_df["k"] == k]

        mae_phys, mae_dl, mae_ml = [], [], []
        for label in x_labels:
            sub   = kdf if label == "Pooled" else kdf[kdf["model_label"] == label]
            valid = sub[~sub["fit_failed"]]
            mae_phys.append(_mae_safe(valid["rho_true"], valid["rho_phys"]))
            mae_dl.append(_mae_safe(sub["rho_true"], sub["rho_dl"]))
            mae_ml.append(_mae_safe(sub["rho_true"], sub["rho_ml"]))

        ax.bar(x_pos + offsets["Physical fit"], mae_phys, bar_w,
               color=COLORS["Physical fit"], label="Physical fit", alpha=0.85, zorder=3)
        if has_dl:
            ax.bar(x_pos + offsets["DL (CNN)"], mae_dl, bar_w,
                   color=COLORS["DL (CNN)"], label="DL (CNN)", alpha=0.85, zorder=3)
        if has_ml:
            ax.bar(x_pos + offsets["Classical ML"], mae_ml, bar_w,
                   color=COLORS["Classical ML"], label="Classical ML", alpha=0.85, zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel("MAE", fontsize=10)
        ax.set_title(f"k = {k} observations", fontsize=11, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    # Scatter panel: k=30, pooled
    ax_s = fig.add_subplot(gs[3])
    k30  = results_df[results_df["k"] == 30]

    valid_phys = k30[~k30["fit_failed"]]
    ax_s.scatter(valid_phys["rho_true"], valid_phys["rho_phys"],
                 c=COLORS["Physical fit"], alpha=0.3, s=6, label="Physical fit", rasterized=True)
    if has_dl:
        dl_valid = k30[k30["rho_dl"].notna()]
        ax_s.scatter(dl_valid["rho_true"], dl_valid["rho_dl"],
                     c=COLORS["DL (CNN)"], alpha=0.3, s=6, label="DL (CNN)", rasterized=True)
    if has_ml:
        ml_valid = k30[k30["rho_ml"].notna()]
        ax_s.scatter(ml_valid["rho_true"], ml_valid["rho_ml"],
                     c=COLORS["Classical ML"], alpha=0.3, s=6, label="Classical ML", rasterized=True)

    ax_s.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="y = x")
    ax_s.set_xlabel(r"$\rho_\mathrm{true}$", fontsize=11)
    ax_s.set_ylabel(r"$\rho_\mathrm{predicted}$", fontsize=11)
    ax_s.set_title("Scatter — k = 30, pooled", fontsize=11, fontweight="bold")
    ax_s.legend(fontsize=8, markerscale=2.5)
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(0, 1)
    ax_s.grid(alpha=0.3)

    fig.suptitle(
        r"Physical model fit vs ML/DL pipeline — $\rho_\mathrm{final}$ prediction",
        fontsize=13, y=1.02,
    )

    out = RESULTS_DIR / "physical_vs_ml_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 8 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading dataset...")
    df, I_series = load_data()
    model_ids = df["model_id"].values.astype(np.int64)
    rho_all   = df["rho_final"].values.astype(np.float32)

    # ── Replicate DL split ────────────────────────────────────────────────────
    print("Computing stratified 70/15/15 split (seed=42)...")
    train_idx, _val_idx, test_idx = get_split_indices(model_ids, seed=42)

    # Restrict test to SIR and SIS only
    phys_id_set = {MODEL_ID_MAP[m] for m in PHYSICAL_MODELS}
    test_phys   = test_idx[np.isin(model_ids[test_idx], list(phys_id_set))]

    print(f"\nTest set — total: {len(test_idx)}, physical models only: {len(test_phys)}")
    for mname in PHYSICAL_MODELS:
        n = int((model_ids[test_phys] == MODEL_ID_MAP[mname]).sum())
        print(f"  {mname}: {n} series")

    # ── DL regressor (loaded once, reused across all k) ───────────────────────
    print("\nLoading DL regressor pkl...")
    dl_reg = load_dl_regressor()
    if dl_reg is None:
        print("  DL predictions will be skipped.")

    # ── Main loop over observation windows ───────────────────────────────────
    all_rows = []

    for k in K_VALUES:
        print(f"\n{'='*60}")
        print(f" k = {k} — fitting HGB on train split...")

        hgb = fit_hgb(I_series[train_idx], rho_all[train_idx], k)

        print(f" k = {k} — running per-series physical fits + predictions...")
        n_series = len(test_phys)

        for step, global_idx in enumerate(test_phys):
            series   = I_series[global_idx]            # (300,) float32, already I/N
            rho_true = float(rho_all[global_idx])
            mname    = ID_TO_MODEL[int(model_ids[global_idx])]
            obs      = np.clip(series[:k].astype(float), 0.0, 1.0)

            # Physical fit
            rho_phys, failed = _FIT_FUNC[mname](obs, k)

            # HGB prediction
            rho_ml = float(predict_hgb(hgb, series[np.newaxis, :], k)[0])

            # DL prediction
            rho_dl = None
            if dl_reg is not None:
                try:
                    rho_dl = float(predict_dl(dl_reg, series[np.newaxis, :], k)[0])
                except Exception:
                    rho_dl = None

            all_rows.append({
                "series_id":   int(global_idx),
                "model_label": mname,
                "rho_true":    rho_true,
                "k":           k,
                "rho_phys":    rho_phys if not failed else np.nan,
                "rho_dl":      rho_dl,
                "rho_ml":      rho_ml,
                "fit_failed":  failed,
            })

            if (step + 1) % 500 == 0:
                print(f"   {step+1}/{n_series} done")

        print(f" k = {k} — done.")

    results_df = pd.DataFrame(all_rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "physical_vs_ml_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nPer-series results saved → {csv_path}")

    # ── Print MAE tables ──────────────────────────────────────────────────────
    for k in K_VALUES:
        print_table(results_df, k)

    # ── Figure ────────────────────────────────────────────────────────────────
    make_figure(results_df)


if __name__ == "__main__":
    main()
