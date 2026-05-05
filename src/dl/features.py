"""
Replicates extract_features() from ml_analysis.py so the DL pipeline can
compute the same 22 handcrafted features and inject them into the model.
"""

import numpy as np

FEATURE_NAMES = [
    "early_growth_rate",
    "log_amplification",
    "doubling_time",
    "curvature",
    "already_peaked",
    "peak_in_window",
    "t_peak_in_window",
    "peak_sharpness",
    "I_at_t_obs",
    "I_mean_window",
    "I_total_change",
    "fraction_above_001",
    "I_std_window",
    "max_single_step_increase",
    "tail_mean",
    "tail_std",
    "endemic_level",
    "decay_rate_after_peak",
    "fraction_decreasing",
    "phase_switch_score",
    "fwhm",
    "autocorr_lag1",
]
N_FEATURES = len(FEATURE_NAMES)


def extract_features(I_series: np.ndarray, t_obs: int) -> np.ndarray:
    """Return a (N_FEATURES,) float32 array for a single series."""
    window = np.asarray(I_series[:t_obs], dtype=float)
    n      = len(window)
    eps    = 1e-8

    log_I             = np.log(window + eps)
    early_growth_rate = float(np.polyfit(np.arange(n), log_I, 1)[0]) if n >= 2 else 0.0
    log_amplification = float(np.log((window[-1] + eps) / (window[0] + eps)))

    doubled      = np.where(window >= 2.0 * (window[0] + eps))[0]
    doubling_time = float(doubled[0]) if len(doubled) else np.nan

    curvature       = float(np.mean(np.diff(window, 2))) if n >= 3 else 0.0
    peak_val        = float(window.max())
    t_peak          = int(window.argmax())
    already_peaked  = 1.0 if window[-1] < peak_val else 0.0
    peak_sharpness  = peak_val / (t_peak + 1)
    I_at_t_obs      = float(window[-1])
    I_mean_window   = float(window.mean())
    I_total_change  = float(window[-1] - window[0])
    fraction_above  = float(np.mean(window > 0.01))
    I_std_window    = float(window.std())

    diffs = np.diff(window)
    max_single_step = float(max(0.0, diffs.max())) if len(diffs) else 0.0

    tail_start = max(1, int(n * 0.8))
    tail       = window[tail_start:]
    tail_mean  = float(tail.mean()) if len(tail) else float(window[-1])
    tail_std   = float(tail.std())  if len(tail) else 0.0
    endemic_level = float(window[-1] / (peak_val + eps))

    post_peak = window[t_peak:]
    decay_rate = float(np.polyfit(np.arange(len(post_peak)), post_peak, 1)[0]) \
        if len(post_peak) >= 2 else 0.0
    fraction_decreasing = float(np.mean(diffs < 0)) if len(diffs) else 0.0

    phase_switch = float(np.max(np.abs(np.diff(window, 2)))) if n >= 5 else 0.0

    half_max   = peak_val / 2.0
    above_half = np.where(window >= half_max)[0]
    fwhm       = float(above_half[-1] - above_half[0] + 1) if len(above_half) >= 2 else 0.0

    if n >= 3 and window[:-1].std() > eps and window[1:].std() > eps:
        autocorr = float(np.corrcoef(window[:-1], window[1:])[0, 1])
    else:
        autocorr = 1.0

    vec = np.array([
        early_growth_rate, log_amplification, doubling_time,
        curvature, already_peaked, peak_val, float(t_peak),
        float(peak_sharpness), I_at_t_obs, I_mean_window,
        I_total_change, fraction_above, I_std_window, max_single_step,
        tail_mean, tail_std, endemic_level, decay_rate,
        fraction_decreasing, phase_switch, fwhm, autocorr,
    ], dtype=np.float32)
    return vec


def extract_features_batch(I_series_matrix: np.ndarray, t_obs: int) -> np.ndarray:
    """Return (n_samples, N_FEATURES) float32 array. NaN/inf replaced with 0."""
    out = np.stack([extract_features(row, t_obs) for row in I_series_matrix])
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)
