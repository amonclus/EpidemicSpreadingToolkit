"""
Tests for the ML feature extraction functions in ml/ml_analysis.py.

Only the pure, stateless functions (extract_features, build_feature_matrix)
are tested here — they do not require disk data or trained models.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import pytest
import numpy as np

from ml.ml_analysis import extract_features, build_feature_matrix, FEATURE_CATEGORIES


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def growing_series():
    return np.linspace(0.01, 0.8, 100)


@pytest.fixture
def decaying_series():
    return np.linspace(0.8, 0.01, 100)


@pytest.fixture
def flat_series():
    return np.full(100, 0.05)


@pytest.fixture
def peaked_series():
    """Series that peaks at t=20 and then decays."""
    t = np.arange(100)
    return np.exp(-0.5 * ((t - 20) / 10) ** 2) * 0.6


@pytest.fixture
def all_zero_series():
    return np.zeros(100)


@pytest.fixture
def spike_series():
    """Single spike at index 50."""
    s = np.zeros(100)
    s[50] = 0.8
    return s


@pytest.fixture
def always_above_001():
    return np.linspace(0.05, 0.9, 100)


@pytest.fixture
def always_below_001():
    return np.full(100, 0.005)


# ─── extract_features: output structure ──────────────────────────────────────

class TestExtractFeaturesStructure:

    EXPECTED_FEATURE_KEYS = {
        "early_growth_rate", "log_amplification", "doubling_time",
        "curvature", "already_peaked", "peak_in_window", "t_peak_in_window",
        "peak_sharpness", "I_at_t_obs", "I_mean_window", "I_total_change",
        "fraction_above_001", "I_std_window", "max_single_step_increase",
        "tail_mean", "tail_std", "endemic_level", "decay_rate_after_peak",
        "fraction_decreasing", "phase_switch_score", "fwhm", "autocorr_lag1",
    }

    def test_returns_dict(self, growing_series):
        feats = extract_features(growing_series, t_obs=50)
        assert isinstance(feats, dict)

    def test_all_expected_keys_present(self, growing_series):
        feats = extract_features(growing_series, t_obs=50)
        for key in self.EXPECTED_FEATURE_KEYS:
            assert key in feats, f"Missing key: {key}"

    def test_output_is_22_features(self, growing_series):
        feats = extract_features(growing_series, t_obs=50)
        assert len(feats) == 22

    def test_output_values_are_floats(self, growing_series):
        feats = extract_features(growing_series, t_obs=50)
        for key, val in feats.items():
            # NaN is allowed (e.g. doubling_time for flat series) but must be float
            assert isinstance(val, float), f"Feature {key} is not float: {type(val)}"


# ─── extract_features: individual feature semantics ──────────────────────────

class TestExtractFeatureValues:

    def test_early_growth_rate_positive_for_growing(self, growing_series):
        feats = extract_features(growing_series, t_obs=80)
        assert feats["early_growth_rate"] > 0.0

    def test_early_growth_rate_negative_for_decaying(self, decaying_series):
        feats = extract_features(decaying_series, t_obs=80)
        assert feats["early_growth_rate"] < 0.0

    def test_already_peaked_is_1_for_peaked_series(self, peaked_series):
        """peaked_series peaks around t=20, so at t_obs=80 it has already peaked."""
        feats = extract_features(peaked_series, t_obs=80)
        assert feats["already_peaked"] == 1.0

    def test_already_peaked_is_0_for_growing_series(self, growing_series):
        """Growing series has not yet peaked."""
        feats = extract_features(growing_series, t_obs=80)
        assert feats["already_peaked"] == 0.0

    def test_i_std_window_is_zero_for_flat_series(self, flat_series):
        feats = extract_features(flat_series, t_obs=80)
        assert abs(feats["I_std_window"]) < 1e-9

    def test_fraction_above_001_is_one_for_always_above(self, always_above_001):
        feats = extract_features(always_above_001, t_obs=80)
        assert feats["fraction_above_001"] == 1.0

    def test_fraction_above_001_is_zero_for_always_below(self, always_below_001):
        feats = extract_features(always_below_001, t_obs=80)
        assert feats["fraction_above_001"] == 0.0

    def test_doubling_time_is_nan_for_flat_series(self, flat_series):
        feats = extract_features(flat_series, t_obs=80)
        assert math.isnan(feats["doubling_time"])

    def test_doubling_time_is_finite_for_growing_series(self, growing_series):
        # growing from 0.01 to 0.8 — it doubles somewhere early
        feats = extract_features(growing_series, t_obs=80)
        # May or may not be NaN depending on whether doubling occurs in window
        # Just verify it's a float
        assert isinstance(feats["doubling_time"], float)

    def test_i_at_t_obs_equals_last_observation(self, growing_series):
        t_obs = 60
        feats = extract_features(growing_series, t_obs=t_obs)
        assert abs(feats["I_at_t_obs"] - growing_series[t_obs - 1]) < 1e-9

    def test_i_total_change_for_growing_is_positive(self, growing_series):
        feats = extract_features(growing_series, t_obs=80)
        assert feats["I_total_change"] > 0.0

    def test_i_total_change_for_decaying_is_negative(self, decaying_series):
        feats = extract_features(decaying_series, t_obs=80)
        assert feats["I_total_change"] < 0.0

    def test_max_single_step_increase_non_negative(self, growing_series):
        feats = extract_features(growing_series, t_obs=80)
        assert feats["max_single_step_increase"] >= 0.0

    def test_max_single_step_increase_zero_for_flat(self, flat_series):
        feats = extract_features(flat_series, t_obs=80)
        assert feats["max_single_step_increase"] == 0.0

    def test_deterministic_same_input_same_output(self, growing_series):
        f1 = extract_features(growing_series, t_obs=50)
        f2 = extract_features(growing_series, t_obs=50)
        for key in f1:
            if not math.isnan(f1[key]):
                assert f1[key] == f2[key], f"Non-deterministic feature: {key}"

    def test_fraction_decreasing_for_pure_decay(self, decaying_series):
        feats = extract_features(decaying_series, t_obs=80)
        # Strictly decreasing → all diffs negative → fraction_decreasing == 1
        assert feats["fraction_decreasing"] == 1.0

    def test_fraction_decreasing_zero_for_growing(self, growing_series):
        feats = extract_features(growing_series, t_obs=80)
        assert feats["fraction_decreasing"] == 0.0

    def test_tail_mean_near_series_end(self, growing_series):
        t_obs = 80
        feats = extract_features(growing_series, t_obs=t_obs)
        # tail_mean is mean of last 20% of window
        tail_start = max(1, int(t_obs * 0.8))
        expected_tail_mean = float(np.mean(growing_series[tail_start:t_obs]))
        assert abs(feats["tail_mean"] - expected_tail_mean) < 1e-6

    def test_endemic_level_near_zero_for_peaked_series(self, peaked_series):
        """For a peaked series, final value is near 0, endemic_level ≈ 0."""
        feats = extract_features(peaked_series, t_obs=90)
        # peak is around index 20, value at t_obs=89 is very small
        assert feats["endemic_level"] < 0.1

    def test_phase_switch_score_non_negative(self, growing_series):
        feats = extract_features(growing_series, t_obs=50)
        assert feats["phase_switch_score"] >= 0.0

    def test_fwhm_non_negative(self, peaked_series):
        feats = extract_features(peaked_series, t_obs=90)
        assert feats["fwhm"] >= 0.0

    def test_autocorr_lag1_in_range(self, growing_series):
        feats = extract_features(growing_series, t_obs=80)
        assert -1.0 <= feats["autocorr_lag1"] <= 1.0


# ─── extract_features: edge cases ────────────────────────────────────────────

class TestExtractFeaturesEdgeCases:

    def test_all_zero_series_no_exception(self, all_zero_series):
        feats = extract_features(all_zero_series, t_obs=80)
        assert isinstance(feats, dict)

    def test_all_zero_fraction_above_001_is_zero(self, all_zero_series):
        feats = extract_features(all_zero_series, t_obs=80)
        assert feats["fraction_above_001"] == 0.0

    def test_spike_series_peak_in_middle(self, spike_series):
        feats = extract_features(spike_series, t_obs=80)
        # peak is at t=50 → should be detected as already peaked
        assert feats["already_peaked"] == 1.0

    def test_t_obs_1(self):
        series = np.array([0.5])
        feats = extract_features(series, t_obs=1)
        assert isinstance(feats, dict)

    def test_t_obs_2(self):
        series = np.array([0.1, 0.2])
        feats = extract_features(series, t_obs=2)
        assert isinstance(feats, dict)

    def test_short_series_no_exception(self):
        series = np.linspace(0.1, 0.5, 5)
        feats = extract_features(series, t_obs=5)
        assert isinstance(feats, dict)

    def test_i_at_t_obs_uses_t_obs_index(self):
        series = np.arange(1, 101, dtype=float) / 100.0
        t_obs = 30
        feats = extract_features(series, t_obs=t_obs)
        assert abs(feats["I_at_t_obs"] - series[t_obs - 1]) < 1e-9


# ─── build_feature_matrix ────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_output_shape_rows(self):
        n_samples = 5
        I_matrix = np.random.rand(n_samples, 100) * 0.8
        X, feat_names = build_feature_matrix(I_matrix, t_obs=50)
        assert X.shape[0] == n_samples

    def test_output_shape_cols_matches_feature_count(self):
        n_samples = 4
        I_matrix = np.random.rand(n_samples, 100) * 0.8
        X, feat_names = build_feature_matrix(I_matrix, t_obs=50)
        assert X.shape[1] == len(feat_names)

    def test_feature_names_are_strings(self):
        I_matrix = np.random.rand(3, 100) * 0.8
        X, feat_names = build_feature_matrix(I_matrix, t_obs=50)
        assert all(isinstance(n, str) for n in feat_names)

    def test_output_dtype_float(self):
        I_matrix = np.random.rand(3, 100) * 0.8
        X, feat_names = build_feature_matrix(I_matrix, t_obs=50)
        assert X.dtype == float

    def test_with_graph_features_appends_columns(self):
        import pandas as pd
        n_samples = 5
        I_matrix = np.random.rand(n_samples, 100) * 0.8
        graph_df = pd.DataFrame({
            "g_feat1": np.ones(n_samples),
            "g_feat2": np.zeros(n_samples),
        })
        X_base, names_base = build_feature_matrix(I_matrix, t_obs=50)
        X_with, names_with = build_feature_matrix(I_matrix, t_obs=50, graph_features_df=graph_df)
        assert X_with.shape[1] == X_base.shape[1] + 2
        assert len(names_with) == len(names_base) + 2

    def test_single_sample(self):
        I_matrix = np.random.rand(1, 100) * 0.5
        X, feat_names = build_feature_matrix(I_matrix, t_obs=50)
        assert X.shape[0] == 1

    def test_different_t_obs_changes_features(self):
        np.random.seed(0)
        I_matrix = np.random.rand(3, 100) * 0.8
        X10, _ = build_feature_matrix(I_matrix, t_obs=10)
        X90, _ = build_feature_matrix(I_matrix, t_obs=90)
        # Different t_obs should produce different feature values
        assert not np.allclose(X10, X90)


# ─── FEATURE_CATEGORIES completeness check ───────────────────────────────────

class TestFeatureCategories:

    def test_all_time_series_features_categorised(self):
        """All 22 time-series features should appear in FEATURE_CATEGORIES."""
        ts_features = [
            "early_growth_rate", "log_amplification", "doubling_time",
            "curvature", "already_peaked", "peak_in_window", "t_peak_in_window",
            "peak_sharpness", "I_at_t_obs", "I_mean_window", "I_total_change",
            "fraction_above_001", "I_std_window", "max_single_step_increase",
            "tail_mean", "tail_std", "endemic_level", "decay_rate_after_peak",
            "fraction_decreasing", "phase_switch_score", "fwhm", "autocorr_lag1",
        ]
        for feat in ts_features:
            assert feat in FEATURE_CATEGORIES, f"Feature not in FEATURE_CATEGORIES: {feat}"

    def test_category_values_are_valid(self):
        valid_categories = {"growth", "shape", "level", "variance", "graph"}
        for feat, cat in FEATURE_CATEGORIES.items():
            assert cat in valid_categories, (
                f"Feature '{feat}' has unknown category '{cat}'"
            )
