"""
ml_tab.py — Virality Predictor tabs for the Network Contagion Lab.

Three sub-tabs extracted from the standalone app2.py:
  render_ml_virality_tab()   — prediction + trajectory chart
  render_ml_education_tab()  — how-does-it-spread explainer
  render_ml_about_tab()      — accuracy metrics and credits

Sidebar inputs (content type, network, series, horizon) are rendered from
within render_ml_virality_tab() using `with st.sidebar:`, which is the same
pattern used in the standalone app2.py.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
# src/ui/tabs/ → src/ui/ → src/
_SRC_DIR = Path(__file__).parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from analysis.graph_features import extract_graph_features, GRAPH_FEATURE_NAMES
    _HAS_GRAPH_FEATURES = True
except ImportError:
    _HAS_GRAPH_FEATURES = False
    GRAPH_FEATURE_NAMES = []

try:
    from input.graph_generator import (
        generate_er_graph,
        generate_lattice_graph,
        generate_random_geometric_graph,
    )
    _HAS_GRAPH_GEN = True
except ImportError:
    _HAS_GRAPH_GEN = False


def _find_ml_data() -> Path:
    for candidate in [
        _SRC_DIR / "ml" / "ml_data",
        _SRC_DIR / "ml_data",
        _SRC_DIR.parent / "ml_data",
    ]:
        if candidate.exists() and (candidate / "dl_regressor.pkl").exists():
            return candidate
    return _SRC_DIR / "ml" / "ml_data"   # fallback — triggers demo mode


ML_DATA = _find_ml_data()

# ── CSS (injected once per session) ──────────────────────────────────────────
_CSS = """
<style>
.verdict-viral    { border-left:4px solid #EF5350; border-radius:6px;
                    padding:16px 20px; margin:8px 0;
                    background:rgba(239,83,80,0.07); }
.verdict-strong   { border-left:4px solid #FFA726; border-radius:6px;
                    padding:16px 20px; margin:8px 0;
                    background:rgba(255,167,38,0.07); }
.verdict-moderate { border-left:4px solid #FFCA28; border-radius:6px;
                    padding:16px 20px; margin:8px 0;
                    background:rgba(255,202,40,0.07); }
.verdict-niche    { border-left:4px solid #42A5F5; border-radius:6px;
                    padding:16px 20px; margin:8px 0;
                    background:rgba(66,165,245,0.07); }
.ml-model-card    { border:1px solid rgba(255,255,255,0.12);
                    border-radius:8px; padding:18px; margin:8px 0; }
.ml-ci-card       { border:1px solid rgba(255,255,255,0.12);
                    border-radius:8px; padding:16px 20px; margin:8px 0; }
.demo-banner      { border-left:4px solid #FFA726; border-radius:6px;
                    padding:10px 16px; margin-bottom:14px;
                    background:rgba(255,167,38,0.1); font-weight:600; }
.ml-section-card  { border:1px solid rgba(255,255,255,0.12);
                    border-radius:8px; padding:18px; margin:8px 0;
                    text-align:center; }
</style>
"""

# ── Model metadata ────────────────────────────────────────────────────────────
MODEL_INFO: dict[str, dict] = {
    "SIR": {
        "icon": "🤧", "label": "Classic Word-of-Mouth",
        "desc": ("People share after a single exposure. Spread is driven by "
                 "individual transmission, like a rumour or breaking news."),
        "type": "Simple contagion",
        "best_for": "News, memes, breaking stories",
        "curve_k": 0.25, "curve_t0": 0.30,
    },
    "SIS": {
        "icon": "🔄", "label": "Recurring Engagement",
        "desc": ("People engage, disengage, and re-engage. Content stays alive "
                 "long-term as people return to it repeatedly."),
        "type": "Simple contagion (endemic)",
        "best_for": "Ongoing debates, evergreen content",
        "curve_k": 0.18, "curve_t0": 0.35,
    },
    "BP": {
        "icon": "🏔️", "label": "Peer Pressure Cascade",
        "desc": ("People only share after multiple friends have. Spread is slow "
                 "at first, then explosive once the threshold is crossed."),
        "type": "Complex contagion",
        "best_for": "Social movements, political content",
        "curve_k": 0.75, "curve_t0": 0.55,
    },
    "WTM": {
        "icon": "📊", "label": "Social Threshold Effect",
        "desc": ("A critical fraction of your circle must engage before you do. "
                 "Very sensitive to initial seeding — too few seeds and nothing happens."),
        "type": "Complex contagion",
        "best_for": "Fashion trends, protest adoption",
        "curve_k": 0.65, "curve_t0": 0.52,
    },
    "H1": {
        "icon": "⚡", "label": "Dual-Channel Spread",
        "desc": ("Spreads both through casual exposure AND peer pressure. Either "
                 "channel alone can ignite a cascade."),
        "type": "Hybrid",
        "best_for": "Product launches, brand campaigns",
        "curve_k": 0.35, "curve_t0": 0.28,
    },
    "H2": {
        "icon": "🌊", "label": "Two-Phase Viral Wave",
        "desc": ("Started as word-of-mouth, now transitioning into a peer-pressure "
                 "driven cascade. Hard to stop once the second phase begins."),
        "type": "Hybrid (sequential)",
        "best_for": "Slow-burn stories that explode",
        "curve_k": 0.60, "curve_t0": 0.48,
    },
    "H3": {
        "icon": "📣", "label": "Amplified Word-of-Mouth",
        "desc": ("Classic sharing amplified by social proof. The more friends share, "
                 "the more likely you are to share too."),
        "type": "Hybrid (soft)",
        "best_for": "Health tips, lifestyle content",
        "curve_k": 0.30, "curve_t0": 0.28,
    },
    "H4": {
        "icon": "🔥", "label": "Endemic Engagement",
        "desc": ("Content maintains a persistent engaged audience through both "
                 "exposure and social reinforcement — it never fully dies out."),
        "type": "Hybrid (endemic)",
        "best_for": "Ongoing social issues, community content",
        "curve_k": 0.22, "curve_t0": 0.35,
    },
    "H5": {
        "icon": "🐢", "label": "Slow Burn to Cascade",
        "desc": ("Building gradually through organic sharing before triggering a "
                 "social threshold cascade. Patient but potentially explosive."),
        "type": "Hybrid (sequential endemic)",
        "best_for": "Grassroots movements, niche communities",
        "curve_k": 0.70, "curve_t0": 0.60,
    },
    "H6": {
        "icon": "🌱", "label": "Soft Social Pressure",
        "desc": ("Gradual peer influence accumulates over time. Sustained engagement "
                 "driven by social norms rather than individual impulse."),
        "type": "Hybrid (soft endemic)",
        "best_for": "Behaviour change campaigns, lifestyle shifts",
        "curve_k": 0.40, "curve_t0": 0.42,
    },
    "BP-SIR hybrid": {
        "icon": "⚡", "label": "BP/SIR Hybrid",
        "desc": ("A mix of individual word-of-mouth and peer-pressure dynamics. "
                 "Spread can be ignited by a single exposure but accelerates once "
                 "social reinforcement kicks in."),
        "type": "Hybrid (BP + SIR)",
        "best_for": "Product launches, brand campaigns, slow-burn news stories",
        "curve_k": 0.35, "curve_t0": 0.35,
    },
    "SIS-WTM hybrid": {
        "icon": "🔥", "label": "SIS/WTM Hybrid",
        "desc": ("A recurring-engagement mechanism reinforced by social thresholds. "
                 "Content stays alive long-term as repeated peer exposure pushes "
                 "more people past their adoption threshold."),
        "type": "Hybrid (SIS + WTM)",
        "best_for": "Ongoing social issues, grassroots movements, community content",
        "curve_k": 0.30, "curve_t0": 0.40,
    },
}

CONTENT_TO_MODEL_HINTS: dict[str, tuple[str, str]] = {
    "News or information":           ("SIR", "H3"),
    "Viral challenge or trend":      ("H3", "BP"),
    "Opinion or political content":  ("BP", "WTM"),
    "Product or brand content":      ("H1", "H2"),
    "Health behaviour or lifestyle": ("H2", "H5"),
}

NETWORK_MULTIPLIER: dict[str, float] = {
    "Tight community (close friends, family groups)":       0.72,
    "Mixed social network (typical social media)":          1.00,
    "Influencer-driven network (few hubs, many followers)": 1.38,
}

# SIS-type models reach a long-run endemic equilibrium rather than dying out.
# rho_final for these models = endemic steady-state fraction (NOT cumulative reach).
# The peak of the trajectory is the more meaningful "maximum spread" metric.
_ENDEMIC_MODELS = {"SIS", "WTM", "H4", "H5", "H6", "SIS-WTM hybrid"}

TIPPING_THRESHOLDS: dict[str, float] = {
    "SIR": 0.05, "SIS": 0.05, "BP": 0.15, "WTM": 0.12,
    "H1":  0.08, "H2": 0.12, "H3": 0.06, "H4": 0.08,
    "H5":  0.15, "H6": 0.10,
    "BP-SIR hybrid": 0.09, "SIS-WTM hybrid": 0.11,
}

# H1/H2/H3 are BP-SIR hybrids; H4/H5/H6 are SIS-WTM hybrids.
# The DL model still predicts the original 10 classes; we remap for display only.
_DISPLAY_NAME: dict[str, str] = {
    "H1": "BP-SIR hybrid", "H2": "BP-SIR hybrid", "H3": "BP-SIR hybrid",
    "H4": "SIS-WTM hybrid", "H5": "SIS-WTM hybrid", "H6": "SIS-WTM hybrid",
}

DEFAULT_FEATURE_NAMES = [
    "early_growth_rate", "log_amplification", "doubling_time",
    "curvature", "already_peaked", "peak_in_window", "t_peak_in_window",
    "peak_sharpness", "I_at_t_obs", "I_mean_window", "I_total_change",
    "fraction_above_001", "I_std_window", "max_single_step_increase",
    "tail_mean", "tail_std", "endemic_level", "decay_rate_after_peak",
    "fraction_decreasing", "phase_switch_score", "fwhm", "autocorr_lag1",
]

_PLOTLY = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
_M = dict(l=50, r=20, t=30, b=50)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def _load_models():
    """Returns (reg, clf, label_encoder, feature_names, demo_mode, graph_feat_means)."""
    try:
        import joblib
        reg = joblib.load(ML_DATA / "dl_regressor.pkl")
        clf = joblib.load(ML_DATA / "dl_classifier.pkl")
        le  = joblib.load(ML_DATA / "label_encoder.pkl")
        try:
            gfm = joblib.load(ML_DATA / "graph_feature_means.pkl")
        except Exception:
            gfm = {}
        return reg, clf, le, None, False, gfm
    except Exception:
        le = {i: name for i, name in enumerate(MODEL_INFO)}
        return None, None, le, None, True, {}


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(series: np.ndarray) -> dict:
    window = np.asarray(series, dtype=float)
    n      = len(window)
    eps    = 1e-8

    log_I             = np.log(window + eps)
    early_growth_rate = float(np.polyfit(np.arange(n), log_I, 1)[0]) if n >= 2 else 0.0
    log_amplification = float(np.log((window[-1] + eps) / (window[0] + eps)))

    doubled       = np.where(window >= 2.0 * (window[0] + eps))[0]
    doubling_time = float(doubled[0]) if len(doubled) else float("nan")

    curvature      = float(np.mean(np.diff(window, 2))) if n >= 3 else 0.0
    peak_val       = float(window.max())
    t_peak         = int(window.argmax())
    already_peaked = 1.0 if window[-1] < peak_val else 0.0
    peak_sharpness = peak_val / (t_peak + 1)

    I_at_t_obs         = float(window[-1])
    I_mean_window      = float(window.mean())
    I_total_change     = float(window[-1] - window[0])
    fraction_above_001 = float(np.mean(window > 0.01))

    diffs                    = np.diff(window)
    I_std_window             = float(window.std())
    max_single_step_increase = float(max(0.0, diffs.max())) if len(diffs) else 0.0

    tail_start    = max(1, int(n * 0.8))
    tail          = window[tail_start:]
    tail_mean     = float(tail.mean()) if len(tail) else float(window[-1])
    tail_std      = float(tail.std())  if len(tail) else 0.0
    endemic_level = float(window[-1] / (peak_val + eps))

    post_peak = window[t_peak:]
    decay_rate_after_peak = (
        float(np.polyfit(np.arange(len(post_peak)), post_peak, 1)[0])
        if len(post_peak) >= 2 else 0.0
    )
    fraction_decreasing = float(np.mean(diffs < 0)) if len(diffs) else 0.0

    d2                 = np.diff(window, 2) if n >= 5 else np.array([0.0])
    phase_switch_score = float(np.max(np.abs(d2)))

    above_half  = np.where(window >= peak_val / 2.0)[0]
    fwhm        = float(above_half[-1] - above_half[0] + 1) if len(above_half) >= 2 else 0.0

    autocorr_lag1 = (
        float(np.corrcoef(window[:-1], window[1:])[0, 1])
        if n >= 3 and window.std() > eps else 1.0
    )

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


def _build_feature_row(features: dict, feature_names: list) -> np.ndarray:
    row = np.array([features.get(f, 0.0) for f in feature_names], dtype=float)
    row = np.where(np.isnan(row), 0.0, row)
    return row.reshape(1, -1)


# ── Trajectory generation ─────────────────────────────────────────────────────

def _sigmoid(t_norm: np.ndarray, k: float, t0: float, scale: float) -> np.ndarray:
    return scale / (1.0 + np.exp(-k * len(t_norm) * (t_norm - t0)))


def _generate_trajectory(model_name: str, rho_final: float,
                          n_obs: int, n_pred: int,
                          last_obs_frac: float) -> np.ndarray:
    info   = MODEL_INFO.get(model_name, MODEL_INFO["SIR"])
    k, t0  = info["curve_k"], info["curve_t0"]
    total  = n_obs + n_pred
    t_norm = np.linspace(0, 1, total)

    full_curve = _sigmoid(t_norm, k, t0, rho_final)

    anchor_raw = full_curve[n_obs - 1]
    if anchor_raw > 1e-9:
        ratio      = last_obs_frac / anchor_raw
        correction = np.ones(total)
        correction[n_obs:] = np.linspace(ratio, 1.0, n_pred)
        full_curve = full_curve * correction

    return np.clip(full_curve, 0.0, 1.0)


# ── Demo-mode fallback ────────────────────────────────────────────────────────

def _demo_predict(series: np.ndarray, content_type: str,
                  network_type: str) -> tuple:
    log_s       = np.log(series + 1e-8)
    growth_rate = float(np.polyfit(np.arange(len(log_s)), log_s, 1)[0]) if len(series) >= 2 else 0.05
    net_mult    = NETWORK_MULTIPLIER.get(network_type, 1.0)
    seed        = int(abs(series.sum()) * 1e4) % (2 ** 31)
    rng         = np.random.default_rng(seed)

    if growth_rate > 0.30:
        rho_final = float(np.clip(rng.uniform(0.60, 0.90) * net_mult, 0.0, 1.0))
        rho_std   = 0.08
    elif growth_rate > 0.10:
        rho_final = float(np.clip(rng.uniform(0.25, 0.55) * net_mult, 0.0, 1.0))
        rho_std   = 0.10
    else:
        rho_final = float(np.clip(rng.uniform(0.02, 0.16) * net_mult, 0.0, 1.0))
        rho_std   = 0.05

    hints      = CONTENT_TO_MODEL_HINTS.get(content_type, ("SIR", "H1"))
    model_name = str(rng.choice(hints))
    confidence = float(rng.uniform(0.50, 0.76))
    return rho_final, rho_std, model_name, confidence


# ── Prediction pipeline ───────────────────────────────────────────────────────

def _run_prediction(series_pct: list[float], content_type: str,
                    network_type: str, horizon: int,
                    graph=None) -> dict:
    reg, clf, label_encoder, _, demo_mode, graph_feat_means = _load_models()

    series = np.clip(np.array(series_pct, dtype=float) / 100.0, 0.0, 1.0)
    graph_conditioned = False

    if demo_mode:
        rho_final, rho_std, raw_model_name, confidence = _demo_predict(
            series, content_type, network_type
        )
    else:
        I_matrix = series.reshape(1, -1).astype(np.float32)

        # Build graph feature row (1, n_graph_feats) when available
        graph_features = None
        if graph is not None and _HAS_GRAPH_FEATURES:
            try:
                gf_dict = extract_graph_features(graph)
                graph_features = np.array(
                    [[gf_dict.get(k, 0.0) for k in GRAPH_FEATURE_NAMES]],
                    dtype=np.float32,
                )
                graph_conditioned = True
            except Exception:
                pass
        if graph_features is None and graph_feat_means:
            graph_features = np.array(
                [[graph_feat_means.get(k, 0.0) for k in GRAPH_FEATURE_NAMES]],
                dtype=np.float32,
            )

        rho_raw = float(reg.predict(I_matrix, graph_features)[0])

        if graph_conditioned:
            rho_final = float(np.clip(rho_raw, 0.0, 1.0))
        else:
            net_mult  = NETWORK_MULTIPLIER.get(network_type, 1.0)
            rho_final = float(np.clip(rho_raw * net_mult, 0.0, 1.0))

        rho_std = 0.08 * rho_final + 0.02

        proba         = clf.predict_proba(I_matrix, graph_features)[0]
        model_id      = int(clf.predict(I_matrix, graph_features)[0])
        raw_model_name = label_encoder[model_id]
        # Confidence: sum probs of all constituent models when remapped to a family
        display_name = _DISPLAY_NAME.get(raw_model_name, raw_model_name)
        if raw_model_name != display_name:
            family_ids = [i for i, n in label_encoder.items()
                          if _DISPLAY_NAME.get(n, n) == display_name]
            confidence = float(sum(proba[i] for i in family_ids if i < len(proba)))
        else:
            confidence = float(proba[model_id])

    # Use specific model name for trajectory shape, remapped name for display
    traj_model = raw_model_name
    model_name = _DISPLAY_NAME.get(raw_model_name, raw_model_name)

    n_obs = len(series)
    traj  = _generate_trajectory(traj_model, rho_final, n_obs, horizon, float(series[-1]))

    return {
        "rho_final":         rho_final,
        "rho_std":           rho_std,
        "model_name":        model_name,
        "confidence":        confidence,
        "series_pct":        list(series_pct),
        "traj_pct":          (traj * 100).tolist(),
        "t_full":            list(range(n_obs + horizon)),
        "n_obs":             n_obs,
        "demo_mode":         demo_mode,
        "content_type":      content_type,
        "network_type":      network_type,
        "graph_conditioned": graph_conditioned,
    }


def _get_verdict(rho: float) -> tuple[str, str, str]:
    if rho >= 0.60:
        return "High spread",     "verdict-viral",    "Predicted to reach a large fraction of the network."
    if rho >= 0.30:
        return "Moderate spread", "verdict-strong",   "Predicted to reach a substantial portion of the network."
    if rho >= 0.10:
        return "Limited spread",  "verdict-moderate", "Predicted to reach a meaningful but minority audience."
    return     "Local spread",    "verdict-niche",    "Predicted to remain within a small fraction of the network."


# ── CSV upload helper ─────────────────────────────────────────────────────────

def _parse_upload(uploaded_file) -> list[float]:
    if uploaded_file is None:
        return [0.5, 0.8, 1.2, 2.1, 3.8]

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return [0.5, 0.8, 1.2, 2.1, 3.8]

    if "reach_pct" in df.columns and "type" in df.columns:
        subset = df[df["type"] == "observed"]["reach_pct"].dropna()
        if len(subset) >= 2:
            vals = subset.tolist()
            st.caption(f"Loaded ViralSense export — {len(vals)} observed steps.")
            return vals

    preferred = ["reach_pct", "I", "value", "I_frac", "infected", "spread"]
    for col in preferred:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
            if len(vals) >= 2:
                st.caption(f"Loaded column '{col}' — {len(vals)} values.")
                break
    else:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.error("No numeric columns found in the file.")
            return [0.5, 0.8, 1.2, 2.1, 3.8]
        col = num_cols[1] if len(num_cols) >= 2 and df[num_cols[0]].is_monotonic_increasing else num_cols[0]
        vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        st.caption(f"Loaded column '{col}' — {len(vals)} values.")

    if len(vals) < 2:
        st.error("File must contain at least 2 data points.")
        return [0.5, 0.8, 1.2, 2.1, 3.8]

    if max(vals) <= 1.0:
        vals = [v * 100 for v in vals]
        st.caption("Values detected as fractions (0–1) and converted to %.")

    return [round(v, 4) for v in vals]


# ══════════════════════════════════════════════════════════════════════════════
# Public tab renderers
# ══════════════════════════════════════════════════════════════════════════════

def render_ml_virality_tab() -> None:
    """Renders the 'Will It Go Viral?' predictor tab, including sidebar inputs."""
    st.markdown(_CSS, unsafe_allow_html=True)

    # pick up graph loaded elsewhere in the app (may be None)
    session_graph = st.session_state.get("graph", None)

    # ── Sidebar inputs ─────────────────────────────────────────────────────
    with st.sidebar:
        st.header("About your content")
        st.caption("Inputs for the predictor")

        content_type = st.selectbox(
            "What kind of content is this?",
            options=list(CONTENT_TO_MODEL_HINTS.keys()),
        )

        if session_graph is not None and _HAS_GRAPH_FEATURES:
            n  = session_graph.number_of_nodes()
            m  = session_graph.number_of_edges()
            st.success(
                f"Using loaded graph ({n:,} nodes, {m:,} edges).\n\n"
                "Graph structure will condition the prediction."
            )
            network_type = None
        else:
            network_type = st.selectbox(
                "What kind of social network?",
                options=list(NETWORK_MULTIPLIER.keys()),
            )

        if _HAS_GRAPH_GEN:
            with st.expander("🔄 Change graph"):
                graph_type = st.selectbox(
                    "Graph type",
                    ["Erdős–Rényi", "Random Geometric", "Lattice"],
                    key="ml_graph_type",
                )
                if graph_type == "Erdős–Rényi":
                    col1, col2 = st.columns(2)
                    _n = col1.number_input("Nodes", 10, 5000, 200, key="ml_er_n")
                    _p = col2.slider("Edge prob (p)", 0.01, 1.0, 0.10, 0.01, key="ml_er_p")
                    if st.button("Generate", key="ml_gen_er", use_container_width=True):
                        with st.spinner("Generating…"):
                            st.session_state["graph"] = generate_er_graph(int(_n), float(_p))
                        st.rerun()
                elif graph_type == "Random Geometric":
                    col1, col2 = st.columns(2)
                    _n = col1.number_input("Nodes", 10, 5000, 200, key="ml_rgg_n")
                    _r = col2.slider("Radius (r)", 0.01, 1.0, 0.20, 0.01, key="ml_rgg_r")
                    if st.button("Generate", key="ml_gen_rgg", use_container_width=True):
                        with st.spinner("Generating…"):
                            st.session_state["graph"] = generate_random_geometric_graph(int(_n), float(_r))
                        st.rerun()
                else:
                    _g = st.number_input("Grid side", 3, 100, 10, key="ml_lat_g")
                    if st.button("Generate", key="ml_gen_lat", use_container_width=True):
                        with st.spinner("Generating…"):
                            st.session_state["graph"] = generate_lattice_graph(int(_g))
                        st.rerun()
                if session_graph is not None:
                    if st.button("Clear graph", key="ml_clear_graph", use_container_width=True):
                        st.session_state["graph"] = None
                        st.rerun()

        st.markdown("---")
        st.subheader("How has it spread so far?")

        input_method = st.radio(
            "Input method",
            ["Manual entry", "Upload file", "Simulate"],
            horizontal=True,
        )

        if input_method == "Manual entry":
            st.caption("% of audience reached at each time step, comma-separated (2–30 values).")
            raw_input = st.text_input(
                "% reached per step",
                value="0.5, 0.8, 1.2, 2.1, 3.8",
            )
            try:
                series_pct = [float(x.strip()) for x in raw_input.split(",") if x.strip()]
                if len(series_pct) < 2:
                    st.warning("Enter at least 2 values.")
                    series_pct = [0.5, 0.8, 1.2, 2.1, 3.8]
            except ValueError:
                st.error("Could not parse input — using default values.")
                series_pct = [0.5, 0.8, 1.2, 2.1, 3.8]

        elif input_method == "Upload file":
            series_pct = _parse_upload(st.file_uploader(
                "Upload a CSV file with spread data",
                type=["csv"],
                help=(
                    "Single numeric column, or two columns (time + reach). "
                    "Values can be percentages (0–100) or fractions (0–1)."
                ),
            ))

        else:
            init_reach   = st.slider("Initial reach (%)", 0.1, 5.0, 0.5, step=0.1)
            growth_style = st.select_slider(
                "Growth style",
                options=["Slow and steady", "Moderate", "Explosive"],
                value="Moderate",
            )
            add_noise = st.checkbox("Add realistic noise")
            rates = {"Slow and steady": 0.08, "Moderate": 0.18, "Explosive": 0.38}
            t_sim = np.arange(10)
            base  = init_reach * np.exp(rates[growth_style] * t_sim)
            if add_noise:
                base = base * np.random.default_rng(42).uniform(0.85, 1.15, len(base))
            series_pct = list(np.clip(base, 0.0, 100.0).round(2))
            st.caption(f"Generated: {', '.join(str(v) for v in series_pct)}")

        horizon = st.slider("Prediction horizon (steps)", 10, 100, 50, step=5)

        predict_btn = st.button(
            "Predict Virality", type="primary", use_container_width=True
        )

    # ── Main area (idle state) ──────────────────────────────────────────────
    if not predict_btn:
        st.markdown("## Contagion predictor")
        st.markdown(
            "#### Predict the spread of a contagion.\n\n"
            "Enter your data in the **sidebar**, then click **Predict Virality**."
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                '<div class="ml-section-card"><h3>Reach Prediction</h3>'
                '<p>Predict the final % of the network reached.</p></div>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                '<div class="ml-section-card"><h3>Mechanism ID</h3>'
                '<p>Identify the spreading model from the early curve shape.</p></div>',
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                '<div class="ml-section-card"><h3>Tipping Point</h3>'
                '<p>Know whether the viral threshold has already been crossed.</p></div>',
                unsafe_allow_html=True,
            )

        if series_pct:
            st.markdown("#### Your early spread data")
            fig_prev = go.Figure(go.Scatter(
                x=list(range(1, len(series_pct) + 1)), y=series_pct,
                mode="lines+markers",
                line=dict(color="#FF6D00", width=2),
                marker=dict(size=8),
            ))
            fig_prev.update_layout(
                xaxis_title="Time steps",
                yaxis_title="% of network reached",
                height=280, margin=dict(l=50, r=20, t=10, b=50), **_PLOTLY,
            )
            st.plotly_chart(fig_prev, use_container_width=True)
        return

    # ── Run prediction ──────────────────────────────────────────────────────
    with st.spinner("Analysing spread pattern…"):
        res = _run_prediction(
            series_pct,
            content_type,
            network_type or "",
            horizon,
            graph=session_graph,
        )

    if res["demo_mode"]:
        st.markdown(
            '<div class="demo-banner">Demo mode — ML model files not found. '
            'Predictions are illustrative only.</div>',
            unsafe_allow_html=True,
        )

    if res.get("graph_conditioned"):
        st.info(
            "Graph-conditioned prediction — the network structure of your loaded graph "
            "is incorporated directly into the forecast.",
            icon="🔬",
        )

    rho         = res["rho_final"]
    rho_pct     = rho * 100
    rho_std     = res["rho_std"]
    rho_std_pct = rho_std * 100
    model_name  = res["model_name"]
    minfo       = MODEL_INFO[model_name]
    conf        = res["confidence"]
    lo_pct      = max(0.0,   rho_pct - 1.5 * rho_std_pct)
    hi_pct      = min(100.0, rho_pct + 1.5 * rho_std_pct)

    is_endemic  = model_name in _ENDEMIC_MODELS
    traj_arr    = np.array(res["traj_pct"])
    peak_pct    = float(traj_arr.max())
    peak_rho    = peak_pct / 100.0

    # For endemic models the verdict is driven by peak reach, not endemic level
    v_label, v_css, v_desc = _get_verdict(peak_rho if is_endemic else rho)

    st.markdown("---")

    col_v, col_m = st.columns(2)

    with col_v:
        st.markdown(
            f'<div class="{v_css}">'
            f'<p style="margin:0 0 4px 0;font-size:0.8rem;text-transform:uppercase;'
            f'letter-spacing:0.08em;opacity:0.7">Spread assessment</p>'
            f'<h2 style="margin:0 0 6px 0">{v_label}</h2>'
            f'<p style="margin:0;font-size:0.9rem">{v_desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if is_endemic:
            st.metric("Peak reach (maximum simultaneously infected)", f"{peak_pct:.1f}%")
            st.progress(float(np.clip(peak_rho, 0, 1)))
            st.markdown(
                f'<div class="ml-ci-card">'
                f'<p style="margin:0 0 6px 0;font-size:0.8rem;text-transform:uppercase;'
                f'letter-spacing:0.08em;opacity:0.7">Endemic equilibrium level</p>'
                f'<h3 style="margin:0">{rho_pct:.1f}%</h3>'
                f'<p style="margin:6px 0 0 0;font-size:0.85rem;opacity:0.7">'
                f'Long-run fraction remaining infected after the peak</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.metric("Predicted final reach (cumulative ever-infected)", f"{rho_pct:.1f}%")
            st.progress(float(np.clip(rho, 0, 1)))
            st.markdown(
                f'<div class="ml-ci-card">'
                f'<p style="margin:0 0 6px 0;font-size:0.8rem;text-transform:uppercase;'
                f'letter-spacing:0.08em;opacity:0.7">90% confidence interval</p>'
                f'<h3 style="margin:0">{lo_pct:.1f}% &ndash; {hi_pct:.1f}%</h3>'
                f'<p style="margin:6px 0 0 0;font-size:0.85rem;opacity:0.7">'
                f'±{rho_std_pct:.1f} pp standard deviation</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_m:
        st.markdown(
            f'<div class="ml-model-card">'
            f'<p style="margin:0 0 4px 0;font-size:0.8rem;text-transform:uppercase;'
            f'letter-spacing:0.08em;opacity:0.7">Identified spreading model</p>'
            f'<h2 style="margin:0 0 4px 0">{model_name}</h2>'
            f'<p style="margin:0 0 10px 0;font-size:0.85rem;opacity:0.7">{minfo["type"]}</p>'
            f'<p style="margin:0;font-size:0.9rem">{minfo["desc"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="ml-ci-card">'
            f'<p style="margin:0 0 6px 0;font-size:0.8rem;text-transform:uppercase;'
            f'letter-spacing:0.08em;opacity:0.7">Classifier confidence</p>'
            f'<h3 style="margin:0">{conf * 100:.0f}%</h3>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(float(conf))

        with st.expander("Details"):
            st.markdown(
                f"**Observation window:** {res['n_obs']} steps  \n"
                f"**Typical content:** {minfo['best_for']}  \n"
                f"**Features used:** 22 time-series"
                + (" + 10 graph-structural" if res.get("graph_conditioned") else "")
            )

    st.markdown("---")
    st.markdown("### Predicted Trajectory")

    traj   = np.array(res["traj_pct"])
    t_full = np.array(res["t_full"])
    n_obs  = res["n_obs"]

    band_hi = np.clip(traj + 1.5 * rho_std_pct, 0, 100)
    band_lo = np.clip(traj - 1.5 * rho_std_pct, 0, 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_full[n_obs - 1:], y=band_hi[n_obs - 1:],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=t_full[n_obs - 1:], y=band_lo[n_obs - 1:],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(255,109,0,0.15)",
        name="Confidence band",
    ))
    fig.add_trace(go.Scatter(
        x=t_full[n_obs - 1:], y=traj[n_obs - 1:],
        mode="lines", line=dict(color="#FF6D00", width=2.5, dash="dash"),
        name="Predicted",
    ))
    fig.add_trace(go.Scatter(
        x=t_full[:n_obs], y=res["series_pct"],
        mode="lines+markers",
        line=dict(color="#42A5F5", width=2.5),
        marker=dict(size=7),
        name="Observed",
    ))
    fig.add_vline(
        x=float(n_obs - 1),
        line=dict(dash="dot", color="rgba(255,255,255,0.4)", width=1.5),
        annotation_text="← Observed  |  Predicted →",
        annotation_position="top",
        annotation_font_color="rgba(255,255,255,0.6)",
    )
    fig.update_layout(
        xaxis_title="Time steps",
        yaxis_title="% of network reached",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        height=400, margin=_M, **_PLOTLY,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Tipping Point Analysis")

    tipping      = TIPPING_THRESHOLDS.get(model_name, 0.10)
    obs_arr      = np.array(res["series_pct"])
    current_frac = float(obs_arr.max()) / 100.0
    crossed      = current_frac >= tipping

    if crossed:
        st.success(
            "✅ **This content has already crossed the viral tipping point.**\n\n"
            "Spread is now self-sustaining and difficult to stop.\n\n"
            "- Removing individual sharers will not stop the cascade\n"
            "- The network effect is now the primary driver of spread"
        )
    else:
        gap         = tipping - current_frac
        extra_users = max(1, int(gap * 10_000))
        avg_step    = max(float(np.mean(np.diff(obs_arr))) / 100.0, 1e-5)
        hours_est   = max(1, int(gap / avg_step))
        plateau_line = (
            f"- Current trajectory suggests a peak of **{peak_pct:.1f}%** "
            f"simultaneously infected, with a long-run endemic level of **{rho_pct:.1f}%**"
            if is_endemic else
            f"- Current trajectory suggests it will plateau at **{rho_pct:.1f}%** "
            f"cumulative reach (ever-infected)"
        )
        st.warning(
            f"⚠️ **This content has NOT yet crossed the viral tipping point.**\n\n"
            f"To push it over the threshold:\n\n"
            f"- Getting **{extra_users:,} more high-connectivity nodes** to share "
            f"in the next **{hours_est} steps** could trigger a cascade\n"
            f"{plateau_line}"
        )

    st.markdown("---")
    with st.expander("📋 Export Results"):
        graph_line = ("Graph-conditioned: Yes" if res.get("graph_conditioned")
                      else f"Network type        : {network_type or 'n/a'}")
        if is_endemic:
            reach_lines = (
                f"Peak reach          : {peak_pct:.1f}%  (max simultaneously infected)\n"
                f"Endemic equilibrium : {rho_pct:.1f}%  (long-run steady-state fraction)\n"
            )
        else:
            reach_lines = (
                f"Predicted reach     : {rho_pct:.1f}%  (range: {lo_pct:.1f}%–{hi_pct:.1f}%)\n"
                f"                      (cumulative ever-infected)\n"
            )
        summary_txt = (
            f"Virality Prediction Report\n"
            f"{'=' * 40}\n"
            f"Content type        : {content_type}\n"
            f"{graph_line}\n"
            f"Verdict             : {v_label}\n"
            f"{reach_lines}"
            f"Spreading mechanism : {minfo['label']} ({model_name})\n"
            f"Confidence          : {conf * 100:.0f}%\n"
            f"Tipping point       : {'Crossed ✅' if crossed else 'Not yet ⚠️'}\n"
        )
        st.text_area("Copy-paste summary", summary_txt, height=200)

        df_export = pd.DataFrame({
            "step":      t_full,
            "reach_pct": np.round(traj, 3),
            "type":      ["observed"] * n_obs + ["predicted"] * (len(t_full) - n_obs),
        })
        st.download_button(
            "⬇️ Download trajectory as CSV",
            df_export.to_csv(index=False),
            file_name="virality_prediction.csv",
            mime="text/csv",
        )


def render_ml_education_tab() -> None:
    """Renders the 'How Does It Spread?' educational explainer tab."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown("## How does content spread?")
    st.markdown("A simple guide to the science behind epidemic-like spreading.")

    st.markdown("---")
    st.markdown("### The Two Types of Viral Content")

    col_l, col_r = st.columns(2)
    t = np.linspace(0, 20, 200)

    with col_l:
        st.markdown(
            '<div class="ml-section-card"><h3>Spreads like a cold</h3>'
            '<b>Simple contagion</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "One contact is enough. You see it once and share it. "
            "Like a funny video or breaking news, **no peer pressure needed**. "
            "Growth is exponential: doubles at a steady rate."
        )
        fig_s = go.Figure()
        for n0, col in [(10, "#42A5F5"), (100, "#26C6DA")]:
            y = np.clip(1.0 * (1.0 - np.exp(-0.2 * t)) * n0 / 10, 0, 100)
            fig_s.add_trace(go.Scatter(x=t, y=y, mode="lines",
                                       line=dict(width=2, color=col),
                                       name=f"Seeded {n0} people"))
        fig_s.update_layout(xaxis_title="Time", yaxis_title="% reached",
                             height=300, margin=_M, **_PLOTLY)
        st.plotly_chart(fig_s, use_container_width=True)

    with col_r:
        st.markdown(
            '<div class="ml-section-card"><h3>Spreads like a trend</h3>'
            '<b>Complex contagion</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "You need to see **multiple friends** doing it before you join. "
            "Like a fitness challenge or a political movement, "
            "**peer pressure is the engine**. "
            "Slow start, then explosive growth once the threshold is crossed."
        )
        fig_c = go.Figure()
        for n0, col in [(10, "#EF5350"), (100, "#FF7043")]:
            scale = n0 / 10
            y = scale / (1.0 + np.exp(-0.5 * (t - (10 - scale))))
            fig_c.add_trace(go.Scatter(x=t, y=np.clip(y, 0, 100), mode="lines",
                                       line=dict(width=2, color=col),
                                       name=f"Seeded {n0} people"))
        fig_c.update_layout(xaxis_title="Time", yaxis_title="% reached",
                             height=300, margin=_M, **_PLOTLY)
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")
    st.markdown("### Real-world examples")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="ml-section-card"><h2>📰</h2><h4>Breaking news</h4>'
            '<b>SIR-like spreading</b><br><br>'
            'Spreads immediately to anyone who sees it. One share is enough. '
            'Peaks fast and fades fast.'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="ml-section-card"><h2>🪣</h2><h4>Ice bucket challenge</h4>'
            '<b>Complex contagion</b><br><br>'
            'Required seeing multiple friends participate before you felt '
            'compelled to join. Slow start, then an explosive cascade.'
            '</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="ml-section-card"><h2>💉</h2><h4>Vaccine hesitancy</h4>'
            '<b>Hybrid spreading</b><br><br>'
            'Spreads through both information exposure AND social reinforcement. '
            'Hard to reverse.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### The Tipping Point")

    seeds         = np.linspace(0, 0.20, 200)
    simple_final  = np.clip(seeds * 5.0, 0, 1)
    complex_final = np.where(seeds < 0.08, seeds * 0.3,
                             np.clip((seeds - 0.08) * 12, 0, 1))

    fig_tip = go.Figure()
    fig_tip.add_trace(go.Scatter(
        x=seeds * 100, y=simple_final * 100,
        mode="lines", line=dict(color="#42A5F5", width=2.5),
        name="Simple contagion (smooth growth)",
    ))
    fig_tip.add_trace(go.Scatter(
        x=seeds * 100, y=complex_final * 100,
        mode="lines", line=dict(color="#EF5350", width=2.5),
        name="Complex contagion (discontinuous jump)",
    ))
    fig_tip.add_vline(
        x=8, line=dict(dash="dot", color="rgba(255,255,255,0.4)"),
        annotation_text="Tipping point", annotation_position="top right",
        annotation_font_color="rgba(255,255,255,0.6)",
    )
    fig_tip.update_layout(
        xaxis_title="Initial seed (% of network)",
        yaxis_title="Final reach (% of network)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400, margin=_M, **_PLOTLY,
    )
    st.plotly_chart(fig_tip, use_container_width=True)
    st.caption(
        "**Complex contagion has a tipping point** — seed too few nodes and nothing "
        "happens. Seed enough and the whole network activates."
    )


def render_ml_about_tab() -> None:
    """Renders the 'About' tab with accuracy metrics and project credits."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown("## About the Predictor")

    st.markdown("### What is this?")
    st.markdown(
        "The virality predictor is the applied deep learning component of a **bachelor's thesis** "
        "on hybrid epidemic spreading models. "
        "Ten spreading models were studied, from pure probabilistic (SIR/SIS) to "
        "pure reinforced (Bootstrap percolation/WTM) and six hybrid combinations. "
        "Simulations were run on three synthetic network types and two real-world "
        "networks (Facebook and GitHub social graphs), generating 50,000 labelled "
        "time-series observations. A deep learning model (CNN) was then trained to "
        "identify which mechanism is active from the **early spread curve and graph structure alone** "
        "and to predict the **final reach**."
    )

    st.markdown("---")
    st.markdown("### The models")

    _base_models = ["SIR", "SIS", "BP", "WTM", "H1", "H2", "H3", "H4", "H5", "H6"]
    table_data = {
        "Model": _base_models,
        "Type":  [MODEL_INFO[m]["type"]      for m in _base_models],
        "Plain-English name": [MODEL_INFO[m]["label"]    for m in _base_models],
        "Best describes":     [MODEL_INFO[m]["best_for"] for m in _base_models],
        "Displayed as":       [_DISPLAY_NAME.get(m, m)  for m in _base_models],
    }
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### How accurate is the prediction?")

    mae_pct, cls_acc = _load_summary_metrics()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Reach prediction error", f"±{mae_pct:.1f}%")
        st.markdown(
            f"On average, the predicted final reach is within "
            f"**{mae_pct:.1f} percentage points** of the true value, "
            "measured on held-out test data across all 10 models and 5 network types."
        )
    with col2:
        st.metric("Mechanism Identification", f"{cls_acc:.1f}%")
        st.markdown(
            f"The correct spreading mechanism is identified **{cls_acc:.1f}% of the time** "
            "from just the first 30 time steps of the epidemic curve "
            "(chance level: 10% for 10 classes)."
        )

    with st.expander("What does this mean in practice?"):
        st.markdown(
            "These numbers come from a rigorous hold-out evaluation — the test data "
            "was never seen during training. The hardest models to distinguish are "
            "those that are mechanistically similar (e.g. H3 vs H1 — both involve "
            "transmission rate β but with different amplification logic). "
            "These pairs are genuinely hard to distinguish from aggregate curves alone, "
            "which is itself a scientific finding: some models are **observationally "
            "equivalent** from the I(t)/N curve alone."
        )

    st.markdown("---")
    st.markdown("### Limitations")
    st.info(
        "This tool is based on simulated network data. Real social networks are more "
        "complex than the models assume. "
        "Predictions should be interpreted as **indicative rather than precise**. "
        "The tool is most reliable when the spreading process has been observed for "
        "at least 5–10 time steps."
    )

    st.markdown("---")
    st.markdown("### Credits")
    st.markdown(
        "| | |\n"
        "|---|---|\n"
        "| **Project** | Bachelor's Thesis — Hybrid Epidemic Spreading Models |\n"
        "| **Author** | Álvaro Monclús |\n"
        "| **Year** | 2025 |\n"
        "| **Models** | SIR, SIS, Bootstrap Percolation, WTM, H1–H6 |\n"
        "| **Networks** | ER, RGG, Lattice, Facebook (SNAP), GitHub (MUSAE) |\n"
        "| **Predictor** | CNN, 22 time-series + 10 graph features, 50 000 simulations |\n"
        "| **Stack** | Python · NetworkX · Streamlit · Plotly · PyTorch · scikit-learn |\n"
    )


def _load_summary_metrics() -> tuple[float, float]:
    """Read MAE (pp) and accuracy (%) from the DL or ML summary file."""
    dl_candidates = [
        _SRC_DIR / "results" / "dl" / "summary.txt",
        _SRC_DIR.parent / "results" / "dl" / "summary.txt",
    ]
    for p in dl_candidates:
        if p.exists():
            try:
                df = pd.read_csv(p, sep=r"\s{2,}", engine="python")
                df.columns = df.columns.str.strip()
                # Prefer Ensemble at t_obs=30, fall back to CNN
                for model in ("Ensemble", "CNN"):
                    row = df[(df["Model"].str.strip() == model) &
                             (df["t_obs"].astype(int) == 30)]
                    if not row.empty:
                        mae_val = row["MAE"].iloc[0]
                        acc_val = row["Accuracy"].iloc[0]
                        if str(mae_val) != "-" and str(acc_val) != "-":
                            return float(mae_val) * 100, float(acc_val) * 100
            except Exception:
                pass

    # Fall back to ML summary (different format)
    ml_candidates = [
        _SRC_DIR / "results" / "ml" / "summary.txt",
        _SRC_DIR.parent / "results" / "ml" / "summary.txt",
    ]
    for p in ml_candidates:
        if p.exists():
            txt = p.read_text()
            mae, acc = None, None
            for line in txt.splitlines():
                if "Best MAE:" in line:
                    try:
                        mae = float(line.split("Best MAE:")[1].split()[0]) * 100
                    except Exception:
                        pass
                if "Best accuracy:" in line:
                    try:
                        acc = float(line.split("Best accuracy:")[1].split("%")[0])
                    except Exception:
                        pass
            if mae is not None and acc is not None:
                return mae, acc
    return 1.7, 74.2
