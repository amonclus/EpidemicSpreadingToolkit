from __future__ import annotations

import os
import tempfile

import streamlit as st

from input.graph_generator import (
    generate_er_graph,
    generate_lattice_graph,
    generate_random_geometric_graph,
)
from input.graph_loader import load_graph_auto
from ui.sidebar import render_sidebar
from ui.state import SessionKeys, clear_sim_results, get_graph_or_stop
from ui.tabs.animation_tab import render_animation_tab
from ui.tabs.simulation_tab import render_simulation_tab
from ui.tabs.stats_tab import render_stats_tab
from ui.tabs.sweep_tab import render_sweep_tab
from ui.tabs.vulnerability_tab import render_vulnerability_tab
from ui.tabs.sir_simulation_tab import render_sir_simulation_tab
from ui.tabs.sir_animation_tab import render_sir_animation_tab
from ui.tabs.sir_vulnerability_tab import render_sir_vulnerability_tab
from ui.tabs.sir_sweep_tab import render_sir_sweep_tab
from ui.tabs.h1_simulation_tab import render_h1_simulation_tab
from ui.tabs.h1_animation_tab import render_h1_animation_tab
from ui.tabs.h1_vulnerability_tab import render_h1_vulnerability_tab
from ui.tabs.h1_sweep_tab import render_h1_sweep_tab
from ui.tabs.h2_simulation_tab import render_h2_simulation_tab
from ui.tabs.h2_animation_tab import render_h2_animation_tab
from ui.tabs.h2_vulnerability_tab import render_h2_vulnerability_tab
from ui.tabs.h2_sweep_tab import render_h2_sweep_tab
from ui.tabs.h3_simulation_tab import render_h3_simulation_tab
from ui.tabs.h3_animation_tab import render_h3_animation_tab
from ui.tabs.h3_vulnerability_tab import render_h3_vulnerability_tab
from ui.tabs.h3_sweep_tab import render_h3_sweep_tab
from ui.tabs.sis_simulation_tab import render_sis_simulation_tab
from ui.tabs.sis_animation_tab import render_sis_animation_tab
from ui.tabs.sis_vulnerability_tab import render_sis_vulnerability_tab
from ui.tabs.sis_sweep_tab import render_sis_sweep_tab
from ui.tabs.wtm_simulation_tab import render_wtm_simulation_tab
from ui.tabs.wtm_animation_tab import render_wtm_animation_tab
from ui.tabs.wtm_vulnerability_tab import render_wtm_vulnerability_tab
from ui.tabs.wtm_sweep_tab import render_wtm_sweep_tab
from ui.tabs.h4_simulation_tab import render_h4_simulation_tab
from ui.tabs.h4_animation_tab import render_h4_animation_tab
from ui.tabs.h4_vulnerability_tab import render_h4_vulnerability_tab
from ui.tabs.h4_sweep_tab import render_h4_sweep_tab
from ui.tabs.h5_simulation_tab import render_h5_simulation_tab
from ui.tabs.h5_animation_tab import render_h5_animation_tab
from ui.tabs.h5_vulnerability_tab import render_h5_vulnerability_tab
from ui.tabs.h5_sweep_tab import render_h5_sweep_tab
from ui.tabs.h6_simulation_tab import render_h6_simulation_tab
from ui.tabs.h6_animation_tab import render_h6_animation_tab
from ui.tabs.h6_vulnerability_tab import render_h6_vulnerability_tab
from ui.tabs.h6_sweep_tab import render_h6_sweep_tab
from ui.tabs.ml_tab import (
    render_ml_virality_tab,
    render_ml_education_tab,
    render_ml_about_tab,
)


_CSS = """
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography ── */
h1 { letter-spacing: -0.5px; font-weight: 700; }
h2 { letter-spacing: -0.3px; font-weight: 600; }
h3 { font-weight: 600; }

/* ── Bordered containers (cards) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    transition: box-shadow 0.18s ease, border-color 0.18s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 6px 24px rgba(37, 99, 235, 0.09) !important;
    border-color: #BFDBFE !important;
}

/* ── Equal-height cards with button pinned to bottom ── */
/* Only targets rows that actually contain bordered cards */
[data-testid="stHorizontalBlock"]:has([data-testid="stVerticalBlockBorderWrapper"]) {
    align-items: stretch;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    display: flex;
    flex-direction: column;
    height: 100%;
}
[data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
    flex: 1;
    display: flex;
    flex-direction: column;
}
[data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] > div:last-child {
    margin-top: auto;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: opacity 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.18) !important;
    opacity: 0.92 !important;
}

/* ── Metric tiles ── */
[data-testid="metric-container"] {
    background-color: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0 !important; font-weight: 500; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { border-right: 1px solid #E2E8F0; }

/* ── Progress bar ── */
.stProgress > div > div { border-radius: 99px; }
</style>
"""


def run_app() -> None:
    st.set_page_config(
        page_title="Network Contagion Lab",
        page_icon="🌐",
        layout="centered",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    graph = st.session_state.get(SessionKeys.GRAPH)
    model = st.session_state.get(SessionKeys.MODEL)

    # Welcome
    if not st.session_state.get(SessionKeys.WELCOMED):
        _render_welcome()
        return

    # Step 1: load a graph
    if graph is None:
        _render_graph_setup()
        return

    # Step 2: choose a model
    if model is None:
        _render_model_selection(graph)
        return

    # Step 3: run the selected model
    if model == "bootstrap":
        st.title("Bootstrap Percolation — Network Risk Analysis")
        config = render_sidebar(model="bootstrap")
        _render_tabs_bootstrap(graph, config)
    elif model == "sir":
        st.title("SIR Epidemic Model — Network Spread Analysis")
        config = render_sidebar(model="sir")
        _render_tabs_sir(graph, config)
    elif model == "h1":
        st.title("H1 — OR-Hybrid Contagion Model")
        config = render_sidebar(model="h1")
        _render_tabs_h1(graph, config)
    elif model == "h2":
        st.title("H2 — Sequential Hybrid (Switching Model)")
        config = render_sidebar(model="h2")
        _render_tabs_h2(graph, config)
    elif model == "h3":
        st.title("H3 — Probabilistic Threshold Hybrid")
        config = render_sidebar(model="h3")
        _render_tabs_h3(graph, config)
    elif model == "sis":
        st.title("SIS Epidemic Model — Reinfection Dynamics")
        config = render_sidebar(model="sis")
        _render_tabs_sis(graph, config)
    elif model == "wtm":
        st.title("WTM — Watts Threshold Model")
        config = render_sidebar(model="wtm")
        _render_tabs_wtm(graph, config)
    elif model == "h4":
        st.title("H4 — OR-Hybrid: SIS + Watts Threshold Model")
        config = render_sidebar(model="h4")
        _render_tabs_h4(graph, config)
    elif model == "h5":
        st.title("H5 — Sequential Hybrid: SIS then WTM")
        config = render_sidebar(model="h5")
        _render_tabs_h5(graph, config)
    elif model == "h6":
        st.title("H6 — Probabilistic WTM (Soft Threshold)")
        config = render_sidebar(model="h6")
        _render_tabs_h6(graph, config)
    elif model == "ml":
        st.title("Cascade predictor")
        with st.sidebar:
            if st.button("← Back to Lab", use_container_width=True):
                st.session_state.pop(SessionKeys.MODEL, None)
                st.rerun()
            st.markdown("---")
        _render_tabs_ml()


def _render_welcome() -> None:
    st.title("🌐 Network Contagion Lab")
    st.markdown(
        "A research platform for studying how things spread through networks — "
        "cascading failures, epidemics, information, and social influence."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🔬 10 Spreading Models")
        st.markdown(
            "From classical SIR epidemics and bootstrap percolation "
            "to six hybrid models that blend threshold and probabilistic dynamics."
        )
    with col2:
        st.markdown("#### 📊 Built-in Analysis")
        st.markdown(
            "Cascade simulations, node vulnerability maps, parameter sweeps, "
            "and animated cascade walkthroughs — all in one place."
        )
    with col3:
        st.markdown("#### 🤖 ML Virality Predictor")
        st.markdown(
            "A Random Forest trained on 50 000 simulations that identifies "
            "the spreading mechanism and estimates final reach from early trajectory data."
        )

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. **Load a network** — generate an Erdős–Rényi, Random Geometric, or Lattice graph, "
        "or upload your own (DIMACS, edge list, GML).\n"
        "2. **Choose a model** — pick the spreading dynamic you want to study.\n"
        "3. **Explore** — run simulations, sweep parameters, and inspect which nodes matter most.\n\n"
        "Your graph stays loaded as you switch between models, so you can compare dynamics "
        "side-by-side without reloading."
    )

    st.markdown("---")
    col_btn, _, _ = st.columns(3)
    if col_btn.button("Get started →", type="primary", use_container_width=True):
        st.session_state[SessionKeys.WELCOMED] = True
        st.rerun()


def _render_graph_setup() -> None:
    st.title("🌐 Network Contagion Lab")
    st.markdown("**Step 1 of 2 — Set up your network.** Choose how to provide the graph you want to analyse.")
    st.markdown("---")

    tab_generate, tab_upload = st.tabs(["Generate a graph", "Upload a file"])

    with tab_generate:
        graph_type = st.selectbox(
            "Graph type", ["Erdős–Rényi", "Random Geometric", "Lattice"]
        )

        if graph_type == "Erdős–Rényi":
            col1, col2 = st.columns(2)
            n = col1.number_input("Number of nodes", 10, 5000, 100)
            p = col2.slider("Edge probability (p)", 0.01, 1.0, 0.1, 0.01)
            if st.button("Generate graph", type="primary"):
                with st.spinner("Generating graph…"):
                    graph = generate_er_graph(int(n), float(p))
                _store_graph(graph)

        elif graph_type == "Random Geometric":
            col1, col2 = st.columns(2)
            n = col1.number_input("Number of nodes", 10, 5000, 100)
            radius = col2.slider("Connection radius (r)", 0.01, 1.0, 0.2, 0.01)
            if st.button("Generate graph", type="primary"):
                with st.spinner("Generating graph…"):
                    graph = generate_random_geometric_graph(int(n), float(radius))
                _store_graph(graph)

        else:  # Lattice
            grid_size = st.number_input("Grid side length", 3, 100, 10)
            if st.button("Generate graph", type="primary"):
                with st.spinner("Generating graph…"):
                    graph = generate_lattice_graph(int(grid_size))
                _store_graph(graph)

    with tab_upload:
        st.caption("Supported formats: DIMACS (`.dimacs`), edge list (`.txt`, `.edgelist`), GML (`.gml`)")
        uploaded = st.file_uploader(
            "Upload graph file", type=["dimacs", "txt", "gml", "edgelist"]
        )

        if uploaded is not None:
            suffix = os.path.splitext(uploaded.name)[1] or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Loading graph…"):
                    graph = load_graph_auto(tmp_path)
                _store_graph(graph)
            except Exception as exc:
                st.error(f"Could not parse the file: {exc}")
            finally:
                os.unlink(tmp_path)


def _store_graph(graph) -> None:
    """Persist a new graph and clear any stale simulation results."""
    clear_sim_results()
    st.session_state[SessionKeys.GRAPH] = graph
    st.rerun()


def _render_model_row(models: list) -> None:
    cols = st.columns(len(models))
    for col, (model_key, title, desc, params, btn_label) in zip(cols, models):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)
                st.caption(f"`{params}`")
                if st.button(btn_label, use_container_width=True, key=f"btn_{model_key}"):
                    st.session_state[SessionKeys.MODEL] = model_key
                    st.rerun()


def _render_model_selection(graph) -> None:
    st.title("🌐 Network Contagion Lab")

    # ── Active graph summary ───────────────────────────────────────────
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    col_info, col_btn = st.columns([3, 1])
    col_info.success(f"Graph loaded — **{n}** nodes, **{m}** edges. Now choose a spreading model.")
    if col_btn.button("Change graph", use_container_width=True):
        clear_sim_results()
        st.session_state.pop(SessionKeys.GRAPH, None)
        st.rerun()

    # ── Network spreading models ───────────────────────────────────────
    st.markdown("## Network Spreading Models")

    st.markdown("#### Base Models")
    _BASE_MODELS = [
        ("bootstrap", "Bootstrap Percolation",
         "Hard threshold — node activates when ≥ k neighbours are infected.",
         "k",  "Select →"),
        ("sir", "SIR",
         "Probabilistic spread with permanent immunity (β, γ).",
         "β · γ", "Select →"),
        ("sis", "SIS",
         "Probabilistic spread, no immunity — reinfection possible.",
         "β · μ", "Select →"),
        ("wtm", "Watts Threshold",
         "Fractional threshold — activates when ≥ φ fraction of neighbours infected.",
         "φ", "Select →"),
    ]
    _render_model_row(_BASE_MODELS)

    st.markdown("#### Hybrid Models")
    st.caption("Combine epidemic and threshold mechanisms.")
    _HYBRID_MODELS = [
        ("h1", "H1 — SIR ∨ Bootstrap",
         "Activates via SIR **or** bootstrap threshold, whichever fires first.",
         "k · β · γ", "Select →"),
        ("h2", "H2 — SIR → Bootstrap",
         "SIR phase until fraction f infected, then switches to bootstrap.",
         "k · β · γ · f", "Select →"),
        ("h3", "H3 — Soft Bootstrap",
         "Infection probability grows linearly with infected-neighbour count.",
         "β · γ", "Select →"),
        ("h4", "H4 — SIS ∨ WTM",
         "Activates via SIS **or** fractional WTM threshold.",
         "φ · β · γ", "Select →"),
        ("h5", "H5 — SIS → WTM",
         "SIS phase until fraction f infected, then switches to WTM.",
         "φ · β · γ · f", "Select →"),
        ("h6", "H6 — Soft WTM",
         "Infection probability scales with the fraction of infected neighbours.",
         "φ · γ", "Select →"),
    ]
    _render_model_row(_HYBRID_MODELS[:3])
    _render_model_row(_HYBRID_MODELS[3:])

    # ── ML Predictor — visually separated ─────────────────────────────
    st.markdown("---")
    st.markdown("## ML Virality Predictor")
    with st.container(border=True):
        col_desc, col_btn2 = st.columns([3, 1])
        col_desc.markdown("**No network needed** — works from early spreading trajectory alone.")
        col_desc.caption(
            "Predict how far a contagion will spread and identify the underlying spreading "
            "mechanism using a Random Forest trained on 50 000 simulations."
        )
        if col_btn2.button("Open Predictor →", use_container_width=True, key="btn_ml"):
            st.session_state[SessionKeys.MODEL] = "ml"
            st.rerun()


def _render_tabs_bootstrap(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Cascade Simulation",
            "🎬 Cascade Animation",
            "🎯 Node Vulnerability",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_simulation_tab(graph, config)

    with tab_anim:
        render_animation_tab(graph, config)

    with tab_vuln:
        render_vulnerability_tab(graph, config)

    with tab_sweep:
        render_sweep_tab(graph, config)


def _render_tabs_sir(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Epidemic Simulation",
            "🎬 Epidemic Animation",
            "🎯 Node Vulnerability",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_sir_simulation_tab(graph, config)

    with tab_anim:
        render_sir_animation_tab(graph, config)

    with tab_vuln:
        render_sir_vulnerability_tab(graph, config)

    with tab_sweep:
        render_sir_sweep_tab(graph, config)


def _render_tabs_h3(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Cascade Simulation",
            "🎬 Cascade Animation",
            "🎯 Node Vulnerability",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_h3_simulation_tab(graph, config)

    with tab_anim:
        render_h3_animation_tab(graph, config)

    with tab_vuln:
        render_h3_vulnerability_tab(graph, config)

    with tab_sweep:
        render_h3_sweep_tab(graph, config)


def _render_tabs_h2(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Cascade Simulation",
            "🎬 Cascade Animation",
            "🎯 Node Vulnerability",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_h2_simulation_tab(graph, config)

    with tab_anim:
        render_h2_animation_tab(graph, config)

    with tab_vuln:
        render_h2_vulnerability_tab(graph, config)

    with tab_sweep:
        render_h2_sweep_tab(graph, config)


def _render_tabs_sis(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        ["📊 Graph Statistics", "🔬 Epidemic Simulation", "🎬 Epidemic Animation",
         "🎯 Node Vulnerability", "📈 Parameter Sweep"]
    )
    with tab_stats:
        render_stats_tab(graph)
    with tab_sim:
        render_sis_simulation_tab(graph, config)
    with tab_anim:
        render_sis_animation_tab(graph, config)
    with tab_vuln:
        render_sis_vulnerability_tab(graph, config)
    with tab_sweep:
        render_sis_sweep_tab(graph, config)


def _render_tabs_wtm(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        ["📊 Graph Statistics", "🔬 Cascade Simulation", "🎬 Cascade Animation",
         "🎯 Node Vulnerability", "📈 Parameter Sweep"]
    )
    with tab_stats:
        render_stats_tab(graph)
    with tab_sim:
        render_wtm_simulation_tab(graph, config)
    with tab_anim:
        render_wtm_animation_tab(graph, config)
    with tab_vuln:
        render_wtm_vulnerability_tab(graph, config)
    with tab_sweep:
        render_wtm_sweep_tab(graph, config)


def _render_tabs_h4(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        ["📊 Graph Statistics", "🔬 Cascade Simulation", "🎬 Cascade Animation",
         "🎯 Node Vulnerability", "📈 Parameter Sweep"]
    )
    with tab_stats:
        render_stats_tab(graph)
    with tab_sim:
        render_h4_simulation_tab(graph, config)
    with tab_anim:
        render_h4_animation_tab(graph, config)
    with tab_vuln:
        render_h4_vulnerability_tab(graph, config)
    with tab_sweep:
        render_h4_sweep_tab(graph, config)


def _render_tabs_h5(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        ["📊 Graph Statistics", "🔬 Cascade Simulation", "🎬 Cascade Animation",
         "🎯 Node Vulnerability", "📈 Parameter Sweep"]
    )
    with tab_stats:
        render_stats_tab(graph)
    with tab_sim:
        render_h5_simulation_tab(graph, config)
    with tab_anim:
        render_h5_animation_tab(graph, config)
    with tab_vuln:
        render_h5_vulnerability_tab(graph, config)
    with tab_sweep:
        render_h5_sweep_tab(graph, config)


def _render_tabs_h6(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        ["📊 Graph Statistics", "🔬 Cascade Simulation", "🎬 Cascade Animation",
         "🎯 Node Vulnerability", "📈 Parameter Sweep"]
    )
    with tab_stats:
        render_stats_tab(graph)
    with tab_sim:
        render_h6_simulation_tab(graph, config)
    with tab_anim:
        render_h6_animation_tab(graph, config)
    with tab_vuln:
        render_h6_vulnerability_tab(graph, config)
    with tab_sweep:
        render_h6_sweep_tab(graph, config)


def _render_tabs_ml() -> None:
    tab_predict, tab_learn, tab_about = st.tabs([
        "Prediction",
        "How Does It Spread?",
        "ℹ️ About",
    ])
    with tab_predict:
        render_ml_virality_tab()
    with tab_learn:
        render_ml_education_tab()
    with tab_about:
        render_ml_about_tab()


def _render_tabs_h1(graph, config) -> None:
    tab_stats, tab_sim, tab_anim, tab_vuln, tab_sweep = st.tabs(
        [
            "📊 Graph Statistics",
            "🔬 Cascade Simulation",
            "🎬 Cascade Animation",
            "🎯 Node Vulnerability",
            "📈 Parameter Sweep",
        ]
    )

    with tab_stats:
        render_stats_tab(graph)

    with tab_sim:
        render_h1_simulation_tab(graph, config)

    with tab_anim:
        render_h1_animation_tab(graph, config)

    with tab_vuln:
        render_h1_vulnerability_tab(graph, config)

    with tab_sweep:
        render_h1_sweep_tab(graph, config)
