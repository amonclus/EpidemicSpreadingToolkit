"""
lab_callbacks.py — Callbacks for the per-model lab page.

Covers all 10 models:
  bootstrap, sir, sis, wtm, h1, h2, h3, h4, h5, h6

Five tabs per model: stats, simulation, animation, vulnerability, sweep.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import dash_bootstrap_components as dbc
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update

from analysis.graph_statistics import compute_graph_statistics, degree_distribution
from simulation.bootstrap import BootstrapPercolation
from simulation.H1 import H1Model
from simulation.H2 import H2Model
from simulation.H3 import H3Model
from simulation.H4 import H4Model
from simulation.H5 import H5Model
from simulation.H6 import H6Model
from simulation.seed_selection import SeedStrategy, select_seeds
from simulation.sir import SIRModel
from simulation.sis import SISModel
from simulation.wtm import WTMModel
from ui.charts import apply_layout_geometry, build_edge_trace, resolve_positions
from ui.components.store import graph_from_store
from ui.state import SidebarConfig
from visualization.visualization import animate_cascade

LARGE_GRAPH_THRESHOLD = 500

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(seed_fraction, num_trials, seed_strategy,
                 threshold, phi, beta, gamma, switch_fraction) -> SidebarConfig:
    """Build a SidebarConfig from raw callback values, with safe defaults."""
    return SidebarConfig(
        threshold=int(threshold or 2),
        seed_fraction=float(seed_fraction or 0.05),
        num_trials=int(num_trials or 50),
        beta=float(beta or 0.3),
        gamma=float(gamma or 0.1),
        seed_strategy=str(seed_strategy or SeedStrategy.RANDOM),
        switch_fraction=float(switch_fraction or 0.2),
        phi=float(phi or 0.3),
    )


def _make_sim(model: str, graph: nx.Graph, config: SidebarConfig):
    """Instantiate the correct simulation class."""
    if model == "bootstrap":
        return BootstrapPercolation(graph, config.threshold)
    if model == "sir":
        return SIRModel(graph, beta=config.beta, gamma=config.gamma)
    if model == "sis":
        return SISModel(graph, beta=config.beta, gamma=config.gamma)
    if model == "wtm":
        return WTMModel(graph, phi=config.phi)
    if model == "h1":
        return H1Model(graph, threshold=config.threshold, beta=config.beta, gamma=config.gamma)
    if model == "h2":
        return H2Model(graph, threshold=config.threshold, beta=config.beta,
                       gamma=config.gamma, switch_fraction=config.switch_fraction)
    if model == "h3":
        return H3Model(graph, beta=config.beta, gamma=config.gamma)
    if model == "h4":
        return H4Model(graph, phi=config.phi, beta=config.beta, gamma=config.gamma)
    if model == "h5":
        return H5Model(graph, phi=config.phi, beta=config.beta,
                       gamma=config.gamma, switch_fraction=config.switch_fraction)
    if model == "h6":
        return H6Model(graph, phi=config.phi, gamma=config.gamma)
    raise ValueError(f"Unknown model: {model}")


def _metric_card(label: str, value: str) -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.P(label, className="text-muted small mb-1"),
                html.H4(value, className="fw-bold mb-0", style={"color": "#1E3A8A"}),
            ])
        ], className="shadow-sm h-100",
           style={"borderLeft": "3px solid #BFDBFE", "borderRadius": "8px"}),
    ], md=3, className="mb-2")


def _empty_state(msg: str = "Click 'Run Simulation' to see results.") -> html.Div:
    return html.Div([
        dbc.Alert(msg, color="light", className="text-muted text-center py-4"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Stats tab content (shared across all models)
# ─────────────────────────────────────────────────────────────────────────────

def _build_stats_content(graph: nx.Graph) -> html.Div:
    n_nodes = graph.number_of_nodes()

    # Graph visualisation
    viz_section = []
    if n_nodes > LARGE_GRAPH_THRESHOLD:
        viz_section = [dbc.Alert(
            f"Graph has {n_nodes:,} nodes — visualisation disabled. Statistics shown below.",
            color="info",
        )]
    else:
        pos, is_lattice, has_pos = resolve_positions(graph)
        node_list = list(graph.nodes())
        edge_width = 0.5 if nx.density(graph) > 0.15 else 0.8
        edge_trace = build_edge_trace(graph, pos, edge_width=edge_width)

        if is_lattice:
            node_marker = dict(size=10, color="steelblue", symbol="square",
                               line=dict(width=1, color="darkgray"))
            hover_text = [f"({n[0]},{n[1]})" for n in node_list]
        elif has_pos:
            degrees = [graph.degree(n) for n in node_list]
            node_marker = dict(size=7, color=degrees, colorscale="Viridis",
                               showscale=True,
                               colorbar=dict(title="Degree", thickness=12, len=0.5),
                               line=dict(width=0.5, color="white"))
            hover_text = [f"Node {n}  (deg {graph.degree(n)})" for n in node_list]
        else:
            degrees = [graph.degree(n) for n in node_list]
            node_marker = dict(
                size=int(max(4, min(8, 300 / n_nodes))),
                color=degrees, colorscale="Plasma", showscale=True,
                colorbar=dict(title="Degree", thickness=12, len=0.5),
                line=dict(width=0.5, color="white"),
            )
            hover_text = [f"Node {n}  (deg {graph.degree(n)})" for n in node_list]

        node_trace = go.Scatter(
            x=[pos[n][0] for n in node_list],
            y=[pos[n][1] for n in node_list],
            mode="markers", text=hover_text, hoverinfo="text",
            marker=node_marker,
        )
        layout_kw = dict(
            title="Network Structure", showlegend=False, hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="#F8FAFC", plot_bgcolor="#F8FAFC",
        )
        apply_layout_geometry(layout_kw, node_list, pos, is_lattice, has_pos)
        layout_kw.pop("width", None)   # let the graph fill its container
        layout_kw.pop("height", None)  # height controlled via CSS below
        fig_graph = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(**layout_kw))
        viz_section = [
            html.H5("Graph Visualisation", className="fw-semibold mb-2"),
            dcc.Graph(figure=fig_graph, responsive=True,
                      style={"height": "450px", "width": "100%"}),
        ]

    # Stats
    stats = compute_graph_statistics(graph)
    dd = degree_distribution(graph)
    df_dd = pd.DataFrame(sorted(dd.items()), columns=["Degree", "Count"])
    fig_dd = px.bar(df_dd, x="Degree", y="Count", title="Degree Distribution",
                    color_discrete_sequence=["#2563EB"])
    fig_dd.update_layout(paper_bgcolor="#F8FAFC", plot_bgcolor="#F8FAFC",
                          margin=dict(l=40, r=20, t=40, b=40))

    diam_val = stats["diameter"] if stats["diameter"] >= 0 else "N/A"

    return html.Div([
        *viz_section,
        html.Hr(),
        html.H5("Structural Statistics", className="fw-semibold mb-3"),
        dbc.Row([
            _metric_card("Nodes", str(stats["nodes"])),
            _metric_card("Edges", str(stats["edges"])),
            _metric_card("Density", f"{stats['density']:.4f}"),
            _metric_card("Avg Degree", f"{stats['average_degree']:.2f}"),
        ]),
        dbc.Row([
            _metric_card("Min Degree", str(stats["min_degree"])),
            _metric_card("Max Degree", str(stats["max_degree"])),
            _metric_card("Avg Clustering", f"{stats['average_clustering']:.4f}"),
            _metric_card("Components", str(stats["num_components"])),
        ]),
        dbc.Row([
            _metric_card("Diameter", str(diam_val)),
            _metric_card("Avg Path Length", f"{stats['average_path_length']:.2f}"),
        ]),
        html.Hr(),
        html.H5("Degree Distribution", className="fw-semibold mb-2"),
        dcc.Graph(figure=fig_dd),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Simulation results renderer (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _render_sim_results(sim_data: dict, model: str, num_trials: int) -> html.Div:
    """Build the simulation results panel from stored sim_data dict."""
    result_d = sim_data.get("result", {})
    metrics_d = sim_data.get("metrics", {})

    # Cascade/epidemic fraction key varies by model
    frac_key = "epidemic_fraction" if model in ("sir", "sis") else "cascade_fraction"
    time_key = "time_to_epidemic"   if model in ("sir", "sis") else "time_to_cascade"
    large_key = "is_large_epidemic" if model in ("sir", "sis") else (
        "is_full_cascade" if model == "bootstrap" else "is_large_cascade"
    )
    size_label = "Epidemic Fraction" if model in ("sir", "sis") else "Cascade Fraction"
    time_label = "Rounds"

    frac = result_d.get(frac_key, 0.0)
    time_val = result_d.get(time_key, 0)
    large = result_d.get(large_key, False)
    robustness = (1 - frac) * (1 / (1 + time_val)) if time_val is not None else 0.0

    single_run = [
        html.H5("Single-run result", className="fw-semibold mb-3"),
        dbc.Row([
            _metric_card(size_label, f"{frac:.4f}"),
            _metric_card(time_label, str(time_val)),
            _metric_card("Large event?", "Yes" if large else "No"),
            _metric_card("Robustness score", f"{robustness:.4f}"),
        ]),
    ]

    # H2 phase breakdown
    h2_section = []
    if model == "h2" and result_d.get("switched") is not None:
        if result_d.get("switched"):
            h2_section = [
                html.H6("Phase breakdown", className="fw-semibold mt-3"),
                dbc.Row([
                    _metric_card("Switch triggered?", "Yes"),
                    _metric_card("Infected at switch", f"{result_d.get('switch_fraction', 0):.2%}"),
                    _metric_card("Phase 1 rounds (SIR)", str(result_d.get("rounds_phase1", "—"))),
                    _metric_card("Phase 2 rounds (BP)", str(result_d.get("rounds_phase2", "—"))),
                ]),
            ]
        else:
            h2_section = [
                dbc.Alert("Switch threshold was never reached — ran as pure SIR.",
                          color="info", className="mt-2"),
            ]

    h5_section = []
    if model == "h5" and result_d.get("switched") is not None:
        if result_d.get("switched"):
            h5_section = [
                html.H6("Phase breakdown", className="fw-semibold mt-3"),
                dbc.Row([
                    _metric_card("Switch triggered?", "Yes"),
                    _metric_card("Infected at switch", f"{result_d.get('switch_fraction', 0):.2%}"),
                    _metric_card("Phase 1 rounds (SIS)", str(result_d.get("rounds_phase1", "—"))),
                    _metric_card("Phase 2 rounds (WTM)", str(result_d.get("rounds_phase2", "—"))),
                ]),
            ]

    # Averaged metrics
    avg_size = metrics_d.get("cascade_size", metrics_d.get("epidemic_size", 0.0))
    avg_prob = metrics_d.get("cascade_probability", metrics_d.get("epidemic_probability", 0.0))
    avg_time = metrics_d.get("time_to_cascade", metrics_d.get("time_to_epidemic", 0.0))
    crit_seed = metrics_d.get("critical_seed_size", "—")
    threshold = metrics_d.get("percolation_threshold", metrics_d.get("epidemic_threshold",
                metrics_d.get("cascade_threshold", 0.0)))

    avg_section = [
        html.Hr(),
        html.H5(f"Averaged metrics ({num_trials} trials)", className="fw-semibold mb-3"),
    ]
    if avg_size == 0.0:
        avg_section.append(
            dbc.Alert("No cascade spread beyond the seeds with these parameters.", color="warning")
        )
    else:
        avg_section += [
            dbc.Row([
                _metric_card("Avg spread fraction", f"{avg_size:.4f}"),
                _metric_card("Cascade probability", f"{avg_prob:.4f}"),
                _metric_card("Critical seed size", str(crit_seed)),
                _metric_card("Avg rounds", f"{avg_time:.2f}"),
            ]),
            dbc.Row([
                _metric_card("Threshold (seed fraction)", f"{threshold:.4f}"),
            ]),
        ]

    # H2 extra metrics
    if model in ("h2", "h5") and metrics_d.get("switch_probability") is not None:
        avg_section += [
            dbc.Row([
                _metric_card("Switch triggered (fraction of trials)",
                             f"{metrics_d['switch_probability']:.2%}"),
            ])
        ]

    return html.Div(single_run + h2_section + h5_section + avg_section)


# ─────────────────────────────────────────────────────────────────────────────
# Animation content builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_animation_content(model: str, graph: nx.Graph, config: SidebarConfig) -> html.Div:
    n = graph.number_of_nodes()
    large = n > LARGE_GRAPH_THRESHOLD

    seed_size = max(1, int(config.seed_fraction * n))
    sim = _make_sim(model, graph, config)
    seed_nodes = set(select_seeds(graph, seed_size, config.seed_strategy))

    if large:
        result, _ = sim.run(seed_nodes, record_sequence=False)
        frac_key = "epidemic_fraction" if model in ("sir", "sis") else "cascade_fraction"
        size_key = "epidemic_size"     if model in ("sir", "sis") else "cascade_size"
        time_key = "time_to_epidemic"  if model in ("sir", "sis") else "time_to_cascade"
        frac = getattr(result, frac_key, 0.0)
        size = getattr(result, size_key, 0)
        time_val = getattr(result, time_key, 0)
        return html.Div([
            dbc.Alert(
                f"Graph has {n:,} nodes — animation disabled. "
                f"Spread reached {size}/{n} nodes ({frac:.2%}) in {time_val} round(s).",
                color="info",
            )
        ])
    else:
        result, activation_sequence = sim.run(seed_nodes, record_sequence=True)
        fig = animate_cascade(graph, activation_sequence, show=False)
        fig.layout.width = None
        fig.layout.height = None
        frac_key = "epidemic_fraction" if model in ("sir", "sis") else "cascade_fraction"
        size_key = "epidemic_size"     if model in ("sir", "sis") else "cascade_size"
        time_key = "time_to_epidemic"  if model in ("sir", "sis") else "time_to_cascade"
        frac = getattr(result, frac_key, 0.0)
        size = getattr(result, size_key, 0)
        time_val = getattr(result, time_key, 0)
        return html.Div([
            dcc.Graph(figure=fig, responsive=True, style={"height": "500px", "width": "100%"}),
            dbc.Alert(
                f"Spread reached {size}/{n} nodes ({frac:.2%}) in {time_val} round(s).",
                color="info", className="mt-2",
            ),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Vulnerability content builder (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _build_vuln_map(graph: nx.Graph, value_map: dict, hover_labels: list,
                    title: str, colorbar_title: str, colorscale: str) -> dcc.Graph:
    pos, is_lattice, has_pos = resolve_positions(graph)
    node_list = list(graph.nodes())
    edge_trace = build_edge_trace(graph, pos, edge_width=0.5)

    n = graph.number_of_nodes()
    if is_lattice:
        marker = dict(size=10, color=[value_map[nd] for nd in node_list],
                      colorscale=colorscale, showscale=True,
                      colorbar=dict(title=colorbar_title, thickness=12, len=0.6),
                      symbol="square", line=dict(width=0.5, color="white"))
    else:
        marker = dict(size=int(max(5, min(10, 400 / n))),
                      color=[value_map[nd] for nd in node_list],
                      colorscale=colorscale, showscale=True,
                      colorbar=dict(title=colorbar_title, thickness=12, len=0.6),
                      symbol="circle", line=dict(width=0.5, color="white"))

    node_trace = go.Scatter(
        x=[pos[nd][0] for nd in node_list], y=[pos[nd][1] for nd in node_list],
        mode="markers", text=hover_labels, hoverinfo="text", marker=marker,
    )
    layout_kw = dict(
        title=title, showlegend=False, hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#F8FAFC", plot_bgcolor="#F8FAFC",
    )
    apply_layout_geometry(layout_kw, node_list, pos, is_lattice, has_pos)
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(**layout_kw))
    return dcc.Graph(figure=fig)


def _build_vuln_content(graph: nx.Graph, vuln_data: list[dict],
                        block_data: list[dict], baseline: dict) -> html.Div:
    # Activation map
    score_map = {row["node"]: row["influence_score"] for row in vuln_data}
    cp_map    = {row["node"]: row["cascade_probability"] for row in vuln_data}
    time_map  = {row["node"]: row["avg_time"] for row in vuln_data}
    hover_a = [
        f"Node {nd}<br>Influence: {score_map[nd]:.4f}<br>"
        f"Cascade Prob: {cp_map[nd]:.2%}<br>Avg Time: {time_map[nd]:.1f} rounds<br>"
        f"Degree: {graph.degree(nd)}"
        for nd in graph.nodes()
    ]
    vuln_graph = _build_vuln_map(graph, score_map, hover_a,
                                  "Node Influence Map", "Influence", "RdBu_r")

    # Blocking map
    red_map      = {row["node"]: row["cascade_reduction"] for row in block_data}
    prob_red_map = {row["node"]: row["prob_reduction"] for row in block_data}
    hover_b = [
        f"Node {nd}<br>Cascade Reduction: {red_map[nd]:.4f}<br>"
        f"Prob Reduction: {prob_red_map[nd]:.2%}<br>Degree: {graph.degree(nd)}"
        for nd in graph.nodes()
    ]
    block_graph = _build_vuln_map(graph, red_map, hover_b,
                                   "Blocking Effectiveness Map", "Cascade<br>Reduction", "Greens")

    df_vuln  = pd.DataFrame(vuln_data)
    df_block = pd.DataFrame(block_data)
    top_n = min(10, len(df_vuln))
    top_b = min(10, len(df_block))

    fmt_v = {"influence_score": "{:.4f}", "cascade_probability": "{:.2%}",
              "avg_time": "{:.1f}", "cascade_std": "{:.4f}",
              "betweenness": "{:.4f}", "closeness": "{:.4f}"}
    fmt_b = {"cascade_reduction": "{:.4f}", "prob_reduction": "{:.2%}",
              "cascade_blocked": "{:.4f}", "prob_blocked": "{:.2%}",
              "time_blocked": "{:.1f}", "betweenness": "{:.4f}", "closeness": "{:.4f}"}

    return html.Div([
        html.H5("Node Activation Analysis", className="fw-semibold mb-2"),
        html.P("Which nodes, if infected, trigger the largest cascades?",
               className="text-muted small"),
        vuln_graph,
        dbc.Row([
            dbc.Col([
                html.H6("Top 10 Most Influential", className="fw-semibold text-danger"),
                _df_table(df_vuln.head(top_n), fmt_v),
            ], md=6),
            dbc.Col([
                html.H6("Top 10 Least Influential", className="fw-semibold text-primary"),
                _df_table(df_vuln.tail(top_n).iloc[::-1], fmt_v),
            ], md=6),
        ]),
        html.Hr(),
        html.H5("Node Blocking Analysis", className="fw-semibold mb-2"),
        dbc.Alert(
            [
                html.Strong("Baseline: "),
                f"avg cascade fraction {baseline['avg']:.4f}, "
                f"full-cascade probability {baseline['prob']:.2%}",
            ],
            color="light", className="py-2",
        ),
        block_graph,
        dbc.Row([
            dbc.Col([
                html.H6("Most Critical to Protect", className="fw-semibold text-success"),
                _df_table(df_block.head(top_b), fmt_b),
            ], md=6),
            dbc.Col([
                html.H6("Least Critical to Protect", className="fw-semibold text-muted"),
                _df_table(df_block.tail(top_b).iloc[::-1], fmt_b),
            ], md=6),
        ]),
    ])


def _df_table(df: pd.DataFrame, fmt: dict) -> dbc.Table:
    styled = df.copy()
    for col, f in fmt.items():
        if col in styled.columns:
            styled[col] = styled[col].apply(lambda v: f.format(v))
    return dbc.Table.from_dataframe(
        styled, striped=True, bordered=False, hover=True,
        responsive=True, size="sm", className="small",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sweep content
# ─────────────────────────────────────────────────────────────────────────────

def _build_sweep_layout(model: str) -> html.Div:
    """Build the sweep tab controls (static layout, no data yet)."""
    sweep_options = _sweep_options_for_model(model)
    return html.Div([
        html.H5("Parameter Sweep", className="fw-semibold mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Sweep type", className="form-label small fw-medium"),
                dbc.Select(id="sweep-type", options=sweep_options,
                           value=sweep_options[0]["value"], size="sm"),
            ], md=4),
            dbc.Col([
                html.Label("Trials per point", className="form-label small fw-medium"),
                dbc.Input(id="sweep-trials", type="number", min=5, max=200,
                          value=20, step=5, size="sm"),
            ], md=3),
        ], className="mb-3"),
        html.Div(id="sweep-param-controls"),
        dbc.Button("Run Sweep", id="btn-run-sweep", color="primary",
                   className="mt-2",
                   style={"backgroundColor": "#2563EB", "borderColor": "#2563EB"}),
        dcc.Loading(html.Div(id="sweep-output"), type="circle"),
    ])


def _sweep_options_for_model(model: str) -> list[dict]:
    base = [{"label": "Seed Fraction", "value": "seed_fraction"}]
    if model in ("sir", "sis", "h1", "h2", "h3", "h4", "h5"):
        base.append({"label": "Transmission rate (β)", "value": "beta"})
    if model in ("wtm", "h4", "h5", "h6"):
        base.append({"label": "Fractional threshold (φ)", "value": "phi"})
    if model == "bootstrap":
        base.append({"label": "Bootstrap threshold (k)", "value": "threshold"})
    base += [
        {"label": "Erdős–Rényi probability", "value": "er_prob"},
        {"label": "Lattice size", "value": "lattice_size"},
    ]
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Callback registration
# ─────────────────────────────────────────────────────────────────────────────

def register(app) -> None:

    # ── Stats tab ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("tab-stats-content", "children"),
        Input("lab-tabs", "active_tab"),
        State("store-graph", "data"),
        State("store-model", "data"),
        prevent_initial_call=False,
    )
    def render_stats_tab(active_tab, graph_data, model):
        if active_tab != "tab-stats":
            return no_update
        graph = graph_from_store(graph_data)
        if graph is None:
            return _empty_state("No graph loaded.")
        return _build_stats_content(graph)

    # ── Run simulation button → store results ─────────────────────────────────
    @app.callback(
        Output("store-sim-results", "data"),
        Output("tab-sim-content",   "children"),
        Input("btn-run-sim",         "n_clicks"),
        State("store-graph",         "data"),
        State("store-model",         "data"),
        State("param-seed-fraction", "value"),
        State("param-num-trials",    "value"),
        State("param-seed-strategy", "value"),
        State("param-threshold",     "value"),
        State("param-phi",           "value"),
        State("param-beta",          "value"),
        State("param-gamma",         "value"),
        State("param-switch-fraction","value"),
        prevent_initial_call=True,
    )
    def run_simulation(n_clicks, graph_data, model,
                       seed_fraction, num_trials, seed_strategy,
                       threshold, phi, beta, gamma, switch_fraction):
        if not n_clicks:
            return no_update, _empty_state()

        graph = graph_from_store(graph_data)
        if graph is None:
            return no_update, _empty_state("No graph loaded.")

        config = _make_config(seed_fraction, num_trials, seed_strategy,
                              threshold, phi, beta, gamma, switch_fraction)
        n = graph.number_of_nodes()
        seed_size = max(1, int(config.seed_fraction * n))

        try:
            sim = _make_sim(model, graph, config)
            seed_nodes = set(select_seeds(graph, seed_size, config.seed_strategy))
            result, _ = sim.run(seed_nodes)
            metrics = sim.collect_metrics(seed_size, num_trials=config.num_trials,
                                          seed=42, strategy=config.seed_strategy)
        except Exception as exc:
            err = dbc.Alert(f"Simulation error: {exc}", color="danger")
            return no_update, err

        # Serialise result / metrics to JSON-safe dicts
        def _to_dict(obj):
            if not hasattr(obj, "__dict__"):
                return {}
            out = {}
            for k, v in vars(obj).items():
                if isinstance(v, set):
                    out[k] = list(v)
                elif isinstance(v, (int, float, bool, str, type(None))):
                    out[k] = v
                elif isinstance(v, (list, tuple)):
                    out[k] = list(v)
                else:
                    out[k] = str(v)
            return out

        sim_data = {
            "result":  _to_dict(result),
            "metrics": _to_dict(metrics),
            "n":       n,
            "seed_size": seed_size,
        }

        content = _render_sim_results(sim_data, model, config.num_trials)
        return sim_data, content

    # ── Simulation tab: re-render from store on tab switch ────────────────────
    @app.callback(
        Output("tab-sim-content", "children", allow_duplicate=True),
        Input("lab-tabs",         "active_tab"),
        State("store-sim-results","data"),
        State("store-model",      "data"),
        State("param-num-trials", "value"),
        prevent_initial_call=True,
    )
    def refresh_sim_tab(active_tab, sim_data, model, num_trials):
        if active_tab != "tab-sim":
            return no_update
        if not sim_data:
            return _empty_state()
        return _render_sim_results(sim_data, model, int(num_trials or 50))

    # ── Animation tab ─────────────────────────────────────────────────────────
    @app.callback(
        Output("tab-anim-content",    "children"),
        Input("lab-tabs",             "active_tab"),
        Input("btn-run-sim",          "n_clicks"),
        State("store-graph",          "data"),
        State("store-model",          "data"),
        State("param-seed-fraction",  "value"),
        State("param-num-trials",     "value"),
        State("param-seed-strategy",  "value"),
        State("param-threshold",      "value"),
        State("param-phi",            "value"),
        State("param-beta",           "value"),
        State("param-gamma",          "value"),
        State("param-switch-fraction","value"),
        prevent_initial_call=True,
    )
    def render_animation_tab(active_tab, n_clicks,
                             graph_data, model,
                             seed_fraction, num_trials, seed_strategy,
                             threshold, phi, beta, gamma, switch_fraction):
        # Only render when the animation tab is active AND sim has been run
        if active_tab != "tab-anim":
            return no_update
        if not n_clicks:
            return _empty_state("Run the simulation first, then switch to this tab.")

        graph = graph_from_store(graph_data)
        if graph is None:
            return _empty_state("No graph loaded.")

        config = _make_config(seed_fraction, num_trials, seed_strategy,
                              threshold, phi, beta, gamma, switch_fraction)
        try:
            return _build_animation_content(model, graph, config)
        except Exception as exc:
            return dbc.Alert(f"Animation error: {exc}", color="danger")

    # ── Vulnerability tab ─────────────────────────────────────────────────────
    @app.callback(
        Output("tab-vuln-content",    "children"),
        Output("store-vuln-results",  "data"),
        Output("store-block-results", "data"),
        Input("lab-tabs",             "active_tab"),
        State("store-graph",          "data"),
        State("store-model",          "data"),
        State("param-seed-fraction",  "value"),
        State("param-num-trials",     "value"),
        State("param-seed-strategy",  "value"),
        State("param-threshold",      "value"),
        State("param-phi",            "value"),
        State("param-beta",           "value"),
        State("param-gamma",          "value"),
        State("param-switch-fraction","value"),
        State("store-vuln-results",   "data"),
        State("store-block-results",  "data"),
        prevent_initial_call=True,
    )
    def render_vuln_tab(active_tab,
                        graph_data, model,
                        seed_fraction, num_trials, seed_strategy,
                        threshold, phi, beta, gamma, switch_fraction,
                        cached_vuln, cached_block):
        if active_tab != "tab-vuln":
            return no_update, no_update, no_update

        graph = graph_from_store(graph_data)
        if graph is None:
            return _empty_state("No graph loaded."), no_update, no_update

        # Return cached results if available
        if cached_vuln and cached_block:
            baseline = {"avg": cached_block[0].get("baseline_avg", 0.0),
                        "prob": cached_block[0].get("baseline_prob", 0.0)}
            # Extract baseline from first entry's stored fields
            if cached_block and len(cached_block) > 0:
                bl_avg  = cached_block[-1].get("_baseline_avg",  0.0)
                bl_prob = cached_block[-1].get("_baseline_prob", 0.0)
            else:
                bl_avg, bl_prob = 0.0, 0.0
            baseline = {"avg": bl_avg, "prob": bl_prob}
            try:
                content = _build_vuln_content(graph, cached_vuln, cached_block, baseline)
                return content, no_update, no_update
            except Exception:
                pass  # fall through to re-compute

        config = _make_config(seed_fraction, num_trials, seed_strategy,
                              threshold, phi, beta, gamma, switch_fraction)
        try:
            sim = _make_sim(model, graph, config)
            vuln_data = sim.node_influence_analysis(
                seed_fraction=config.seed_fraction,
                num_trials=15, seed=42,
            )
            block_data, bl_avg, bl_prob = sim.node_blocking_analysis(
                seed_fraction=config.seed_fraction,
                num_trials=15, seed=42,
            )
        except Exception as exc:
            return dbc.Alert(f"Vulnerability analysis error: {exc}", color="danger"), no_update, no_update

        # Tag baseline into last element for caching
        block_data_stored = list(block_data)
        if block_data_stored:
            block_data_stored[-1]["_baseline_avg"]  = bl_avg
            block_data_stored[-1]["_baseline_prob"] = bl_prob

        baseline = {"avg": bl_avg, "prob": bl_prob}
        content  = _build_vuln_content(graph, vuln_data, block_data_stored, baseline)
        return content, vuln_data, block_data_stored

    # ── Sweep tab layout ──────────────────────────────────────────────────────
    @app.callback(
        Output("tab-sweep-content", "children"),
        Input("lab-tabs",           "active_tab"),
        State("store-model",        "data"),
        prevent_initial_call=True,
    )
    def render_sweep_tab(active_tab, model):
        if active_tab != "tab-sweep":
            return no_update
        if not model:
            return _empty_state("No model selected.")
        return _build_sweep_layout(model)

    # ── Sweep param controls ──────────────────────────────────────────────────
    @app.callback(
        Output("sweep-param-controls", "children"),
        Input("sweep-type",            "value"),
        State("store-model",           "data"),
        State("param-beta",            "value"),
        State("param-phi",             "value"),
        prevent_initial_call=True,
    )
    def update_sweep_controls(sweep_type, model, beta, phi):
        if sweep_type == "seed_fraction":
            return _sweep_range_controls("sf", "Seed fraction", 0.01, 0.5, 0.5, 10)
        if sweep_type == "beta":
            return _sweep_range_controls("beta", "β", 0.01, 1.0, 0.8, 10)
        if sweep_type == "phi":
            return _sweep_range_controls("phi-sw", "φ", 0.01, 1.0, 0.8, 10)
        if sweep_type == "threshold":
            return _sweep_range_controls("k-sw", "k (integer)", 1, 10, 8, 8, int_mode=True)
        if sweep_type == "er_prob":
            return html.Div([
                html.Label("Number of ER nodes", className="form-label small fw-medium"),
                dbc.Input(id="sweep-er-n", type="number", min=10, max=2000, value=100, size="sm", className="mb-2"),
                html.Label("Probabilities (comma-separated)", className="form-label small fw-medium"),
                dbc.Input(id="sweep-er-probs", type="text", value="0.01,0.05,0.1,0.2,0.3", size="sm"),
            ])
        if sweep_type == "lattice_size":
            return html.Div([
                html.Label("Grid sizes (comma-separated)", className="form-label small fw-medium"),
                dbc.Input(id="sweep-lat-sizes", type="text", value="5,10,15,20", size="sm"),
            ])
        return html.Div()

    # ── Run sweep ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("sweep-output",         "children"),
        Output("store-sweep-results",  "data"),
        Input("btn-run-sweep",         "n_clicks"),
        State("store-graph",           "data"),
        State("store-model",           "data"),
        State("sweep-type",            "value"),
        State("sweep-trials",          "value"),
        State("param-seed-fraction",   "value"),
        State("param-beta",            "value"),
        State("param-gamma",           "value"),
        State("param-phi",             "value"),
        State("param-threshold",       "value"),
        State("param-switch-fraction", "value"),
        prevent_initial_call=True,
    )
    def run_sweep(n_clicks, graph_data, model,
                  sweep_type, sweep_trials,
                  seed_fraction, beta, gamma, phi, threshold, switch_fraction):
        if not n_clicks:
            return no_update, no_update

        graph = graph_from_store(graph_data)
        if graph is None:
            return dbc.Alert("No graph loaded.", color="danger"), no_update

        trials = int(sweep_trials or 20)
        beta   = float(beta or 0.3)
        gamma  = float(gamma or 0.1)
        phi    = float(phi or 0.3)
        k      = int(threshold or 2)
        sf     = float(seed_fraction or 0.05)
        swfr   = float(switch_fraction or 0.2)

        try:
            df, x_col, y_cols, title = _run_sweep_logic(
                model, graph, sweep_type, trials,
                beta=beta, gamma=gamma, phi=phi, k=k,
                seed_fraction=sf, switch_fraction=swfr,
            )
        except Exception as exc:
            return dbc.Alert(f"Sweep error: {exc}", color="danger"), no_update

        figs = []
        for y_col, y_label in y_cols:
            fig = px.line(df, x=x_col, y=y_col, markers=True,
                          title=f"{y_label} vs {x_col}",
                          color_discrete_sequence=["#2563EB"])
            fig.update_layout(paper_bgcolor="#F8FAFC", plot_bgcolor="#F8FAFC",
                               margin=dict(l=40, r=20, t=40, b=40))
            figs.append(dcc.Graph(figure=fig))

        table = dbc.Table.from_dataframe(
            df.round(4), striped=True, bordered=False,
            hover=True, responsive=True, size="sm",
        )

        return html.Div([html.H5(title, className="fw-semibold mb-3"), *figs, table]), df.to_dict("records")


def _sweep_range_controls(prefix, label, min_v, max_v, default_max, default_steps,
                           int_mode=False) -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label(f"Min {label}", className="form-label small fw-medium"),
                dbc.Input(id=f"sweep-{prefix}-min", type="number",
                          min=min_v, max=max_v, value=min_v,
                          step=1 if int_mode else 0.01, size="sm"),
            ], md=4),
            dbc.Col([
                html.Label(f"Max {label}", className="form-label small fw-medium"),
                dbc.Input(id=f"sweep-{prefix}-max", type="number",
                          min=min_v, max=max_v, value=default_max,
                          step=1 if int_mode else 0.01, size="sm"),
            ], md=4),
            dbc.Col([
                html.Label("Steps", className="form-label small fw-medium"),
                dbc.Input(id=f"sweep-{prefix}-steps", type="number",
                          min=3, max=30, value=default_steps, step=1, size="sm"),
            ], md=4),
        ]),
    ])


def _run_sweep_logic(model, graph, sweep_type, trials,
                     beta, gamma, phi, k, seed_fraction, switch_fraction):
    """Execute the correct parameter sweep and return (df, x_col, y_cols, title)."""

    # Dynamic import of sweep modules to avoid circular imports
    if sweep_type == "seed_fraction":
        fracs = _linspace(0.01, 0.5, 10)
        return _sweep_seed_fraction(model, graph, fracs, beta, gamma, phi, k,
                                    switch_fraction, trials)

    if sweep_type == "beta":
        betas = _linspace(0.01, 0.8, 10)
        return _sweep_beta(model, graph, betas, gamma, phi, k,
                           seed_fraction, switch_fraction, trials)

    if sweep_type == "phi":
        phis = _linspace(0.01, 0.9, 10)
        return _sweep_phi(model, graph, phis, beta, gamma, k,
                          seed_fraction, switch_fraction, trials)

    if sweep_type == "threshold":
        ks = list(range(1, min(9, k + 5)))
        return _sweep_k(model, graph, ks, beta, gamma, seed_fraction, trials)

    if sweep_type == "er_prob":
        from input.graph_generator import generate_er_graph
        n_er = 100
        probs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        rows = []
        for p in probs:
            g2 = generate_er_graph(n_er, p)
            cfg = SidebarConfig(threshold=k, seed_fraction=seed_fraction, num_trials=trials,
                                beta=beta, gamma=gamma, seed_strategy="random",
                                switch_fraction=switch_fraction, phi=phi)
            sim = _make_sim(model, g2, cfg)
            ss  = max(1, int(seed_fraction * n_er))
            met = sim.collect_metrics(ss, num_trials=trials, seed=42, strategy="random")
            rows.append({"p": p,
                         "epidemic_probability": getattr(met, "cascade_probability",
                                                getattr(met, "epidemic_probability", 0.0)),
                         "epidemic_size": getattr(met, "cascade_size",
                                         getattr(met, "epidemic_size", 0.0))})
        df = pd.DataFrame(rows)
        return df, "p", [("epidemic_probability", "Epidemic Probability"),
                         ("epidemic_size", "Epidemic Size")], "ER Probability Sweep"

    if sweep_type == "lattice_size":
        from input.graph_generator import generate_lattice_graph
        sizes = [5, 10, 15, 20]
        rows = []
        for sz in sizes:
            g2 = generate_lattice_graph(sz)
            n2 = g2.number_of_nodes()
            cfg = SidebarConfig(threshold=k, seed_fraction=seed_fraction, num_trials=trials,
                                beta=beta, gamma=gamma, seed_strategy="random",
                                switch_fraction=switch_fraction, phi=phi)
            sim = _make_sim(model, g2, cfg)
            ss  = max(1, int(seed_fraction * n2))
            met = sim.collect_metrics(ss, num_trials=trials, seed=42, strategy="random")
            rows.append({"grid_size": sz,
                         "epidemic_probability": getattr(met, "cascade_probability",
                                                getattr(met, "epidemic_probability", 0.0)),
                         "epidemic_size": getattr(met, "cascade_size",
                                         getattr(met, "epidemic_size", 0.0))})
        df = pd.DataFrame(rows)
        return df, "grid_size", [("epidemic_probability", "Epidemic Probability"),
                                  ("epidemic_size", "Epidemic Size")], "Lattice Size Sweep"

    raise ValueError(f"Unknown sweep type: {sweep_type}")


def _linspace(start, stop, n):
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _sweep_seed_fraction(model, graph, fracs, beta, gamma, phi, k, swfr, trials):
    rows = []
    for frac in fracs:
        cfg = SidebarConfig(threshold=k, seed_fraction=frac, num_trials=trials,
                            beta=beta, gamma=gamma, seed_strategy="random",
                            switch_fraction=swfr, phi=phi)
        sim = _make_sim(model, graph, cfg)
        ss  = max(1, int(frac * graph.number_of_nodes()))
        met = sim.collect_metrics(ss, num_trials=trials, seed=42, strategy="random")
        rows.append({"seed_fraction": round(frac, 4),
                     "epidemic_probability": getattr(met, "cascade_probability",
                                            getattr(met, "epidemic_probability", 0.0)),
                     "epidemic_size": getattr(met, "cascade_size",
                                     getattr(met, "epidemic_size", 0.0))})
    df = pd.DataFrame(rows)
    return df, "seed_fraction", [("epidemic_probability", "Epidemic Probability"),
                                  ("epidemic_size", "Epidemic Size")], "Seed Fraction Sweep"


def _sweep_beta(model, graph, betas, gamma, phi, k, sf, swfr, trials):
    rows = []
    for b in betas:
        cfg = SidebarConfig(threshold=k, seed_fraction=sf, num_trials=trials,
                            beta=b, gamma=gamma, seed_strategy="random",
                            switch_fraction=swfr, phi=phi)
        sim = _make_sim(model, graph, cfg)
        ss  = max(1, int(sf * graph.number_of_nodes()))
        met = sim.collect_metrics(ss, num_trials=trials, seed=42, strategy="random")
        rows.append({"beta": round(b, 4),
                     "epidemic_probability": getattr(met, "cascade_probability",
                                            getattr(met, "epidemic_probability", 0.0)),
                     "epidemic_size": getattr(met, "cascade_size",
                                     getattr(met, "epidemic_size", 0.0))})
    df = pd.DataFrame(rows)
    return df, "beta", [("epidemic_probability", "Epidemic Probability"),
                        ("epidemic_size", "Epidemic Size")], "Transmission Rate Sweep"


def _sweep_phi(model, graph, phis, beta, gamma, k, sf, swfr, trials):
    rows = []
    for p in phis:
        cfg = SidebarConfig(threshold=k, seed_fraction=sf, num_trials=trials,
                            beta=beta, gamma=gamma, seed_strategy="random",
                            switch_fraction=swfr, phi=p)
        sim = _make_sim(model, graph, cfg)
        ss  = max(1, int(sf * graph.number_of_nodes()))
        met = sim.collect_metrics(ss, num_trials=trials, seed=42, strategy="random")
        rows.append({"phi": round(p, 4),
                     "epidemic_probability": getattr(met, "cascade_probability",
                                            getattr(met, "epidemic_probability", 0.0)),
                     "epidemic_size": getattr(met, "cascade_size",
                                     getattr(met, "epidemic_size", 0.0))})
    df = pd.DataFrame(rows)
    return df, "phi", [("epidemic_probability", "Epidemic Probability"),
                       ("epidemic_size", "Epidemic Size")], "Phi (φ) Threshold Sweep"


def _sweep_k(model, graph, ks, beta, gamma, sf, trials):
    rows = []
    for ki in ks:
        cfg = SidebarConfig(threshold=ki, seed_fraction=sf, num_trials=trials,
                            beta=beta, gamma=gamma, seed_strategy="random",
                            switch_fraction=0.2, phi=0.3)
        sim = _make_sim(model, graph, cfg)
        ss  = max(1, int(sf * graph.number_of_nodes()))
        met = sim.collect_metrics(ss, num_trials=trials, seed=42, strategy="random")
        rows.append({"k": ki,
                     "epidemic_probability": getattr(met, "cascade_probability",
                                            getattr(met, "epidemic_probability", 0.0)),
                     "epidemic_size": getattr(met, "cascade_size",
                                     getattr(met, "epidemic_size", 0.0))})
    df = pd.DataFrame(rows)
    return df, "k", [("epidemic_probability", "Epidemic Probability"),
                     ("epidemic_size", "Epidemic Size")], "Threshold (k) Sweep"
