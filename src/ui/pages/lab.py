"""
lab.py — Lab page layout (5-tab analysis for each model).

The layout is model-agnostic; callbacks populate the content.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from ui.components.param_panel import build_param_panel


_MODEL_TITLES = {
    "bootstrap": "Bootstrap Percolation — Network Risk Analysis",
    "sir":       "SIR Epidemic Model — Network Spread Analysis",
    "sis":       "SIS Epidemic Model — Reinfection Dynamics",
    "wtm":       "WTM — Watts Threshold Model",
    "h1":        "H1 — OR-Hybrid: SIR ∨ Bootstrap",
    "h2":        "H2 — Sequential Hybrid (SIR → Bootstrap)",
    "h3":        "H3 — Probabilistic Threshold Hybrid",
    "h4":        "H4 — OR-Hybrid: SIS ∨ WTM",
    "h5":        "H5 — Sequential Hybrid (SIS → WTM)",
    "h6":        "H6 — Soft WTM (Probabilistic Threshold)",
}


def layout(model: str, graph_n: int | None = None,
           graph_m: int | None = None) -> html.Div:
    title = _MODEL_TITLES.get(model, model.upper())

    return html.Div([
        dbc.Container([
            # Title row
            dbc.Row([
                dbc.Col([
                    html.H3(title, className="fw-bold mb-0",
                            style={"color": "#1E3A8A", "letterSpacing": "-0.3px"}),
                ], md=9),
                dbc.Col([
                    dbc.Button("Back to Models", href="/model", color="secondary",
                               outline=True, size="sm"),
                ], md=3, className="d-flex align-items-center justify-content-end"),
            ], className="mb-4"),

            dbc.Row([
                # ── Left column: parameter panel ──────────────────────────
                dbc.Col([
                    build_param_panel(model, graph_n, graph_m),
                    # Hidden stores for this panel's values
                    dcc.Store(id="store-run-trigger", data=0),
                ], md=3),

                # ── Right column: 5-tab panel ─────────────────────────────
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(
                            label="Graph Statistics",
                            tab_id="tab-stats",
                            children=dcc.Loading(
                                html.Div(id="tab-stats-content",
                                         className="pt-3"),
                                type="circle",
                            ),
                        ),
                        dbc.Tab(
                            label="Simulation",
                            tab_id="tab-sim",
                            children=dcc.Loading(
                                html.Div(id="tab-sim-content",
                                         className="pt-3"),
                                type="circle",
                            ),
                        ),
                        dbc.Tab(
                            label="Animation",
                            tab_id="tab-anim",
                            children=dcc.Loading(
                                html.Div(id="tab-anim-content",
                                         className="pt-3"),
                                type="circle",
                            ),
                        ),
                        dbc.Tab(
                            label="Node Vulnerability",
                            tab_id="tab-vuln",
                            children=dcc.Loading(
                                html.Div(id="tab-vuln-content",
                                         className="pt-3"),
                                type="circle",
                            ),
                        ),
                        dbc.Tab(
                            label="Parameter Sweep",
                            tab_id="tab-sweep",
                            children=dcc.Loading(
                                html.Div(id="tab-sweep-content",
                                         className="pt-3"),
                                type="circle",
                            ),
                        ),
                    ], id="lab-tabs", active_tab="tab-stats",
                       className="nav-tabs-custom"),

                    # Store sim results for the current model
                    dcc.Store(id="store-sim-results"),
                    dcc.Store(id="store-vuln-results"),
                    dcc.Store(id="store-block-results"),
                    dcc.Store(id="store-sweep-results"),
                ], md=9),
            ]),
        ], fluid=False, className="py-4"),
    ])
