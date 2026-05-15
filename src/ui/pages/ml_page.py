"""
ml_page.py — ML Virality Predictor page layout.
Callbacks are in ui/callbacks/ml_callbacks.py.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


_CONTENT_OPTIONS = [
    {"label": "News or information", "value": "News or information"},
    {"label": "Viral challenge or trend", "value": "Viral challenge or trend"},
    {"label": "Opinion or political content", "value": "Opinion or political content"},
    {"label": "Product or brand content", "value": "Product or brand content"},
    {"label": "Health behaviour or lifestyle", "value": "Health behaviour or lifestyle"},
]

_NETWORK_OPTIONS = [
    {"label": "Tight community (close friends, family groups)",
     "value": "Tight community (close friends, family groups)"},
    {"label": "Mixed social network (typical social media)",
     "value": "Mixed social network (typical social media)"},
    {"label": "Influencer-driven network (few hubs, many followers)",
     "value": "Influencer-driven network (few hubs, many followers)"},
]


def layout() -> html.Div:
    return html.Div([
        dbc.Container([
            html.H2("ML Virality Predictor",
                    className="fw-bold mb-1", style={"color": "#1E3A8A"}),
            html.P("Identify the spreading mechanism and estimate final network reach from early trajectory data.",
                   className="text-muted mb-4"),

            dbc.Row([
                # ── Left panel: inputs ─────────────────────────────────────
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("About your content", className="text-uppercase text-muted small fw-bold mb-3"),

                            html.Label("Content type", className="form-label small fw-medium"),
                            dbc.Select(
                                id="ml-content-type",
                                options=_CONTENT_OPTIONS,
                                value="News or information",
                                size="sm",
                                className="mb-3",
                            ),

                            html.Label("Social network type", className="form-label small fw-medium"),
                            dbc.Select(
                                id="ml-network-type",
                                options=_NETWORK_OPTIONS,
                                value="Mixed social network (typical social media)",
                                size="sm",
                                className="mb-3",
                            ),

                            html.Hr(className="my-2"),

                            html.H6("Spread trajectory", className="text-uppercase text-muted small fw-bold mb-2"),
                            dbc.RadioItems(
                                id="ml-input-method",
                                options=[
                                    {"label": "Manual entry", "value": "manual"},
                                    {"label": "Upload CSV",   "value": "upload"},
                                    {"label": "Simulate",     "value": "simulate"},
                                ],
                                value="manual",
                                inline=True,
                                className="mb-3",
                            ),

                            # Manual entry
                            html.Div(id="ml-manual-div", children=[
                                html.Label("% reached per step (comma-separated)",
                                           className="form-label small fw-medium"),
                                dbc.Input(
                                    id="ml-series-input",
                                    type="text",
                                    value="0.5, 0.8, 1.2, 2.1, 3.8",
                                    size="sm",
                                    className="mb-3",
                                ),
                            ]),

                            # Upload
                            html.Div(id="ml-upload-div", children=[
                                dcc.Upload(
                                    id="ml-upload",
                                    children=html.Div([
                                        "Drag & drop or ",
                                        html.A("select CSV", className="text-primary"),
                                    ]),
                                    style={
                                        "width": "100%", "height": "60px",
                                        "lineHeight": "60px", "borderWidth": "2px",
                                        "borderStyle": "dashed", "borderRadius": "8px",
                                        "borderColor": "#BFDBFE", "textAlign": "center",
                                        "backgroundColor": "#F8FAFC", "cursor": "pointer",
                                    },
                                    multiple=False,
                                    accept=".csv",
                                ),
                                html.Div(id="ml-upload-status", className="small text-muted mt-1"),
                            ], style={"display": "none"}),

                            # Simulate
                            html.Div(id="ml-sim-div", style={"display": "none"}, children=[
                                html.Label("Model to simulate", className="form-label small fw-medium"),
                                dbc.Select(
                                    id="ml-sim-model",
                                    options=[
                                        {"label": "SIR",                    "value": "sir"},
                                        {"label": "SIS",                    "value": "sis"},
                                        {"label": "Bootstrap Percolation",  "value": "bootstrap"},
                                        {"label": "Watts Threshold (WTM)",  "value": "wtm"},
                                        {"label": "H1 — SIR ∨ Bootstrap",  "value": "h1"},
                                        {"label": "H2 — SIR → Bootstrap",  "value": "h2"},
                                        {"label": "H3 — Soft Bootstrap",    "value": "h3"},
                                        {"label": "H4 — SIS ∨ WTM",        "value": "h4"},
                                        {"label": "H5 — SIS → WTM",        "value": "h5"},
                                        {"label": "H6 — Soft WTM",         "value": "h6"},
                                    ],
                                    value="sir",
                                    size="sm",
                                    className="mb-2",
                                ),
                                # β slider (sir, sis, h1, h2, h3, h4, h5)
                                html.Div(id="ml-sim-beta-group", children=[
                                    html.Label("β (transmission rate)",
                                               className="form-label small fw-medium"),
                                    dcc.Slider(id="ml-sim-beta", min=0.05, max=0.60,
                                               step=0.05, value=0.30,
                                               tooltip={"placement": "bottom",
                                                        "always_visible": False},
                                               className="mb-2"),
                                ]),
                                # γ slider (sir, sis, h1, h2, h3, h4, h5, h6)
                                html.Div(id="ml-sim-gamma-group", children=[
                                    html.Label("γ (recovery rate)",
                                               className="form-label small fw-medium"),
                                    dcc.Slider(id="ml-sim-gamma", min=0.05, max=0.50,
                                               step=0.05, value=0.10,
                                               tooltip={"placement": "bottom",
                                                        "always_visible": False},
                                               className="mb-2"),
                                ]),
                                # k slider (bootstrap, h1, h2)
                                html.Div(id="ml-sim-k-group", style={"display": "none"},
                                         children=[
                                    html.Label("k (min. infected neighbours)",
                                               className="form-label small fw-medium"),
                                    dcc.Slider(id="ml-sim-k", min=1, max=8, step=1, value=3,
                                               tooltip={"placement": "bottom",
                                                        "always_visible": False},
                                               className="mb-2"),
                                ]),
                                # φ slider (wtm, h4, h5, h6)
                                html.Div(id="ml-sim-phi-group", style={"display": "none"},
                                         children=[
                                    html.Label("φ (fraction threshold)",
                                               className="form-label small fw-medium"),
                                    dcc.Slider(id="ml-sim-phi", min=0.10, max=0.60,
                                               step=0.05, value=0.25,
                                               tooltip={"placement": "bottom",
                                                        "always_visible": False},
                                               className="mb-2"),
                                ]),
                                # switch fraction (h2, h5)
                                html.Div(id="ml-sim-sf-group", style={"display": "none"},
                                         children=[
                                    html.Label("Switch fraction f",
                                               className="form-label small fw-medium"),
                                    dcc.Slider(id="ml-sim-sf", min=0.05, max=0.50,
                                               step=0.05, value=0.20,
                                               tooltip={"placement": "bottom",
                                                        "always_visible": False},
                                               className="mb-2"),
                                ]),
                                html.Label("Seed fraction (% of nodes)",
                                           className="form-label small fw-medium"),
                                dcc.Slider(id="ml-sim-seed-frac", min=0.1, max=10.0,
                                           step=0.1, value=1.0,
                                           tooltip={"placement": "bottom",
                                                    "always_visible": False},
                                           className="mb-2"),
                                html.Div(id="ml-sim-seed-caption",
                                         className="small text-muted mb-2"),
                                dbc.Button("Run Simulation", id="ml-sim-run-btn",
                                           color="primary", size="sm",
                                           className="w-100 mb-2"),
                                html.Div(id="ml-sim-result", className="small"),
                            ]),

                            html.Label("Prediction horizon (steps)",
                                       className="form-label small fw-medium"),
                            dcc.Slider(
                                id="ml-horizon",
                                min=10, max=100, step=5, value=50,
                                marks={10: "10", 50: "50", 100: "100"},
                                tooltip={"placement": "bottom", "always_visible": False},
                                className="mb-3",
                            ),

                            dbc.Button(
                                "Predict Virality",
                                id="ml-predict-btn",
                                color="warning",
                                className="w-100 fw-bold",
                                style={"backgroundColor": "#FF6D00", "borderColor": "#FF6D00"},
                            ),
                        ])
                    ], className="shadow-sm border-0",
                       style={"backgroundColor": "#F8FAFC"}),
                ], md=3),

                # ── Right panel: results tabs ──────────────────────────────
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(
                            label="Prediction",
                            tab_id="ml-tab-predict",
                            children=dcc.Loading(
                                html.Div(id="ml-predict-content", className="pt-3"),
                                type="circle",
                            ),
                        ),
                        dbc.Tab(
                            label="How Does It Spread?",
                            tab_id="ml-tab-learn",
                            children=html.Div(id="ml-learn-content", className="pt-3"),
                        ),
                        dbc.Tab(
                            label="About",
                            tab_id="ml-tab-about",
                            children=html.Div(id="ml-about-content", className="pt-3"),
                        ),
                    ], id="ml-tabs", active_tab="ml-tab-predict"),

                    dcc.Store(id="ml-series-store"),
                ], md=9),
            ]),
        ], fluid=False, className="py-4"),
    ])
