"""
model_selection.py — Step 2: Choose a spreading model (layout only).
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


_BASE_MODELS = [
    ("bootstrap", "Bootstrap Percolation",
     "Hard threshold — node activates when ≥ k neighbours are infected.",
     "k"),
    ("sir", "SIR",
     "Probabilistic spread with permanent immunity (β, γ).",
     "β · γ"),
    ("sis", "SIS",
     "Probabilistic spread, no immunity — reinfection possible.",
     "β · μ"),
    ("wtm", "Watts Threshold",
     "Fractional threshold — activates when ≥ φ fraction of neighbours infected.",
     "φ"),
]

_HYBRID_MODELS = [
    ("h1", "H1 — SIR ∨ Bootstrap",
     "Activates via SIR or bootstrap threshold, whichever fires first.",
     "k · β · γ"),
    ("h2", "H2 — SIR → Bootstrap",
     "SIR phase until fraction f infected, then switches to bootstrap.",
     "k · β · γ · f"),
    ("h3", "H3 — Soft Bootstrap",
     "Infection probability grows linearly with infected-neighbour count.",
     "β · γ"),
    ("h4", "H4 — SIS ∨ WTM",
     "Activates via SIS or fractional WTM threshold.",
     "φ · β · γ"),
    ("h5", "H5 — SIS → WTM",
     "SIS phase until fraction f infected, then switches to WTM.",
     "φ · β · γ · f"),
    ("h6", "H6 — Soft WTM",
     "Infection probability scales with the fraction of infected neighbours.",
     "φ · γ"),
]


def _model_card(model_key: str, title: str, desc: str, params: str,
                color: str = "#2563EB") -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6(title, className="fw-bold mb-1"),
                    html.P(desc, className="text-muted small mb-2"),
                    dbc.Badge(params, color="light", text_color="dark",
                              className="font-monospace small mb-3"),
                ], className="flex-grow-1"),
                dbc.Button(
                    "Select",
                    href=f"/lab/{model_key}",
                    color="primary",
                    outline=True,
                    size="sm",
                    className="w-100 mt-auto",
                    style={"borderColor": color, "color": color},
                ),
            ], className="d-flex flex-column h-100"),
        ], className="h-100 shadow-sm",
           style={"borderTop": f"3px solid {color}", "borderRadius": "8px"}),
    ], md=3, className="mb-3")


def layout(graph_n: int | None = None, graph_m: int | None = None) -> html.Div:
    graph_info = ""
    change_btn = html.Span()
    if graph_n is not None:
        graph_info = f"Graph loaded: {graph_n:,} nodes, {graph_m:,} edges"
        change_btn = dbc.Button(
            "Change Graph",
            href="/graph",
            color="secondary",
            outline=True,
            size="sm",
        )

    return html.Div([
        dbc.Container([
            html.H2("Step 2 — Choose a spreading model",
                    className="fw-bold mb-1", style={"color": "#1E3A8A"}),

            dbc.Row([
                dbc.Col([
                    dbc.Alert(graph_info, color="success", className="py-2 mb-0") if graph_n else html.Span(),
                ], md=9),
                dbc.Col([change_btn], md=3, className="text-end"),
            ], className="align-items-center mb-4"),

            # Base models
            html.H5("Base Models", className="fw-semibold mb-3 mt-2",
                    style={"color": "#374151"}),
            dbc.Row([
                _model_card(*m) for m in _BASE_MODELS
            ]),

            # Hybrid models
            html.H5("Hybrid Models", className="fw-semibold mb-1 mt-3",
                    style={"color": "#374151"}),
            html.P("Combine epidemic and threshold mechanisms.",
                   className="text-muted small mb-3"),
            dbc.Row([
                _model_card(*m) for m in _HYBRID_MODELS[:3]
            ]),
            dbc.Row([
                _model_card(*m) for m in _HYBRID_MODELS[3:]
            ]),

            html.Hr(className="my-4"),

            # ML predictor
            html.H5("ML Virality Predictor", className="fw-semibold mb-3",
                    style={"color": "#374151"}),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("No network needed — works from early spreading trajectory alone.",
                                    className="fw-bold mb-2"),
                            html.P(
                                "Predict how far a contagion will spread and identify the underlying spreading "
                                "mechanism using a deep learning model trained on 50,000 simulations.",
                                className="text-muted small mb-0",
                            ),
                        ], md=9),
                        dbc.Col([
                            dbc.Button(
                                "Open Predictor",
                                href="/ml",
                                color="warning",
                                className="w-100 fw-semibold",
                                style={"backgroundColor": "#FF6D00", "borderColor": "#FF6D00"},
                            ),
                        ], md=3, className="d-flex align-items-center"),
                    ]),
                ])
            ], style={"borderLeft": "4px solid #FF6D00", "borderRadius": "8px"},
               className="shadow-sm"),

        ], fluid=False, className="py-4"),
    ])
