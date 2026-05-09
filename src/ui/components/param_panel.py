"""
param_panel.py — Parameter sidebar panel component.

Builds the left-column parameter controls for each model.
Returns a dbc.Card containing the controls.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


_STRATEGY_OPTIONS = [
    {"label": "Random", "value": "random"},
    {"label": "High Degree", "value": "high_degree"},
    {"label": "High k-core", "value": "high_kcore"},
]


def _slider_row(label: str, id: str, min_val: float, max_val: float,
                step: float, value: float, marks: dict | None = None) -> html.Div:
    return html.Div([
        html.Label(label, className="form-label small fw-medium mb-1"),
        dcc.Slider(
            id=id,
            min=min_val,
            max=max_val,
            step=step,
            value=value,
            marks=marks or {min_val: str(min_val), max_val: str(max_val)},
            tooltip={"placement": "bottom", "always_visible": False},
            className="mb-3",
        ),
    ])


def _number_row(label: str, id: str, min_val: int, max_val: int,
                value: int, step: int = 1) -> html.Div:
    return html.Div([
        html.Label(label, className="form-label small fw-medium mb-1"),
        dbc.Input(
            id=id,
            type="number",
            min=min_val,
            max=max_val,
            step=step,
            value=value,
            size="sm",
            className="mb-3",
        ),
    ])


def build_param_panel(model: str, graph_n: int | None = None,
                      graph_m: int | None = None) -> dbc.Card:
    """Return the full parameter control card for the given model.

    All parameter components are always rendered (to ensure Dash IDs exist in the
    DOM), but irrelevant ones are hidden with display:none.
    """
    graph_section = []
    if graph_n is not None:
        graph_section = [
            dbc.Alert(
                [
                    html.Strong(f"{graph_n:,} nodes"),
                    html.Span(f" · {graph_m:,} edges"),
                ],
                color="primary",
                className="py-2 px-3 mb-3",
            ),
        ]

    nav_btns = dbc.ButtonGroup([
        dbc.Button(
            "Change Graph", id="btn-change-graph", color="secondary",
            outline=True, size="sm", className="flex-fill",
            href="/graph",
        ),
        dbc.Button(
            "Change Model", id="btn-change-model", color="secondary",
            outline=True, size="sm", className="flex-fill",
            href="/model",
        ),
    ], className="w-100 mb-3")

    common_params = [
        _slider_row(
            "Initial infection fraction", "param-seed-fraction",
            0.01, 1.0, 0.01, 0.05,
            marks={0.01: "1%", 0.25: "25%", 0.5: "50%", 1.0: "100%"},
        ),
        _number_row("Number of trials", "param-num-trials", 10, 500, 50, 10),
        html.Div([
            html.Label("Seeding strategy", className="form-label small fw-medium mb-1"),
            dbc.Select(
                id="param-seed-strategy",
                options=_STRATEGY_OPTIONS,
                value="random",
                size="sm",
                className="mb-3",
            ),
        ]),
    ]

    # ── Always render all model params; hide irrelevant ones ──────────────
    needs_threshold     = model in ("bootstrap", "h1", "h2")
    needs_phi           = model in ("wtm", "h4", "h5", "h6")
    needs_beta          = model in ("sir", "sis", "h1", "h2", "h3", "h4", "h5")
    needs_gamma         = model in ("sir", "sis", "h1", "h2", "h3", "h4", "h5", "h6")
    needs_switch        = model in ("h2", "h5")
    gamma_label         = "Recovery rate (μ)" if model == "sis" else "Recovery rate (γ)"

    def _vis(show: bool) -> dict:
        return {} if show else {"display": "none"}

    model_params = [
        html.Div(
            _number_row("Bootstrap threshold (k)", "param-threshold", 1, 50, 2),
            style=_vis(needs_threshold),
        ),
        html.Div(
            _slider_row("Fractional threshold (φ)", "param-phi",
                        0.01, 1.0, 0.01, 0.3,
                        marks={0.01: "1%", 0.3: "30%", 0.5: "50%", 1.0: "100%"}),
            style=_vis(needs_phi),
        ),
        html.Div(
            _slider_row("Transmission rate (β)", "param-beta",
                        0.01, 1.0, 0.01, 0.3,
                        marks={0.01: "0.01", 0.3: "0.3", 1.0: "1.0"}),
            style=_vis(needs_beta),
        ),
        html.Div(
            _slider_row(gamma_label, "param-gamma",
                        0.01, 1.0, 0.01, 0.1,
                        marks={0.01: "0.01", 0.1: "0.1", 1.0: "1.0"}),
            style=_vis(needs_gamma),
        ),
        html.Div(
            _slider_row("Switch threshold (f)", "param-switch-fraction",
                        0.01, 1.0, 0.01, 0.2,
                        marks={0.01: "1%", 0.2: "20%", 0.5: "50%", 1.0: "100%"}),
            style=_vis(needs_switch),
        ),
    ]

    run_btn = dbc.Button(
        "Run Simulation",
        id="btn-run-sim",
        color="warning",
        className="w-100 fw-bold mt-2",
        style={"backgroundColor": "#FF6D00", "borderColor": "#FF6D00"},
    )

    return dbc.Card([
        dbc.CardBody([
            html.H6("Parameters", className="text-uppercase text-muted small fw-bold mb-3"),
            *graph_section,
            nav_btns,
            html.Hr(className="my-2"),
            *common_params,
            *model_params,
            run_btn,
        ])
    ], className="shadow-sm border-0", style={"backgroundColor": "#F8FAFC"})
