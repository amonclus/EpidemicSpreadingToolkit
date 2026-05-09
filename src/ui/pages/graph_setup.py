"""
graph_setup.py — Step 1: Set up your network (layout only).
Callbacks are registered in ui/callbacks/graph_callbacks.py.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def layout() -> html.Div:
    return html.Div([
        dbc.Container([
            html.H2("Step 1 — Set up your network",
                    className="fw-bold mb-1", style={"color": "#1E3A8A"}),
            html.P("Choose how to provide the graph you want to analyse.",
                   className="text-muted mb-4"),

            dbc.Tabs([
                dbc.Tab(label="Generate a graph", tab_id="tab-generate",
                        children=_generate_tab()),
                dbc.Tab(label="Upload a file", tab_id="tab-upload",
                        children=_upload_tab()),
            ], id="graph-setup-tabs", active_tab="tab-generate"),

            # Status / spinner feedback
            html.Div(id="graph-setup-status", className="mt-3"),

            # Hidden redirect trigger
            dcc.Location(id="graph-setup-redirect", refresh=True),

        ], fluid=False, className="py-4"),
    ])


def _generate_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Graph type", className="form-label fw-medium"),
                dbc.Select(
                    id="graph-type-select",
                    options=[
                        {"label": "Erdős–Rényi", "value": "er"},
                        {"label": "Random Geometric", "value": "rgg"},
                        {"label": "Lattice", "value": "lattice"},
                    ],
                    value="er",
                ),
            ], md=4, className="mb-3"),
        ]),

        # ER params
        html.Div(id="er-params", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Number of nodes", className="form-label small"),
                    dbc.Input(id="er-n", type="number", min=10, max=5000,
                              value=100, size="sm"),
                ], md=4),
                dbc.Col([
                    html.Label("Edge probability (p)", className="form-label small"),
                    dcc.Slider(id="er-p", min=0.01, max=1.0, step=0.01, value=0.1,
                               marks={0.01: "0.01", 0.25: "0.25", 0.5: "0.5", 1.0: "1.0"},
                               tooltip={"placement": "bottom", "always_visible": False}),
                ], md=6),
            ], className="mb-3"),
        ]),

        # RGG params
        html.Div(id="rgg-params", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Number of nodes", className="form-label small"),
                    dbc.Input(id="rgg-n", type="number", min=10, max=5000,
                              value=100, size="sm"),
                ], md=4),
                dbc.Col([
                    html.Label("Connection radius (r)", className="form-label small"),
                    dcc.Slider(id="rgg-r", min=0.01, max=1.0, step=0.01, value=0.2,
                               marks={0.01: "0.01", 0.2: "0.2", 0.5: "0.5", 1.0: "1.0"},
                               tooltip={"placement": "bottom", "always_visible": False}),
                ], md=6),
            ], className="mb-3"),
        ], style={"display": "none"}),

        # Lattice params
        html.Div(id="lattice-params", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Grid side length", className="form-label small"),
                    dbc.Input(id="lattice-size", type="number", min=3, max=100,
                              value=10, size="sm"),
                ], md=4),
            ], className="mb-3"),
        ], style={"display": "none"}),

        dbc.Button(
            "Generate graph",
            id="btn-generate-graph",
            color="primary",
            style={"backgroundColor": "#2563EB", "borderColor": "#2563EB"},
        ),

        # Loading spinner
        dcc.Loading(
            id="loading-generate",
            children=html.Div(id="generate-output"),
            type="circle",
        ),
    ], className="pt-3")


def _upload_tab() -> html.Div:
    return html.Div([
        html.P(
            "Supported formats: DIMACS (.dimacs), edge list (.txt, .edgelist), GML (.gml)",
            className="text-muted small",
        ),
        dcc.Upload(
            id="upload-graph",
            children=html.Div([
                "Drag and drop or ",
                html.A("select a file", className="text-primary fw-medium"),
            ]),
            style={
                "width": "100%",
                "height": "80px",
                "lineHeight": "80px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "8px",
                "borderColor": "#BFDBFE",
                "textAlign": "center",
                "backgroundColor": "#F8FAFC",
                "cursor": "pointer",
            },
            multiple=False,
            accept=".dimacs,.txt,.gml,.edgelist",
        ),
        dcc.Loading(
            id="loading-upload",
            children=html.Div(id="upload-output"),
            type="circle",
        ),
    ], className="pt-3")
