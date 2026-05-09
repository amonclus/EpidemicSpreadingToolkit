"""
welcome.py — Welcome / landing page layout.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


def layout() -> html.Div:
    return html.Div([
        # Hero section
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1(
                        "Network Contagion Lab",
                        className="display-5 fw-bold mb-2",
                        style={"color": "#1E3A8A", "letterSpacing": "-0.5px"},
                    ),
                    html.P(
                        "A research platform for studying how things spread through networks — "
                        "cascading failures, epidemics, information, and social influence.",
                        className="lead text-muted mb-4",
                    ),
                    dbc.Button(
                        "Get started",
                        href="/graph",
                        color="primary",
                        size="lg",
                        className="fw-semibold px-5",
                        style={"backgroundColor": "#2563EB", "borderColor": "#2563EB"},
                    ),
                ], md=8, className="py-5"),
            ]),

            html.Hr(className="my-4"),

            # Feature cards row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("10 Spreading Models", className="fw-bold mb-2",
                                    style={"color": "#2563EB"}),
                            html.P(
                                "From classical SIR epidemics and bootstrap percolation "
                                "to six hybrid models that blend threshold and probabilistic dynamics.",
                                className="text-muted small mb-0",
                            ),
                        ])
                    ], className="h-100 shadow-sm",
                       style={"borderTop": "3px solid #2563EB", "borderRadius": "8px"}),
                ], md=4, className="mb-3"),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Built-in Analysis", className="fw-bold mb-2",
                                    style={"color": "#2563EB"}),
                            html.P(
                                "Cascade simulations, node vulnerability maps, parameter sweeps, "
                                "and animated cascade walkthroughs — all in one place.",
                                className="text-muted small mb-0",
                            ),
                        ])
                    ], className="h-100 shadow-sm",
                       style={"borderTop": "3px solid #2563EB", "borderRadius": "8px"}),
                ], md=4, className="mb-3"),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("ML Virality Predictor", className="fw-bold mb-2",
                                    style={"color": "#FF6D00"}),
                            html.P(
                                "A deep learning model trained on 50,000 simulations that identifies "
                                "the spreading mechanism and estimates final reach from early trajectory data.",
                                className="text-muted small mb-0",
                            ),
                        ])
                    ], className="h-100 shadow-sm",
                       style={"borderTop": "3px solid #FF6D00", "borderRadius": "8px"}),
                ], md=4, className="mb-3"),
            ]),

            html.Hr(className="my-4"),

            dbc.Row([
                dbc.Col([
                    html.H4("How it works", className="fw-bold mb-3"),
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.Span("1", className="badge bg-primary me-2"),
                            html.Strong("Load a network"),
                            html.Span(
                                " — generate an Erdős–Rényi, Random Geometric, or Lattice graph, "
                                "or upload your own (DIMACS, edge list, GML).",
                                className="text-muted",
                            ),
                        ], className="border-0 ps-0"),
                        dbc.ListGroupItem([
                            html.Span("2", className="badge bg-primary me-2"),
                            html.Strong("Choose a model"),
                            html.Span(
                                " — pick the spreading dynamic you want to study.",
                                className="text-muted",
                            ),
                        ], className="border-0 ps-0"),
                        dbc.ListGroupItem([
                            html.Span("3", className="badge bg-primary me-2"),
                            html.Strong("Explore"),
                            html.Span(
                                " — run simulations, sweep parameters, and inspect which nodes matter most.",
                                className="text-muted",
                            ),
                        ], className="border-0 ps-0"),
                    ], flush=True),
                ], md=8),
            ]),

        ], fluid=False, className="py-4"),
    ])
