"""
navbar.py — Top navigation bar component.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def build_navbar(graph_info: str | None = None) -> dbc.NavbarSimple:
    brand_children = [
        html.Span("Network Contagion Lab", className="fw-bold"),
    ]
    if graph_info:
        brand_children.append(
            dbc.Badge(graph_info, color="primary", className="ms-2 fs-6 fw-normal")
        )

    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(
                dbc.NavLink("Lab", href="/model", id="nav-lab"),
            ),
            dbc.NavItem(
                dbc.NavLink("ML Predictor", href="/ml"),
            ),
        ],
        brand=html.Span(brand_children),
        brand_href="/",
        color="#1E40AF",
        dark=True,
        className="mb-0 shadow-sm",
        fluid=True,
        style={"borderBottom": "1px solid #1e3a8a"},
    )
