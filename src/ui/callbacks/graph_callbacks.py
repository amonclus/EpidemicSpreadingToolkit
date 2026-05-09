"""
graph_callbacks.py — Callbacks for the graph setup page.
"""
from __future__ import annotations

import base64
import os
import tempfile

import dash_bootstrap_components as dbc
import networkx as nx
from dash import Input, Output, State, callback, html, no_update

from input.graph_generator import (
    generate_er_graph,
    generate_lattice_graph,
    generate_random_geometric_graph,
)
from input.graph_loader import load_graph_auto
from ui.components.store import graph_to_store


def register(app) -> None:
    """Register all graph-setup callbacks on the given Dash app."""

    # ── Toggle param sections based on graph type ─────────────────────
    @app.callback(
        Output("er-params",      "style"),
        Output("rgg-params",     "style"),
        Output("lattice-params", "style"),
        Input("graph-type-select", "value"),
    )
    def toggle_params(graph_type):
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if graph_type == "er"      else hide,
            show if graph_type == "rgg"     else hide,
            show if graph_type == "lattice" else hide,
        )

    # ── Generate graph button ──────────────────────────────────────────
    @app.callback(
        Output("store-graph",        "data",    allow_duplicate=True),
        Output("generate-output",    "children"),
        Output("url",                "pathname", allow_duplicate=True),
        Input("btn-generate-graph",  "n_clicks"),
        State("graph-type-select",   "value"),
        State("er-n",                "value"),
        State("er-p",                "value"),
        State("rgg-n",               "value"),
        State("rgg-r",               "value"),
        State("lattice-size",        "value"),
        prevent_initial_call=True,
    )
    def generate_graph(n_clicks, graph_type,
                       er_n, er_p,
                       rgg_n, rgg_r,
                       lattice_size):
        if not n_clicks:
            return no_update, no_update, no_update

        try:
            if graph_type == "er":
                g = generate_er_graph(int(er_n or 100), float(er_p or 0.1))
            elif graph_type == "rgg":
                g = generate_random_geometric_graph(int(rgg_n or 100), float(rgg_r or 0.2))
            else:
                g = generate_lattice_graph(int(lattice_size or 10))
        except Exception as exc:
            return no_update, dbc.Alert(f"Error generating graph: {exc}", color="danger"), no_update

        return graph_to_store(g), no_update, "/model"

    # ── Upload graph ───────────────────────────────────────────────────
    @app.callback(
        Output("store-graph",    "data",    allow_duplicate=True),
        Output("upload-output",  "children"),
        Output("url",            "pathname", allow_duplicate=True),
        Input("upload-graph",    "contents"),
        State("upload-graph",    "filename"),
        prevent_initial_call=True,
    )
    def upload_graph(contents, filename):
        if contents is None:
            return no_update, no_update, no_update

        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        suffix = os.path.splitext(filename)[1] or ".txt"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name

        try:
            g = load_graph_auto(tmp_path)
        except Exception as exc:
            os.unlink(tmp_path)
            return no_update, dbc.Alert(f"Could not parse the file: {exc}", color="danger"), no_update
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return graph_to_store(g), no_update, "/model"
