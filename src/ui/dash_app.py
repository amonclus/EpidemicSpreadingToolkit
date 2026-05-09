"""
dash_app.py — Dash application instance + root layout.

Creates the Dash app, sets the global layout with URL routing,
registers all callbacks, and exposes the server for WSGI.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when run directly
_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from ui.components.store import graph_from_store

# ── Create app ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        dbc.icons.BOOTSTRAP,
    ],
    suppress_callback_exceptions=True,
    title="Network Contagion Lab",
    update_title=None,
)
server = app.server  # for WSGI / gunicorn


# ── Custom CSS ────────────────────────────────────────────────────────────────
_CUSTOM_CSS = """
/* ── Typography ── */
body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
h1,h2,h3 { letter-spacing: -0.3px; }

/* ── Cards ── */
.card { transition: box-shadow 0.18s ease, border-color 0.18s ease; }
.card:hover { box-shadow: 0 6px 24px rgba(37,99,235,0.09) !important; }

/* ── Tabs ── */
.nav-tabs .nav-link { border-radius: 8px 8px 0 0 !important; font-weight: 500; }
.nav-tabs .nav-link.active {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
    background-color: #fff !important;
}

/* ── Buttons ── */
.btn { border-radius: 8px !important; font-weight: 500 !important; }
.btn:hover { box-shadow: 0 2px 8px rgba(37,99,235,0.18) !important; }

/* ── Navbar ── */
.navbar { padding: 0.5rem 1rem; }

/* ── Form controls ── */
.form-control, .form-select { border-radius: 6px !important; }

/* ── Slider ── */
.rc-slider-track { background-color: #2563EB !important; }
.rc-slider-handle { border-color: #2563EB !important; }

/* ── Metric tiles ── */
.metric-tile {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 20px;
}

/* ── Page content padding ── */
#page-content { min-height: calc(100vh - 80px); }
"""


# ── Root layout ───────────────────────────────────────────────────────────────
app.index_string = app.index_string.replace(
    "</head>",
    f"<style>{_CUSTOM_CSS}</style></head>",
)

app.layout = html.Div([
    # URL / routing
    dcc.Location(id="url", refresh=False),

    # Global stores
    dcc.Store(id="store-graph",  storage_type="session"),
    dcc.Store(id="store-model",  storage_type="session"),

    # Dynamic navbar
    html.Div(id="navbar-container"),

    # Page content
    html.Div(id="page-content", className="pb-5"),
])


# ── Page routing callback ─────────────────────────────────────────────────────
@app.callback(
    Output("page-content",      "children"),
    Output("navbar-container",  "children"),
    Output("store-model",       "data"),
    Input("url",                "pathname"),
    State("store-graph",        "data"),
    State("store-model",        "data"),
)
def route(pathname, graph_data, current_model):
    from ui.components.navbar import build_navbar
    from ui.pages.welcome import layout as welcome_layout
    from ui.pages.graph_setup import layout as graph_setup_layout
    from ui.pages.model_selection import layout as model_selection_layout
    from ui.pages.lab import layout as lab_layout
    from ui.pages.ml_page import layout as ml_layout

    # Determine graph summary for navbar
    graph = graph_from_store(graph_data)
    graph_info = None
    if graph is not None:
        n, m = graph.number_of_nodes(), graph.number_of_edges()
        graph_info = f"{n:,} nodes · {m:,} edges"

    navbar = build_navbar(graph_info)
    new_model = no_update

    if pathname == "/" or pathname is None:
        return welcome_layout(), navbar, new_model

    if pathname == "/graph":
        return graph_setup_layout(), navbar, new_model

    if pathname == "/model":
        gn, gm = (graph.number_of_nodes(), graph.number_of_edges()) if graph else (None, None)
        return model_selection_layout(gn, gm), navbar, new_model

    if pathname and pathname.startswith("/lab/"):
        model = pathname.split("/lab/", 1)[1]
        valid = {"bootstrap", "sir", "sis", "wtm", "h1", "h2", "h3", "h4", "h5", "h6"}
        if model in valid:
            gn = graph.number_of_nodes() if graph else None
            gm = graph.number_of_edges() if graph else None
            new_model = model
            return lab_layout(model, gn, gm), navbar, new_model
        return html.Div(dbc.Alert(f"Unknown model: {model}", color="danger")), navbar, no_update

    if pathname == "/ml":
        return ml_layout(), navbar, no_update

    # 404
    return html.Div([
        dbc.Container([
            dbc.Alert(f"Page not found: {pathname}", color="warning"),
            dbc.Button("Back to home", href="/", color="primary"),
        ], className="py-5")
    ]), navbar, no_update


# ── Register all callbacks ────────────────────────────────────────────────────
def _register_callbacks() -> None:
    from ui.callbacks.graph_callbacks import register as reg_graph
    from ui.callbacks.lab_callbacks   import register as reg_lab
    from ui.callbacks.ml_callbacks    import register as reg_ml

    reg_graph(app)
    reg_lab(app)
    reg_ml(app)


_register_callbacks()
