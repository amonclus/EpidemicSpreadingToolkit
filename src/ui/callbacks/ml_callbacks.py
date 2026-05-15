"""
ml_callbacks.py — Callbacks for the ML Virality Predictor page.

All ML prediction logic is delegated to the existing pure functions in
ui/tabs/ml_tab.py (which are framework-agnostic once the Streamlit calls
are removed).  We call those functions directly.
"""
from __future__ import annotations

import base64
import io

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update

from ui.tabs.ml_tab import (
    MODEL_INFO,
    NETWORK_MULTIPLIER,
    TIPPING_THRESHOLDS,
    _ENDEMIC_MODELS,
    _DISPLAY_NAME,
    _ENDEMIC_SIM,
    _extract_features,
    _generate_trajectory,
    _get_verdict,
    _load_models,
    _run_prediction,
    _run_epidemic_series,
    _load_summary_metrics,
)
from ui.components.store import graph_from_store

_PLOTLY = dict(
    paper_bgcolor="#F8FAFC",
    plot_bgcolor="#F8FAFC",
)
_M = dict(l=50, r=20, t=30, b=50)


def register(app) -> None:

    # ── Toggle manual / upload / simulate divs ───────────────────────────────
    @app.callback(
        Output("ml-manual-div", "style"),
        Output("ml-upload-div", "style"),
        Output("ml-sim-div",    "style"),
        Input("ml-input-method", "value"),
    )
    def toggle_input_method(method):
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if method == "manual"   else hide,
            show if method == "upload"   else hide,
            show if method == "simulate" else hide,
        )

    # ── Show/hide simulation param groups based on model ─────────────────────
    @app.callback(
        Output("ml-sim-beta-group",  "style"),
        Output("ml-sim-gamma-group", "style"),
        Output("ml-sim-k-group",     "style"),
        Output("ml-sim-phi-group",   "style"),
        Output("ml-sim-sf-group",    "style"),
        Input("ml-sim-model", "value"),
    )
    def toggle_sim_params(model):
        show = {"display": "block"}
        hide = {"display": "none"}
        beta_models  = {"sir", "sis", "h1", "h2", "h3", "h4", "h5"}
        gamma_models = {"sir", "sis", "h1", "h2", "h3", "h4", "h5", "h6"}
        k_models     = {"bootstrap", "h1", "h2"}
        phi_models   = {"wtm", "h4", "h5", "h6"}
        sf_models    = {"h2", "h5"}
        return (
            show if model in beta_models  else hide,
            show if model in gamma_models else hide,
            show if model in k_models     else hide,
            show if model in phi_models   else hide,
            show if model in sf_models    else hide,
        )

    # ── Update seed caption ───────────────────────────────────────────────────
    @app.callback(
        Output("ml-sim-seed-caption", "children"),
        Input("ml-sim-seed-frac",     "value"),
        State("store-graph",          "data"),
    )
    def update_seed_caption(seed_frac, graph_data):
        graph = graph_from_store(graph_data)
        if graph is None:
            return "No graph loaded"
        n = graph.number_of_nodes()
        n_seeds = max(1, round((seed_frac or 1.0) / 100 * n))
        return f"≈ {n_seeds} seed node{'s' if n_seeds != 1 else ''} on this graph ({n:,} nodes)"

    # ── Run epidemic simulation and store resulting series ────────────────────
    @app.callback(
        Output("ml-series-store",  "data",     allow_duplicate=True),
        Output("ml-sim-result",    "children"),
        Input("ml-sim-run-btn",    "n_clicks"),
        State("store-graph",       "data"),
        State("ml-sim-model",      "value"),
        State("ml-sim-beta",       "value"),
        State("ml-sim-gamma",      "value"),
        State("ml-sim-k",          "value"),
        State("ml-sim-phi",        "value"),
        State("ml-sim-sf",         "value"),
        State("ml-sim-seed-frac",  "value"),
        prevent_initial_call=True,
    )
    def run_ml_simulation(n_clicks, graph_data, model_key,
                          beta, gamma, k, phi, sf, seed_frac):
        if not n_clicks:
            return no_update, no_update

        graph = graph_from_store(graph_data)
        if graph is None:
            return no_update, dbc.Alert("No graph loaded — load one in the Lab first.",
                                        color="warning")

        params = {}
        if model_key in {"sir", "sis", "h1", "h2", "h3", "h4", "h5"}:
            params["beta"]  = float(beta  or 0.30)
            params["gamma"] = float(gamma or 0.10)
        if model_key == "h6":
            params["gamma"] = float(gamma or 0.10)
        if model_key in {"bootstrap", "h1", "h2"}:
            params["threshold"] = int(k or 3)
        if model_key in {"wtm", "h4", "h5", "h6"}:
            params["phi"] = float(phi or 0.25)
        if model_key in {"h2", "h5"}:
            params["switch_fraction"] = float(sf or 0.20)

        n = graph.number_of_nodes()
        n_seeds = max(1, round((seed_frac or 1.0) / 100 * n))

        try:
            series, rho_final = _run_epidemic_series(
                graph, model_key, params,
                n_seeds=n_seeds, max_steps=20, critical_frac=0.30, min_steps=10,
            )
        except Exception as exc:
            return no_update, dbc.Alert(f"Simulation error: {exc}", color="danger")

        if not series:
            return no_update, dbc.Alert("Epidemic died immediately — try different parameters.",
                                        color="warning")

        series_pct = [round(v * 100, 4) for v in series]
        preview = ", ".join(f"{v:.1f}" for v in series_pct[:6])
        suffix  = "…" if len(series_pct) > 6 else ""
        result_info = html.Div([
            html.P(f"{len(series_pct)} steps recorded — {preview}{suffix}",
                   className="mb-1"),
            dbc.Alert(
                [html.Strong("True final reach: "),
                 f"{rho_final * 100:.1f}%"],
                color="info", className="py-1 px-2 mb-0",
            ),
        ])
        return series_pct, result_info

    # ── Parse upload and cache series in store ────────────────────────────────
    @app.callback(
        Output("ml-series-store",   "data",     allow_duplicate=True),
        Output("ml-upload-status",  "children"),
        Input("ml-upload",          "contents"),
        State("ml-upload",          "filename"),
        prevent_initial_call=True,
    )
    def parse_upload(contents, filename):
        if contents is None:
            return no_update, no_update
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.BytesIO(decoded))
        except Exception as exc:
            return no_update, dbc.Alert(f"Could not read file: {exc}", color="danger")

        preferred = ["reach_pct", "I", "value", "I_frac", "infected", "spread"]
        vals = None
        for col in preferred:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
                if len(vals) >= 2:
                    break

        if vals is None:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                return no_update, dbc.Alert("No numeric columns found.", color="danger")
            col = num_cols[0]
            vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()

        if len(vals) < 2:
            return no_update, dbc.Alert("Need at least 2 data points.", color="danger")
        if max(vals) <= 1.0:
            vals = [v * 100 for v in vals]
        return [round(v, 4) for v in vals], f"Loaded {len(vals)} steps from {filename}"

    # ── Run prediction ────────────────────────────────────────────────────────
    @app.callback(
        Output("ml-predict-content", "children"),
        Input("ml-predict-btn",      "n_clicks"),
        State("ml-input-method",     "value"),
        State("ml-series-input",     "value"),
        State("ml-series-store",     "data"),
        State("ml-content-type",     "value"),
        State("ml-network-type",     "value"),
        State("ml-horizon",          "value"),
        State("store-graph",         "data"),
        prevent_initial_call=True,
    )
    def run_prediction(n_clicks, input_method, manual_text, stored_series,
                       content_type, network_type, horizon, graph_data):
        if not n_clicks:
            return _idle_state()

        # Parse series
        if input_method == "manual":
            try:
                series_pct = [float(x.strip()) for x in manual_text.split(",") if x.strip()]
            except ValueError:
                return dbc.Alert("Could not parse input values.", color="danger")
        else:
            series_pct = stored_series or [0.5, 0.8, 1.2, 2.1, 3.8]
            if input_method == "simulate" and not stored_series:
                return dbc.Alert("Run a simulation first before predicting.", color="warning")

        if len(series_pct) < 2:
            return dbc.Alert("Need at least 2 data points.", color="danger")

        graph = graph_from_store(graph_data)
        try:
            res = _run_prediction(series_pct, content_type, network_type or "",
                                  int(horizon or 50), graph=graph)
        except Exception as exc:
            return dbc.Alert(f"Prediction error: {exc}", color="danger")

        return _build_prediction_results(res)

    # ── Education tab ─────────────────────────────────────────────────────────
    @app.callback(
        Output("ml-learn-content", "children"),
        Input("ml-tabs",           "active_tab"),
    )
    def render_learn_tab(active_tab):
        if active_tab != "ml-tab-learn":
            return no_update
        return _build_education_content()

    # ── About tab ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("ml-about-content", "children"),
        Input("ml-tabs",           "active_tab"),
    )
    def render_about_tab(active_tab):
        if active_tab != "ml-tab-about":
            return no_update
        return _build_about_content()

    # ── Download trajectory CSV (only available after prediction) ─────────────
    @app.callback(
        Output("ml-download",     "data"),
        Input("ml-download-btn",  "n_clicks"),
        State("ml-export-store",  "data"),
        prevent_initial_call=True,
    )
    def download_trajectory(n_clicks, export_data):
        if not n_clicks or not export_data:
            return no_update
        df = pd.DataFrame(export_data)
        return dcc.send_data_frame(df.to_csv, "virality_prediction.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Content builders
# ─────────────────────────────────────────────────────────────────────────────

def _idle_state() -> html.Div:
    return html.Div([
        html.H4("Contagion Predictor", className="fw-bold mb-2"),
        html.P("Enter your spread data in the panel on the left, then click Predict Virality.",
               className="text-muted"),
        dbc.Row([
            dbc.Col([
                dbc.Card([dbc.CardBody([
                    html.H5("Reach Prediction", className="fw-bold"),
                    html.P("Predict the final % of the network reached.", className="text-muted small"),
                ])], className="text-center shadow-sm"),
            ], md=4),
            dbc.Col([
                dbc.Card([dbc.CardBody([
                    html.H5("Mechanism ID", className="fw-bold"),
                    html.P("Identify the spreading model from the early curve shape.", className="text-muted small"),
                ])], className="text-center shadow-sm"),
            ], md=4),
            dbc.Col([
                dbc.Card([dbc.CardBody([
                    html.H5("Tipping Point", className="fw-bold"),
                    html.P("Know whether the viral threshold has already been crossed.", className="text-muted small"),
                ])], className="text-center shadow-sm"),
            ], md=4),
        ]),
    ])


def _build_prediction_results(res: dict) -> html.Div:
    rho        = res["rho_final"]
    rho_pct    = rho * 100
    rho_std    = res["rho_std"]
    rho_std_pct = rho_std * 100
    model_name = res["model_name"]
    minfo      = MODEL_INFO.get(model_name, MODEL_INFO["SIR"])
    conf       = res["confidence"]
    lo_pct     = max(0.0,   rho_pct - 1.5 * rho_std_pct)
    hi_pct     = min(100.0, rho_pct + 1.5 * rho_std_pct)

    is_endemic = model_name in _ENDEMIC_MODELS
    traj_arr   = np.array(res["traj_pct"])
    peak_pct   = float(traj_arr.max())
    peak_rho   = peak_pct / 100.0

    v_label, v_css, v_desc = _get_verdict(peak_rho if is_endemic else rho)

    _CSS_COLORS = {
        "verdict-viral":    ("#FEE2E2", "#B91C1C"),
        "verdict-strong":   ("#FEF3C7", "#92400E"),
        "verdict-moderate": ("#FEF9C3", "#713F12"),
        "verdict-niche":    ("#DBEAFE", "#1E40AF"),
    }
    bg, fg = _CSS_COLORS.get(v_css, ("#F8FAFC", "#111"))

    # Demo banner
    demo_banner = []
    if res.get("demo_mode"):
        demo_banner = [dbc.Alert(
            "Demo mode — ML model files not found. Predictions are illustrative only.",
            color="warning", className="mb-3",
        )]

    # Verdict card
    verdict_card = dbc.Card([
        dbc.CardBody([
            html.P("Spread assessment", className="text-uppercase small fw-bold mb-1",
                   style={"opacity": "0.7", "letterSpacing": "0.08em"}),
            html.H3(v_label, className="fw-bold mb-1", style={"color": fg}),
            html.P(v_desc, className="small mb-3"),
            *(
                [
                    html.P(f"Peak reach: {peak_pct:.1f}%", className="fw-semibold mb-1"),
                    dbc.Progress(value=int(peak_rho * 100), color="danger", className="mb-2"),
                    html.P(f"Endemic level: {rho_pct:.1f}%", className="small text-muted"),
                ]
                if is_endemic else
                [
                    html.P(f"Predicted final reach: {rho_pct:.1f}%", className="fw-semibold mb-1"),
                    dbc.Progress(value=int(min(rho, 1.0) * 100), color="primary", className="mb-2"),
                    html.P(f"90% CI: {lo_pct:.1f}% – {hi_pct:.1f}%", className="small text-muted"),
                ]
            ),
        ])
    ], style={"backgroundColor": bg, "border": f"1px solid {fg}30", "borderRadius": "8px"})

    # Model card
    model_card = dbc.Card([
        dbc.CardBody([
            html.P("Spreading mechanism", className="text-uppercase small fw-bold mb-1",
                   style={"opacity": "0.7", "letterSpacing": "0.08em"}),
            html.H4(model_name, className="fw-bold mb-1"),
            html.P(minfo["type"], className="small text-muted mb-2"),
            html.P(minfo["desc"], className="small mb-3"),
            html.P(f"Classifier confidence: {conf * 100:.0f}%", className="fw-semibold mb-1"),
            dbc.Progress(value=int(conf * 100), color="info", className="mb-1"),
            html.P(f"Typical content: {minfo['best_for']}", className="small text-muted"),
        ])
    ], className="shadow-sm", style={"borderRadius": "8px"})

    # Trajectory chart
    traj   = np.array(res["traj_pct"])
    t_full = np.array(res["t_full"])
    n_obs  = res["n_obs"]
    band_hi = np.clip(traj + 1.5 * rho_std_pct, 0, 100)
    band_lo = np.clip(traj - 1.5 * rho_std_pct, 0, 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_full[n_obs - 1:], y=band_hi[n_obs - 1:],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=t_full[n_obs - 1:], y=band_lo[n_obs - 1:],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(255,109,0,0.15)",
        name="Confidence band",
    ))
    fig.add_trace(go.Scatter(
        x=t_full[n_obs - 1:], y=traj[n_obs - 1:],
        mode="lines", line=dict(color="#FF6D00", width=2.5, dash="dash"),
        name="Predicted",
    ))
    fig.add_trace(go.Scatter(
        x=t_full[:n_obs], y=res["series_pct"],
        mode="lines+markers",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=7),
        name="Observed",
    ))
    fig.add_vline(
        x=float(n_obs - 1),
        line=dict(dash="dot", color="rgba(100,100,100,0.4)", width=1.5),
        annotation_text="Observed | Predicted",
        annotation_position="top",
        annotation_font_color="rgba(80,80,80,0.8)",
    )
    fig.update_layout(
        xaxis_title="Time steps",
        yaxis_title="% of network reached",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        height=400, margin=_M, **_PLOTLY,
    )

    # Tipping point
    tipping      = TIPPING_THRESHOLDS.get(model_name, 0.10)
    obs_arr      = np.array(res["series_pct"])
    current_frac = float(obs_arr.max()) / 100.0
    crossed      = current_frac >= tipping

    if crossed:
        tipping_content = dbc.Alert(
            [
                html.Strong("This content has already crossed the viral tipping point. "),
                "Spread is now self-sustaining.",
            ],
            color="success",
        )
    else:
        gap         = tipping - current_frac
        extra_users = max(1, int(gap * 10_000))
        avg_step    = max(float(np.mean(np.diff(obs_arr))) / 100.0, 1e-5)
        hours_est   = max(1, int(gap / avg_step))
        tipping_content = dbc.Alert(
            [
                html.Strong("This content has NOT yet crossed the viral tipping point. "),
                f"Getting {extra_users:,} more high-connectivity nodes to share "
                f"in the next {hours_est} steps could trigger a cascade.",
            ],
            color="warning",
        )

    # Export
    df_export = pd.DataFrame({
        "step":      t_full,
        "reach_pct": np.round(traj, 3),
        "type":      ["observed"] * n_obs + ["predicted"] * (len(t_full) - n_obs),
    })

    return html.Div([
        *demo_banner,
        html.Hr(className="my-2"),
        dbc.Row([
            dbc.Col([verdict_card], md=6),
            dbc.Col([model_card], md=6),
        ], className="mb-4"),
        html.H5("Predicted Trajectory", className="fw-semibold mb-2"),
        dcc.Graph(figure=fig),
        html.H5("Tipping Point Analysis", className="fw-semibold mb-2"),
        tipping_content,
        html.Hr(),
        dbc.Accordion([
            dbc.AccordionItem([
                dcc.Download(id="ml-download"),
                dbc.Button(
                    "Download trajectory CSV",
                    id="ml-download-btn",
                    color="secondary",
                    size="sm",
                    outline=True,
                ),
                dcc.Store(id="ml-export-store",
                          data=df_export.to_dict("records")),
            ], title="Export Results"),
        ], start_collapsed=True, className="mt-2"),
    ])


def _build_education_content() -> html.Div:
    t = np.linspace(0, 20, 200)

    # Simple contagion curve
    fig_s = go.Figure()
    for n0, col, label in [(10, "#2563EB", "Low seed"), (100, "#60A5FA", "High seed")]:
        y = np.clip(1.0 * (1.0 - np.exp(-0.2 * t)) * n0 / 10, 0, 100)
        fig_s.add_trace(go.Scatter(x=t, y=y, mode="lines",
                                   line=dict(width=2, color=col), name=label))
    fig_s.update_layout(xaxis_title="Time", yaxis_title="% reached",
                         height=300, margin=_M, **_PLOTLY)

    # Complex contagion curve
    fig_c = go.Figure()
    for n0, col, label in [(10, "#EF5350", "Low seed"), (100, "#FF7043", "High seed")]:
        scale = n0 / 10
        y = scale / (1.0 + np.exp(-0.5 * (t - (10 - scale))))
        fig_c.add_trace(go.Scatter(x=t, y=np.clip(y, 0, 100), mode="lines",
                                   line=dict(width=2, color=col), name=label))
    fig_c.update_layout(xaxis_title="Time", yaxis_title="% reached",
                         height=300, margin=_M, **_PLOTLY)

    # Tipping point chart
    seeds         = np.linspace(0, 0.20, 200)
    simple_final  = np.clip(seeds * 5.0, 0, 1)
    complex_final = np.where(seeds < 0.08, seeds * 0.3,
                             np.clip((seeds - 0.08) * 12, 0, 1))
    fig_tip = go.Figure()
    fig_tip.add_trace(go.Scatter(x=seeds * 100, y=simple_final * 100,
                                  mode="lines", line=dict(color="#2563EB", width=2.5),
                                  name="Simple contagion"))
    fig_tip.add_trace(go.Scatter(x=seeds * 100, y=complex_final * 100,
                                  mode="lines", line=dict(color="#EF5350", width=2.5),
                                  name="Complex contagion"))
    fig_tip.add_vline(x=8, line=dict(dash="dot", color="rgba(100,100,100,0.5)"),
                       annotation_text="Tipping point",
                       annotation_position="top right")
    fig_tip.update_layout(
        xaxis_title="Initial seed (% of network)",
        yaxis_title="Final reach (% of network)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400, margin=_M, **_PLOTLY,
    )

    return html.Div([
        html.H4("How does content spread?", className="fw-bold mb-3"),
        html.P("A simple guide to the science behind epidemic-like spreading.",
               className="text-muted"),
        html.Hr(),
        html.H5("The Two Types of Viral Content", className="fw-semibold mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([dbc.CardBody([
                    html.H6("Spreads like a cold — Simple contagion", className="fw-bold mb-2"),
                    html.P("One contact is enough. Growth is exponential: doubles at a steady rate. "
                           "Like a funny video or breaking news.", className="text-muted small mb-3"),
                    dcc.Graph(figure=fig_s),
                ])], className="shadow-sm h-100"),
            ], md=6),
            dbc.Col([
                dbc.Card([dbc.CardBody([
                    html.H6("Spreads like a trend — Complex contagion", className="fw-bold mb-2"),
                    html.P("You need multiple friends doing it. Slow start, explosive once "
                           "the threshold is crossed. Like a fitness challenge.", className="text-muted small mb-3"),
                    dcc.Graph(figure=fig_c),
                ])], className="shadow-sm h-100"),
            ], md=6),
        ]),
        html.Hr(),
        html.H5("Real-world examples", className="fw-semibold mb-3"),
        dbc.Row([
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H5("Breaking news", className="fw-bold"),
                dbc.Badge("SIR-like spreading", color="primary", className="mb-2"),
                html.P("Spreads immediately to anyone who sees it. Peaks fast, fades fast.",
                       className="text-muted small"),
            ])], className="text-center shadow-sm")], md=4),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H5("Ice bucket challenge", className="fw-bold"),
                dbc.Badge("Complex contagion", color="danger", className="mb-2"),
                html.P("Required seeing multiple friends participate. Slow start, then explosive.",
                       className="text-muted small"),
            ])], className="text-center shadow-sm")], md=4),
            dbc.Col([dbc.Card([dbc.CardBody([
                html.H5("Vaccine hesitancy", className="fw-bold"),
                dbc.Badge("Hybrid spreading", color="warning", className="mb-2"),
                html.P("Spreads through both information exposure AND social reinforcement.",
                       className="text-muted small"),
            ])], className="text-center shadow-sm")], md=4),
        ], className="mb-4"),
        html.Hr(),
        html.H5("The Tipping Point", className="fw-semibold mb-3"),
        dcc.Graph(figure=fig_tip),
        html.P(
            "Complex contagion has a tipping point — seed too few nodes and nothing happens. "
            "Seed enough and the whole network activates.",
            className="text-muted small mt-2",
        ),
    ])


def _build_about_content() -> html.Div:
    mae_pct, cls_acc = _load_summary_metrics()

    _base_models = ["SIR", "SIS", "BP", "WTM", "H1", "H2", "H3", "H4", "H5", "H6"]
    table_data = {
        "Model":              _base_models,
        "Type":               [MODEL_INFO[m]["type"]      for m in _base_models],
        "Plain-English name": [MODEL_INFO[m]["label"]     for m in _base_models],
        "Best describes":     [MODEL_INFO[m]["best_for"]  for m in _base_models],
        "Displayed as":       [_DISPLAY_NAME.get(m, m)   for m in _base_models],
    }
    df_models = pd.DataFrame(table_data)

    return html.Div([
        html.H4("About the Predictor", className="fw-bold mb-3"),
        html.P(
            "The virality predictor is the applied deep learning component of a bachelor's thesis "
            "on hybrid epidemic spreading models. Ten spreading models were studied, simulations "
            "were run on three synthetic network types and two real-world networks (Facebook and GitHub), "
            "generating 50,000 labelled time-series observations.",
            className="text-muted",
        ),
        html.Hr(),
        html.H5("The models", className="fw-semibold mb-3"),
        dbc.Table.from_dataframe(df_models, striped=True, bordered=False,
                                  hover=True, responsive=True, size="sm"),
        html.Hr(),
        html.H5("How accurate is the prediction?", className="fw-semibold mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([dbc.CardBody([
                    html.P("Reach prediction error", className="text-muted small mb-1"),
                    html.H3(f"±{mae_pct:.1f}%", className="fw-bold text-primary"),
                    html.P(f"Average error across all models and network types.",
                           className="text-muted small"),
                ])], className="text-center shadow-sm"),
            ], md=6),
            dbc.Col([
                dbc.Card([dbc.CardBody([
                    html.P("Mechanism identification", className="text-muted small mb-1"),
                    html.H3(f"{cls_acc:.1f}%", className="fw-bold text-success"),
                    html.P("Correct spreading model identified from first 30 time steps.",
                           className="text-muted small"),
                ])], className="text-center shadow-sm"),
            ], md=6),
        ], className="mb-4"),
        html.Hr(),
        html.H5("Limitations", className="fw-semibold mb-2"),
        dbc.Alert(
            "This tool is based on simulated network data. Real social networks are more complex. "
            "Predictions should be interpreted as indicative rather than precise. "
            "Most reliable when at least 5–10 time steps have been observed.",
            color="info",
        ),
        html.Hr(),
        html.H5("Credits", className="fw-semibold mb-3"),
        dbc.Table([
            html.Tbody([
                html.Tr([html.Td(html.Strong("Project")),
                         html.Td("Bachelor's Thesis — Hybrid Epidemic Spreading Models")]),
                html.Tr([html.Td(html.Strong("Author")),  html.Td("Álvaro Monclús")]),
                html.Tr([html.Td(html.Strong("Year")),    html.Td("2025")]),
                html.Tr([html.Td(html.Strong("Models")),
                         html.Td("SIR, SIS, Bootstrap Percolation, WTM, H1–H6")]),
                html.Tr([html.Td(html.Strong("Networks")),
                         html.Td("ER, RGG, Lattice, Facebook (SNAP), GitHub (MUSAE)")]),
                html.Tr([html.Td(html.Strong("Predictor")),
                         html.Td("CNN, 22 time-series + 10 graph features, 50,000 simulations")]),
                html.Tr([html.Td(html.Strong("Stack")),
                         html.Td("Python · NetworkX · Plotly Dash · PyTorch · scikit-learn")]),
            ])
        ], bordered=False, striped=True, size="sm", responsive=True),
    ])
