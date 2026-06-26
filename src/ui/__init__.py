# `run_app` is the Streamlit entrypoint. It is imported lazily so that importing
# the `ui` package for the Dash app does not pull in the whole Streamlit tab
# suite (which would apply `@st.cache_data` decorators with no Streamlit runtime
# and emit "No runtime found" warnings).

__all__ = ["run_app"]


def __getattr__(name):
    if name == "run_app":
        from ui.app_entry import run_app
        return run_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
