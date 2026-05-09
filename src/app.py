"""
app.py — Entry point for the Network Contagion Lab (Dash version).

Run:
    python src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the Python path
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ui.dash_app import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
