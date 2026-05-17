# Network Contagion Lab

An interactive platform for simulating, analysing, and predicting epidemic and cascade spreading processes on complex networks.  
Developed as part of a Bachelor's thesis at ISEP.

---

## Motivation

Understanding how diseases, failures, or information propagate through a network is a central challenge in epidemiology, network science, and risk analysis.  
This project explores that question through a unified toolbox that combines:

- **Mathematical spreading models** (Bootstrap Percolation, SIR, SIS, and six hybrid variants)
- **Graph topology analysis** (structural features, robustness metrics, parameter sweeps)
- **Machine-learning forecasting** (predicting epidemic outcomes from early observations)
- **Real-world epidemic validation** against influenza and COVID-19 datasets

The interactive dashboard allows researchers and students to experiment with all models and datasets without writing a single line of code.

---

## Requirements

- **Python 3.10+**
- The following packages (also listed in `src/requirements.txt`):

| Package | Purpose |
|---|---|
| `networkx` | Graph construction and algorithms |
| `matplotlib` / `plotly` | Static and interactive visualisation |
| `dash` + `dash-bootstrap-components` | Web-based interactive UI |
| `pandas` / `scipy` / `statsmodels` | Data processing and statistics |
| `torch` | Deep-learning forecasting module |
| `scikit-learn` | ML pipeline (Random Forest, Gradient Boosting, etc.) |
| `tqdm` | Progress bars during long simulations |
| `streamlit` | Legacy CLI prototype (optional) |
| `watchdog` | File-system watcher for hot-reload |

---

## Installation & Running

### Quickstart (recommended)

From the project root directory, run the provided shell script. It will automatically create a virtual environment, install all dependencies, and launch the app:

```bash
bash run.sh
```

Then open your browser at **http://localhost:8050**.

### Manual setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r src/requirements.txt

# 3. Launch the Dash web app
python src/app.py
```

Open **http://localhost:8050** in your browser.

### CLI mode (legacy)

A command-line interface is also available for headless execution:

```bash
python src/Main.py
```

---

## Main Functionalities

### 1. Graph Input

Load or generate the network on which simulations will run:

- **Generate synthetic graphs** — Erdős–Rényi (ER), Random Geometric Graph (RGG), square Lattice, and more
- **Load real-world graphs** — edge-list (`.txt`/`.csv`), GML, or DIMACS formats
- **Bundled datasets** — Facebook social network, GitHub collaboration network

### 2. Spreading Models

Eight epidemic / cascade models are implemented, all on a common interface:

| Model | Description |
|---|---|
| **Bootstrap Percolation (BP)** | A node activates when it has ≥ *k* infected neighbours (threshold model) |
| **SIR** | Susceptible → Infected → Recovered with transmission rate β and recovery rate γ |
| **SIS** | Like SIR but recovered nodes re-enter the susceptible pool |
| **H1 — OR-Hybrid** | Node infects if *either* the SIR channel fires OR the bootstrap threshold is met |
| **H2 — Sequential Hybrid** | SIR dynamics switch to Bootstrap mode once a fraction *f* of nodes is infected |
| **H3 — AND-Hybrid** | Both SIR and Bootstrap conditions must hold simultaneously to infect |
| **H4 — Weighted Hybrid** | Infection probability is a weighted combination of SIR and Bootstrap channels |
| **H5 / H6** | Additional hybrid variants exploring alternative coupling mechanisms |
| **WTM (Watts Threshold Model)** | Classic fractional-threshold model |

### 3. Seed Selection Strategies

Control how the initial infected set is chosen:

- **Random** — uniformly at random
- **High-degree** — prefer highest-degree nodes
- **Betweenness-centrality** — prefer bridge nodes
- **PageRank** — prefer influential nodes in the network flow

### 4. Parameter Sweeps & Analysis

- Sweep β, γ, threshold *k*, or seed fraction across a range and observe how metrics evolve
- Per-graph topology statistics (degree distribution, clustering, diameter, etc.)
- Comparative plots across graph types and model configurations
- Percolation threshold estimation and robustness scoring

### 5. Machine-Learning Forecasting

Predict epidemic outcomes from early-stage time-series observations:

- **Feature extraction** from graph topology and partial infection curves
- **Models**: Ridge Regression, Random Forest, Histogram Gradient Boosting, Logistic Regression
- **Targets**: final epidemic size (regression) and large-epidemic flag (classification)
- **Deep-learning module** (PyTorch) for sequence-to-outcome prediction
- Trained on synthetic simulation data; evaluated on held-out trials

### 6. Real-World Epidemic Validation

Two Jupyter notebooks validate the models against real outbreak data:

| Notebook | Dataset | Task |
|---|---|---|
| `ml/boarding_school_pred.ipynb` | 1978 UK boarding-school influenza outbreak | Fit SIR/hybrid models; compare with observed daily cases |
| `ml/flunet_pred.ipynb` | WHO FluNet multi-season surveillance data | Season-level forecasting; cross-season generalisation |

Additional datasets included: COVID-19 global confirmed cases, Ebola historical records, US CDC ILINet flu surveillance.

### 7. Interactive Dashboard

The Dash web app provides a single-page interface to:

- Configure graph source and model parameters via sidebar controls
- Visualise the network with animated cascade propagation
- Inspect per-round infection curves and summary metric cards
- Switch between models and re-run experiments without restarting

---

## Project Structure

```
Bootstrap/
├── run.sh                  # One-command launcher
└── src/
    ├── app.py              # Dash entry point
    ├── Main.py             # CLI entry point
    ├── requirements.txt
    ├── simulation/         # All spreading models (BP, SIR, SIS, H1–H6, WTM)
    ├── analysis/           # Parameter sweeps, graph statistics, feature extraction
    ├── input/              # Graph generators and file loaders
    ├── ui/                 # Dash layout, callbacks, sidebar, charts
    ├── visualization/      # Standalone visualisation utilities
    ├── ml/                 # ML pipeline and validation notebooks
    ├── data/               # Bundled real-world datasets
    └── results/            # Output figures and summaries
```

---

## Credits

Developed by **Álvaro Monclús**  
Bachelor's Thesis — *Network Contagion Lab: Simulation, Analysis and Prediction of Epidemic Spreading on Complex Networks*  
**ISEP — Institut Supérieur d'Électronique de Paris**
