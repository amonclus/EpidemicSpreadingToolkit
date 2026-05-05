#!/usr/bin/env python3
"""
Generate labelled trial simulations for testing the ML virality predictor.

For each trial the script:
  - Picks a random spreading model, network type, and parameter value.
  - Runs a full simulation and records I(t)/N.
  - Saves the first T_OBS timesteps (10–20, chosen randomly) as a CSV.
  - Saves the graph used to a paired .gml file.
  - Embeds the true final outbreak size and model name as comment lines
    at the top of the CSV so predictions can be compared to ground truth.

Output directory:  data/ml_test/
  trial_0001_SIR_ER.csv       — observed I(t)/N series + metadata comments
  trial_0001_SIR_ER.gml       — the graph used for that trial

Usage:
  python ml_scripts/generate_test_trials.py [--n-trials 50] [--seed 0]
"""
import argparse
import math
import random
import sys
import traceback
from pathlib import Path

import networkx as nx
import numpy as np

# ── Make src/ importable regardless of working directory ──────────────────────
_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

from simulation.sir import SIRModel
from simulation.sis import SISModel
from simulation.bootstrap import BootstrapPercolation
from simulation.wtm import WTMModel
from simulation.H1 import H1Model
from simulation.H2 import H2Model
from simulation.H3 import H3Model
from simulation.H4 import H4Model
from simulation.H5 import H5Model
from simulation.H6 import H6Model
from input.graph_generator import (
    generate_er_graph,
    generate_random_geometric_graph,
    generate_lattice_graph,
)

# ── Constants ─────────────────────────────────────────────────────────────────
N         = 500       # nodes for synthetic graphs
LATTICE_SZ = 22       # lattice is LATTICE_SZ × LATTICE_SZ (≈ 484 nodes)
MU        = 0.14      # recovery rate (gamma) — fixed across all SIR-like models
K         = 3         # bootstrap threshold
PHI_FIXED = 0.15      # WTM/H4/H5 phi when not being swept
N_SEEDS   = 5         # initial seed count for non-BP/WTM models
T_OBS_MIN = 10        # minimum observation window
T_OBS_MAX = 20        # maximum observation window

OUTPUT_DIR   = _SRC / "data" / "ml_test"
GRAPH_DATA   = _SRC / "data"

ALL_MODELS   = ["SIR", "SIS", "BP", "WTM", "H1", "H2", "H3", "H4", "H5", "H6"]
SYNTH_NETS   = ["ER", "RGG", "Lattice"]
REAL_NETS    = ["Facebook", "GitHub"]
ALL_NETS     = SYNTH_NETS + REAL_NETS

GITHUB_SAMPLE = 500   # BFS-sampled subgraph size for GitHub

# Parameter ranges used to draw a random value for each trial
PARAM_RANGES = {
    "SIR": ("beta",  (0.01,  0.12)),
    "SIS": ("beta",  (0.01,  0.12)),
    "BP":  ("rho_0", (0.02,  0.25)),
    "WTM": ("phi",   (0.05,  0.40)),
    "H1":  ("beta",  (0.01,  0.12)),
    "H2":  ("f",     (0.05,  0.45)),
    "H3":  ("beta",  (0.01,  0.12)),
    "H4":  ("beta",  (0.01,  0.12)),
    "H5":  ("f",     (0.05,  0.45)),
    "H6":  ("phi",   (0.01,  0.12)),
}


# ── Graph factories ───────────────────────────────────────────────────────────

def _make_synth_graph(network_type: str, rng: random.Random) -> nx.Graph:
    if network_type == "ER":
        p = 6 / (N - 1)
        return generate_er_graph(N, p)
    if network_type == "RGG":
        r = math.sqrt(6 / (N * math.pi))
        return generate_random_geometric_graph(N, r)
    if network_type == "Lattice":
        G = generate_lattice_graph(LATTICE_SZ)
        return nx.convert_node_labels_to_integers(G)
    raise ValueError(f"Unknown synthetic network type: {network_type}")


def _load_real_graphs() -> dict[str, nx.Graph]:
    graphs: dict[str, nx.Graph] = {}

    fb_path = GRAPH_DATA / "facebook_combined.txt"
    if fb_path.exists():
        G = nx.read_edgelist(str(fb_path), nodetype=int)
        lcc = max(nx.connected_components(G), key=len)
        graphs["Facebook"] = nx.convert_node_labels_to_integers(
            G.subgraph(lcc).copy()
        )
        print(f"  Facebook: {graphs['Facebook'].number_of_nodes()} nodes, "
              f"{graphs['Facebook'].number_of_edges()} edges")
    else:
        print(f"  Facebook graph not found at {fb_path} — skipping")

    gh_path = GRAPH_DATA / "musae_git_edges.csv"
    if gh_path.exists():
        G = nx.read_edgelist(str(gh_path), nodetype=int, delimiter=",")
        lcc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc)
        start = max(G_lcc.degree(), key=lambda x: x[1])[0]
        sampled = list(nx.bfs_tree(G_lcc, start).nodes())[:GITHUB_SAMPLE]
        graphs["GitHub"] = nx.convert_node_labels_to_integers(
            G_lcc.subgraph(sampled).copy()
        )
        print(f"  GitHub:   {graphs['GitHub'].number_of_nodes()} nodes, "
              f"{graphs['GitHub'].number_of_edges()} edges")
    else:
        print(f"  GitHub graph not found at {gh_path} — skipping")

    return graphs


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_model(model_name: str, G: nx.Graph, param_val: float):
    if model_name == "SIR":
        return SIRModel(G, beta=param_val, gamma=MU)
    if model_name == "SIS":
        return SISModel(G, beta=param_val, gamma=MU)
    if model_name == "BP":
        return BootstrapPercolation(G, threshold=K)
    if model_name == "WTM":
        return WTMModel(G, phi=param_val)
    if model_name == "H1":
        return H1Model(G, threshold=K, beta=param_val, gamma=MU)
    if model_name == "H2":
        return H2Model(G, threshold=K, beta=0.05, gamma=MU, switch_fraction=param_val)
    if model_name == "H3":
        return H3Model(G, beta=param_val, gamma=MU)
    if model_name == "H4":
        return H4Model(G, phi=PHI_FIXED, beta=param_val, gamma=MU)
    if model_name == "H5":
        return H5Model(G, phi=PHI_FIXED, beta=0.05, gamma=MU, switch_fraction=param_val)
    if model_name == "H6":
        return H6Model(G, phi=param_val, gamma=MU)
    raise ValueError(f"Unknown model: {model_name}")


def _make_seeds(G: nx.Graph, model_name: str, param_val: float) -> set:
    nodes = list(G.nodes())
    n = len(nodes)
    if model_name == "BP":
        size = max(1, round(param_val * n))
        return set(random.sample(nodes, min(size, n)))
    if model_name == "WTM":
        size = max(1, round(0.05 * n))
        return set(random.sample(nodes, min(size, n)))
    return set(random.sample(nodes, min(N_SEEDS, n)))


# ── I(t)/N series reconstruction ─────────────────────────────────────────────

def _build_i_series(model_name: str, result, act_seq: list, n: int) -> list[float]:
    """Return the full I(t)/N time series as a plain list of floats."""
    if model_name in ("SIS", "H4"):
        return [x / n for x in result.infected_series]

    if model_name in ("BP", "WTM"):
        cumul, raw = 0, []
        for step_set in act_seq:
            cumul += len(step_set)
            raw.append(min(1.0, cumul / n))
        return raw

    # SIR-type: sparse (new_infected, new_recovered) per active round
    if not act_seq:
        return [0.0]
    I = len(act_seq[0][0])
    raw = [I / n]
    for new_inf, new_rec in act_seq[1:]:
        I = max(0, I + len(new_inf) - len(new_rec))
        raw.append(I / n)
    return raw


def _rho_final(model_name: str, result, n: int) -> float:
    if model_name == "SIR":
        return result.epidemic_fraction
    return result.cascade_fraction


# ── Output writers ────────────────────────────────────────────────────────────

def _write_csv(path: Path, series: list[float], t_obs: int) -> None:
    """Write the observed I(t)/N series as a plain two-column CSV."""
    lines = ["step,I_over_N"]
    for step, val in enumerate(series[:t_obs]):
        lines.append(f"{step},{val:.8f}")
    path.write_text("\n".join(lines) + "\n")


def _write_graph(path: Path, G: nx.Graph) -> None:
    nx.write_gml(G, str(path))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_trials: int = 50, seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading real-world graphs...")
    real_graphs = _load_real_graphs()
    available_nets = SYNTH_NETS + [n for n in REAL_NETS if n in real_graphs]

    # Distribute trials evenly across models so every model appears at least once
    model_queue = (ALL_MODELS * (n_trials // len(ALL_MODELS) + 1))[:n_trials]
    random.shuffle(model_queue)

    ok = 0
    failed = 0

    for trial_idx, model_name in enumerate(model_queue, start=1):
        network_type = random.choice(available_nets)
        t_obs = random.randint(T_OBS_MIN, T_OBS_MAX)
        param_name, (lo, hi) = PARAM_RANGES[model_name]
        param_val = random.uniform(lo, hi)

        try:
            # Build graph
            if network_type in real_graphs:
                G = real_graphs[network_type]
            else:
                G = _make_synth_graph(network_type, random)

            n = G.number_of_nodes()

            # Run simulation
            model = _make_model(model_name, G, param_val)
            seeds = _make_seeds(G, model_name, param_val)
            result, act_seq = model.run(seeds, record_sequence=True)

            i_series = _build_i_series(model_name, result, act_seq, n)
            rho      = _rho_final(model_name, result, n)

            if len(i_series) < t_obs:
                t_obs = max(2, len(i_series))

            # Encode model, network, and true rho_final directly in the filename
            rho_str    = f"{rho:.3f}".replace(".", "p")
            stem       = f"trial_{trial_idx:04d}_{model_name}_{network_type}_rho{rho_str}"
            csv_path   = OUTPUT_DIR / f"{stem}.csv"
            graph_path = OUTPUT_DIR / f"{stem}.gml"

            _write_csv(csv_path, i_series, t_obs)
            _write_graph(graph_path, G)

            ok += 1
            print(f"  [{trial_idx:4d}/{n_trials}] {stem}  "
                  f"t_obs={t_obs}  |I|={len(i_series)}")

        except Exception as exc:
            failed += 1
            print(f"  [{trial_idx:4d}/{n_trials}] FAILED {stem}: {exc}")
            if "--verbose" in sys.argv:
                traceback.print_exc()

    print(f"\nDone. {ok} trials saved to {OUTPUT_DIR}  ({failed} failed)")
    if ok:
        print("Each trial has a paired .csv and .gml file.")
        print("The true outbreak size is encoded in the filename as rho<value>.")
        print("Example: trial_0001_SIR_ER_rho0p423.csv  →  ρ_final = 0.423")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labelled ML test trials.")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of trials to generate (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global random seed (default: 0)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full tracebacks on failure")
    args = parser.parse_args()
    main(n_trials=args.n_trials, seed=args.seed)
