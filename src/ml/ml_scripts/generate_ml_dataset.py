#!/usr/bin/env python3
"""
Generate an ML dataset by running all 10 epidemic/cascade models over a grid of
parameters and network types.

Outputs (in ml_data/):
  ml_dataset.csv       — one row per realisation (no I_series column)
  ml_I_series.npy      — float32 array of shape (n_samples, T_MAX); rows match CSV
  ml_dataset_summary.txt

Usage:
  python generate_ml_dataset.py
"""
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Make project importable from any working directory ────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.graph_features import extract_graph_features

# ── Fixed parameters ──────────────────────────────────────────────────────────
MU          = 0.14   # recovery rate (gamma)
K           = 3      # bootstrap threshold
PHI_FIXED   = 0.15   # fixed WTM/H4/H5 phi when not being swept
RHO_0_FIXED = 0.05   # initial seed fraction (WTM seeding)
T_MAX       = 300    # series length (pad / truncate to this)
N_SEEDS     = 5      # fixed seed count for non-BP/WTM models
N_REAL      = 50     # realisations per configuration
N           = 1000   # nodes for ER and RGG
LATTICE_SZ  = 32     # lattice is LATTICE_SZ × LATTICE_SZ (= 1024 nodes)

OUTPUT_DIR      = Path("../ml_data")
GRAPH_DATA_DIR  = Path("../../data")
GITHUB_SAMPLE   = 2000   # BFS-sampled node count for GitHub (full graph: 37 700 nodes)

MODEL_IDS = {
    "SIR": 0, "SIS": 1, "BP": 2, "WTM": 3,
    "H1": 4,  "H2": 5,  "H3": 6, "H4": 7, "H5": 8, "H6": 9,
}
NETWORKS = ["ER", "RGG", "Lattice", "Facebook", "GitHub"]

# Primary parameter sweep per model (param_name, 20 values)
PARAM_SWEEPS = {
    "SIR": ("beta",  np.logspace(np.log10(0.001), np.log10(0.15), 20)),
    "SIS": ("beta",  np.logspace(np.log10(0.001), np.log10(0.15), 20)),
    "BP":  ("rho_0", np.linspace(0.01, 0.30, 20)),
    "WTM": ("phi",   np.linspace(0.05, 0.50, 20)),
    "H1":  ("beta",  np.logspace(np.log10(0.001), np.log10(0.15), 20)),
    "H2":  ("f",     np.linspace(0.02, 0.50, 20)),
    "H3":  ("beta",  np.logspace(np.log10(0.001), np.log10(0.15), 20)),
    "H4":  ("beta",  np.logspace(np.log10(0.001), np.log10(0.15), 20)),
    "H5":  ("f",     np.linspace(0.02, 0.50, 20)),
    # H6 uses phi as its spread-rate parameter (see H6 docstring)
    "H6":  ("phi",   np.logspace(np.log10(0.001), np.log10(0.15), 20)),
}


# ── Real-world graph loading ──────────────────────────────────────────────────

def load_real_graphs() -> dict:
    """
    Load Facebook and GitHub once.  Returns {name: nx.Graph}.

    Facebook: largest connected component of the ego-network (~4 039 nodes).
    GitHub:   BFS-sampled subgraph of GITHUB_SAMPLE nodes starting from the
              highest-degree node, preserving the dense core of the network.
    """
    graphs = {}

    fb_path = GRAPH_DATA_DIR / "facebook_combined.txt"
    G_fb = nx.read_edgelist(str(fb_path), nodetype=int)
    lcc_fb = max(nx.connected_components(G_fb), key=len)
    graphs["Facebook"] = nx.convert_node_labels_to_integers(
        G_fb.subgraph(lcc_fb).copy()
    )
    print(f"  Facebook loaded: {graphs['Facebook'].number_of_nodes()} nodes, "
          f"{graphs['Facebook'].number_of_edges()} edges")

    gh_path = GRAPH_DATA_DIR / "musae_git_edges.csv"
    G_gh = nx.read_edgelist(str(gh_path), nodetype=int, delimiter=",")
    lcc_gh = max(nx.connected_components(G_gh), key=len)
    G_gh_lcc = G_gh.subgraph(lcc_gh)
    start = max(G_gh_lcc.degree(), key=lambda x: x[1])[0]
    sampled = list(nx.bfs_tree(G_gh_lcc, start).nodes())[:GITHUB_SAMPLE]
    graphs["GitHub"] = nx.convert_node_labels_to_integers(
        G_gh_lcc.subgraph(sampled).copy()
    )
    print(f"  GitHub loaded:   {graphs['GitHub'].number_of_nodes()} nodes, "
          f"{graphs['GitHub'].number_of_edges()} edges")

    return graphs


# ── Graph factories ───────────────────────────────────────────────────────────

def make_graph(network_type: str) -> nx.Graph:
    """Generate a fresh synthetic graph with roughly avg_degree ≈ 6."""
    if network_type == "ER":
        p = 6 / (N - 1)
        return generate_er_graph(N, p)
    if network_type == "RGG":
        # For a unit square: avg_degree ≈ N·π·r² → r = sqrt(6 / (N·π))
        r = math.sqrt(6 / (N * math.pi))
        return generate_random_geometric_graph(N, r)
    if network_type == "Lattice":
        G = generate_lattice_graph(LATTICE_SZ)
        return nx.convert_node_labels_to_integers(G)
    raise ValueError(f"Unknown network type: {network_type}")


# ── Model factories ───────────────────────────────────────────────────────────

def make_model(model_name: str, G: nx.Graph, param_val: float):
    """Instantiate the named model with its swept parameter set."""
    if model_name == "SIR":
        return SIRModel(G, beta=param_val, gamma=MU)
    if model_name == "SIS":
        return SISModel(G, beta=param_val, gamma=MU)
    if model_name == "BP":
        # param_val is rho_0 (seed fraction); threshold K is fixed
        return BootstrapPercolation(G, threshold=K)
    if model_name == "WTM":
        # param_val is phi; seed fraction fixed at RHO_0_FIXED
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


# ── Seed selection ────────────────────────────────────────────────────────────

def select_seeds(G: nx.Graph, model_name: str, param_val: float, rng: np.random.Generator) -> set:
    """Return initial seed nodes for one run."""
    nodes = list(G.nodes())
    n = len(nodes)
    if model_name == "BP":
        size = max(1, round(param_val * n))
        chosen = rng.choice(n, min(size, n), replace=False)
        return {nodes[i] for i in chosen}
    if model_name == "WTM":
        size = max(1, round(RHO_0_FIXED * n))
        chosen = rng.choice(n, min(size, n), replace=False)
        return {nodes[i] for i in chosen}
    # All other models: fixed count of N_SEEDS random nodes
    chosen = rng.choice(n, min(N_SEEDS, n), replace=False)
    return {nodes[i] for i in chosen}


# ── I(t)/N series reconstruction ─────────────────────────────────────────────

def build_i_series(
    model_name: str,
    result,
    activation_seq: list,
    n: int,
) -> np.ndarray:
    """
    Reconstruct a dense I(t)/N array of length T_MAX.

    SIS and H4 record infected_series natively (one entry per round).
    BP and WTM activation_sequence contains [seed_set, new_inf_1, ...];
      cumulative sum gives I(t) (no recovery).
    All other SIR-type models: activation_sequence[0] = (initial_infected, {});
      subsequent entries = (newly_infected, newly_recovered) per active round
      (sparse: rounds with no events are omitted, so time is slightly compressed).
    Pads to T_MAX with the last value; truncates if longer.
    """
    if model_name in ("SIS", "H4"):
        raw = [x / n for x in result.infected_series]

    elif model_name in ("BP", "WTM"):
        cumul = 0
        raw = []
        for step_set in activation_seq:
            cumul += len(step_set)
            raw.append(min(1.0, cumul / n))

    else:
        # SIR-type: reconstruct from sparse (new_inf, new_rec) tuples
        if activation_seq:
            I = len(activation_seq[0][0])
            raw = [I / n]
            for new_inf, new_rec in activation_seq[1:]:
                I = max(0, I + len(new_inf) - len(new_rec))
                raw.append(I / n)
        else:
            raw = [0.0]

    # Pad or truncate to exactly T_MAX
    pad = raw[-1] if raw else 0.0
    if len(raw) < T_MAX:
        raw = raw + [pad] * (T_MAX - len(raw))
    return np.array(raw[:T_MAX], dtype=np.float32)


def get_rho_final(model_name: str, result, n: int) -> float:
    """Extract the scalar 'epidemic size' from the result object."""
    if model_name == "SIR":
        return result.epidemic_fraction          # ever-infected fraction (= R/N at end)
    return result.cascade_fraction               # peak for SIS/H4; final for cascade models


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows: list[dict] = []
    i_series_list: list[np.ndarray] = []
    failures: list[dict] = []

    model_names = list(MODEL_IDS.keys())
    n_params = 20
    total_runs = len(model_names) * len(NETWORKS) * n_params * N_REAL  # 50 000

    print("Loading real-world graphs...")
    real_graphs = load_real_graphs()

    print("Pre-computing real-world graph features...")
    real_graph_features = {name: extract_graph_features(G) for name, G in real_graphs.items()}
    for name, feats in real_graph_features.items():
        print(f"  {name}: density={feats['graph_density']:.4f}  "
              f"mean_deg={feats['graph_mean_degree']:.2f}  "
              f"clustering={feats['graph_clustering']:.3f}")

    pbar = tqdm(total=total_runs, desc="Generating", unit="run", dynamic_ncols=True)

    for model_name in model_names:
        model_id = MODEL_IDS[model_name]
        param_name, param_values = PARAM_SWEEPS[model_name]

        for net_idx, network_type in enumerate(NETWORKS):

            for param_idx, param_val in enumerate(param_values):
                if network_type in real_graphs:
                    # Real-world graphs are fixed — no regeneration per param.
                    G = real_graphs[network_type]
                    graph_feats = real_graph_features[network_type]
                else:
                    # Fresh synthetic graph per (model, network, param) so the
                    # classifier cannot learn topology fingerprints.
                    graph_seed = model_id * 10_000 + net_idx * 100 + param_idx
                    np.random.seed(graph_seed)
                    random.seed(graph_seed)
                    G = make_graph(network_type)
                    graph_feats = extract_graph_features(G)
                n = G.number_of_nodes()
                model = make_model(model_name, G, float(param_val))

                for real_idx in range(N_REAL):
                    run_seed = (
                        model_id * 10_000
                        + net_idx * 1_000
                        + param_idx * 50
                        + real_idx
                    )
                    # Seed both Python random and numpy
                    random.seed(run_seed)
                    np.random.seed(run_seed)
                    rng = np.random.default_rng(run_seed)

                    try:
                        seeds = select_seeds(G, model_name, float(param_val), rng)
                        result, act_seq = model.run(seeds, record_sequence=True)

                        rho_final = get_rho_final(model_name, result, n)
                        i_arr = build_i_series(model_name, result, act_seq, n)
                        peak_I = float(np.max(i_arr))
                        t_peak = int(np.argmax(i_arr))

                        rows.append(
                            dict(
                                model_name=model_name,
                                model_id=model_id,
                                network_type=network_type,
                                primary_param_name=param_name,
                                primary_param_value=float(param_val),
                                rho_final=float(rho_final),
                                peak_I=peak_I,
                                t_peak=t_peak,
                                is_supercritical=bool(rho_final > 0.05),
                                **graph_feats,
                            )
                        )
                        i_series_list.append(i_arr)

                    except Exception as exc:
                        failures.append(
                            dict(
                                model=model_name,
                                network=network_type,
                                param_val=float(param_val),
                                realisation=real_idx,
                                error=repr(exc),
                                traceback=traceback.format_exc(),
                            )
                        )
                        rows.append(
                            dict(
                                model_name=model_name,
                                model_id=model_id,
                                network_type=network_type,
                                primary_param_name=param_name,
                                primary_param_value=float(param_val),
                                rho_final=float("nan"),
                                peak_I=float("nan"),
                                t_peak=-1,
                                is_supercritical=False,
                                **graph_feats,
                            )
                        )
                        i_series_list.append(np.full(T_MAX, np.nan, dtype=np.float32))

                    pbar.update(1)

    pbar.close()

    # ── Persist outputs ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "ml_dataset.csv"
    df.to_csv(csv_path, index=False)

    I_arr = np.stack(i_series_list)          # (n_samples, T_MAX)
    npy_path = OUTPUT_DIR / "ml_I_series.npy"
    np.save(npy_path, I_arr)

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = df.dropna(subset=["rho_final"])
    elapsed = time.time() - t_start

    lines: list[str] = [
        "=" * 62,
        "ML Dataset Summary",
        "=" * 62,
        f"Total samples            : {len(df):,}",
        f"Failed runs              : {len(failures)}",
        f"I_series array shape     : {I_arr.shape}",
        "",
        "Samples per model:",
    ]
    for name in MODEL_IDS:
        lines.append(f"  {name:<5}: {(df['model_name'] == name).sum():>6,}")
    lines += ["", "Samples per network:"]
    for net in NETWORKS:
        lines.append(f"  {net:<8}: {(df['network_type'] == net).sum():>6,}")
    lines += ["", "Mean rho_final per model:"]
    for name in MODEL_IDS:
        m = valid[valid["model_name"] == name]["rho_final"].mean()
        lines.append(f"  {name:<5}: {m:.4f}")
    lines += ["", "Fraction supercritical per model:"]
    for name in MODEL_IDS:
        fsc = valid[valid["model_name"] == name]["is_supercritical"].mean()
        lines.append(f"  {name:<5}: {fsc:.4f}")
    if failures:
        lines += ["", f"Failed runs ({len(failures)} total):"]
        for f in failures[:20]:
            lines.append(f"  {f['model']} | {f['network']} | param={f['param_val']:.4f}"
                         f" | real={f['realisation']} | {f['error']}")
        if len(failures) > 20:
            lines.append(f"  … and {len(failures) - 20} more (see logs)")
    lines += [
        "",
        f"Total runtime            : {elapsed:.1f} s  ({elapsed / 60:.1f} min)",
        "=" * 62,
    ]

    summary = "\n".join(lines)
    print(summary)
    (OUTPUT_DIR / "ml_dataset_summary.txt").write_text(summary)

    print(f"\nSaved: {csv_path}  ({len(df):,} rows)")
    print(f"Saved: {npy_path}  (shape {I_arr.shape})")
    print(f"Saved: {OUTPUT_DIR / 'ml_dataset_summary.txt'}")


if __name__ == "__main__":
    main()
