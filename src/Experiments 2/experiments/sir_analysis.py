#!/usr/bin/env python3
"""
sir_analysis.py — SIR epidemic model on three network types.

Produces three PNG figures:
  1. sir_epidemic_size.png  — ρ_final vs β/μ, all networks, N=50 realisations
  2. sir_time_series.png    — I(t)/N for 3 representative β values per network
  3. sir_phase_diagram.png  — 2-D heatmap of ρ_final in (β, μ) for ER network

Usage:
    python src/experiments/sir_analysis.py

Runtime: ~3–8 min on a modern laptop (dominated by the 20×20 phase diagram).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR = Path("results/sir")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
N_NODES    = 600      # nodes per network
MU_DEFAULT = 0.1      # default recovery rate
SEED       = 42       # master RNG seed

plt.rcParams.update({"font.size": 11})


# ─────────────────────────────────────────────────────────────────────────────
# Network construction
# ─────────────────────────────────────────────────────────────────────────────

def build_networks(n: int = N_NODES, seed: int = SEED) -> dict[str, nx.Graph]:
    """
    Three networks chosen so epidemic thresholds fall in a convenient β range:
      · ER  avg-degree ≈ 6  → β_c/μ ≈ 1/6  ≈ 0.17
      · BA  m=3              → β_c/μ << ER due to hubs
      · Lattice 20×30       → β_c/μ ≈ 1/4  ≈ 0.25  (regular degree 4)
    """
    rng = np.random.default_rng(seed)
    s = lambda: int(rng.integers(1_000_000))

    G_er  = nx.erdos_renyi_graph(n, 6 / (n - 1), seed=s())
    G_ba  = nx.barabasi_albert_graph(n, 3, seed=s())

    # 20 × 30 = 600-node grid; interior nodes have degree 4
    G_lat = nx.grid_2d_graph(20, 30)
    G_lat = nx.convert_node_labels_to_integers(G_lat)

    return {"ER": G_er, "BA (scale-free)": G_ba, "Lattice": G_lat}


def _degree_stats(G: nx.Graph) -> tuple[float, float]:
    deg = np.array([d for _, d in G.degree()])
    return float(deg.mean()), float((deg ** 2).mean())


def _edge_arrays(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Return (src, dst) including both directions for an undirected graph."""
    u, v = zip(*G.edges())
    u, v = np.array(u, dtype=np.int32), np.array(v, dtype=np.int32)
    return np.concatenate([u, v]), np.concatenate([v, u])


# ─────────────────────────────────────────────────────────────────────────────
# SIR simulation — discrete time, fully vectorised
# ─────────────────────────────────────────────────────────────────────────────

def sir_run(
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    beta: float,
    mu: float,
    n_seeds: int = 6,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """
    Run one realisation of discrete-time SIR on a graph given as edge arrays.

    At each step:
      · Each I→S edge fires (infection) independently with probability β.
        A susceptible node becomes I if *any* adjacent infected node fires.
      · Each infected node recovers independently with probability μ.
    Both events use the state at the *start* of the step (synchronous update).

    Returns
    -------
    I_t  : array, I(t)/N for each time step until extinction
    rho  : float, final epidemic size R(∞)/N
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialise: small random seed set
    state = np.zeros(N, dtype=np.int8)          # 0=S, 1=I, 2=R
    state[rng.choice(N, n_seeds, replace=False)] = 1

    I_series: list[float] = []

    while True:
        infected    = state == 1
        susceptible = state == 0
        n_i = int(infected.sum())
        I_series.append(n_i / N)
        if n_i == 0:
            break

        new_state = state.copy()

        # --- Transmission ---
        active = infected[src] & susceptible[dst]
        idx    = np.where(active)[0]
        if idx.size:
            fires = rng.random(idx.size) < beta
            new_state[np.unique(dst[idx[fires]])] = 1

        # --- Recovery ---
        new_state[infected & (rng.random(N) < mu)] = 2

        state = new_state

    rho = float((state == 2).sum()) / N
    return np.array(I_series), rho


def _avg_rho(
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    beta: float,
    mu: float,
    n_runs: int,
    base_seed: int = 0,
) -> float:
    """Average ρ_final over n_runs independent realisations."""
    return float(np.mean([
        sir_run(src, dst, N, beta, mu,
                rng=np.random.default_rng(base_seed + i))[1]
        for i in range(n_runs)
    ]))


# ─────────────────────────────────────────────────────────────────────────────
# Epidemic threshold (heterogeneous mean-field)
# ─────────────────────────────────────────────────────────────────────────────

def epidemic_threshold(G: nx.Graph, mu: float) -> float:
    """
    β_c = μ · ⟨k⟩ / ⟨k²⟩

    Reduces to μ/⟨k⟩ for homogeneous (Poisson / regular) networks.
    Gives a lower β_c for heavy-tailed (BA) networks because ⟨k²⟩ >> ⟨k⟩².
    """
    k1, k2 = _degree_stats(G)
    return mu * k1 / k2


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Final epidemic size curve
# ─────────────────────────────────────────────────────────────────────────────

def fig_epidemic_size(
    graphs: dict[str, nx.Graph],
    mu: float = MU_DEFAULT,
    n_betas: int = 30,
    n_runs: int = 50,
) -> None:
    print("Figure 1 — final epidemic size curve …")

    # Cover 0 → 0.12 in β; covers threshold for all three networks
    betas  = np.linspace(0.0, 0.12, n_betas + 1)[1:]   # skip exact 0
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for (name, G), color in zip(graphs.items(), colors):
        src, dst = _edge_arrays(G)
        N  = G.number_of_nodes()
        bc = epidemic_threshold(G, mu)
        k1, _ = _degree_stats(G)
        print(f"  {name}: ⟨k⟩={k1:.1f}, β_c={bc:.4f}, β_c/μ={bc/mu:.3f}")

        rhos = []
        for b in betas:
            rhos.append(_avg_rho(src, dst, N, b, mu, n_runs))
            sys.stdout.write(".")
            sys.stdout.flush()
        print()

        ax.plot(betas / mu, rhos, "o-", color=color,
                markersize=4, lw=1.8, label=name)

        # Threshold marker — only draw if within plot range
        if bc / mu < betas[-1] / mu * 1.05:
            ax.axvline(bc / mu, color=color, ls="--", lw=1.2, alpha=0.75)
            ypos = 0.55 + list(graphs).index(name) * 0.15
            ax.text(bc / mu + 0.015 * betas[-1] / mu, ypos,
                    f"β_c ({name.split()[0]})",
                    color=color, fontsize=8, va="center")

    ax.set_xlabel("β / μ", fontsize=13)
    ax.set_ylabel("Final epidemic size  ρ", fontsize=13)
    ax.set_title(
        f"SIR Final Epidemic Size  (μ={mu}, N={N_NODES}, {n_runs} realisations)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, betas[-1] / mu)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "sir_epidemic_size.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Time series I(t)/N
# ─────────────────────────────────────────────────────────────────────────────

def fig_time_series(
    graphs: dict[str, nx.Graph],
    mu: float = MU_DEFAULT,
    avg_runs: int = 15,
) -> None:
    print("Figure 2 — time series …")

    colors  = ["#1565C0", "#E65100", "#1B5E20"]
    lstyles = ["-", "--", ":"]
    beta_factors = [0.5, 1.0, 2.0]
    factor_labels = ["0.5 β_c  (sub-critical)",
                     "1.0 β_c  (critical)",
                     "2.0 β_c  (super-critical)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (name, G) in zip(axes, graphs.items()):
        src, dst = _edge_arrays(G)
        N  = G.number_of_nodes()
        bc = epidemic_threshold(G, mu)

        for factor, lbl, color, ls in zip(beta_factors, factor_labels, colors, lstyles):
            beta = factor * bc
            series_list = []
            for i in range(avg_runs):
                I_t, _ = sir_run(src, dst, N, beta, mu, n_seeds=6,
                                 rng=np.random.default_rng(300 + i))
                series_list.append(I_t)

            # Pad to common length and average
            T = max(len(s) for s in series_list)
            mat = np.zeros((avg_runs, T))
            for i, s in enumerate(series_list):
                mat[i, :len(s)] = s
            I_mean = mat.mean(axis=0)

            ax.plot(I_mean, color=color, ls=ls, lw=2.0,
                    label=f"β={beta:.4f}\n{lbl}")

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (steps)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("I(t) / N", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(bottom=-0.01)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"SIR Infected Fraction  I(t)/N  (μ={mu}, avg of {avg_runs} runs)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out = OUT_DIR / "sir_time_series.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase diagram (β, μ) for ER
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_diagram(
    G: nx.Graph,
    n_grid: int = 30,
    n_runs: int = 30,
) -> None:
    print(f"Figure 3 — phase diagram ({n_grid}×{n_grid} grid, {n_runs} runs each) …")
    print("  This takes several minutes — progress shown per row.")

    src, dst = _edge_arrays(G)
    N = G.number_of_nodes()

    betas = np.linspace(0.005, 0.12,  n_grid)
    mus   = np.linspace(0.02,  0.40,  n_grid)

    rho_grid = np.zeros((n_grid, n_grid))
    for i, mu in enumerate(mus):
        for j, beta in enumerate(betas):
            rho_grid[i, j] = _avg_rho(src, dst, N, beta, mu, n_runs,
                                       base_seed=i * n_grid + j)
        print(f"  row {i + 1:2d}/{n_grid}  μ={mu:.3f}  "
              f"ρ_max={rho_grid[i].max():.2f}")

    # Theoretical threshold curve: β_c = μ · ⟨k⟩/⟨k²⟩
    k1, k2 = _degree_stats(G)
    bc_line = mus * (k1 / k2)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(betas, mus, rho_grid,
                       cmap="plasma", shading="auto", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final epidemic size  ρ", fontsize=12)

    # Overlay analytical threshold where it falls inside the plot
    in_range = bc_line <= betas[-1]
    ax.plot(bc_line[in_range], mus[in_range], "w--", lw=2.0,
            label=f"β_c = μ·⟨k⟩/⟨k²⟩  (⟨k⟩={k1:.1f})")

    ax.set_xlabel("β  (transmission rate)", fontsize=13)
    ax.set_ylabel("μ  (recovery rate)", fontsize=13)
    ax.set_title(
        f"SIR Phase Diagram — Erdős–Rényi  (N={N}, {n_runs} runs/point)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()

    out = OUT_DIR / "sir_phase_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Building networks  (N={N_NODES}, seed={SEED}) …")
    graphs = build_networks()

    for name, G in graphs.items():
        k1, k2 = _degree_stats(G)
        print(f"  {name:20s}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
              f"⟨k⟩={k1:.1f}, ⟨k²⟩={k2:.1f}, "
              f"β_c(μ=0.1)={epidemic_threshold(G, MU_DEFAULT):.4f}")

    print()
    fig_epidemic_size(graphs)
    print()
    fig_time_series(graphs)
    print()
    fig_phase_diagram(graphs["ER"])

    print(f"\nAll outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
