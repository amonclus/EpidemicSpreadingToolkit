#!/usr/bin/env python3
"""
bootstrap_analysis.py — Bootstrap percolation on three network types.

A node becomes infected when it has ≥ k already-infected neighbours.
Starting from a random seed fraction ρ_0, the cascade propagates
synchronously until no new nodes can be activated.

Produces three PNG figures:
  1. bootstrap_rho_vs_rho0.png  — ρ_final vs ρ_0 for k=2,3,4, all networks
  2. bootstrap_rho_vs_k.png     — ρ_final vs k (1–6), ρ_0=0.05, all networks
  3. bootstrap_phase_diagram.png — 2-D heatmap (ρ_0, k) for ER only

Usage:
    python src/experiments/bootstrap_analysis.py

Runtime: ~2–5 min depending on hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import networkx as nx

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("results/bootstrap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
N_NODES = 600
SEED    = 42

plt.rcParams.update({"font.size": 11})


# ─────────────────────────────────────────────────────────────────────────────
# Network construction
# ─────────────────────────────────────────────────────────────────────────────

def build_networks(n: int = N_NODES, seed: int = SEED) -> dict[str, nx.Graph]:
    """
    Three networks with avg-degree spread to reveal different cascade regimes:
      · ER       ⟨k⟩≈8  — above the k=2,3 bootstrap threshold for moderate ρ_0
      · BA       m=4     — hubs lower the effective threshold vs ER
      · Lattice  20×30   — regular degree 4; resistant to k≥3 cascades
    """
    rng = np.random.default_rng(seed)
    s = lambda: int(rng.integers(1_000_000))

    G_er  = nx.erdos_renyi_graph(n, 8 / (n - 1), seed=s())
    G_ba  = nx.barabasi_albert_graph(n, 4, seed=s())

    G_lat = nx.grid_2d_graph(20, 30)          # 600 nodes, degree 4 interior
    G_lat = nx.convert_node_labels_to_integers(G_lat)

    return {"ER": G_er, "BA (scale-free)": G_ba, "Lattice": G_lat}


def _edge_arrays(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Return (src, dst) with both directions for an undirected graph."""
    u, v = zip(*G.edges())
    u, v = np.array(u, dtype=np.int32), np.array(v, dtype=np.int32)
    return np.concatenate([u, v]), np.concatenate([v, u])


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap percolation — vectorised
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_run(
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    k: int,
    seed_indices: np.ndarray,
) -> float:
    """
    One deterministic bootstrap percolation run.

    Each step:
      · Count infected neighbours for every node via edge sums.
      · Activate all susceptible nodes whose count ≥ k simultaneously.
    Repeats until no new activations (guaranteed to terminate in ≤ N rounds).

    Returns ρ_final = (total infected) / N.
    """
    infected = np.zeros(N, dtype=bool)
    infected[seed_indices] = True

    while True:
        # Vectorised infected-neighbour count
        infected_nbr = np.zeros(N, dtype=np.int32)
        np.add.at(infected_nbr, dst, infected[src].astype(np.int32))

        newly = ~infected & (infected_nbr >= k)
        if not newly.any():
            break
        infected |= newly

    return float(infected.sum()) / N


def avg_rho(
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    k: int,
    rho_0: float,
    n_runs: int,
    base_seed: int = 0,
) -> float:
    """Average ρ_final over n_runs realisations with random seed sets."""
    n_seeds = max(1, round(rho_0 * N))
    rhos = []
    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i)
        seeds = rng.choice(N, n_seeds, replace=False)
        rhos.append(bootstrap_run(src, dst, N, k, seeds))
    return float(np.mean(rhos))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ρ_final vs ρ_0  (k = 2, 3, 4;  all networks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rho_vs_rho0(
    graphs: dict[str, nx.Graph],
    k_values: list[int] = [2, 3, 4],
    n_rho0: int = 25,
    n_runs: int = 50,
) -> None:
    print("Figure 1 — ρ_final vs ρ_0 …")

    rho0_arr = np.linspace(0.0, 0.35, n_rho0 + 1)[1:]   # skip exact 0

    # Colour per network, line-style per k
    net_colors  = {"ER": "#2196F3", "BA (scale-free)": "#FF5722", "Lattice": "#4CAF50"}
    k_styles    = {2: "-", 3: "--", 4: ":"}
    k_markers   = {2: "o", 3: "s", 4: "^"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Draw a faint diagonal (ρ_final = ρ_0, i.e. no cascade)
    ax.plot([0, rho0_arr[-1]], [0, rho0_arr[-1]], color="gray",
            lw=0.8, ls="-.", zorder=0, label="ρ_final = ρ_0  (no cascade)")

    for name, G in graphs.items():
        src, dst = _edge_arrays(G)
        N  = G.number_of_nodes()
        color = net_colors[name]

        for k in k_values:
            rhos = []
            for rho_0 in rho0_arr:
                rhos.append(avg_rho(src, dst, N, k, rho_0, n_runs))
                sys.stdout.write(".")
                sys.stdout.flush()
            print(f"  {name} k={k}")

            ax.plot(
                rho0_arr, rhos,
                color=color,
                ls=k_styles[k], marker=k_markers[k], markersize=4, lw=1.8,
                label=f"{name}, k={k}",
            )

    ax.set_xlabel("Initial seed fraction  ρ₀", fontsize=13)
    ax.set_ylabel("Final infected fraction  ρ_final", fontsize=13)
    ax.set_title(
        f"Bootstrap Percolation — ρ_final vs ρ₀\n"
        f"(N={N_NODES}, {n_runs} realisations per point)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, rho0_arr[-1])
    ax.set_ylim(-0.02, 1.05)

    # Compact legend: two-column layout
    ax.legend(fontsize=8, ncol=3, loc="upper left",
              framealpha=0.9, columnspacing=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "bootstrap_rho_vs_rho0.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — ρ_final vs k  (ρ_0 = 0.05 fixed;  all networks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rho_vs_k(
    graphs: dict[str, nx.Graph],
    rho_0: float = 0.05,
    k_max: int = 6,
    n_runs: int = 50,
) -> None:
    print(f"Figure 2 — ρ_final vs k  (ρ₀={rho_0}) …")

    k_values = list(range(1, k_max + 1))
    colors   = ["#2196F3", "#FF5722", "#4CAF50"]
    markers  = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for (name, G), color, marker in zip(graphs.items(), colors, markers):
        src, dst = _edge_arrays(G)
        N  = G.number_of_nodes()

        rhos = []
        for k in k_values:
            rhos.append(avg_rho(src, dst, N, k, rho_0, n_runs))
            sys.stdout.write(".")
            sys.stdout.flush()
        print(f"  {name}")

        ax.plot(k_values, rhos, color=color, marker=marker,
                markersize=7, lw=2.0, label=name)

    ax.set_xlabel("Bootstrap threshold  k", fontsize=13)
    ax.set_ylabel("Final infected fraction  ρ_final", fontsize=13)
    ax.set_title(
        f"Bootstrap Percolation — ρ_final vs k\n"
        f"(ρ₀={rho_0}, N={N_NODES}, {n_runs} realisations)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(k_values)
    ax.set_xlim(0.5, k_max + 0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "bootstrap_rho_vs_k.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase diagram (ρ_0, k)  for ER
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_diagram(
    G_er: nx.Graph,
    k_max: int = 8,
    n_rho0: int = 20,
    n_runs: int = 30,
) -> None:
    print(f"Figure 3 — phase diagram  (k=1–{k_max}, {n_rho0} ρ₀ steps, "
          f"{n_runs} runs each) …")

    src, dst = _edge_arrays(G_er)
    N = G_er.number_of_nodes()

    rho0_arr = np.linspace(0.0, 0.30, n_rho0 + 1)[1:]   # skip 0
    k_arr    = np.arange(1, k_max + 1)

    # rho_grid[i, j] = avg ρ_final for (k_arr[i], rho0_arr[j])
    rho_grid = np.zeros((len(k_arr), n_rho0))

    for i, k in enumerate(k_arr):
        for j, rho_0 in enumerate(rho0_arr):
            rho_grid[i, j] = avg_rho(
                src, dst, N, k, rho_0, n_runs,
                base_seed=i * n_rho0 + j,
            )
        print(f"  k={k} done  (ρ_max={rho_grid[i].max():.2f})")

    fig, ax = plt.subplots(figsize=(8, 6))

    # pcolormesh needs cell edges, not centres
    rho0_edges = np.concatenate([[0], (rho0_arr[:-1] + rho0_arr[1:]) / 2, [rho0_arr[-1] + (rho0_arr[1] - rho0_arr[0]) / 2]])
    k_edges    = np.arange(0.5, k_max + 1.5)

    im = ax.pcolormesh(
        rho0_edges, k_edges, rho_grid,
        cmap="plasma", shading="flat", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final infected fraction  ρ_final", fontsize=12)

    # Mark approximate percolation boundary: cells where ρ_final just
    # exceeds twice the seed fraction (cascade started)
    cascade_mask = np.zeros_like(rho_grid, dtype=bool)
    for j, rho_0 in enumerate(rho0_arr):
        for i in range(len(k_arr)):
            cascade_mask[i, j] = rho_grid[i, j] > rho_0 * 2 + 0.05

    # Contour at ρ_final = 0.5 as the transition line
    rho0_centres = rho0_arr
    k_centres    = k_arr.astype(float)
    try:
        cs = ax.contour(
            rho0_centres, k_centres, rho_grid,
            levels=[0.5], colors=["white"], linewidths=[2.0], linestyles=["--"],
        )
        ax.clabel(cs, fmt="ρ=0.5", fontsize=9, colors="white")
    except Exception:
        pass   # contour may fail if grid is flat

    ax.set_xlabel("Initial seed fraction  ρ₀", fontsize=13)
    ax.set_ylabel("Bootstrap threshold  k", fontsize=13)
    ax.set_yticks(k_arr)
    ax.set_title(
        f"Bootstrap Percolation — Phase Diagram  (Erdős–Rényi, N={N})\n"
        f"({n_runs} realisations per cell)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out = OUT_DIR / "bootstrap_phase_diagram.png"
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
        deg = np.array([d for _, d in G.degree()])
        print(f"  {name:20s}: nodes={G.number_of_nodes()}, "
              f"edges={G.number_of_edges()}, "
              f"⟨k⟩={deg.mean():.1f}, max_k={deg.max()}")

    print()
    fig_rho_vs_rho0(graphs)
    print()
    fig_rho_vs_k(graphs)
    print()
    fig_phase_diagram(graphs["ER"])

    print(f"\nAll outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
