#!/usr/bin/env python3
"""
wtm_analysis.py — Watts Threshold Model (WTM) on three network types.

A susceptible node activates when the *fraction* of its neighbours that are
infected meets or exceeds the threshold φ:

    activates  iff  infected_neighbours / degree(v) >= φ

This is bootstrap percolation normalised by degree.  The cascade is
deterministic given the seed set; randomness comes only from seed selection.

Limiting cases:
  φ → 0  :  any infected neighbour activates (≡ k=1 bootstrap)
  φ → 1  :  all neighbours must be infected (rarely cascades)

The interesting region is intermediate φ, where global cascades depend
sensitively on the network structure and ρ₀.

Produces three PNG figures:
  1. wtm_rho_vs_phi.png       — ρ_final vs φ for ρ₀=0.01,0.05,0.10, all networks
  2. wtm_rho_vs_rho0.png      — ρ_final vs ρ₀ for φ=0.1,0.2,0.3, all networks
  3. wtm_phase_diagram.png    — 2-D heatmap of ρ_final in (φ, ρ₀) for ER only

Usage:
    python src/experiments/wtm_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("results/wtm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
N_NODES = 600
SEED    = 42

plt.rcParams.update({"font.size": 11})


# ─────────────────────────────────────────────────────────────────────────────
# Network construction
# ─────────────────────────────────────────────────────────────────────────────

def build_networks(n: int = N_NODES, seed: int = SEED) -> dict[str, nx.Graph]:
    rng = np.random.default_rng(seed)
    s = lambda: int(rng.integers(1_000_000))

    G_er  = nx.erdos_renyi_graph(n, 8 / (n - 1), seed=s())
    G_ba  = nx.barabasi_albert_graph(n, 4, seed=s())
    G_lat = nx.grid_2d_graph(20, 30)
    G_lat = nx.convert_node_labels_to_integers(G_lat)

    return {"ER": G_er, "BA (scale-free)": G_ba, "Lattice": G_lat}


def _edge_arrays(G: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    u, v = zip(*G.edges())
    u, v = np.array(u, dtype=np.int32), np.array(v, dtype=np.int32)
    return np.concatenate([u, v]), np.concatenate([v, u])


def _degree_array(G: nx.Graph) -> np.ndarray:
    """Degree of each node as a float array indexed by node id."""
    N = G.number_of_nodes()
    deg = np.zeros(N, dtype=np.float32)
    for node, d in G.degree():
        deg[node] = d
    return deg


# ─────────────────────────────────────────────────────────────────────────────
# WTM simulation — vectorised
# ─────────────────────────────────────────────────────────────────────────────

def wtm_run(
    degrees: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    phi: float,
    seed_indices: np.ndarray,
) -> float:
    """
    One deterministic WTM cascade.

    Each synchronous round:
      · Count infected neighbours for every node via edge sums.
      · Activate every susceptible node v (with degree > 0) where
        infected_count[v] / degree[v] >= phi.
    Repeats until stable.  Returns ρ_final = infected / N.
    """
    infected = np.zeros(N, dtype=bool)
    infected[seed_indices] = True

    while True:
        infected_count = np.zeros(N, dtype=np.int32)
        np.add.at(infected_count, dst, infected[src].astype(np.int32))

        # Fraction of infected neighbours; 0 for isolated nodes (degrees=0)
        frac = np.where(degrees > 0, infected_count / degrees, 0.0)

        newly = ~infected & (degrees > 0) & (frac >= phi)
        if not newly.any():
            break
        infected |= newly

    return float(infected.sum()) / N


def _avg_rho(
    degrees: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    phi: float,
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
        rhos.append(wtm_run(degrees, src, dst, N, phi, seeds))
    return float(np.mean(rhos))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ρ_final vs φ  (ρ₀ = 0.01, 0.05, 0.10;  all networks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rho_vs_phi(
    graphs: dict[str, nx.Graph],
    rho0_values: list[float] = [0.01, 0.05, 0.10],
    n_phi: int = 25,
    n_runs: int = 50,
) -> None:
    print("Figure 1 — ρ_final vs φ …")

    phi_arr = np.linspace(0.05, 0.50, n_phi)

    net_colors   = {"ER": "#2196F3", "BA (scale-free)": "#FF5722", "Lattice": "#4CAF50"}
    rho0_styles  = {0.01: "-",  0.05: "--", 0.10: ":"}
    rho0_markers = {0.01: "o",  0.05: "s",  0.10: "^"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Faint diagonal: ρ_final = ρ₀ (no cascade beyond seeds)
    ax.axhline(0.01,  color="gray", lw=0.6, ls="-.", alpha=0.4, zorder=0)
    ax.axhline(0.05,  color="gray", lw=0.6, ls="-.", alpha=0.4, zorder=0)
    ax.axhline(0.10,  color="gray", lw=0.6, ls="-.", alpha=0.4, zorder=0)

    for name, G in graphs.items():
        deg = _degree_array(G)
        src, dst = _edge_arrays(G)
        N  = G.number_of_nodes()
        color = net_colors[name]

        for rho_0 in rho0_values:
            rhos = []
            for phi in phi_arr:
                rhos.append(_avg_rho(deg, src, dst, N, phi, rho_0, n_runs))
                sys.stdout.write(".")
                sys.stdout.flush()
            print(f"  {name} ρ₀={rho_0}")

            ax.plot(
                phi_arr, rhos,
                color=color, ls=rho0_styles[rho_0],
                marker=rho0_markers[rho_0], markersize=4, lw=1.8,
                label=f"{name}, ρ₀={rho_0}",
            )

    ax.set_xlabel("Threshold  φ", fontsize=13)
    ax.set_ylabel("Final infected fraction  ρ_final", fontsize=13)
    ax.set_title(
        f"WTM — ρ_final vs φ\n"
        f"(N={N_NODES}, {n_runs} realisations per point;  "
        f"faint lines = ρ_final = ρ₀ baseline)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(phi_arr[0], phi_arr[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "wtm_rho_vs_phi.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — ρ_final vs ρ₀  (φ = 0.1, 0.2, 0.3;  all networks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rho_vs_rho0(
    graphs: dict[str, nx.Graph],
    phi_values: list[float] = [0.1, 0.2, 0.3],
    n_rho0: int = 25,
    n_runs: int = 50,
) -> None:
    print("Figure 2 — ρ_final vs ρ₀ …")

    rho0_arr = np.linspace(0.0, 0.20, n_rho0 + 1)[1:]   # skip exact 0

    net_colors  = {"ER": "#2196F3", "BA (scale-free)": "#FF5722", "Lattice": "#4CAF50"}
    phi_styles  = {0.1: "-",  0.2: "--", 0.3: ":"}
    phi_markers = {0.1: "o",  0.2: "s",  0.3: "^"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # No-cascade diagonal
    ax.plot([0, rho0_arr[-1]], [0, rho0_arr[-1]], color="gray",
            lw=0.8, ls="-.", zorder=0, label="ρ_final = ρ₀  (no cascade)")

    for name, G in graphs.items():
        deg = _degree_array(G)
        src, dst = _edge_arrays(G)
        N  = G.number_of_nodes()
        color = net_colors[name]

        for phi in phi_values:
            rhos = []
            for rho_0 in rho0_arr:
                rhos.append(_avg_rho(deg, src, dst, N, phi, rho_0, n_runs))
                sys.stdout.write(".")
                sys.stdout.flush()
            print(f"  {name} φ={phi}")

            ax.plot(
                rho0_arr, rhos,
                color=color, ls=phi_styles[phi],
                marker=phi_markers[phi], markersize=4, lw=1.8,
                label=f"{name}, φ={phi}",
            )

    ax.set_xlabel("Initial seed fraction  ρ₀", fontsize=13)
    ax.set_ylabel("Final infected fraction  ρ_final", fontsize=13)
    ax.set_title(
        f"WTM — ρ_final vs ρ₀\n"
        f"(N={N_NODES}, {n_runs} realisations per point)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, rho0_arr[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "wtm_rho_vs_rho0.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase diagram (φ, ρ₀) for ER
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_diagram(
    G_er: nx.Graph,
    n_phi: int = 20,
    n_rho0: int = 20,
    n_runs: int = 30,
) -> None:
    print(f"Figure 3 — phase diagram  ({n_phi} φ × {n_rho0} ρ₀,  "
          f"{n_runs} runs each) …")

    deg = _degree_array(G_er)
    src, dst = _edge_arrays(G_er)
    N = G_er.number_of_nodes()
    k_mean = float(deg.mean())

    phi_arr  = np.linspace(0.05, 0.50, n_phi)
    rho0_arr = np.linspace(0.00, 0.20, n_rho0 + 1)[1:]   # skip exact 0

    # rho_grid[i, j] = avg ρ_final for (rho0_arr[i], phi_arr[j])
    rho_grid = np.zeros((n_rho0, n_phi))

    for i, rho_0 in enumerate(rho0_arr):
        for j, phi in enumerate(phi_arr):
            rho_grid[i, j] = _avg_rho(
                deg, src, dst, N, phi, rho_0, n_runs,
                base_seed=i * n_phi + j,
            )
        print(f"  ρ₀={rho_0:.3f} done  (ρ_max={rho_grid[i].max():.2f})")

    fig, ax = plt.subplots(figsize=(8, 6))

    phi_edges  = np.concatenate([
        [phi_arr[0]  - (phi_arr[1]  - phi_arr[0])  / 2],
        (phi_arr[:-1]  + phi_arr[1:])  / 2,
        [phi_arr[-1]  + (phi_arr[-1]  - phi_arr[-2])  / 2],
    ])
    rho0_edges = np.concatenate([
        [rho0_arr[0] - (rho0_arr[1] - rho0_arr[0]) / 2],
        (rho0_arr[:-1] + rho0_arr[1:]) / 2,
        [rho0_arr[-1] + (rho0_arr[-1] - rho0_arr[-2]) / 2],
    ])

    im = ax.pcolormesh(
        phi_edges, rho0_edges, rho_grid,
        cmap="plasma", shading="flat", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final infected fraction  ρ_final", fontsize=12)

    # ρ = 0.5 contour as the cascade boundary
    try:
        cs = ax.contour(
            phi_arr, rho0_arr, rho_grid,
            levels=[0.5], colors=["white"], linewidths=[2.0], linestyles=["--"],
        )
        ax.clabel(cs, fmt="ρ=0.5", fontsize=9, colors="white")
    except Exception:
        pass

    # Annotate the approximate analytical cascade window for ER:
    # A global cascade is possible when phi < 1/⟨k⟩ (Watts 2002)
    phi_c = 1.0 / k_mean
    if phi_arr[0] <= phi_c <= phi_arr[-1]:
        ax.axvline(phi_c, color="cyan", lw=1.8, ls=":",
                   label=f"φ_c ≈ 1/⟨k⟩ = {phi_c:.3f}")

    ax.set_xlabel("Threshold  φ", fontsize=13)
    ax.set_ylabel("Initial seed fraction  ρ₀", fontsize=13)
    ax.set_title(
        f"WTM Phase Diagram — Erdős–Rényi  (N={N}, ⟨k⟩={k_mean:.1f})\n"
        f"({n_runs} realisations per cell)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()

    out = OUT_DIR / "wtm_phase_diagram.png"
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
        deg = _degree_array(G)
        print(f"  {name:20s}: nodes={G.number_of_nodes()}, "
              f"edges={G.number_of_edges()}, "
              f"⟨k⟩={deg.mean():.1f},  max_k={int(deg.max())},  "
              f"φ_c≈1/⟨k⟩={1/deg.mean():.3f}")

    print()
    fig_rho_vs_phi(graphs)
    print()
    fig_rho_vs_rho0(graphs)
    print()
    fig_phase_diagram(graphs["ER"])

    print(f"\nAll outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
