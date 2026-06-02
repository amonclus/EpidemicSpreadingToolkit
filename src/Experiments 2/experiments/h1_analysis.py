#!/usr/bin/env python3
"""
h1_analysis.py — H1 (OR-Hybrid) contagion model on three network types.

A susceptible node is infected each round if EITHER:
  (1) SIR channel  : at least one infected neighbour transmits with probability β, OR
  (2) Bootstrap    : it has ≥ k infected neighbours simultaneously (deterministic).
Infected nodes recover with probability γ each round.
Limiting case k → ∞ reduces to pure SIR.

Produces three PNG figures:
  1. h1_rho_vs_beta.png      — ρ_final vs β for k=1,2,3 + pure SIR, all networks
  2. h1_time_series.png      — I(t)/N: k=2 hybrid vs SIR for 3 β values (ER only)
  3. h1_phase_diagram.png    — 2-D heatmap of ρ_final in (β, k) for ER only

Usage:
    python src/experiments/h1_analysis.py

Runtime: ~5–10 min (dominated by the phase diagram).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("results/h1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
N_NODES = 600
GAMMA   = 0.1    # recovery rate (fixed throughout)
N_SEEDS = 6      # initial infected nodes (~1 % of N)
SEED    = 42

plt.rcParams.update({"font.size": 11})


# ─────────────────────────────────────────────────────────────────────────────
# Network construction  (identical topology to sir_analysis / bootstrap_analysis)
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
    """Both directions for an undirected graph."""
    u, v = zip(*G.edges())
    u, v = np.array(u, dtype=np.int32), np.array(v, dtype=np.int32)
    return np.concatenate([u, v]), np.concatenate([v, u])


def _degree_stats(G: nx.Graph) -> float:
    return float(np.mean([d for _, d in G.degree()]))


# ─────────────────────────────────────────────────────────────────────────────
# H1 simulation — vectorised
# ─────────────────────────────────────────────────────────────────────────────

def h1_run(
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    k: int,
    beta: float,
    gamma: float,
    n_seeds: int = N_SEEDS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """
    One realisation of discrete-time H1 (OR-Hybrid).

    Each round:
      · SIR channel   : sample transmission along every I→S edge (prob β).
      · Bootstrap     : activate every S node with ≥ k infected neighbours.
      · Union (OR)    : a S node becomes I if either channel fires.
      · Recovery      : each I node recovers independently with prob γ.
    Both channels use the state at the START of the round (synchronous).

    Returns I(t)/N array and ρ_final (fraction ever infected).
    """
    if rng is None:
        rng = np.random.default_rng()

    infected      = np.zeros(N, dtype=bool)
    susceptible   = np.ones(N, dtype=bool)
    ever_infected = np.zeros(N, dtype=bool)

    seeds = rng.choice(N, n_seeds, replace=False)
    infected[seeds]      = True
    susceptible[seeds]   = False
    ever_infected[seeds] = True

    I_series: list[float] = []

    while True:
        n_i = int(infected.sum())
        I_series.append(n_i / N)
        if n_i == 0:
            break

        # ── SIR channel ──────────────────────────────────────────────────
        sir_mask = infected[src] & susceptible[dst]
        idx = np.where(sir_mask)[0]
        sir_newly = np.zeros(N, dtype=bool)
        if idx.size:
            fires = rng.random(idx.size) < beta
            sir_newly[np.unique(dst[idx[fires]])] = True

        # ── Bootstrap channel ────────────────────────────────────────────
        infect_count = np.zeros(N, dtype=np.int32)
        np.add.at(infect_count, dst, infected[src].astype(np.int32))
        boot_newly = susceptible & (infect_count >= k)

        # ── OR union ─────────────────────────────────────────────────────
        newly_infected = sir_newly | boot_newly

        # ── Recovery (uses start-of-round infected set) ──────────────────
        newly_recovered = infected & (rng.random(N) < gamma)

        # ── State update ─────────────────────────────────────────────────
        infected    = (infected | newly_infected) & ~newly_recovered
        susceptible = susceptible & ~newly_infected
        ever_infected |= newly_infected

    rho = float(ever_infected.sum()) / N
    return np.array(I_series), rho


def sir_run(
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    beta: float,
    gamma: float,
    n_seeds: int = N_SEEDS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Pure SIR — equivalent to H1 with k → ∞ (bootstrap channel disabled)."""
    if rng is None:
        rng = np.random.default_rng()

    infected      = np.zeros(N, dtype=bool)
    susceptible   = np.ones(N, dtype=bool)
    ever_infected = np.zeros(N, dtype=bool)

    seeds = rng.choice(N, n_seeds, replace=False)
    infected[seeds]      = True
    susceptible[seeds]   = False
    ever_infected[seeds] = True

    I_series: list[float] = []

    while True:
        n_i = int(infected.sum())
        I_series.append(n_i / N)
        if n_i == 0:
            break

        mask = infected[src] & susceptible[dst]
        idx  = np.where(mask)[0]
        newly_infected = np.zeros(N, dtype=bool)
        if idx.size:
            fires = rng.random(idx.size) < beta
            newly_infected[np.unique(dst[idx[fires]])] = True

        newly_recovered = infected & (rng.random(N) < gamma)

        infected    = (infected | newly_infected) & ~newly_recovered
        susceptible = susceptible & ~newly_infected
        ever_infected |= newly_infected

    rho = float(ever_infected.sum()) / N
    return np.array(I_series), rho


def _avg_rho_h1(src, dst, N, k, beta, gamma, n_runs, base_seed=0):
    return float(np.mean([
        h1_run(src, dst, N, k, beta, gamma,
               rng=np.random.default_rng(base_seed + i))[1]
        for i in range(n_runs)
    ]))


def _avg_rho_sir(src, dst, N, beta, gamma, n_runs, base_seed=0):
    return float(np.mean([
        sir_run(src, dst, N, beta, gamma,
                rng=np.random.default_rng(base_seed + i))[1]
        for i in range(n_runs)
    ]))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ρ_final vs β  (k=1,2,3 + pure SIR, all networks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rho_vs_beta(
    graphs: dict[str, nx.Graph],
    gamma: float = GAMMA,
    n_betas: int = 25,
    n_runs: int = 50,
) -> None:
    print("Figure 1 — ρ_final vs β …")

    # Range chosen to span sub-threshold → fully supercritical for all networks
    betas = np.linspace(0.0, 0.10, n_betas + 1)[1:]   # skip exact 0

    k_values  = [1, 2, 3]
    k_styles  = {1: "-",  2: "--",  3: ":"}
    k_markers = {1: "o",  2: "s",   3: "^"}
    net_colors = {"ER": "#2196F3", "BA (scale-free)": "#FF5722", "Lattice": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, G in graphs.items():
        src, dst = _edge_arrays(G)
        N = G.number_of_nodes()
        color = net_colors[name]
        k1 = _degree_stats(G)

        # --- H1 curves (k = 1, 2, 3) ---
        for k in k_values:
            rhos = []
            for beta in betas:
                rhos.append(_avg_rho_h1(src, dst, N, k, beta, gamma, n_runs))
                sys.stdout.write(".")
                sys.stdout.flush()
            print(f"  {name} k={k}")

            ax.plot(
                betas, rhos,
                color=color, ls=k_styles[k], marker=k_markers[k],
                markersize=4, lw=1.8,
                label=f"{name}, k={k}",
            )

        # --- Pure SIR (k → ∞) ---
        sir_rhos = []
        for beta in betas:
            sir_rhos.append(_avg_rho_sir(src, dst, N, beta, gamma, n_runs))
            sys.stdout.write(".")
            sys.stdout.flush()
        print(f"  {name} SIR")

        ax.plot(
            betas, sir_rhos,
            color=color, ls="-.", marker="D", markersize=4, lw=1.5, alpha=0.6,
            label=f"{name}, SIR (k=∞)",
        )

        # Mark the mean-field SIR threshold
        beta_c = gamma / k1
        if beta_c <= betas[-1]:
            ax.axvline(beta_c, color=color, lw=0.8, ls=":", alpha=0.4)

    ax.set_xlabel("Transmission rate  β", fontsize=13)
    ax.set_ylabel("Final epidemic size  ρ_final", fontsize=13)
    ax.set_title(
        f"H1 (OR-Hybrid) — ρ_final vs β\n"
        f"(γ={gamma}, N={N_NODES}, {n_runs} realisations per point)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, betas[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, ncol=4, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "h1_rho_vs_beta.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Time series I(t)/N  (ER only, k=2 hybrid vs SIR)
# ─────────────────────────────────────────────────────────────────────────────

def fig_time_series(
    G_er: nx.Graph,
    gamma: float = GAMMA,
    avg_runs: int = 15,
) -> None:
    print("Figure 2 — time series (ER only) …")

    src, dst = _edge_arrays(G_er)
    N  = G_er.number_of_nodes()
    k1 = _degree_stats(G_er)
    beta_c = gamma / k1   # SIR mean-field threshold

    rep_betas = [
        ("β = 0.5 β_c  (sub-critical)",  0.5 * beta_c),
        ("β ≈ β_c       (critical)",      1.0 * beta_c),
        ("β = 4 β_c    (super-critical)", 4.0 * beta_c),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {"H1  k=2": "#1565C0", "Pure SIR": "#C62828"}
    lstyles = {"H1  k=2": "-", "Pure SIR": "--"}

    for ax, (beta_label, beta) in zip(axes, rep_betas):
        for model_name, runner in [
            ("H1  k=2", lambda b, s: h1_run(src, dst, N, 2, b, gamma, rng=s)),
            ("Pure SIR", lambda b, s: sir_run(src, dst, N, b, gamma, rng=s)),
        ]:
            series_list = []
            for i in range(avg_runs):
                I_t, _ = runner(beta, np.random.default_rng(400 + i))
                series_list.append(I_t)

            T = max(len(s) for s in series_list)
            mat = np.zeros((avg_runs, T))
            for i, s in enumerate(series_list):
                mat[i, :len(s)] = s
            I_mean = mat.mean(axis=0)

            ax.plot(
                I_mean,
                color=colors[model_name], ls=lstyles[model_name],
                lw=2.2, label=model_name,
            )

        ax.set_title(beta_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (steps)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("I(t) / N", fontsize=11)

        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=-0.005)

    fig.suptitle(
        f"H1 Time Series — I(t)/N  (ER, γ={gamma}, avg of {avg_runs} runs)\n"
        f"Solid = H1 k=2 hybrid,  Dashed = pure SIR",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out = OUT_DIR / "h1_time_series.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase diagram (β, k)  for ER
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_diagram(
    G_er: nx.Graph,
    gamma: float = GAMMA,
    k_max: int = 8,
    n_betas: int = 20,
    n_runs: int = 30,
) -> None:
    print(f"Figure 3 — phase diagram  (β: 20 steps, k=1–{k_max}, "
          f"{n_runs} runs each) …")

    src, dst = _edge_arrays(G_er)
    N  = G_er.number_of_nodes()
    k1 = _degree_stats(G_er)

    # β range centred on the SIR threshold (β_c = γ/⟨k⟩ ≈ 0.012 for this graph).
    # Going to 0.5 saturates everything; 0.04 (≈3×β_c) reveals the transition.
    betas  = np.linspace(0.0, 0.04, n_betas + 1)[1:]   # skip 0
    k_arr  = np.arange(1, k_max + 1)

    # rho_grid[i, j] = avg ρ_final  for  (k_arr[i], betas[j])
    rho_grid = np.zeros((len(k_arr), n_betas))

    for i, k in enumerate(k_arr):
        for j, beta in enumerate(betas):
            rho_grid[i, j] = _avg_rho_h1(
                src, dst, N, k, beta, gamma, n_runs,
                base_seed=i * n_betas + j,
            )
        print(f"  k={k} done  (ρ_max={rho_grid[i].max():.2f})")

    # SIR mean-field threshold line: β_c = γ / ⟨k⟩
    beta_c = gamma / k1

    fig, ax = plt.subplots(figsize=(8, 6))

    beta_edges = np.concatenate([
        [betas[0] - (betas[1] - betas[0]) / 2],
        (betas[:-1] + betas[1:]) / 2,
        [betas[-1] + (betas[-1] - betas[-2]) / 2],
    ])
    k_edges = np.arange(0.5, k_max + 1.5)

    im = ax.pcolormesh(
        beta_edges, k_edges, rho_grid,
        cmap="plasma", shading="flat", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final epidemic size  ρ_final", fontsize=12)

    # Contour at ρ = 0.5 to mark the effective transition boundary
    try:
        cs = ax.contour(
            betas, k_arr.astype(float), rho_grid,
            levels=[0.5], colors=["white"], linewidths=[2.0], linestyles=["--"],
        )
        ax.clabel(cs, fmt="ρ=0.5", fontsize=9, colors="white")
    except Exception:
        pass

    # SIR threshold vertical line (independent of k)
    ax.axvline(beta_c, color="cyan", lw=1.8, ls=":",
               label=f"SIR threshold  β_c={beta_c:.4f}")

    ax.set_xlabel("Transmission rate  β", fontsize=13)
    ax.set_ylabel("Bootstrap threshold  k", fontsize=13)
    ax.set_yticks(k_arr)
    ax.set_title(
        f"H1 Phase Diagram — Erdős–Rényi  (N={N}, γ={gamma})\n"
        f"({n_runs} realisations per cell)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()

    out = OUT_DIR / "h1_phase_diagram.png"
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
        k1 = _degree_stats(G)
        beta_c = GAMMA / k1
        print(f"  {name:20s}: nodes={G.number_of_nodes()}, "
              f"edges={G.number_of_edges()}, "
              f"⟨k⟩={k1:.1f},  SIR β_c={beta_c:.4f}")

    print()
    fig_rho_vs_beta(graphs)
    print()
    fig_time_series(graphs["ER"])
    print()
    fig_phase_diagram(graphs["ER"])

    print(f"\nAll outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
