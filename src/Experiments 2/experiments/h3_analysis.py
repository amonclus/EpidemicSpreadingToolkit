#!/usr/bin/env python3
"""
h3_analysis.py — H3 (Probabilistic Threshold Hybrid) contagion model.

A susceptible node with m infected neighbours is infected each round with
probability:

    P(m) = (1 − α) · [1 − (1−β)^m]  +  α · min(1, m·β)
            \___ pure-SIR term ___/       \___ H3 linear term ___/

α = 0  →  pure SIR  (P = 1−(1−β)^m, independent per-contact transmissions)
α = 1  →  pure H3   (P = min(1, m·β), linear social reinforcement)
Intermediate α interpolates between the two mechanisms.

Recovery is identical to SIR: each infected node recovers with prob γ.
Terminates when no infected nodes remain.

Fixed in this script: γ = 0.1, N = 600, ~1 % initial seeds.

Produces three PNG figures:
  1. h3_rho_vs_beta.png      — ρ_final vs β for α=0,0.1,0.3,0.5, all networks
  2. h3_infection_kernel.png — P(m) vs m (static, no simulation)
  3. h3_phase_diagram.png    — 2-D heatmap of ρ_final in (β, α) for ER only

Usage:
    python src/experiments/h3_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("results/h3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
N_NODES = 900
GAMMA   = 0.1
N_SEEDS = 6      # ~1 % of N
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


def _mean_degree(G: nx.Graph) -> float:
    return float(np.mean([d for _, d in G.degree()]))


# ─────────────────────────────────────────────────────────────────────────────
# Infection kernel  (analytical, no simulation needed)
# ─────────────────────────────────────────────────────────────────────────────

def p_infect(m: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    """
    P(infection | m infected neighbours) for the H3-α model.

    P(m) = (1−α)·[1−(1−β)^m]  +  α·min(1, m·β)

    α=0 → pure SIR (independent per-contact events)
    α=1 → pure H3  (linear reinforcement, saturates at m·β = 1)
    """
    sir_term = 1.0 - (1.0 - beta) ** m
    h3_term  = np.minimum(1.0, m * beta)
    return (1.0 - alpha) * sir_term + alpha * h3_term


# ─────────────────────────────────────────────────────────────────────────────
# H3-α simulation — vectorised
# ─────────────────────────────────────────────────────────────────────────────

def h3_run(
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    beta: float,
    alpha: float,
    gamma: float,
    n_seeds: int = N_SEEDS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """
    One H3-α realisation.

    Each round:
      · Count infected neighbours m[v] for every node v via edge-sum.
      · Compute P(m[v]) for each susceptible node.
      · Sample one Bernoulli per susceptible node — avoids per-edge loops.
      · Recover each infected node independently with prob γ.
    All transitions use the state at the START of the round (synchronous).

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

        # Count infected neighbours for each node (vectorised)
        m = np.zeros(N, dtype=np.int32)
        np.add.at(m, dst, infected[src].astype(np.int32))

        # Compute infection probability per node, sample once per node
        p = p_infect(m.astype(float), beta, alpha)
        newly_infected = susceptible & (rng.random(N) < p)

        # Recovery
        newly_recovered = infected & (rng.random(N) < gamma)

        infected    = (infected | newly_infected) & ~newly_recovered
        susceptible = susceptible & ~newly_infected
        ever_infected |= newly_infected

    rho = float(ever_infected.sum()) / N
    return np.array(I_series), rho


def _avg_rho(src, dst, N, beta, alpha, gamma, n_runs, base_seed=0):
    return float(np.mean([
        h3_run(src, dst, N, beta, alpha, gamma,
               rng=np.random.default_rng(base_seed + i))[1]
        for i in range(n_runs)
    ]))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ρ_final vs β  (α = 0, 0.1, 0.3, 0.5;  all networks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rho_vs_beta(
    graphs: dict[str, nx.Graph],
    alphas: list[float] = [0.0, 0.1, 0.3, 0.5],
    n_betas: int = 25,
    n_runs: int = 50,
) -> None:
    print("Figure 1 — ρ_final vs β …")

    betas = np.linspace(0.0, 0.12, n_betas + 1)[1:]   # skip exact 0

    net_colors    = {"ER": "#2196F3", "BA (scale-free)": "#FF5722", "Lattice": "#4CAF50"}
    alpha_styles  = {0.0: "-",  0.1: "--", 0.3: ":",  0.5: "-."}
    alpha_markers = {0.0: "o",  0.1: "s",  0.3: "^",  0.5: "D"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, G in graphs.items():
        src, dst = _edge_arrays(G)
        N     = G.number_of_nodes()
        color = net_colors[name]
        k1    = _mean_degree(G)

        for alpha in alphas:
            rhos = []
            for beta in betas:
                rhos.append(_avg_rho(src, dst, N, beta, alpha, GAMMA, n_runs))
                sys.stdout.write(".")
                sys.stdout.flush()
            print(f"  {name} α={alpha}")

            ax.plot(
                betas, rhos,
                color=color, ls=alpha_styles[alpha],
                marker=alpha_markers[alpha], markersize=4, lw=1.8,
                label=f"{name}, α={alpha}",
            )

        # SIR mean-field threshold (same for all α, shifts with ⟨k⟩)
        beta_c = GAMMA / k1
        if beta_c <= betas[-1]:
            ax.axvline(beta_c, color=color, lw=0.8, ls=":", alpha=0.35)

    ax.set_xlabel("Transmission rate  β", fontsize=13)
    ax.set_ylabel("Final epidemic size  ρ_final", fontsize=13)
    ax.set_title(
        f"H3 (Probabilistic Threshold Hybrid) — ρ_final vs β\n"
        f"(γ={GAMMA}, N={N_NODES}, {n_runs} realisations per point)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, betas[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, ncol=4, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "h3_rho_vs_beta.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Infection kernel P(m) vs m  (static, no simulation)
# ─────────────────────────────────────────────────────────────────────────────

def fig_infection_kernel(
    beta: float = 0.1,
    alphas: list[float] = [0.0, 0.1, 0.3, 0.5],
    degree: int = 6,
) -> None:
    print("Figure 2 — infection kernel P(m) vs m  (analytical) …")

    m_arr = np.arange(0, degree + 1)   # m = 0, 1, …, degree

    colors  = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]
    markers = ["o", "s", "^", "D"]
    labels  = {
        0.0: "α=0  (pure SIR)",
        0.1: "α=0.1",
        0.3: "α=0.3",
        0.5: "α=0.5",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for alpha, color, marker in zip(alphas, colors, markers):
        p = p_infect(m_arr.astype(float), beta, alpha)
        ax.plot(m_arr, p, color=color, marker=marker, markersize=8,
                lw=2.0, label=labels[alpha])

    # Annotate the natural soft threshold m* = 1/β where H3 saturates
    m_star = 1.0 / beta
    ax.axvline(m_star, color="gray", ls="--", lw=1.2, alpha=0.7,
               label=f"m* = 1/β = {m_star:.0f}  (H3 saturation)")

    ax.set_xlabel("Number of infected neighbours  m", fontsize=13)
    ax.set_ylabel("Infection probability  P(m)", fontsize=13)
    ax.set_title(
        f"H3 Infection Kernel  (β={beta}, node degree={degree})\n"
        f"P(m) = (1−α)·[1−(1−β)^m] + α·min(1, m·β)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(m_arr)
    ax.set_xlim(-0.3, degree + 0.3)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "h3_infection_kernel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase diagram (β, α)  for ER
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_diagram(
    G_er: nx.Graph,
    n_betas: int = 20,
    n_alphas: int = 20,
    n_runs: int = 30,
) -> None:
    print(f"Figure 3 — phase diagram  ({n_betas} β × {n_alphas} α,  "
          f"{n_runs} runs each) …")

    src, dst = _edge_arrays(G_er)
    N  = G_er.number_of_nodes()
    k1 = _mean_degree(G_er)

    # β range centred on the SIR threshold (β_c ≈ 0.012 for this graph).
    # Going to 0.5 saturates everything; 0.04 (≈3×β_c) reveals the transition.
    betas  = np.linspace(0.0, 0.04, n_betas + 1)[1:]   # skip exact 0
    alphas = np.linspace(0.0, 1.0, n_alphas)

    # rho_grid[i, j] = avg ρ_final for (alphas[i], betas[j])
    rho_grid = np.zeros((n_alphas, n_betas))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            rho_grid[i, j] = _avg_rho(
                src, dst, N, beta, alpha, GAMMA, n_runs,
                base_seed=i * n_betas + j,
            )
        print(f"  α={alpha:.2f} done  (ρ_max={rho_grid[i].max():.2f})")

    # SIR mean-field threshold: β_c = γ/⟨k⟩ (independent of α)
    beta_c = GAMMA / k1

    fig, ax = plt.subplots(figsize=(8, 6))

    beta_edges = np.concatenate([
        [betas[0] - (betas[1] - betas[0]) / 2],
        (betas[:-1] + betas[1:]) / 2,
        [betas[-1] + (betas[-1] - betas[-2]) / 2],
    ])
    alpha_edges = np.concatenate([
        [alphas[0] - (alphas[1] - alphas[0]) / 2],
        (alphas[:-1] + alphas[1:]) / 2,
        [alphas[-1] + (alphas[-1] - alphas[-2]) / 2],
    ])

    im = ax.pcolormesh(
        beta_edges, alpha_edges, rho_grid,
        cmap="plasma", shading="flat", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final epidemic size  ρ_final", fontsize=12)

    # ρ = 0.5 contour as the effective transition boundary
    try:
        cs = ax.contour(
            betas, alphas, rho_grid,
            levels=[0.5], colors=["white"], linewidths=[2.0], linestyles=["--"],
        )
        ax.clabel(cs, fmt="ρ=0.5", fontsize=9, colors="white")
    except Exception:
        pass

    # SIR threshold: vertical line (independent of α)
    ax.axvline(beta_c, color="cyan", lw=1.8, ls=":",
               label=f"SIR threshold  β_c={beta_c:.4f}")

    # Annotate regimes
    ax.text(0.005, 0.03, "α=0\n(SIR)", fontsize=9, color="white",
            va="bottom", ha="left")
    ax.text(0.005, 0.97, "α=1\n(H3)", fontsize=9, color="white",
            va="top", ha="left")

    ax.set_xlabel("Transmission rate  β", fontsize=13)
    ax.set_ylabel("Reinforcement parameter  α", fontsize=13)
    ax.set_title(
        f"H3 Phase Diagram — Erdős–Rényi  (N={N}, γ={GAMMA})\n"
        f"({n_runs} realisations per cell)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()

    out = OUT_DIR / "h3_phase_diagram.png"
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
        k1 = _mean_degree(G)
        print(f"  {name:20s}: nodes={G.number_of_nodes()}, "
              f"edges={G.number_of_edges()}, "
              f"⟨k⟩={k1:.1f},  SIR β_c={GAMMA/k1:.4f}")

    print()
    fig_rho_vs_beta(graphs)
    print()
    fig_infection_kernel()
    print()
    fig_phase_diagram(graphs["ER"])

    print(f"\nAll outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
