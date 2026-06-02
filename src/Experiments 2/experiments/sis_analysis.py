#!/usr/bin/env python3
"""
sis_analysis.py — SIS (Susceptible-Infected-Susceptible) epidemic model.

Unlike SIR, recovered nodes return immediately to the susceptible pool and
can be re-infected.  Above the epidemic threshold the infection persists as
an endemic equilibrium I* > 0; below it the epidemic dies out (I* = 0).

Mean-field endemic prevalence: I*/N = 1 − μ / (β·⟨k⟩)
Mean-field threshold:          β_c  = μ·⟨k⟩ / ⟨k²⟩  (heterogeneous networks)

Endemic prevalence I*/N is estimated by averaging I(t)/N over the last 200
of 500 simulation steps.

Fixed: μ = 0.1, ~5 % initial seeds (enough to survive near-threshold noise).

Produces three PNG figures:
  1. sis_endemic_prevalence.png  — I*/N vs β/μ for all 3 networks
  2. sis_time_series.png         — I(t)/N for 3 representative β, one panel/network
  3. sis_phase_diagram.png       — 2-D heatmap of I*/N in (β, μ) for ER only

Usage:
    python src/experiments/sis_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("results/sis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
N_NODES    = 600
MU_DEFAULT = 0.1      # recovery rate (fixed for figures 1 & 2)
SEED       = 42

T_RUN     = 500   # total steps per simulation
T_STEADY  = 300   # start of steady-state averaging window (last 200 steps)
T_SERIES  = 250   # steps shown in time-series plots

# Use 5 % seeds so near-threshold epidemics have a fair chance to persist
N_SEEDS = max(1, N_NODES // 20)

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


def _degree_stats(G: nx.Graph) -> tuple[float, float]:
    deg = np.array([d for _, d in G.degree()])
    return float(deg.mean()), float((deg ** 2).mean())


def _threshold(G: nx.Graph, mu: float) -> float:
    """Heterogeneous mean-field epidemic threshold: β_c = μ·⟨k⟩/⟨k²⟩."""
    k1, k2 = _degree_stats(G)
    return mu * k1 / k2


# ─────────────────────────────────────────────────────────────────────────────
# SIS simulation — vectorised
# ─────────────────────────────────────────────────────────────────────────────

def sis_run(
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    beta: float,
    mu: float,
    n_steps: int,
    n_seeds: int = N_SEEDS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Run SIS for n_steps rounds.  Returns I(t)/N array of length n_steps+1.

    Each round:
      · Transmission: each I→S edge fires with prob β; susceptible target
        becomes infected if any edge fires (vectorised via np.unique).
      · Recovery: each infected node independently recovers with prob μ
        and returns to susceptible (not removed).
    Both transitions use the state at the START of the round.

    If the epidemic dies out before n_steps the series is padded with zeros.
    """
    if rng is None:
        rng = np.random.default_rng()

    infected    = np.zeros(N, dtype=bool)
    susceptible = np.ones(N, dtype=bool)

    seeds = rng.choice(N, n_seeds, replace=False)
    infected[seeds]    = True
    susceptible[seeds] = False

    I_series = np.empty(n_steps + 1)
    I_series[0] = n_seeds / N

    for t in range(n_steps):
        if not infected.any():
            I_series[t + 1:] = 0.0
            break

        # Transmission along I→S edges
        mask = infected[src] & susceptible[dst]
        idx  = np.where(mask)[0]
        newly_infected = np.zeros(N, dtype=bool)
        if idx.size:
            fires = rng.random(idx.size) < beta
            newly_infected[np.unique(dst[idx[fires]])] = True

        # Recovery → back to susceptible
        newly_recovered = infected & (rng.random(N) < mu)

        # Synchronous update
        infected    = (infected | newly_infected) & ~newly_recovered
        susceptible = (susceptible | newly_recovered) & ~newly_infected

        I_series[t + 1] = infected.sum() / N

    return I_series


def _endemic_prevalence(
    src, dst, N, beta, mu, n_runs, base_seed=0
) -> float:
    """
    Average I*/N over n_runs realisations.
    I* is the mean of I(t)/N over the last (T_RUN - T_STEADY) steps.
    """
    steady_vals = []
    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i)
        series = sis_run(src, dst, N, beta, mu, T_RUN, rng=rng)
        steady_vals.append(series[T_STEADY:].mean())
    return float(np.mean(steady_vals))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Endemic prevalence I*/N vs β/μ
# ─────────────────────────────────────────────────────────────────────────────

def fig_endemic_prevalence(
    graphs: dict[str, nx.Graph],
    mu: float = MU_DEFAULT,
    n_betas: int = 30,
    n_runs: int = 50,
) -> None:
    print("Figure 1 — endemic prevalence I*/N vs β/μ …")

    # β/μ range covers well past the highest threshold (lattice ~0.25)
    ratio_arr = np.linspace(0.0, 1.5, n_betas + 1)[1:]   # skip exact 0
    betas     = ratio_arr * mu

    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for (name, G), color in zip(graphs.items(), colors):
        src, dst = _edge_arrays(G)
        N  = G.number_of_nodes()
        k1, k2 = _degree_stats(G)
        bc = _threshold(G, mu)
        bc_ratio = bc / mu

        print(f"  {name}: ⟨k⟩={k1:.1f}, β_c={bc:.4f}, (β/μ)_c={bc_ratio:.3f}")

        I_stars = []
        for beta in betas:
            I_stars.append(_endemic_prevalence(src, dst, N, beta, mu, n_runs))
            sys.stdout.write(".")
            sys.stdout.flush()
        print()

        ax.plot(ratio_arr, I_stars, "o-", color=color,
                markersize=4, lw=1.8, label=name)

        # Epidemic threshold marker
        ax.axvline(bc_ratio, color=color, ls="--", lw=1.2, alpha=0.7)
        ax.text(bc_ratio + 0.01, 0.85 - list(graphs).index(name) * 0.12,
                f"β_c ({name.split()[0]})",
                color=color, fontsize=8.5, va="top")

    # Theoretical mean-field I*/N for ER (largest ⟨k⟩ baseline)
    G_er = graphs["ER"]
    k1_er, _ = _degree_stats(G_er)
    theo_ratio = np.linspace(1.0 / k1_er, ratio_arr[-1], 200)
    theo_Istar = np.maximum(0.0, 1.0 - 1.0 / (theo_ratio * k1_er))
    ax.plot(theo_ratio, theo_Istar, color="#2196F3", ls=":", lw=1.5, alpha=0.6,
            label="MF theory (ER)")

    ax.set_xlabel("β / μ", fontsize=13)
    ax.set_ylabel("Endemic prevalence  I* / N", fontsize=13)
    ax.set_title(
        f"SIS Endemic Prevalence  (μ={mu}, N={N_NODES}, {n_runs} realisations)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, ratio_arr[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "sis_endemic_prevalence.png"
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (name, G) in zip(axes, graphs.items()):
        src, dst = _edge_arrays(G)
        N  = G.number_of_nodes()
        bc = _threshold(G, mu)

        rep_betas = [0.5 * bc, bc, 3.0 * bc]
        beta_labels = [
            f"β=0.5 β_c  (sub-critical)",
            f"β=β_c       (critical)",
            f"β=3 β_c    (endemic)",
        ]

        for beta, lbl, color, ls in zip(rep_betas, beta_labels, colors, lstyles):
            # Average T_SERIES-step runs
            mat = np.zeros((avg_runs, T_SERIES + 1))
            for i in range(avg_runs):
                rng = np.random.default_rng(200 + i)
                mat[i] = sis_run(src, dst, N, beta, mu, T_SERIES, rng=rng)
            I_mean = mat.mean(axis=0)

            ax.plot(I_mean, color=color, ls=ls, lw=2.0,
                    label=f"β={beta:.4f}\n{lbl}")

        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (steps)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("I(t) / N", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(-0.01, 1.0)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"SIS Time Series — I(t)/N  (μ={mu}, avg of {avg_runs} runs)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out = OUT_DIR / "sis_time_series.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase diagram (β, μ) for ER
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_diagram(
    G_er: nx.Graph,
    n_grid: int = 20,
    n_runs: int = 30,
) -> None:
    print(f"Figure 3 — phase diagram  ({n_grid}×{n_grid}, {n_runs} runs each) …")

    src, dst = _edge_arrays(G_er)
    N = G_er.number_of_nodes()
    k1, k2 = _degree_stats(G_er)

    betas = np.linspace(0.005, 0.15, n_grid)
    mus   = np.linspace(0.02,  0.40, n_grid)

    # Istar_grid[i, j] = avg I*/N for (mus[i], betas[j])
    Istar_grid = np.zeros((n_grid, n_grid))

    for i, mu in enumerate(mus):
        for j, beta in enumerate(betas):
            Istar_grid[i, j] = _endemic_prevalence(
                src, dst, N, beta, mu, n_runs,
                base_seed=i * n_grid + j,
            )
        print(f"  μ={mu:.3f} done  (I*_max={Istar_grid[i].max():.2f})")

    # Theoretical threshold curve: β_c = μ·⟨k⟩/⟨k²⟩
    bc_line = mus * (k1 / k2)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(betas, mus, Istar_grid,
                       cmap="plasma", shading="auto", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Endemic prevalence  I* / N", fontsize=12)

    # Theoretical threshold
    in_range = bc_line <= betas[-1]
    ax.plot(bc_line[in_range], mus[in_range], "w--", lw=2.0,
            label=f"β_c = μ·⟨k⟩/⟨k²⟩  (⟨k⟩={k1:.1f})")

    ax.set_xlabel("β  (transmission rate)", fontsize=13)
    ax.set_ylabel("μ  (recovery rate)", fontsize=13)
    ax.set_title(
        f"SIS Phase Diagram — Erdős–Rényi  (N={N}, {n_runs} runs/point)\n"
        f"Dashed: heterogeneous mean-field threshold",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()

    out = OUT_DIR / "sis_phase_diagram.png"
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
        bc = _threshold(G, MU_DEFAULT)
        print(f"  {name:20s}: nodes={G.number_of_nodes()}, "
              f"edges={G.number_of_edges()}, "
              f"⟨k⟩={k1:.1f},  β_c(μ={MU_DEFAULT})={bc:.4f},  "
              f"(β/μ)_c={bc/MU_DEFAULT:.3f}")

    print(f"\nFixed: μ={MU_DEFAULT}, {N_SEEDS} seeds, "
          f"T_run={T_RUN}, steady-state avg over last {T_RUN-T_STEADY} steps\n")

    fig_endemic_prevalence(graphs)
    print()
    fig_time_series(graphs)
    print()
    fig_phase_diagram(graphs["ER"])

    print(f"\nAll outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
