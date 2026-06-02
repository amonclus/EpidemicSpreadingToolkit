#!/usr/bin/env python3
"""
h6_analysis.py — H6 (Probabilistic Threshold Hybrid: Soft WTM + SIR recovery).

Each round (synchronous):
  1. Infection: susceptible node v with m infected neighbours and degree d is
     infected with probability  min(1.0, (m/d) / phi).
     At m/d == phi the probability saturates to 1 (matching hard WTM).
     Below phi it scales linearly — "soft" reinforcement.
  2. Recovery: each infected node recovers permanently with probability gamma
     (SIR — no return to susceptible).
Runs until no infected nodes remain (or T_MAX steps).

Because recovery is permanent the long-run metric is the cascade fraction
R_final/N = (ever-infected) / N, not endemic prevalence.

Fixed: T_MAX=300 steps, 5 seeds.

Produces three PNG figures:
  1. h6_cascade_fraction.png — R_final/N vs phi for gamma=0.1, 0.3, 0.5,
                               all 3 networks
  2. h6_time_series.png      — I(t)/N for phi=0.10, 0.35, 0.60 with gamma=0.2,
                               ER only; epidemic peak marked
  3. h6_phase_diagram.png    — 2-D heatmap of R_final/N in (phi, gamma) for ER

Usage:
    python src/experiments/h6_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("results/h6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
N_NODES  = 600
N_SEEDS  = 5      # fixed initial infected nodes
T_MAX    = 300    # max simulation steps per run
SEED     = 42

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
    N = G.number_of_nodes()
    deg = np.zeros(N, dtype=np.float32)
    for node, d in G.degree():
        deg[node] = d
    return deg


def _degree_stats(G: nx.Graph) -> tuple[float, float]:
    deg = np.array([d for _, d in G.degree()])
    return float(deg.mean()), float((deg ** 2).mean())


# ─────────────────────────────────────────────────────────────────────────────
# H6 simulation — vectorised
# ─────────────────────────────────────────────────────────────────────────────

def h6_run(
    degrees: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    phi: float,
    gamma: float,
    n_seeds: int = N_SEEDS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """
    Run one H6 realisation.

    Each step (start-of-round state used for both sub-steps):
      · Infection : P(v infected) = min(1, (inf_count[v] / deg[v]) / phi)
      · Recovery  : each I node recovers permanently with prob gamma

    Returns:
        I_series       : I(t)/N array (length ≤ T_MAX + 1)
        cascade_fraction: (ever infected) / N at termination
    """
    if rng is None:
        rng = np.random.default_rng()

    infected    = np.zeros(N, dtype=bool)
    susceptible = np.ones(N, dtype=bool)

    seeds = rng.choice(N, n_seeds, replace=False)
    infected[seeds]    = True
    susceptible[seeds] = False

    I_series: list[float] = [n_seeds / N]

    for _ in range(T_MAX):
        if not infected.any():
            break

        # ── Infection: soft WTM probability ──────────────────────────────
        inf_cnt = np.zeros(N, dtype=np.int32)
        np.add.at(inf_cnt, dst, infected[src].astype(np.int32))

        # Only susceptible nodes with at least one infected neighbour matter
        raw_prob = np.where(
            degrees > 0,
            np.minimum(1.0, (inf_cnt / np.maximum(degrees, 1.0)) / phi),
            0.0,
        )
        new_inf = susceptible & (rng.random(N) < raw_prob)

        # ── Recovery: permanent (SIR) ────────────────────────────────────
        new_rec = infected & (rng.random(N) < gamma)

        infected    = (infected | new_inf) & ~new_rec
        susceptible = susceptible & ~new_inf

        I_series.append(infected.sum() / N)

    cascade_fraction = 1.0 - susceptible.sum() / N  # ever infected = I + R_final
    return np.array(I_series), cascade_fraction


def _avg_cascade(
    degrees, src, dst, N, phi, gamma, n_runs, base_seed=0
) -> float:
    """Average cascade fraction over n_runs realisations."""
    vals = []
    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i)
        _, cf = h6_run(degrees, src, dst, N, phi, gamma, rng=rng)
        vals.append(cf)
    return float(np.mean(vals))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Cascade fraction vs phi  (gamma=0.1, 0.3, 0.5;  all networks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_cascade_fraction(
    graphs: dict[str, nx.Graph],
    gammas: list[float] = [0.3, 0.5, 0.7],
    n_phis: int = 20,
    n_runs: int = 50,
) -> None:
    print("Figure 1 — Cascade fraction vs phi …")

    # phi_c = 1/((k-1)*gamma).  Lattice(k=4) needs phi up to ~1.5 for gamma=0.3,
    # and BA hubs shift the threshold higher than ER.  Extend range to 1.5.
    phis = np.linspace(0.05, 1.5, n_phis)

    net_colors    = {"ER": "#2196F3", "BA (scale-free)": "#FF5722", "Lattice": "#4CAF50"}
    gamma_styles  = {0.3: "-",  0.5: "--", 0.7: ":"}
    gamma_markers = {0.3: "o",  0.5: "s",  0.7: "^"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, G in graphs.items():
        deg = _degree_array(G)
        src, dst = _edge_arrays(G)
        N   = G.number_of_nodes()
        k1, _ = _degree_stats(G)
        color = net_colors[name]

        for gamma in gammas:
            cfs = []
            for phi in phis:
                cfs.append(_avg_cascade(deg, src, dst, N, phi, gamma, n_runs))
                sys.stdout.write(".")
                sys.stdout.flush()
            print(f"  {name} γ={gamma}")

            ax.plot(
                phis, cfs,
                color=color, ls=gamma_styles[gamma],
                marker=gamma_markers[gamma], markersize=4, lw=1.8,
                label=f"{name}, γ={gamma}",
            )

        # Approximate threshold line  phi_c ≈ 1 / ((k-1) * gamma_mid)
        for gamma in gammas:
            phi_c = 1.0 / ((k1 - 1) * gamma)
            if phis[0] <= phi_c <= phis[-1]:
                ax.axvline(phi_c, color=net_colors[name], lw=0.7, ls=":",
                           alpha=0.35)

    ax.set_xlabel("Soft WTM threshold  φ", fontsize=13)
    ax.set_ylabel("Cascade fraction  R_final / N", fontsize=13)
    ax.set_title(
        f"H6 (Soft WTM + SIR) — Cascade Fraction vs φ\n"
        f"(N={N_NODES}, {N_SEEDS} seeds, {n_runs} realisations)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(phis[0], phis[-1])
    ax.axvline(1.0, color="gray", lw=0.8, ls=":", alpha=0.5)  # hard WTM threshold reference
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "h6_cascade_fraction.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Time series  (ER only, phi = 0.10 / 0.35 / 0.60, gamma=0.2)
# ─────────────────────────────────────────────────────────────────────────────

def fig_time_series(
    G_er: nx.Graph,
    gamma: float = 0.4,
    phis: list[float] = [0.10, 0.36, 0.70],
    avg_runs: int = 15,
) -> None:
    print("Figure 2 — time series (ER only) …")

    deg = _degree_array(G_er)
    src, dst = _edge_arrays(G_er)
    N = G_er.number_of_nodes()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, phi in zip(axes, phis):
        all_series: list[np.ndarray] = []

        for i in range(avg_runs):
            rng = np.random.default_rng(800 + i)
            series, _ = h6_run(deg, src, dst, N, phi, gamma, rng=rng)
            all_series.append(series)

        # Pad to same length for averaging (pad with 0 — epidemic has died)
        max_len = max(len(s) for s in all_series)
        padded  = np.zeros((avg_runs, max_len))
        for i, s in enumerate(all_series):
            padded[i, : len(s)] = s

        I_mean = padded.mean(axis=0)
        I_std  = padded.std(axis=0)
        steps  = np.arange(max_len)

        ax.plot(steps, I_mean, color="#1565C0", lw=2.2, label=f"φ = {phi}")
        ax.fill_between(
            steps, np.maximum(0, I_mean - I_std), I_mean + I_std,
            alpha=0.20, color="#1565C0",
        )

        # Mark epidemic peak
        peak_t = int(np.argmax(I_mean))
        ax.axvline(peak_t, color="#C62828", lw=1.5, ls="--",
                   label=f"peak  t = {peak_t}")
        ax.scatter([peak_t], [I_mean[peak_t]], color="#C62828", zorder=5, s=50)

        ax.set_title(f"φ = {phi}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (steps)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("I(t) / N", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(-0.01, 1.0)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"H6 Time Series — ER  (γ={gamma}, {N_SEEDS} seeds, avg of {avg_runs} runs)\n"
        f"Dashed red = mean epidemic peak",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out = OUT_DIR / "h6_time_series.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase diagram  (phi, gamma)  for ER
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_diagram(
    G_er: nx.Graph,
    n_phis: int = 20,
    n_gammas: int = 20,
    n_runs: int = 30,
) -> None:
    print(f"Figure 3 — phase diagram  ({n_phis} φ × {n_gammas} γ,"
          f"  {n_runs} runs each) …")

    deg = _degree_array(G_er)
    src, dst = _edge_arrays(G_er)
    N   = G_er.number_of_nodes()
    k1, _ = _degree_stats(G_er)

    phis   = np.linspace(0.0, 2.0, n_phis   + 1)[1:]   # extend to 2.0; threshold curve enters at ~phi=1.4 for gamma=0.1
    gammas = np.linspace(0.0, 1.0, n_gammas + 1)[1:]   # extend to 1.0; phi_c drops to 0.14 at gamma=1.0

    # grid[i, j] = avg cascade fraction for (phis[i], gammas[j])
    grid = np.zeros((n_phis, n_gammas))

    for i, phi in enumerate(phis):
        for j, gamma in enumerate(gammas):
            grid[i, j] = _avg_cascade(
                deg, src, dst, N, phi, gamma, n_runs,
                base_seed=i * n_gammas + j,
            )
        print(f"  φ={phi:.2f} done  (cf_max={grid[i].max():.2f})")

    fig, ax = plt.subplots(figsize=(8, 6))

    phi_edges = np.concatenate([
        [phis[0]   - (phis[1]   - phis[0])   / 2],
        (phis[:-1]   + phis[1:])   / 2,
        [phis[-1]   + (phis[-1]   - phis[-2])   / 2],
    ])
    gamma_edges = np.concatenate([
        [gammas[0] - (gammas[1] - gammas[0]) / 2],
        (gammas[:-1] + gammas[1:]) / 2,
        [gammas[-1] + (gammas[-1] - gammas[-2]) / 2],
    ])

    im = ax.pcolormesh(
        phi_edges, gamma_edges, grid.T,
        cmap="plasma", shading="flat", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cascade fraction  R_final / N", fontsize=12)

    # Cascade-size contours
    try:
        cs = ax.contour(
            phis, gammas, grid.T,
            levels=[0.1, 0.5], colors=["white", "cyan"],
            linewidths=[1.5, 2.0], linestyles=[":", "--"],
        )
        ax.clabel(cs, fmt={0.1: "cf=0.1", 0.5: "cf=0.5"},
                  fontsize=9, colors="white")
    except Exception:
        pass

    # Theoretical epidemic threshold curve: phi_c = 1 / ((k-1) * gamma)
    gamma_line = np.linspace(gammas[0], gammas[-1], 400)
    phi_c_line = 1.0 / ((k1 - 1) * gamma_line)
    mask = phi_c_line <= phis[-1]
    if mask.any():
        ax.plot(phi_c_line[mask], gamma_line[mask],
                color="yellow", lw=1.8, ls="--",
                label=f"threshold  φ_c = 1/((⟨k⟩-1)·γ)")

    ax.set_xlabel("Soft WTM threshold  φ", fontsize=13)
    ax.set_ylabel("Recovery rate  γ", fontsize=13)
    ax.set_title(
        f"H6 Phase Diagram — Erdős–Rényi  (N={N}, {N_SEEDS} seeds)\n"
        f"({n_runs} realisations per cell)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    out = OUT_DIR / "h6_phase_diagram.png"
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
        # Rough threshold: phi_c(gamma) = 1/((k1-1)*gamma)
        print(f"  {name:20s}: nodes={G.number_of_nodes()}, "
              f"edges={G.number_of_edges()}, "
              f"⟨k⟩={k1:.1f},  φ_c(γ=0.2) ≈ {1/((k1-1)*0.2):.3f}")

    print(f"\nFixed: T_MAX={T_MAX} steps, {N_SEEDS} seeds\n")

    fig_cascade_fraction(graphs)
    print()
    fig_time_series(graphs["ER"])
    print()
    fig_phase_diagram(graphs["ER"])

    print(f"\nAll outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
