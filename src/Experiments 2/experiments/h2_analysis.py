#!/usr/bin/env python3
"""
h2_analysis.py — H2 (Sequential Hybrid / Switching) contagion model.

Phase 1 — SIR: standard SIR dynamics (β, γ) run until the ever-infected
fraction reaches the switch threshold f.
Phase 2 — Bootstrap: SIR stops.  All ever-infected nodes (I ∪ R) count
toward the bootstrap threshold k for susceptible neighbours.  Runs
deterministically until no new nodes can be activated.
If f is never reached the simulation runs as pure SIR.

Key parameter: f ∈ [0, 1] — switch threshold (f≈0 → near-pure bootstrap,
f→1 → near-pure SIR).

Fixed in this script: γ=0.1, k=2 (Phase 2 bootstrap threshold).

Produces three PNG figures:
  1. h2_rho_vs_f.png        — ρ_final vs f for β=0.1,0.2,0.3 and all networks
  2. h2_time_series.png     — I(t)/N + cumulative for f=0.1,0.3,0.5 (ER, β=0.2)
  3. h2_phase_diagram.png   — 2-D heatmap of ρ_final in (f, β) for ER only

Usage:
    python src/experiments/h2_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import networkx as nx

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("results/h2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
N_NODES  = 600
GAMMA    = 0.1   # recovery rate (Phase 1)
K_BOOT   = 2     # bootstrap threshold (Phase 2)
N_SEEDS  = 6     # initial infected nodes (~1 % of N)
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


# ─────────────────────────────────────────────────────────────────────────────
# H2 simulation — vectorised
# ─────────────────────────────────────────────────────────────────────────────

def h2_run(
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    k: int,
    beta: float,
    gamma: float,
    f: float,
    n_seeds: int = N_SEEDS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int | None, float]:
    """
    One H2 realisation.

    Phase 1 (SIR):
      · Each round: transmit along I→S edges with prob β, recover I with prob γ.
      · Before each round: if ever_infected/N >= f, switch to Phase 2.
      · If infected empties before f is reached, run ends as pure SIR.

    Phase 2 (Bootstrap):
      · active = all ever-infected nodes (I ∪ R) — they all count as 'signal'.
      · Synchronously activate every susceptible node with ≥ k active neighbours.
      · No SIR transmission or recovery in Phase 2.
      · Repeats until stable.

    Returns
    -------
    I_series   : I(t)/N at each time step (Phase 1 + Phase 2 concatenated)
    cum_series : ever_infected(t)/N — monotone non-decreasing
    switch_t   : time index when Phase 2 began (None if no switch occurred)
    rho_final  : total fraction ever infected
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

    I_series:   list[float] = []
    cum_series: list[float] = []
    switch_t: int | None = None

    # ── Phase 1: SIR ──────────────────────────────────────────────────────
    while True:
        n_i    = int(infected.sum())
        n_ever = int(ever_infected.sum())
        I_series.append(n_i / N)
        cum_series.append(n_ever / N)

        # Switch condition checked before running the round
        if n_ever / N >= f:
            switch_t = len(I_series) - 1
            break

        if n_i == 0:
            break   # epidemic died without hitting f → pure SIR result

        # SIR transmission
        mask = infected[src] & susceptible[dst]
        idx  = np.where(mask)[0]
        newly_infected = np.zeros(N, dtype=bool)
        if idx.size:
            fires = rng.random(idx.size) < beta
            newly_infected[np.unique(dst[idx[fires]])] = True

        # Recovery (start-of-round infected set)
        newly_recovered = infected & (rng.random(N) < gamma)

        infected    = (infected | newly_infected) & ~newly_recovered
        susceptible = susceptible & ~newly_infected
        ever_infected |= newly_infected

    # ── Phase 2: Bootstrap ────────────────────────────────────────────────
    if switch_t is not None:
        # All ever-infected (I ∪ R) serve as 'active' signal nodes
        active = ever_infected.copy()

        while True:
            infect_count = np.zeros(N, dtype=np.int32)
            np.add.at(infect_count, dst, active[src].astype(np.int32))

            newly = susceptible & (infect_count >= k)
            if not newly.any():
                break

            # Newly activated: S → I in Phase 2 (no recovery in this phase)
            susceptible   &= ~newly
            active        |= newly
            ever_infected |= newly
            infected      |= newly   # stay infected — no recovery in Phase 2

            I_series.append(infected.sum() / N)
            cum_series.append(ever_infected.sum() / N)

    rho = float(ever_infected.sum()) / N
    return np.array(I_series), np.array(cum_series), switch_t, rho


def _avg_rho(src, dst, N, k, beta, gamma, f, n_runs, base_seed=0):
    """Average ρ_final over n_runs realisations."""
    return float(np.mean([
        h2_run(src, dst, N, k, beta, gamma, f,
               rng=np.random.default_rng(base_seed + i))[3]
        for i in range(n_runs)
    ]))


def _avg_series(src, dst, N, k, beta, gamma, f, avg_runs, base_seed=0):
    """
    Average I(t)/N and cumulative series across avg_runs runs.
    Returns (I_mean, cum_mean, mean_switch_t).
    """
    all_I, all_cum, sw_times = [], [], []
    for i in range(avg_runs):
        I_t, cum_t, sw_t, _ = h2_run(
            src, dst, N, k, beta, gamma, f,
            rng=np.random.default_rng(base_seed + i),
        )
        all_I.append(I_t)
        all_cum.append(cum_t)
        if sw_t is not None:
            sw_times.append(sw_t)

    T = max(len(s) for s in all_I)
    I_mat   = np.zeros((avg_runs, T))
    cum_mat = np.zeros((avg_runs, T))

    for i in range(avg_runs):
        n = len(all_I[i])
        # Pad by holding the last value (epidemic has ended)
        I_mat[i, :n]   = all_I[i]
        I_mat[i, n:]   = all_I[i][-1]
        cum_mat[i, :n] = all_cum[i]
        cum_mat[i, n:] = all_cum[i][-1]

    mean_sw_t = float(np.mean(sw_times)) if sw_times else None
    return I_mat.mean(axis=0), cum_mat.mean(axis=0), mean_sw_t


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ρ_final vs f  (β = 0.1, 0.2, 0.3;  all networks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_rho_vs_f(
    graphs: dict[str, nx.Graph],
    betas: list[float] = [0.1, 0.2, 0.3],
    n_f: int = 25,
    n_runs: int = 50,
) -> None:
    print("Figure 1 — ρ_final vs f …")

    f_arr = np.linspace(0.0, 1.0, n_f)

    net_colors = {"ER": "#2196F3", "BA (scale-free)": "#FF5722", "Lattice": "#4CAF50"}
    beta_styles  = {0.1: "-", 0.2: "--", 0.3: ":"}
    beta_markers = {0.1: "o",  0.2: "s",  0.3: "^"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, G in graphs.items():
        src, dst = _edge_arrays(G)
        N  = G.number_of_nodes()
        color = net_colors[name]

        for beta in betas:
            rhos = []
            for f in f_arr:
                rhos.append(_avg_rho(src, dst, N, K_BOOT, beta, GAMMA, f, n_runs))
                sys.stdout.write(".")
                sys.stdout.flush()
            print(f"  {name} β={beta}")

            ax.plot(
                f_arr, rhos,
                color=color, ls=beta_styles[beta],
                marker=beta_markers[beta], markersize=4, lw=1.8,
                label=f"{name}, β={beta}",
            )

    # Axis labels and annotations
    ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(1, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.text(0.01, 0.03, "f≈0\n(≈bootstrap)", fontsize=8, color="gray", va="bottom")
    ax.text(0.99, 0.03, "f≈1\n(≈SIR)", fontsize=8, color="gray", va="bottom", ha="right")

    ax.set_xlabel("Switch threshold  f  (ever-infected fraction that triggers Phase 2)", fontsize=12)
    ax.set_ylabel("Final epidemic size  ρ_final", fontsize=13)
    ax.set_title(
        f"H2 (Sequential Hybrid) — ρ_final vs f\n"
        f"(γ={GAMMA}, k={K_BOOT}, N={N_NODES}, {n_runs} realisations per point)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "h2_rho_vs_f.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Time series  (ER only, β=0.2, f=0.1 / 0.3 / 0.5)
# ─────────────────────────────────────────────────────────────────────────────

def fig_time_series(
    G_er: nx.Graph,
    beta: float = 0.2,
    f_values: list[float] = [0.1, 0.3, 0.5],
    avg_runs: int = 15,
) -> None:
    print("Figure 2 — time series (ER, β=0.2) …")

    src, dst = _edge_arrays(G_er)
    N = G_er.number_of_nodes()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, f in zip(axes, f_values):
        I_mean, cum_mean, mean_sw_t = _avg_series(
            src, dst, N, K_BOOT, beta, GAMMA, f,
            avg_runs=avg_runs, base_seed=600,
        )
        t = np.arange(len(I_mean))

        ax.plot(t, I_mean,   color="#1565C0", lw=2.2, ls="-",  label="I(t) / N  (infected)")
        ax.plot(t, cum_mean, color="#B71C1C", lw=2.2, ls="--", label="Cumulative ever-infected / N")

        # Mark switching point
        if mean_sw_t is not None:
            sw_idx = int(round(mean_sw_t))
            ax.axvline(sw_idx, color="black", lw=1.5, ls=":",
                       label=f"Switch  (t≈{sw_idx})")
            # Mark on cumulative curve
            if sw_idx < len(cum_mean):
                ax.plot(sw_idx, cum_mean[sw_idx], "ko", markersize=8, zorder=5)
            # Annotate f level
            ax.axhline(f, color="orange", lw=1.0, ls="-.", alpha=0.8,
                       label=f"f = {f}")

        ax.set_title(f"f = {f}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (steps)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("I(t) / N", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(-0.01, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"H2 Time Series — ER Network  (β={beta}, γ={GAMMA}, k={K_BOOT},  "
        f"avg of {avg_runs} runs)\n"
        f"Solid = I(t)/N,   Dashed = cumulative infected,   ● = switch point",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out = OUT_DIR / "h2_time_series.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase diagram (f, β)  for ER
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_diagram(
    G_er: nx.Graph,
    n_f: int = 20,
    n_betas: int = 20,
    n_runs: int = 30,
) -> None:
    print(f"Figure 3 — phase diagram  ({n_f} f × {n_betas} β,  {n_runs} runs each) …")

    src, dst = _edge_arrays(G_er)
    N = G_er.number_of_nodes()

    f_arr    = np.linspace(0.0, 1.0, n_f)
    # β range centred on the SIR threshold (β_c ≈ 0.012 for this graph).
    # Going to 0.5 saturates everything; 0.04 (≈3×β_c) reveals the transition.
    beta_arr = np.linspace(0.0, 0.04, n_betas + 1)[1:]   # skip exact 0

    # rho_grid[i, j] = avg ρ_final for (beta_arr[i], f_arr[j])
    rho_grid = np.zeros((n_betas, n_f))

    for i, beta in enumerate(beta_arr):
        for j, f in enumerate(f_arr):
            rho_grid[i, j] = _avg_rho(
                src, dst, N, K_BOOT, beta, GAMMA, f, n_runs,
                base_seed=i * n_f + j,
            )
        print(f"  β={beta:.3f} done  (ρ_max={rho_grid[i].max():.2f})")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Cell edges for pcolormesh
    f_edges = np.concatenate([
        [f_arr[0] - (f_arr[1] - f_arr[0]) / 2],
        (f_arr[:-1] + f_arr[1:]) / 2,
        [f_arr[-1] + (f_arr[1] - f_arr[0]) / 2],
    ])
    beta_edges = np.concatenate([
        [beta_arr[0] - (beta_arr[1] - beta_arr[0]) / 2],
        (beta_arr[:-1] + beta_arr[1:]) / 2,
        [beta_arr[-1] + (beta_arr[-1] - beta_arr[-2]) / 2],
    ])

    im = ax.pcolormesh(
        f_edges, beta_edges, rho_grid,
        cmap="plasma", shading="flat", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Final epidemic size  ρ_final", fontsize=12)

    # ρ = 0.5 contour as transition boundary
    try:
        cs = ax.contour(
            f_arr, beta_arr, rho_grid,
            levels=[0.5], colors=["white"], linewidths=[2.0], linestyles=["--"],
        )
        ax.clabel(cs, fmt="ρ=0.5", fontsize=9, colors="white")
    except Exception:
        pass

    # SIR threshold: β_c = γ / ⟨k⟩ (independent of f)
    k_mean = float(np.mean([d for _, d in G_er.degree()]))
    beta_c = GAMMA / k_mean
    ax.axhline(beta_c, color="cyan", lw=1.8, ls=":",
               label=f"SIR threshold  β_c≈{beta_c:.3f}")

    # Annotate regime boundaries
    ax.text(0.02, ax.get_ylim()[0] + 0.01, "← bootstrap dominant", fontsize=9,
            color="white", va="bottom")
    ax.text(0.98, ax.get_ylim()[0] + 0.01, "SIR dominant →", fontsize=9,
            color="white", va="bottom", ha="right")

    ax.set_xlabel("Switch threshold  f", fontsize=13)
    ax.set_ylabel("Transmission rate  β", fontsize=13)
    ax.set_title(
        f"H2 Phase Diagram — Erdős–Rényi  (N={N}, γ={GAMMA}, k={K_BOOT})\n"
        f"({n_runs} realisations per cell)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()

    out = OUT_DIR / "h2_phase_diagram.png"
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
        k_mean = float(np.mean([d for _, d in G.degree()]))
        print(f"  {name:20s}: nodes={G.number_of_nodes()}, "
              f"edges={G.number_of_edges()}, "
              f"⟨k⟩={k_mean:.1f},  SIR β_c={GAMMA/k_mean:.4f}")

    print(f"\nFixed parameters: γ={GAMMA}, k_bootstrap={K_BOOT}\n")
    fig_rho_vs_f(graphs)
    print()
    fig_time_series(graphs["ER"])
    print()
    fig_phase_diagram(graphs["ER"])

    print(f"\nAll outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
