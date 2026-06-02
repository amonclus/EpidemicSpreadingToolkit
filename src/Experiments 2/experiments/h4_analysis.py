#!/usr/bin/env python3
"""
h4_analysis.py — H4 (OR-Hybrid: SIS + Watts Threshold Model).

Each round a susceptible node becomes infected if EITHER:
  (1) SIS channel  : at least one infected neighbour transmits with prob β, OR
  (2) WTM channel  : infected_neighbours / degree(v) >= φ  (fractional threshold).
Infected nodes recover with prob μ and return to susceptible (SIS recovery —
no permanent immunity).

H4 is the SIS analogue of H1, with the absolute bootstrap threshold k replaced
by the degree-normalised WTM threshold φ.

Limiting cases:
  φ = 1.0  →  WTM channel almost never fires  →  effectively pure SIS.
  β = 0    →  SIS channel silent  →  effectively pure WTM (with recovery).

Fixed: μ=0.1, T=300 steps, I* = mean I(t)/N over last 50 steps, 5 seeds.

Produces three PNG figures:
  1. h4_endemic_prevalence.png — I*/N vs β for φ=0.1,0.2,0.3 + pure SIS,
                                 all 3 networks
  2. h4_time_series.png        — I(t)/N: φ=0.2 hybrid vs pure SIS for 3 β
                                 values (ER only)
  3. h4_phase_diagram.png      — 2-D heatmap of I*/N in (β, φ) for ER only

Usage:
    python src/experiments/h4_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("results/h4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
N_NODES  = 600
MU       = 0.1    # recovery rate
N_SEEDS  = 5      # fixed initial infected nodes
T_RUN    = 300    # total simulation steps
T_STEADY = 250    # start of steady-state window  (last 50 steps)
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
# H4 simulation — vectorised
# ─────────────────────────────────────────────────────────────────────────────

def h4_run(
    degrees: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    phi: float,
    beta: float,
    mu: float,
    n_seeds: int = N_SEEDS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Run H4 for T_RUN steps.  Returns I(t)/N array of length T_RUN+1.

    Each round (synchronous, start-of-round state used throughout):
      · SIS channel  : batch-sample transmissions along all I→S edges (prob β).
      · WTM channel  : activate every S node v where infected_count[v]/deg[v] >= φ.
      · OR union     : newly_infected = sis_newly | wtm_newly.
      · Recovery     : each I node independently recovers with prob μ → S.

    If the epidemic dies before T_RUN the series is zero-padded.
    """
    if rng is None:
        rng = np.random.default_rng()

    infected    = np.zeros(N, dtype=bool)
    susceptible = np.ones(N, dtype=bool)

    seeds = rng.choice(N, n_seeds, replace=False)
    infected[seeds]    = True
    susceptible[seeds] = False

    I_series = np.zeros(T_RUN + 1)
    I_series[0] = n_seeds / N

    for t in range(T_RUN):
        if not infected.any():
            # Epidemic died — pad the rest with zeros
            break

        # ── SIS channel ──────────────────────────────────────────────────
        mask = infected[src] & susceptible[dst]
        idx  = np.where(mask)[0]
        sis_newly = np.zeros(N, dtype=bool)
        if idx.size:
            fires = rng.random(idx.size) < beta
            sis_newly[np.unique(dst[idx[fires]])] = True

        # ── WTM channel ──────────────────────────────────────────────────
        infected_count = np.zeros(N, dtype=np.int32)
        np.add.at(infected_count, dst, infected[src].astype(np.int32))
        frac = np.where(degrees > 0, infected_count / degrees, 0.0)
        wtm_newly = susceptible & (degrees > 0) & (frac >= phi)

        # ── OR union ─────────────────────────────────────────────────────
        newly_infected = sis_newly | wtm_newly

        # ── Recovery → susceptible (SIS) ─────────────────────────────────
        newly_recovered = infected & (rng.random(N) < mu)

        infected    = (infected | newly_infected) & ~newly_recovered
        susceptible = (susceptible | newly_recovered) & ~newly_infected

        I_series[t + 1] = infected.sum() / N

    return I_series


def _endemic_prevalence(
    degrees, src, dst, N, phi, beta, mu, n_runs, base_seed=0
) -> float:
    """Average I*/N over n_runs; I* = mean of last (T_RUN−T_STEADY) steps."""
    vals = []
    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i)
        series = h4_run(degrees, src, dst, N, phi, beta, mu, rng=rng)
        vals.append(series[T_STEADY:].mean())
    return float(np.mean(vals))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — I*/N vs β  (φ = 0.1, 0.2, 0.3 + pure SIS;  all networks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_endemic_prevalence(
    graphs: dict[str, nx.Graph],
    phis: list[float] = [0.1, 0.2, 0.3],
    n_betas: int = 25,
    n_runs: int = 50,
) -> None:
    print("Figure 1 — I*/N vs β …")

    betas = np.linspace(0.0, 0.12, n_betas + 1)[1:]   # skip exact 0

    net_colors  = {"ER": "#2196F3", "BA (scale-free)": "#FF5722", "Lattice": "#4CAF50"}
    phi_styles  = {0.1: "-",  0.2: "--", 0.3: ":",  1.0: "-."}
    phi_markers = {0.1: "o",  0.2: "s",  0.3: "^",  1.0: "D"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, G in graphs.items():
        deg = _degree_array(G)
        src, dst = _edge_arrays(G)
        N  = G.number_of_nodes()
        k1, k2 = _degree_stats(G)
        color = net_colors[name]

        # H4 curves for each φ
        for phi in phis:
            I_stars = []
            for beta in betas:
                I_stars.append(_endemic_prevalence(deg, src, dst, N, phi, beta, MU, n_runs))
                sys.stdout.write(".")
                sys.stdout.flush()
            print(f"  {name} φ={phi}")

            ax.plot(betas, I_stars,
                    color=color, ls=phi_styles[phi],
                    marker=phi_markers[phi], markersize=4, lw=1.8,
                    label=f"{name}, φ={phi}")

        # Pure SIS overlay (φ=1 — WTM channel effectively disabled)
        sis_stars = []
        for beta in betas:
            sis_stars.append(_endemic_prevalence(deg, src, dst, N, 1.0, beta, MU, n_runs))
            sys.stdout.write(".")
            sys.stdout.flush()
        print(f"  {name} SIS (φ=1)")

        ax.plot(betas, sis_stars,
                color=color, ls=phi_styles[1.0],
                marker=phi_markers[1.0], markersize=4, lw=1.5, alpha=0.6,
                label=f"{name}, SIS (φ=1)")

        # SIS mean-field threshold marker
        beta_c = MU * k1 / k2
        if beta_c <= betas[-1]:
            ax.axvline(beta_c, color=color, lw=0.8, ls=":", alpha=0.35)

    ax.set_xlabel("Transmission rate  β", fontsize=13)
    ax.set_ylabel("Endemic prevalence  I* / N", fontsize=13)
    ax.set_title(
        f"H4 (SIS + WTM OR-Hybrid) — I*/N vs β\n"
        f"(μ={MU}, N={N_NODES}, {N_SEEDS} seeds, {n_runs} realisations)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, betas[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, ncol=4, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "h4_endemic_prevalence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Time series  (ER only, φ=0.2 hybrid vs pure SIS)
# ─────────────────────────────────────────────────────────────────────────────

def fig_time_series(
    G_er: nx.Graph,
    avg_runs: int = 15,
) -> None:
    print("Figure 2 — time series (ER only) …")

    deg = _degree_array(G_er)
    src, dst = _edge_arrays(G_er)
    N  = G_er.number_of_nodes()
    k1, k2 = _degree_stats(G_er)
    beta_c = MU * k1 / k2   # SIS mean-field threshold

    rep_betas = [0.5 * beta_c, beta_c, 4.0 * beta_c]
    beta_labels = [
        "β = 0.5 β_c  (sub-critical)",
        "β ≈ β_c       (critical)",
        "β = 4 β_c    (endemic)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors  = {"H4  φ=0.2": "#1565C0", "Pure SIS": "#C62828"}
    lstyles = {"H4  φ=0.2": "-",       "Pure SIS": "--"}

    for ax, beta, blabel in zip(axes, rep_betas, beta_labels):
        for model_name, phi in [("H4  φ=0.2", 0.2), ("Pure SIS", 1.0)]:
            mat = np.zeros((avg_runs, T_RUN + 1))
            for i in range(avg_runs):
                rng = np.random.default_rng(400 + i)
                mat[i] = h4_run(deg, src, dst, N, phi, beta, MU, rng=rng)
            I_mean = mat.mean(axis=0)

            ax.plot(I_mean,
                    color=colors[model_name], ls=lstyles[model_name],
                    lw=2.2, label=model_name)

        # Mark steady-state window
        ax.axvspan(T_STEADY, T_RUN, color="gray", alpha=0.08,
                   label="I* window" if ax is axes[0] else None)

        ax.set_title(blabel, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (steps)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("I(t) / N", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(-0.01, 1.0)
        ax.set_xlim(0, T_RUN)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"H4 Time Series — ER  (μ={MU}, {N_SEEDS} seeds, avg of {avg_runs} runs)\n"
        f"Solid = H4 φ=0.2 hybrid,   Dashed = pure SIS (φ=1),   "
        f"Shaded = I* averaging window",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out = OUT_DIR / "h4_time_series.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase diagram (β, φ)  for ER
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_diagram(
    G_er: nx.Graph,
    n_betas: int = 20,
    n_phis: int = 20,
    n_runs: int = 30,
) -> None:
    print(f"Figure 3 — phase diagram  ({n_betas} β × {n_phis} φ,  "
          f"{n_runs} runs each) …")

    deg = _degree_array(G_er)
    src, dst = _edge_arrays(G_er)
    N  = G_er.number_of_nodes()
    k1, k2 = _degree_stats(G_er)

    # β range centred on the SIS threshold (β_c ≈ 0.011 for this graph).
    # Going to 0.5 saturates everything; 0.04 (≈4×β_c) reveals the transition.
    betas = np.linspace(0.0,  0.04, n_betas + 1)[1:]   # skip exact 0
    phis  = np.linspace(0.05, 0.5,  n_phis)

    # Istar_grid[i, j] = avg I*/N for (phis[i], betas[j])
    Istar_grid = np.zeros((n_phis, n_betas))

    for i, phi in enumerate(phis):
        for j, beta in enumerate(betas):
            Istar_grid[i, j] = _endemic_prevalence(
                deg, src, dst, N, phi, beta, MU, n_runs,
                base_seed=i * n_betas + j,
            )
        print(f"  φ={phi:.2f} done  (I*_max={Istar_grid[i].max():.2f})")

    # SIS mean-field threshold (independent of φ)
    beta_c = MU * k1 / k2

    fig, ax = plt.subplots(figsize=(8, 6))

    beta_edges = np.concatenate([
        [betas[0] - (betas[1] - betas[0]) / 2],
        (betas[:-1] + betas[1:]) / 2,
        [betas[-1] + (betas[-1] - betas[-2]) / 2],
    ])
    phi_edges = np.concatenate([
        [phis[0] - (phis[1] - phis[0]) / 2],
        (phis[:-1] + phis[1:]) / 2,
        [phis[-1] + (phis[-1] - phis[-2]) / 2],
    ])

    im = ax.pcolormesh(
        beta_edges, phi_edges, Istar_grid,
        cmap="plasma", shading="flat", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Endemic prevalence  I* / N", fontsize=12)

    # I* = 0.1 and 0.5 contours
    try:
        cs = ax.contour(
            betas, phis, Istar_grid,
            levels=[0.1, 0.5], colors=["white", "cyan"],
            linewidths=[1.5, 2.0], linestyles=[":", "--"],
        )
        ax.clabel(cs, fmt={0.1: "I*=0.1", 0.5: "I*=0.5"},
                  fontsize=9, colors="white")
    except Exception:
        pass

    # SIS threshold vertical line
    ax.axvline(beta_c, color="white", lw=1.8, ls=":",
               label=f"SIS β_c ≈ {beta_c:.4f}")

    # Annotate WTM regime boundary: φ_c = 1/⟨k⟩
    phi_c_wtm = 1.0 / k1
    if phis[0] <= phi_c_wtm <= phis[-1]:
        ax.axhline(phi_c_wtm, color="yellow", lw=1.5, ls=":",
                   label=f"WTM φ_c ≈ 1/⟨k⟩ = {phi_c_wtm:.3f}")

    ax.set_xlabel("Transmission rate  β", fontsize=13)
    ax.set_ylabel("WTM threshold  φ", fontsize=13)
    ax.set_title(
        f"H4 Phase Diagram — Erdős–Rényi  (N={N}, μ={MU}, {N_SEEDS} seeds)\n"
        f"({n_runs} realisations per cell)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    out = OUT_DIR / "h4_phase_diagram.png"
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
        beta_c = MU * k1 / k2
        print(f"  {name:20s}: nodes={G.number_of_nodes()}, "
              f"edges={G.number_of_edges()}, "
              f"⟨k⟩={k1:.1f},  SIS β_c={beta_c:.4f},  "
              f"WTM φ_c≈{1/k1:.3f}")

    print(f"\nFixed: μ={MU}, T={T_RUN} steps, "
          f"I* = mean I(t)/N over steps {T_STEADY}–{T_RUN}, "
          f"{N_SEEDS} seeds\n")

    fig_endemic_prevalence(graphs)
    print()
    fig_time_series(graphs["ER"])
    print()
    fig_phase_diagram(graphs["ER"])

    print(f"\nAll outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
