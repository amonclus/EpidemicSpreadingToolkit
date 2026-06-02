#!/usr/bin/env python3
"""
h5_analysis.py — H5 (Sequential Hybrid: SIS → WTM with recovery).

Phase 1 — SIS mode: the network runs SIS dynamics (beta, mu=0.1).
  Switch condition: when simultaneously infected fraction I(t)/N >= f,
  transition to Phase 2.

Phase 2 — WTM mode with SIS recovery: new infections follow WTM rules
  (susceptible node v infected if infected_count(v)/degree(v) >= phi), but
  infected nodes still recover with prob mu per timestep.
  Phase 2 always runs for T2=200 additional timesteps.

Endemic prevalence: I* = mean I(t)/N over last 50 timesteps of Phase 2.
If f is never reached, Phase 2 continues with SIS dynamics; I* from last 50 steps.

Fixed: mu=0.1, phi=0.2, T1_MAX=300 steps, T2=200 steps, 5 seeds.

Produces three PNG figures:
  1. h5_endemic_prevalence.png — I*/N vs f for beta=0.1, 0.2, 0.3, all 3 networks
  2. h5_time_series.png        — I(t)/N for f=0.05, 0.15, 0.30 with beta=0.2, ER only
  3. h5_phase_diagram.png      — 2D heatmap of I*/N in (f, beta) for ER only

Usage:
    python src/experiments/h5_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("results/h5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global constants ──────────────────────────────────────────────────────────
N_NODES  = 600
MU       = 0.3    # recovery rate (both phases)
PHI      = 0.4    # fixed WTM threshold (Phase 2)
N_SEEDS  = 5      # fixed initial infected nodes
T1_MAX   = 300    # max Phase 1 (SIS) steps
T2       = 200    # Phase 2 steps (always runs)
T_STEADY = 50     # I* = mean of last T_STEADY steps
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
# H5 simulation — vectorised
# ─────────────────────────────────────────────────────────────────────────────

def h5_run(
    degrees: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    N: int,
    f: float,
    phi: float,
    beta: float,
    mu: float,
    n_seeds: int = N_SEEDS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int | None]:
    """
    Run one H5 realisation.

    Phase 1 — SIS for at most T1_MAX steps, stopping when I(t)/N >= f.
    Phase 2 — T2 steps: WTM infection rule + mu recovery if switch triggered,
              otherwise continues as SIS.

    Returns:
        I_series: I(t)/N array, length = (Phase 1 steps + 1) + T2.
        switch_t: index in I_series when Phase 2 began (None if f never reached).
    """
    if rng is None:
        rng = np.random.default_rng()

    infected    = np.zeros(N, dtype=bool)
    susceptible = np.ones(N, dtype=bool)

    seeds = rng.choice(N, n_seeds, replace=False)
    infected[seeds]    = True
    susceptible[seeds] = False

    I_list: list[float] = [n_seeds / N]
    switch_t: int | None = None

    # ── Phase 1: SIS ─────────────────────────────────────────────────────
    for t in range(T1_MAX):
        if not infected.any():
            break

        # Switch check at current state (index t in I_list)
        if infected.sum() / N >= f:
            switch_t = t
            break

        # SIS step
        mask = infected[src] & susceptible[dst]
        idx  = np.where(mask)[0]
        new_inf = np.zeros(N, dtype=bool)
        if idx.size:
            fires = rng.random(idx.size) < beta
            new_inf[np.unique(dst[idx[fires]])] = True

        new_rec = infected & (rng.random(N) < mu)
        infected    = (infected | new_inf) & ~new_rec
        susceptible = (susceptible | new_rec) & ~new_inf
        I_list.append(infected.sum() / N)

    # ── Phase 2: T2 steps ────────────────────────────────────────────────
    for _ in range(T2):
        if switch_t is not None:
            # WTM infection rule: susceptible v infected if inf_count/deg >= phi
            inf_cnt = np.zeros(N, dtype=np.int32)
            np.add.at(inf_cnt, dst, infected[src].astype(np.int32))
            frac    = np.where(degrees > 0, inf_cnt / degrees, 0.0)
            new_inf = susceptible & (degrees > 0) & (frac >= phi)
        else:
            # f never reached: continue SIS
            mask = infected[src] & susceptible[dst]
            idx  = np.where(mask)[0]
            new_inf = np.zeros(N, dtype=bool)
            if idx.size:
                fires = rng.random(idx.size) < beta
                new_inf[np.unique(dst[idx[fires]])] = True

        new_rec = infected & (rng.random(N) < mu)
        infected    = (infected | new_inf) & ~new_rec
        susceptible = (susceptible | new_rec) & ~new_inf
        I_list.append(infected.sum() / N)

    return np.array(I_list), switch_t


def _endemic_prevalence(
    degrees, src, dst, N, f, phi, beta, mu, n_runs, base_seed=0
) -> float:
    """Average I*/N over n_runs; I* = mean of last T_STEADY steps."""
    vals = []
    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i)
        series, _ = h5_run(degrees, src, dst, N, f, phi, beta, mu, rng=rng)
        vals.append(series[-T_STEADY:].mean())
    return float(np.mean(vals))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — I*/N vs f  (beta = 0.1, 0.2, 0.3;  all networks)
# ─────────────────────────────────────────────────────────────────────────────

def fig_endemic_prevalence(
    graphs: dict[str, nx.Graph],
    betas: list[float] = [0.1, 0.2, 0.3],
    n_fs: int = 20,
    n_runs: int = 50,
    phi: float = PHI,
) -> None:
    print("Figure 1 — I*/N vs f …")

    fs = np.linspace(0.02, 0.50, n_fs)

    net_colors   = {"ER": "#2196F3", "BA (scale-free)": "#FF5722", "Lattice": "#4CAF50"}
    beta_styles  = {0.1: "-",  0.2: "--", 0.3: ":"}
    beta_markers = {0.1: "o",  0.2: "s",  0.3: "^"}

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, G in graphs.items():
        deg = _degree_array(G)
        src, dst = _edge_arrays(G)
        N   = G.number_of_nodes()
        color = net_colors[name]

        for beta in betas:
            I_stars = []
            for f in fs:
                I_stars.append(
                    _endemic_prevalence(deg, src, dst, N, f, phi, beta, MU, n_runs)
                )
                sys.stdout.write(".")
                sys.stdout.flush()
            print(f"  {name} β={beta}")

            ax.plot(
                fs, I_stars,
                color=color, ls=beta_styles[beta],
                marker=beta_markers[beta], markersize=4, lw=1.8,
                label=f"{name}, β={beta}",
            )

    ax.set_xlabel("Switching fraction  f", fontsize=13)
    ax.set_ylabel("Endemic prevalence  I* / N", fontsize=13)
    ax.set_title(
        f"H5 (SIS → WTM) — Endemic Prevalence vs Switching Fraction\n"
        f"(μ={MU}, φ={phi}, N={N_NODES}, {N_SEEDS} seeds, {n_runs} realisations)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, fs[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "h5_endemic_prevalence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Time series  (ER only, f = 0.05 / 0.15 / 0.30, beta=0.2)
# ─────────────────────────────────────────────────────────────────────────────

def fig_time_series(
    G_er: nx.Graph,
    beta: float = 0.2,
    fs: list[float] = [0.05, 0.15, 0.30],
    avg_runs: int = 15,
    phi: float = PHI,
) -> None:
    print("Figure 2 — time series (ER only) …")

    deg = _degree_array(G_er)
    src, dst = _edge_arrays(G_er)
    N = G_er.number_of_nodes()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, f in zip(axes, fs):
        all_series: list[np.ndarray] = []
        switch_times: list[int] = []

        for i in range(avg_runs):
            rng = np.random.default_rng(700 + i)
            series, switch_t = h5_run(deg, src, dst, N, f, phi, beta, MU, rng=rng)
            all_series.append(series)
            if switch_t is not None:
                switch_times.append(switch_t)

        # Pad all series to the same length for averaging
        max_len = max(len(s) for s in all_series)
        padded  = np.zeros((avg_runs, max_len))
        for i, s in enumerate(all_series):
            padded[i, : len(s)] = s
            if len(s) < max_len:
                padded[i, len(s):] = s[-1]   # hold last value

        I_mean = padded.mean(axis=0)
        I_std  = padded.std(axis=0)
        steps  = np.arange(max_len)

        ax.plot(steps, I_mean, color="#1565C0", lw=2.2, label=f"f = {f}")
        ax.fill_between(
            steps, I_mean - I_std, I_mean + I_std,
            alpha=0.20, color="#1565C0",
        )

        # Mark mean switching point
        if switch_times:
            mean_sw = float(np.mean(switch_times))
            ax.axvline(mean_sw, color="#C62828", lw=1.8, ls="--",
                       label=f"switch  t ≈ {mean_sw:.0f}")
        else:
            ax.text(0.5, 0.95, "f never reached", transform=ax.transAxes,
                    ha="center", va="top", color="#C62828", fontsize=9)

        # Mark I* averaging window (last T_STEADY steps)
        ax.axvspan(max_len - T_STEADY, max_len, color="gray", alpha=0.10,
                   label="I* window" if ax is axes[0] else None)

        ax.set_title(f"f = {f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (steps)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("I(t) / N", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(-0.01, 1.0)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"H5 Time Series — ER  (β={beta}, μ={MU}, φ={phi}, {N_SEEDS} seeds,"
        f" avg of {avg_runs} runs)\n"
        f"Dashed red = mean Phase-2 switch point,   "
        f"Shaded grey = I* averaging window",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out = OUT_DIR / "h5_time_series.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Phase diagram  (f, beta)  for ER
# ─────────────────────────────────────────────────────────────────────────────

def fig_phase_diagram(
    G_er: nx.Graph,
    n_fs: int = 20,
    n_betas: int = 20,
    n_runs: int = 30,
    phi: float = PHI,
) -> None:
    print(f"Figure 3 — phase diagram  ({n_fs} f × {n_betas} β,"
          f"  {n_runs} runs each) …")

    deg = _degree_array(G_er)
    src, dst = _edge_arrays(G_er)
    N   = G_er.number_of_nodes()
    k1, k2 = _degree_stats(G_er)

    fs    = np.linspace(0.0, 0.5, n_fs  + 1)[1:]   # skip exact 0
    betas = np.linspace(0.0, 0.5, n_betas + 1)[1:]

    # Istar_grid[i, j] = avg I*/N for (fs[i], betas[j])
    Istar_grid = np.zeros((n_fs, n_betas))

    for i, f in enumerate(fs):
        for j, beta in enumerate(betas):
            Istar_grid[i, j] = _endemic_prevalence(
                deg, src, dst, N, f, phi, beta, MU, n_runs,
                base_seed=i * n_betas + j,
            )
        print(f"  f={f:.2f} done  (I*_max={Istar_grid[i].max():.2f})")

    # SIS mean-field epidemic threshold
    beta_c = MU * k1 / k2

    fig, ax = plt.subplots(figsize=(8, 6))

    beta_edges = np.concatenate([
        [betas[0] - (betas[1] - betas[0]) / 2],
        (betas[:-1] + betas[1:]) / 2,
        [betas[-1] + (betas[-1] - betas[-2]) / 2],
    ])
    f_edges = np.concatenate([
        [fs[0] - (fs[1] - fs[0]) / 2],
        (fs[:-1] + fs[1:]) / 2,
        [fs[-1] + (fs[-1] - fs[-2]) / 2],
    ])

    im = ax.pcolormesh(
        beta_edges, f_edges, Istar_grid,
        cmap="plasma", shading="flat", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Endemic prevalence  I* / N", fontsize=12)

    # I* = 0.1 and 0.5 contours
    try:
        cs = ax.contour(
            betas, fs, Istar_grid,
            levels=[0.1, 0.5], colors=["white", "cyan"],
            linewidths=[1.5, 2.0], linestyles=[":", "--"],
        )
        ax.clabel(cs, fmt={0.1: "I*=0.1", 0.5: "I*=0.5"},
                  fontsize=9, colors="white")
    except Exception:
        pass

    # SIS mean-field threshold (vertical line, independent of f)
    ax.axvline(beta_c, color="white", lw=1.8, ls=":",
               label=f"SIS β_c ≈ {beta_c:.4f}")

    ax.set_xlabel("Transmission rate  β", fontsize=13)
    ax.set_ylabel("Switching fraction  f", fontsize=13)
    ax.set_title(
        f"H5 Phase Diagram — Erdős–Rényi  (N={N}, μ={MU}, φ={phi}, {N_SEEDS} seeds)\n"
        f"({n_runs} realisations per cell)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    out = OUT_DIR / "h5_phase_diagram.png"
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
              f"⟨k⟩={k1:.1f},  SIS β_c={beta_c:.4f}")

    print(
        f"\nFixed: μ={MU}, φ={PHI}, T1_MAX={T1_MAX} steps, T2={T2} steps, "
        f"I* = mean I(t)/N over last {T_STEADY} steps, {N_SEEDS} seeds\n"
    )

    fig_endemic_prevalence(graphs)
    print()
    fig_time_series(graphs["ER"])
    print()
    fig_phase_diagram(graphs["ER"])

    print(f"\nAll outputs written to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
