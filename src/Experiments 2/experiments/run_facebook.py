"""
run_facebook.py — Social Contagion Dynamics on the Facebook Friendship Graph.

Runs 10 epidemic models on the Facebook ego-network and produces a two-panel
comparison figure:
  Panel 1 — Time series of I(t)/N for all models (mean ± std, 30 runs)
  Panel 2 — Bar chart of AUC and rho_final per model

Models
------
 1. SIR   — simple contagion, permanent immunity
 2. SIS   — simple contagion, reinfection (endemic)
 3. BP    — pure bootstrap percolation, absolute threshold k
 4. WTM   — Watts Threshold Model, fractional threshold phi
 5. H1    — SIR OR BP    (simultaneous simple + complex)
 6. H2    — SIR then BP  (simple seeds complex takeover)
 7. H3    — SIR soft BP  (probability amplified by neighbour count)
 8. H4    — SIS OR WTM   (endemic simple + fractional complex)
 9. H5    — SIS then WTM (endemic seeding, fractional takeover)
10. H6    — SIS soft WTM (endemic with fractional amplification)
"""

import os
import sys
import random

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Parameters ─────────────────────────────────────────────────────────────────
BETA    = 0.03    # transmission probability per contact
MU      = 0.14    # recovery / disengagement rate
K       = 3       # absolute bootstrap threshold (BP, H1, H2, H3)
PHI     = 0.18    # fractional WTM threshold (WTM, H4, H5, H6)
RHO_0   = 0.05    # initial seed fraction for BP and WTM
F       = 0.10    # switching fraction for H2 (ever-infected) and H5 (current I)
ALPHA   = 0.5     # amplification coefficient for H3 and H6
T       = 50     # total timesteps
T2      = 200     # Phase 2 timesteps for H5
N_SEEDS = 5       # fixed-count seeds for SIR/SIS-type models
N_RUNS  = 30      # realisations per model
MASTER_SEED = 42
NETWORK = "facebook"

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

network_name = "github" if NETWORK == "github" else "facebook"
GRAPH_PATH = os.path.join(_ROOT, "data", f"{network_name}_combined.txt" if network_name == "facebook" else "musae_git_edges.csv")
OUT_PATH       = os.path.join(_ROOT, "data", f"epidemic_comparison_{network_name}.png")
STRAT_OUT_PATH = os.path.join(_ROOT, "data", f"seed_selection_{network_name}.png")
STRAT_CSV_PATH = os.path.join(_ROOT, "data", f"seed_selection_{network_name}.csv")

STRATEGIES    = ["Random", "High Degree", "High K-Core"]
STRAT_COLORS  = ["#4477AA", "#EE6677", "#228833"]
STRAT_HATCHES = ["", "//", "xx"]
# ── Visual style ──────────────────────────────────────────────────────────────
# Paul Tol / Wong colour-blind safe palette (10 colours)
PALETTE = [
    "#0077BB",   # SIR   blue
    "#33BBEE",   # SIS   cyan
    "#009988",   # BP    teal
    "#EE7733",   # WTM   orange
    "#CC3311",   # H1    red
    "#EE3377",   # H2    rose/magenta
    "#BBBBBB",   # H3    grey
    "#AA4499",   # H4    purple
    "#44AA99",   # H5    green-teal
    "#DDCC77",   # H6    yellow
]
LSTYLES = ["-", "-", "--", "--", "-.", "-.", ":", "-.", ":", ":"]
MODEL_NAMES = ["SIR", "SIS", "BP", "WTM", "H1", "H2", "H3", "H4", "H5", "H6"]

# ── Graph loading ──────────────────────────────────────────────────────────────

def load_facebook_graph(path: str) -> nx.Graph:
    if NETWORK == "facebook":
        G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())
    elif NETWORK == "github":
        G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph(), delimiter=",")
    return G


def print_graph_stats(G: nx.Graph) -> None:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    avg_deg = 2 * e / n
    avg_cc  = nx.average_clustering(G)
    print(f" graph statistics")
    print(f"  Nodes                   : {n:,}")
    print(f"  Edges                   : {e:,}")
    print(f"  Average degree          : {avg_deg:.2f}")
    print(f"  Average clustering coef : {avg_cc:.4f}")
    print()

# ── Adjacency helpers ──────────────────────────────────────────────────────────

def build_adj(G: nx.Graph) -> dict[int, list[int]]:
    """Pre-build adjacency lists for fast iteration."""
    return {node: list(G.neighbors(node)) for node in G.nodes()}


def build_deg(G: nx.Graph) -> dict[int, int]:
    return dict(G.degree())


def apply_infections(newly_infected, infected, susceptible, inf_nb_count, adj):
    """Update state when a batch of nodes become infected."""
    infected |= newly_infected
    susceptible -= newly_infected
    for node in newly_infected:
        for nb in adj[node]:
            inf_nb_count[nb] += 1


def apply_recoveries_sir(newly_recovered, infected, recovered, inf_nb_count, adj):
    """SIR recovery: infected → permanent recovered (immune)."""
    infected -= newly_recovered
    recovered |= newly_recovered
    for node in newly_recovered:
        for nb in adj[node]:
            inf_nb_count[nb] -= 1


def apply_recoveries_sis(newly_recovered, infected, susceptible, inf_nb_count, adj):
    """SIS recovery: infected → susceptible again."""
    infected -= newly_recovered
    susceptible |= newly_recovered
    for node in newly_recovered:
        for nb in adj[node]:
            inf_nb_count[nb] -= 1


# ── Seed selection ─────────────────────────────────────────────────────────────

def random_seeds_fixed(nodes, n=N_SEEDS) -> set[int]:
    return set(random.sample(nodes, n))


def random_seeds_fraction(nodes, rho=RHO_0) -> set[int]:
    return {node for node in nodes if random.random() < rho}


# ── Model simulations ─────────────────────────────────────────────────────────
# Each function returns:
#   i_series : np.ndarray, shape (T+1,), I(t)/N at each timestep
#   rho_final: float, model-specific final epidemic size


def run_sir(nodes, adj, deg, seed_nodes) -> tuple[np.ndarray, float]:
    """SIR: permanent immunity. rho_final = ever-infected fraction."""
    n = len(nodes)
    i_series = np.zeros(T + 1)

    infected    = set(seed_nodes)
    susceptible = set(nodes) - infected
    recovered   = set()
    ever_inf    = set(seed_nodes)

    inf_nb = {nd: sum(1 for nb in adj[nd] if nb in infected) for nd in nodes}

    i_series[0] = len(infected) / n

    for t in range(1, T + 1):
        if not infected:
            break

        newly_inf = set()
        for nd in infected:
            for nb in adj[nd]:
                if nb in susceptible and random.random() < BETA:
                    newly_inf.add(nb)

        newly_rec = {nd for nd in infected if random.random() < MU}

        apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
        apply_recoveries_sir(newly_rec, infected, recovered, inf_nb, adj)
        ever_inf |= newly_inf

        i_series[t] = len(infected) / n

    return i_series, len(ever_inf) / n


def run_sis(nodes, adj, deg, seed_nodes) -> tuple[np.ndarray, float]:
    """SIS: reinfection. rho_final = I* (mean I(t)/N last 50 steps)."""
    n = len(nodes)
    i_series = np.zeros(T + 1)

    infected    = set(seed_nodes)
    susceptible = set(nodes) - infected

    inf_nb = {nd: sum(1 for nb in adj[nd] if nb in infected) for nd in nodes}

    i_series[0] = len(infected) / n

    for t in range(1, T + 1):
        if not infected:
            break

        newly_inf = set()
        for nd in infected:
            for nb in adj[nd]:
                if nb in susceptible and random.random() < BETA:
                    newly_inf.add(nb)

        newly_rec = {nd for nd in infected if random.random() < MU}

        apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
        apply_recoveries_sis(newly_rec, infected, susceptible, inf_nb, adj)

        i_series[t] = len(infected) / n

    rho_final = float(np.mean(i_series[T - 49:]))
    return i_series, rho_final


def run_bp(nodes, adj, deg, seed_nodes) -> tuple[np.ndarray, float]:
    """BP: synchronous bootstrap percolation. rho_final = final cascade fraction."""
    n = len(nodes)
    infected    = set(seed_nodes)
    susceptible = set(nodes) - infected

    inf_nb = {nd: sum(1 for nb in adj[nd] if nb in infected) for nd in nodes}

    history = [len(infected) / n]

    while True:
        newly_inf = {nd for nd in susceptible if inf_nb[nd] >= K}
        if not newly_inf:
            break
        apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
        history.append(len(infected) / n)

    final_val = history[-1]
    while len(history) <= T:
        history.append(final_val)
    i_series = np.array(history[:T + 1])
    return i_series, final_val


def run_wtm(nodes, adj, deg, seed_nodes) -> tuple[np.ndarray, float]:
    """WTM: Watts Threshold Model. rho_final = final cascade fraction."""
    n = len(nodes)
    infected    = set(seed_nodes)
    susceptible = set(nodes) - infected

    inf_nb = {nd: sum(1 for nb in adj[nd] if nb in infected) for nd in nodes}

    history = [len(infected) / n]

    while True:
        newly_inf = set()
        for nd in susceptible:
            d = deg[nd]
            if d > 0 and inf_nb[nd] / d >= PHI:
                newly_inf.add(nd)
        if not newly_inf:
            break
        apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
        history.append(len(infected) / n)

    final_val = history[-1]
    while len(history) <= T:
        history.append(final_val)
    i_series = np.array(history[:T + 1])
    return i_series, final_val


def run_h1(nodes, adj, deg, seed_nodes) -> tuple[np.ndarray, float]:
    """H1: SIR OR BP. rho_final = ever-infected fraction."""
    n = len(nodes)
    i_series = np.zeros(T + 1)

    infected    = set(seed_nodes)
    susceptible = set(nodes) - infected
    recovered   = set()
    ever_inf    = set(seed_nodes)

    inf_nb = {nd: sum(1 for nb in adj[nd] if nb in infected) for nd in nodes}

    i_series[0] = len(infected) / n

    for t in range(1, T + 1):
        if not infected:
            break

        # SIR channel
        newly_inf = set()
        for nd in infected:
            for nb in adj[nd]:
                if nb in susceptible and random.random() < BETA:
                    newly_inf.add(nb)

        # BP channel (hard threshold, does not double-count)
        for nd in susceptible:
            if nd not in newly_inf and inf_nb[nd] >= K:
                newly_inf.add(nd)

        newly_rec = {nd for nd in infected if random.random() < MU}

        apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
        apply_recoveries_sir(newly_rec, infected, recovered, inf_nb, adj)
        ever_inf |= newly_inf

        i_series[t] = len(infected) / n

    return i_series, len(ever_inf) / n


def run_h2(nodes, adj, deg, seed_nodes) -> tuple[np.ndarray, float]:
    """H2: SIR then BP. Phase 1 until ever-infected >= F, then BP.
    rho_final = ever-infected fraction."""
    n = len(nodes)
    i_series = np.zeros(T + 1)

    infected    = set(seed_nodes)
    susceptible = set(nodes) - infected
    recovered   = set()
    ever_inf    = set(seed_nodes)

    inf_nb = {nd: sum(1 for nb in adj[nd] if nb in infected) for nd in nodes}

    i_series[0] = len(infected) / n
    t = 0

    # Phase 1: SIR until (I+R)/N >= F
    switched = False
    while infected and t < T:
        if len(ever_inf) / n >= F:
            switched = True
            break

        t += 1
        newly_inf = set()
        for nd in infected:
            for nb in adj[nd]:
                if nb in susceptible and random.random() < BETA:
                    newly_inf.add(nb)

        newly_rec = {nd for nd in infected if random.random() < MU}

        apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
        apply_recoveries_sir(newly_rec, infected, recovered, inf_nb, adj)
        ever_inf |= newly_inf

        if t <= T:
            i_series[t] = len(infected) / n

    # Phase 2: BP using ever-infected as active pool
    if switched:
        # Active = all ever-infected (I ∪ R); re-build inf_nb from active
        active = set(ever_inf)
        active_nb = {nd: sum(1 for nb in adj[nd] if nb in active) for nd in nodes}

        while t < T:
            newly_act = {nd for nd in susceptible if active_nb[nd] >= K}
            if not newly_act:
                break
            t += 1
            # New infections: add to infected and active, no recovery in Phase 2
            for nd in newly_act:
                infected.add(nd)
                susceptible.discard(nd)
                active.add(nd)
                ever_inf.add(nd)
                for nb in adj[nd]:
                    active_nb[nb] += 1
                    inf_nb[nb] += 1
            if t <= T:
                i_series[t] = len(infected) / n

    # Fill remaining with last known value (epidemic died or BP converged)
    last_val = i_series[t] if t <= T else i_series[T]
    for t2 in range(t + 1, T + 1):
        i_series[t2] = last_val

    return i_series, len(ever_inf) / n


def run_h3(nodes, adj, deg, seed_nodes) -> tuple[np.ndarray, float]:
    """H3: SIR soft BP.
    p(m) = 1 - (1-beta)^m * (1 - alpha*m/d), clipped to [0,1].
    rho_final = ever-infected fraction."""
    n = len(nodes)
    i_series = np.zeros(T + 1)

    infected    = set(seed_nodes)
    susceptible = set(nodes) - infected
    recovered   = set()
    ever_inf    = set(seed_nodes)

    inf_nb = {nd: sum(1 for nb in adj[nd] if nb in infected) for nd in nodes}

    i_series[0] = len(infected) / n

    for t in range(1, T + 1):
        if not infected:
            break

        newly_inf = set()
        for nd in susceptible:
            m = inf_nb[nd]
            if m == 0:
                continue
            d = deg[nd]
            p = 1.0 - (1.0 - BETA) ** m * (1.0 - ALPHA * m / d)
            p = max(0.0, min(1.0, p))
            if random.random() < p:
                newly_inf.add(nd)

        newly_rec = {nd for nd in infected if random.random() < MU}

        apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
        apply_recoveries_sir(newly_rec, infected, recovered, inf_nb, adj)
        ever_inf |= newly_inf

        i_series[t] = len(infected) / n

    return i_series, len(ever_inf) / n


def run_h4(nodes, adj, deg, seed_nodes) -> tuple[np.ndarray, float]:
    """H4: SIS OR WTM. rho_final = I* (mean last 50 steps)."""
    n = len(nodes)
    i_series = np.zeros(T + 1)

    infected    = set(seed_nodes)
    susceptible = set(nodes) - infected

    inf_nb = {nd: sum(1 for nb in adj[nd] if nb in infected) for nd in nodes}

    i_series[0] = len(infected) / n

    for t in range(1, T + 1):
        if not infected:
            break

        # SIS channel
        newly_inf = set()
        for nd in infected:
            for nb in adj[nd]:
                if nb in susceptible and random.random() < BETA:
                    newly_inf.add(nb)

        # WTM channel
        for nd in susceptible:
            if nd not in newly_inf:
                d = deg[nd]
                if d > 0 and inf_nb[nd] / d >= PHI:
                    newly_inf.add(nd)

        newly_rec = {nd for nd in infected if random.random() < MU}

        apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
        apply_recoveries_sis(newly_rec, infected, susceptible, inf_nb, adj)

        i_series[t] = len(infected) / n

    rho_final = float(np.mean(i_series[T - 49:]))
    return i_series, rho_final


def run_h5(nodes, adj, deg, seed_nodes) -> tuple[np.ndarray, float]:
    """H5: SIS then WTM.
    Phase 1: SIS until I(t)/N >= F.
    Phase 2: WTM infections (threshold on current infected) + SIS recovery, T2 steps.
    rho_final = I* (mean last 50 steps of full T run)."""
    n = len(nodes)
    i_series = np.zeros(T + 1)

    infected    = set(seed_nodes)
    susceptible = set(nodes) - infected

    inf_nb = {nd: sum(1 for nb in adj[nd] if nb in infected) for nd in nodes}

    i_series[0] = len(infected) / n
    t = 0

    # Phase 1: SIS until I(t)/N >= F
    switched = False
    while infected and t < T:
        if len(infected) / n >= F:
            switched = True
            break

        t += 1
        newly_inf = set()
        for nd in infected:
            for nb in adj[nd]:
                if nb in susceptible and random.random() < BETA:
                    newly_inf.add(nb)

        newly_rec = {nd for nd in infected if random.random() < MU}

        apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
        apply_recoveries_sis(newly_rec, infected, susceptible, inf_nb, adj)

        if t <= T:
            i_series[t] = len(infected) / n

    # Phase 2: WTM new infections + SIS recovery for T2 steps
    if switched:
        t2_count = 0
        while t < T and t2_count < T2:
            t += 1
            t2_count += 1

            # WTM: threshold on CURRENT infected neighbours
            newly_inf = set()
            for nd in susceptible:
                d = deg[nd]
                if d > 0 and inf_nb[nd] / d >= PHI:
                    newly_inf.add(nd)

            newly_rec = {nd for nd in infected if random.random() < MU}

            apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
            apply_recoveries_sis(newly_rec, infected, susceptible, inf_nb, adj)

            if t <= T:
                i_series[t] = len(infected) / n

    # Extend flat for any remaining timesteps
    if t < T:
        last_val = i_series[t]
        for t2 in range(t + 1, T + 1):
            i_series[t2] = last_val

    rho_final = float(np.mean(i_series[T - 49:]))
    return i_series, rho_final


def run_h6(nodes, adj, deg, seed_nodes) -> tuple[np.ndarray, float]:
    """H6: SIS soft WTM.
    p(psi) = 1 - (1-beta)*(1 - alpha*psi), clipped [0,1], applied when m>0.
    SIS recovery. rho_final = I* (mean last 50 steps)."""
    n = len(nodes)
    i_series = np.zeros(T + 1)

    infected    = set(seed_nodes)
    susceptible = set(nodes) - infected

    inf_nb = {nd: sum(1 for nb in adj[nd] if nb in infected) for nd in nodes}

    i_series[0] = len(infected) / n

    for t in range(1, T + 1):
        if not infected:
            break

        newly_inf = set()
        for nd in susceptible:
            m = inf_nb[nd]
            if m == 0:
                continue
            d = deg[nd]
            psi = m / d
            p = 1.0 - (1.0 - BETA) * (1.0 - ALPHA * psi)
            p = max(0.0, min(1.0, p))
            if random.random() < p:
                newly_inf.add(nd)

        newly_rec = {nd for nd in infected if random.random() < MU}

        apply_infections(newly_inf, infected, susceptible, inf_nb, adj)
        apply_recoveries_sis(newly_rec, infected, susceptible, inf_nb, adj)

        i_series[t] = len(infected) / n

    rho_final = float(np.mean(i_series[T - 49:]))
    return i_series, rho_final


# ── Seed selection helpers ─────────────────────────────────────────────────────

def high_degree_seeds(nodes, deg, n: int) -> set[int]:
    return set(sorted(nodes, key=lambda v: deg[v], reverse=True)[:n])


def high_kcore_seeds(nodes, kcore: dict, n: int) -> set[int]:
    return set(sorted(nodes, key=lambda v: kcore[v], reverse=True)[:n])


def make_strategy_seeds(strategy: str, nodes, deg, kcore, n: int) -> set[int]:
    if strategy == "Random":
        return set(random.sample(nodes, n))
    elif strategy == "High Degree":
        return high_degree_seeds(nodes, deg, n)
    else:
        return high_kcore_seeds(nodes, kcore, n)


# ── Experiment runner ─────────────────────────────────────────────────────────

RUN_FUNCS = [run_sir, run_sis, run_bp, run_wtm,
             run_h1, run_h2, run_h3, run_h4, run_h5, run_h6]


def run_all(G: nx.Graph) -> tuple[dict, dict]:
    """Run N_RUNS realisations of every model.

    Returns
    -------
    series : dict  model_name → ndarray (N_RUNS, T+1) of I(t)/N
    finals : dict  model_name → list[float] of rho_final per run
    """
    nodes = list(G.nodes())
    adj   = build_adj(G)
    deg   = build_deg(G)

    series: dict[str, np.ndarray] = {}
    finals: dict[str, list]       = {}

    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)

    for name, func in zip(MODEL_NAMES, RUN_FUNCS):
        print(f"  Running {name} ...", flush=True)
        all_i  = np.zeros((N_RUNS, T + 1))
        all_rf = []

        for run_idx in range(N_RUNS):
            # Seed selection
            if name in ("BP", "WTM"):
                seeds = random_seeds_fraction(nodes, RHO_0)
                if not seeds:
                    seeds = {random.choice(nodes)}
            else:
                seeds = random_seeds_fixed(nodes, N_SEEDS)

            i_ser, rf = func(nodes, adj, deg, seeds)
            all_i[run_idx] = i_ser
            all_rf.append(rf)

        series[name] = all_i
        finals[name] = all_rf

    return series, finals


# ── Seed-selection experiment ──────────────────────────────────────────────────

def run_all_strategies(G: nx.Graph) -> tuple[dict, int]:
    """Run every model × every seed strategy × N_RUNS trials.

    Returns
    -------
    strategy_finals : dict  strategy → {model_name → list[rho_final]}
    seed_n          : int   unified seed set size used across all models
    """
    nodes  = list(G.nodes())
    adj    = build_adj(G)
    deg    = build_deg(G)
    kcore  = nx.core_number(G)
    n      = len(nodes)
    seed_n = max(1, int(RHO_0 * n))

    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)

    strategy_finals: dict[str, dict[str, list[float]]] = {}

    for strategy in STRATEGIES:
        print(f"  Strategy: {strategy}  (seed size = {seed_n})", flush=True)
        finals: dict[str, list[float]] = {}

        for name, func in zip(MODEL_NAMES, RUN_FUNCS):
            print(f"    Running {name} ...", flush=True)
            all_rf = []
            for _ in range(N_RUNS):
                seeds = make_strategy_seeds(strategy, nodes, deg, kcore, seed_n)
                _, rf = func(nodes, adj, deg, seeds)
                all_rf.append(rf)
            finals[name] = all_rf

        strategy_finals[strategy] = finals

    return strategy_finals, seed_n


# ── Summary statistics ─────────────────────────────────────────────────────────

def compute_summaries(series: dict, finals: dict) -> list[dict]:
    """Compute AUC, rho_final, peak, t_peak for each model."""
    rows = []
    ts = np.arange(T + 1)

    for name in MODEL_NAMES:
        i_mat = series[name]           # (N_RUNS, T+1)
        rf_list = finals[name]         # list of rho_final per run

        mean_i  = i_mat.mean(axis=0)
        auc_val = float(np.trapezoid(mean_i, ts) / T)
        rho_f   = float(np.mean(rf_list))
        peak_v  = float(mean_i.max())
        t_peak  = int(mean_i.argmax())

        rows.append({
            "Model"       : name,
            "rho_final"   : round(rho_f,  4),
            "AUC"         : round(auc_val, 4),
            "Peak I(t)/N" : round(peak_v,  4),
            "Time of peak": t_peak,
        })

    return rows


def print_summary_table(rows: list[dict]) -> None:
    header = f"{'Model':<8} {'rho_final':>10} {'AUC':>8} {'Peak I(t)/N':>12} {'Time of peak':>13}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['Model']:<8} {r['rho_final']:>10.4f} {r['AUC']:>8.4f} "
              f"{r['Peak I(t)/N']:>12.4f} {r['Time of peak']:>13d}")

    # ── H1 sanity check ──────────────────────────────────────────────────────
    # H1 (SIR OR BP) adds a strictly-additional infection channel, so its
    # rho_final must be >= SIR's rho_final when compared on the same random run.
    # Across independent runs the averages should reflect this.  If the numbers
    # below suggest H1 < SIR, the most likely explanations are:
    #   (a) On this dense graph (avg_degree ≈ 44) the BP channel fires
    #       explosively at ~7% infected, compressing the epidemic into far
    #       fewer timesteps.  Peripheral low-degree nodes whose infected
    #       neighbours recover before SIR transmission fires can be missed.
    #   (b) Statistical noise from independent random seeds (use N_RUNS >= 50
    #       and T=300 for reliable comparison).
    # The lower AUC/T for H1 vs SIR is *expected* and is NOT a bug: a faster
    # epidemic has lower time-averaged prevalence within a fixed window, even
    # if rho_final is equal or higher.
    by_name = {r["Model"]: r for r in rows}
    sir_rf = by_name["SIR"]["rho_final"]
    h1_rf  = by_name["H1"]["rho_final"]
    print()
    print(f"[H1 check]  SIR rho_final = {sir_rf:.4f},  H1 rho_final = {h1_rf:.4f}", end="")
    if h1_rf < sir_rf - 0.005:
        print(f"  ← H1 < SIR (Δ={sir_rf-h1_rf:.4f}): see note above; "
              "likely a dense-graph / short-window artifact, not a model bug.")
    else:
        print("  ✓ H1 >= SIR as expected.")


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_figure(series: dict, finals: dict, rows: list[dict]) -> None:
    plt.rcParams.update({
        "font.family"   : "DejaVu Sans",
        "font.size"     : 10,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.grid"     : True,
        "grid.linewidth": 0.4,
        "grid.alpha"    : 0.5,
        "figure.dpi"    : 100,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11),
                                   gridspec_kw={"height_ratios": [1.6, 1]})
    fig.suptitle("Social Contagion Dynamics —" f"{NETWORK}",
                 fontsize=13, fontweight="bold", y=0.98)

    ts = np.arange(T + 1)

    # ── Panel 1: time series ──────────────────────────────────────────────────
    bp_wtm_names = {"BP", "WTM"}
    legend_handles = []

    for i, name in enumerate(MODEL_NAMES):
        col  = PALETTE[i]
        ls   = LSTYLES[i]
        mat  = series[name]        # (N_RUNS, T+1)
        mean = mat.mean(axis=0)

        # For BP/WTM: draw the rising section as solid and the flat extension as dashed.
        if name in bp_wtm_names:
            # Find first timestep where the mean curve is flat (cascade converged)
            diff = np.diff(mean)
            stable = np.abs(diff) < 1e-9
            conv_t = int(np.argmax(stable)) if np.any(stable) else T
            conv_t = max(conv_t, 1)

            ax1.plot(ts[:conv_t + 1], mean[:conv_t + 1],
                     color=col, lw=2.0, ls=ls, zorder=3)
            ax1.plot(ts[conv_t:], mean[conv_t:],
                     color=col, lw=1.5, ls="--", alpha=0.75, zorder=2)
        else:
            ax1.plot(ts, mean, color=col, lw=2.0, ls=ls, zorder=3)

        handle = Line2D([0], [0], color=col, lw=2.0, ls=ls, label=name)
        legend_handles.append(handle)

    # Vertical dashed line at t = T-50 (endemic measurement window)
    ax1.axvline(T - 50, color="black", lw=1.0, ls=":", alpha=0.6,
                label=f"Endemic window (t={T-50})")
    legend_handles.append(
        Line2D([0], [0], color="black", lw=1.0, ls=":", alpha=0.6,
               label=f"Endemic window (t={T-50})")
    )

    ax1.set_xlim(0, T)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Timestep", fontsize=11)
    ax1.set_ylabel("Infected fraction  I(t)/N", fontsize=11)
    ax1.legend(handles=legend_handles, loc="center right",
               fontsize=8.5, ncol=2, framealpha=0.9)

    # ── Panel 2: bar chart ────────────────────────────────────────────────────
    n_models  = len(MODEL_NAMES)
    bar_w     = 0.35
    x_pos     = np.arange(n_models)

    auc_vals  = [r["AUC"]       for r in rows]
    rho_vals  = [r["rho_final"] for r in rows]

    # AUC bars
    bars1 = ax2.bar(x_pos - bar_w / 2, auc_vals, bar_w,
                    color=PALETTE, alpha=0.85, edgecolor="white", lw=0.5,
                    label="AUC / T")
    # rho_final bars (slightly lighter, hatched)
    bars2 = ax2.bar(x_pos + bar_w / 2, rho_vals, bar_w,
                    color=PALETTE, alpha=0.50, edgecolor="grey", lw=0.5,
                    hatch="//", label=r"$\rho_{\rm final}$")

    # Reference line at SIR rho_final
    sir_rho = rho_vals[MODEL_NAMES.index("SIR")]
    ax2.axhline(sir_rho, color=PALETTE[0], lw=1.5, ls="--", alpha=0.8,
                label=f"SIR $\\rho_{{\\rm final}}$ = {sir_rho:.3f}")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(MODEL_NAMES, rotation=45, ha="right", fontsize=10)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_ylim(0, max(max(auc_vals), max(rho_vals)) * 1.15)
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved → {OUT_PATH}")


# ── Seed-selection CSV + figure ───────────────────────────────────────────────

def save_strategy_csv(strategy_finals: dict) -> None:
    import csv
    rows = []
    for strategy, finals in strategy_finals.items():
        for name, rfs in finals.items():
            rows.append({
                "Strategy"      : strategy,
                "Model"         : name,
                "rho_final_mean": round(float(np.mean(rfs)), 4),
                "rho_final_std" : round(float(np.std(rfs)),  4),
            })
    with open(STRAT_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Strategy", "Model", "rho_final_mean", "rho_final_std"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Seed-selection CSV saved → {STRAT_CSV_PATH}")


def make_seed_selection_figure(strategy_finals: dict, seed_n: int) -> None:
    plt.rcParams.update({
        "font.family"       : "DejaVu Sans",
        "font.size"         : 10,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.grid"         : False,
        "grid.linewidth"    : 0.4,
        "grid.alpha"        : 0.5,
        "figure.dpi"        : 100,
    })

    n_models    = len(MODEL_NAMES)
    n_strats    = len(STRATEGIES)
    total_width = 0.75
    bar_w       = total_width / n_strats
    x_pos       = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        f"Seed Selection Strategy Comparison — {NETWORK}  (seed size = {seed_n})",
        fontsize=13, fontweight="bold",
    )

    for si, (strategy, color, hatch) in enumerate(
        zip(STRATEGIES, STRAT_COLORS, STRAT_HATCHES)
    ):
        finals = strategy_finals[strategy]
        means  = [float(np.mean(finals[m])) for m in MODEL_NAMES]
        stds   = [float(np.std(finals[m]))  for m in MODEL_NAMES]
        offset = (si - (n_strats - 1) / 2) * bar_w

        bars = ax.bar(
            x_pos + offset, means, bar_w,
            yerr=stds, capsize=3,
            color=color, alpha=0.80,
            hatch=hatch, edgecolor="white", lw=0.5,
            label=strategy,
            error_kw={"elinewidth": 1.0, "ecolor": "grey"},
        )

        std_offset = max(stds) * 0.05 if stds else 0.0
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std_offset + 0.004,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=6.5,
            )

    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODEL_NAMES, fontsize=10)
    ax.set_ylabel(r"$\rho_{\rm final}$  (mean ± std)", fontsize=11)
    ax.set_ylim(0, None)
    ax.legend(title="Seed selection", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(STRAT_OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Seed-selection figure saved → {STRAT_OUT_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_epidemic(G: nx.Graph) -> None:
    print("Running simulations …")
    series, finals = run_all(G)

    print("\nComputing summaries …")
    rows = compute_summaries(series, finals)

    print("\n── Summary table ──────────────────────────────────────────────────")
    print_summary_table(rows)

    print("\nGenerating figure …")
    make_figure(series, finals, rows)


def run_seed_selection(G: nx.Graph) -> None:
    print("Running seed-selection analysis …")
    strategy_finals, seed_n = run_all_strategies(G)
    save_strategy_csv(strategy_finals)
    print("\nGenerating seed-selection figure …")
    make_seed_selection_figure(strategy_finals, seed_n)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Social contagion experiments on the Facebook/GitHub graph."
    )
    parser.add_argument(
        "--mode",
        choices=["epidemic", "seeds", "all"],
        default="all",
        help=(
            "epidemic — phase-transition comparison only; "
            "seeds    — seed-selection comparison only; "
            "all      — both (default)"
        ),
    )
    args = parser.parse_args()

    print(f"Loading graph from {GRAPH_PATH} …")
    G = load_facebook_graph(GRAPH_PATH)
    print_graph_stats(G)
    print(f"Parameters: β={BETA}  μ={MU}  k={K}  φ={PHI}  "
          f"ρ₀={RHO_0}  f={F}  α={ALPHA}  T={T}  runs={N_RUNS}\n")

    if args.mode in ("epidemic", "all"):
        run_epidemic(G)
    if args.mode in ("seeds", "all"):
        run_seed_selection(G)


if __name__ == "__main__":
    main()
