"""
Shared fixtures for the epidemic spreading test suite.
"""
import sys
import os

# Ensure the project root is on sys.path so all source modules are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random

import networkx as nx
import numpy as np
import pytest


# ── Small synthetic graphs ─────────────────────────────────────────────────────

@pytest.fixture
def small_er_graph():
    """ER graph with N=30, edge probability 0.25 (dense enough for epidemics)."""
    random.seed(0)
    g = nx.erdos_renyi_graph(30, 0.25, seed=0)
    # Ensure connectivity: add a spanning path if needed
    if not nx.is_connected(g):
        nodes = list(nx.nodes(g))
        for i in range(len(nodes) - 1):
            if not nx.has_path(g, nodes[i], nodes[i + 1]):
                g.add_edge(nodes[i], nodes[i + 1])
    return g


@pytest.fixture
def small_er_graph_20():
    """ER graph with N=20, edge probability 0.3."""
    return nx.erdos_renyi_graph(20, 0.3, seed=7)


@pytest.fixture
def small_lattice():
    """4x4 square lattice (16 nodes, interior nodes have degree 4)."""
    g = nx.grid_2d_graph(4, 4)
    return nx.convert_node_labels_to_integers(g)


@pytest.fixture
def small_path_graph():
    """Simple 10-node path graph."""
    return nx.path_graph(10)


@pytest.fixture
def small_complete_graph():
    """Complete graph K_10."""
    return nx.complete_graph(10)


@pytest.fixture
def star_graph():
    """Star graph with 10 leaves (hub + 10 leaves = 11 nodes)."""
    return nx.star_graph(10)


# ── SIR parameter presets ──────────────────────────────────────────────────────

@pytest.fixture
def sir_supercritical():
    """High beta, low gamma — epidemic spreads widely."""
    return {"beta": 0.5, "gamma": 0.05}


@pytest.fixture
def sir_subcritical():
    """Very low beta, high gamma — epidemic dies almost immediately."""
    return {"beta": 0.01, "gamma": 0.9}


@pytest.fixture
def sir_moderate():
    """Moderate parameters — some spread."""
    return {"beta": 0.15, "gamma": 0.10}


# ── SIS parameter presets ──────────────────────────────────────────────────────

@pytest.fixture
def sis_supercritical():
    return {"beta": 0.5, "gamma": 0.05}


@pytest.fixture
def sis_subcritical():
    return {"beta": 0.01, "gamma": 0.9}


# ── Time-series fixtures for ML feature tests ─────────────────────────────────

@pytest.fixture
def synthetic_growing_series():
    """Deterministic growing I(t)/N series of length 100."""
    return np.linspace(0.01, 0.8, 100)


@pytest.fixture
def synthetic_flat_series():
    """Constant series at 0.05."""
    return np.full(100, 0.05)


@pytest.fixture
def synthetic_peaked_series():
    """Series that peaks at t=20 and then decays."""
    t = np.arange(100)
    return np.exp(-0.5 * ((t - 20) / 10) ** 2) * 0.6


@pytest.fixture
def synthetic_decaying_series():
    """Monotonically decaying series."""
    return np.linspace(0.8, 0.01, 100)


@pytest.fixture
def synthetic_all_zero_series():
    """All-zero series."""
    return np.zeros(100)


@pytest.fixture
def synthetic_spike_series():
    """Series with a single nonzero spike at t=50."""
    s = np.zeros(100)
    s[50] = 0.8
    return s


@pytest.fixture
def synthetic_always_above_001():
    """Series always above 0.01."""
    return np.linspace(0.05, 0.9, 100)


@pytest.fixture
def synthetic_always_below_001():
    """Series always below 0.01."""
    return np.full(100, 0.005)
