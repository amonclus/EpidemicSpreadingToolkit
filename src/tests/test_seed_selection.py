"""
Tests for simulation/seed_selection.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import pytest
import networkx as nx

from simulation.seed_selection import (
    SeedStrategy,
    select_seeds,
    random_seeds,
    high_degree_seeds,
    high_kcore_seeds,
)


@pytest.fixture
def er_graph():
    return nx.erdos_renyi_graph(20, 0.3, seed=42)


@pytest.fixture
def path_graph():
    return nx.path_graph(15)


@pytest.fixture
def complete_graph():
    return nx.complete_graph(12)


# ─── SeedStrategy enum ────────────────────────────────────────────────────────

class TestSeedStrategyEnum:

    def test_strategy_values(self):
        assert SeedStrategy.RANDOM.value    == "random"
        assert SeedStrategy.HIGH_DEGREE.value == "high_degree"
        assert SeedStrategy.HIGH_KCORE.value  == "high_kcore"

    def test_strategy_from_string(self):
        assert SeedStrategy("random")     == SeedStrategy.RANDOM
        assert SeedStrategy("high_degree") == SeedStrategy.HIGH_DEGREE
        assert SeedStrategy("high_kcore")  == SeedStrategy.HIGH_KCORE

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            SeedStrategy("nonexistent")


# ─── random_seeds ─────────────────────────────────────────────────────────────

class TestRandomSeeds:

    def test_returns_correct_count(self, er_graph):
        seeds = random_seeds(er_graph, 5)
        assert len(seeds) == 5

    def test_all_nodes_valid(self, er_graph):
        seeds = random_seeds(er_graph, 8)
        assert all(n in er_graph.nodes() for n in seeds)

    def test_no_duplicates(self, er_graph):
        seeds = random_seeds(er_graph, 10)
        assert len(seeds) == len(set(seeds))

    def test_returns_list(self, er_graph):
        seeds = random_seeds(er_graph, 3)
        assert isinstance(seeds, list)

    def test_count_1(self, er_graph):
        seeds = random_seeds(er_graph, 1)
        assert len(seeds) == 1

    def test_count_equals_n(self, er_graph):
        n = er_graph.number_of_nodes()
        seeds = random_seeds(er_graph, n)
        assert len(seeds) == n
        assert set(seeds) == set(er_graph.nodes())


# ─── high_degree_seeds ────────────────────────────────────────────────────────

class TestHighDegreeSeeds:

    def test_returns_correct_count(self, er_graph):
        seeds = high_degree_seeds(er_graph, 5)
        assert len(seeds) == 5

    def test_returns_highest_degree_nodes(self, er_graph):
        k = 4
        seeds = set(high_degree_seeds(er_graph, k))
        # Seeds should all have degree >= any non-selected node's degree
        all_nodes = sorted(er_graph.nodes(), key=lambda v: er_graph.degree(v), reverse=True)
        expected = set(all_nodes[:k])
        # When multiple nodes share the same degree, multiple valid answers exist;
        # just verify degree ordering is respected
        seed_degrees = sorted([er_graph.degree(v) for v in seeds], reverse=True)
        non_seed_nodes = [v for v in er_graph.nodes() if v not in seeds]
        if non_seed_nodes:
            min_seed_deg = min(er_graph.degree(v) for v in seeds)
            max_non_seed_deg = max(er_graph.degree(v) for v in non_seed_nodes)
            assert min_seed_deg >= max_non_seed_deg

    def test_all_nodes_valid(self, er_graph):
        seeds = high_degree_seeds(er_graph, 5)
        assert all(n in er_graph.nodes() for n in seeds)

    def test_deterministic(self, er_graph):
        s1 = high_degree_seeds(er_graph, 5)
        s2 = high_degree_seeds(er_graph, 5)
        assert s1 == s2

    def test_star_hub_selected(self):
        """In a star graph, hub (node 0) must always be selected first."""
        g = nx.star_graph(9)  # hub is node 0
        seeds = high_degree_seeds(g, 1)
        assert seeds[0] == 0


# ─── high_kcore_seeds ────────────────────────────────────────────────────────

class TestHighKcoreSeeds:

    def test_returns_correct_count(self, er_graph):
        seeds = high_kcore_seeds(er_graph, 5)
        assert len(seeds) == 5

    def test_all_nodes_valid(self, er_graph):
        seeds = high_kcore_seeds(er_graph, 5)
        assert all(n in er_graph.nodes() for n in seeds)

    def test_returns_highest_coreness_nodes(self, er_graph):
        k = 4
        seeds = set(high_kcore_seeds(er_graph, k))
        coreness = nx.core_number(er_graph)
        min_selected_core = min(coreness[v] for v in seeds)
        non_seeds = [v for v in er_graph.nodes() if v not in seeds]
        if non_seeds:
            max_non_selected = max(coreness[v] for v in non_seeds)
            assert min_selected_core >= max_non_selected

    def test_deterministic(self, er_graph):
        s1 = high_kcore_seeds(er_graph, 5)
        s2 = high_kcore_seeds(er_graph, 5)
        assert s1 == s2


# ─── select_seeds dispatcher ──────────────────────────────────────────────────

class TestSelectSeeds:

    def test_random_strategy_returns_correct_count(self, er_graph):
        random.seed(42)
        seeds = select_seeds(er_graph, 5, SeedStrategy.RANDOM)
        assert len(seeds) == 5

    def test_high_degree_strategy(self, er_graph):
        seeds = select_seeds(er_graph, 5, SeedStrategy.HIGH_DEGREE)
        assert len(seeds) == 5
        assert all(n in er_graph.nodes() for n in seeds)

    def test_high_kcore_strategy(self, er_graph):
        seeds = select_seeds(er_graph, 5, SeedStrategy.HIGH_KCORE)
        assert len(seeds) == 5
        assert all(n in er_graph.nodes() for n in seeds)

    def test_string_strategy_random(self, er_graph):
        random.seed(99)
        seeds = select_seeds(er_graph, 3, "random")
        assert len(seeds) == 3

    def test_string_strategy_high_degree(self, er_graph):
        seeds = select_seeds(er_graph, 3, "high_degree")
        assert len(seeds) == 3

    def test_string_strategy_high_kcore(self, er_graph):
        seeds = select_seeds(er_graph, 3, "high_kcore")
        assert len(seeds) == 3

    def test_all_selected_nodes_in_graph(self, er_graph):
        for strategy in SeedStrategy:
            seeds = select_seeds(er_graph, 4, strategy)
            assert all(n in er_graph.nodes() for n in seeds)

    def test_count_1_returns_single_node(self, er_graph):
        for strategy in SeedStrategy:
            seeds = select_seeds(er_graph, 1, strategy)
            assert len(seeds) == 1

    def test_seeds_on_path_graph(self, path_graph):
        """Path graph has heterogeneous degrees (endpoints have degree 1)."""
        seeds_deg = high_degree_seeds(path_graph, 2)
        # Internal nodes have degree 2, endpoints degree 1
        for s in seeds_deg:
            assert path_graph.degree(s) >= 1

    def test_high_degree_on_complete_graph_returns_any_nodes(self, complete_graph):
        """All nodes have equal degree on K_n; any k of them should be returned."""
        n = complete_graph.number_of_nodes()
        k = n // 2
        seeds = high_degree_seeds(complete_graph, k)
        assert len(seeds) == k

    def test_random_seeds_reproducible_with_same_random_state(self, er_graph):
        random.seed(123)
        s1 = random_seeds(er_graph, 5)
        random.seed(123)
        s2 = random_seeds(er_graph, 5)
        assert s1 == s2
