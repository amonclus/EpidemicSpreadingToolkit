"""
Tests for:
  analysis/graph_statistics.py
  analysis/graph_features.py
  analysis/parameter_sweep.py   (Bootstrap sweeps)
  analysis/sir_parameter_sweep.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import pytest
import networkx as nx

from analysis.graph_statistics import (
    compute_graph_statistics,
    degree_distribution,
)
from analysis.graph_features import (
    extract_graph_features,
    GRAPH_FEATURE_NAMES,
)
from analysis.parameter_sweep import (
    sweep_seed_fraction,
    sweep_er_probability,
    sweep_lattice_size,
)
from analysis.sir_parameter_sweep import (
    sir_sweep_beta,
    sir_sweep_seed_fraction,
    sir_sweep_er_probability,
)


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def small_er():
    random.seed(0)
    g = nx.erdos_renyi_graph(20, 0.3, seed=0)
    if not nx.is_connected(g):
        nodes = list(g.nodes())
        for i in range(len(nodes) - 1):
            g.add_edge(nodes[i], nodes[i + 1])
    return g


@pytest.fixture
def small_lattice():
    g = nx.grid_2d_graph(4, 4)
    return nx.convert_node_labels_to_integers(g)


@pytest.fixture
def small_path():
    return nx.path_graph(10)


@pytest.fixture
def empty_graph():
    return nx.Graph()


# ─── compute_graph_statistics ─────────────────────────────────────────────────

class TestComputeGraphStatistics:

    def test_returns_dict(self, small_er):
        stats = compute_graph_statistics(small_er)
        assert isinstance(stats, dict)

    def test_empty_graph_returns_empty(self, empty_graph):
        stats = compute_graph_statistics(empty_graph)
        assert stats == {}

    def test_required_keys_present(self, small_er):
        stats = compute_graph_statistics(small_er)
        expected_keys = {
            "nodes", "edges", "density", "average_degree", "max_degree",
            "min_degree", "average_clustering", "num_components",
            "largest_component_size", "average_path_length", "diameter",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_node_count_correct(self, small_er):
        stats = compute_graph_statistics(small_er)
        assert stats["nodes"] == small_er.number_of_nodes()

    def test_edge_count_correct(self, small_er):
        stats = compute_graph_statistics(small_er)
        assert stats["edges"] == small_er.number_of_edges()

    def test_density_in_range(self, small_er):
        stats = compute_graph_statistics(small_er)
        assert 0.0 <= stats["density"] <= 1.0

    def test_average_degree_positive(self, small_er):
        stats = compute_graph_statistics(small_er)
        assert stats["average_degree"] >= 0.0

    def test_average_degree_formula(self, small_er):
        """avg_degree = 2 * edges / nodes."""
        stats = compute_graph_statistics(small_er)
        n = small_er.number_of_nodes()
        m = small_er.number_of_edges()
        expected = 2 * m / n
        assert abs(stats["average_degree"] - expected) < 1e-9

    def test_max_degree_at_least_avg(self, small_er):
        stats = compute_graph_statistics(small_er)
        assert stats["max_degree"] >= stats["average_degree"]

    def test_min_degree_at_most_avg(self, small_er):
        stats = compute_graph_statistics(small_er)
        assert stats["min_degree"] <= stats["average_degree"]

    def test_num_components_positive(self, small_er):
        stats = compute_graph_statistics(small_er)
        assert stats["num_components"] >= 1

    def test_largest_component_size_leq_n(self, small_er):
        stats = compute_graph_statistics(small_er)
        assert stats["largest_component_size"] <= small_er.number_of_nodes()

    def test_average_clustering_in_range(self, small_er):
        stats = compute_graph_statistics(small_er)
        assert 0.0 <= stats["average_clustering"] <= 1.0

    def test_average_path_length_positive_for_connected(self, small_er):
        if nx.is_connected(small_er):
            stats = compute_graph_statistics(small_er)
            assert stats["average_path_length"] > 0.0

    def test_diameter_positive_for_connected(self, small_er):
        if nx.is_connected(small_er):
            stats = compute_graph_statistics(small_er)
            assert stats["diameter"] >= 1

    def test_single_node_graph(self):
        g = nx.Graph()
        g.add_node(0)
        stats = compute_graph_statistics(g)
        assert stats["nodes"] == 1
        assert stats["edges"] == 0

    def test_lattice_stats(self, small_lattice):
        stats = compute_graph_statistics(small_lattice)
        assert stats["nodes"] == 16
        assert stats["num_components"] == 1


# ─── degree_distribution ──────────────────────────────────────────────────────

class TestDegreeDistribution:

    def test_returns_dict(self, small_er):
        dist = degree_distribution(small_er)
        assert isinstance(dist, dict)

    def test_counts_sum_to_n(self, small_er):
        dist = degree_distribution(small_er)
        assert sum(dist.values()) == small_er.number_of_nodes()

    def test_all_degrees_non_negative(self, small_er):
        dist = degree_distribution(small_er)
        assert all(d >= 0 for d in dist.keys())

    def test_path_graph_distribution(self):
        """Path of length n has 2 nodes with degree 1 and n-2 with degree 2."""
        n = 8
        g = nx.path_graph(n)
        dist = degree_distribution(g)
        assert dist.get(1, 0) == 2
        assert dist.get(2, 0) == n - 2

    def test_complete_graph_all_same_degree(self):
        n = 6
        g = nx.complete_graph(n)
        dist = degree_distribution(g)
        assert list(dist.keys()) == [n - 1]
        assert dist[n - 1] == n

    def test_empty_graph(self, empty_graph):
        dist = degree_distribution(empty_graph)
        assert dist == {}


# ─── extract_graph_features ───────────────────────────────────────────────────

class TestExtractGraphFeatures:

    def test_returns_dict(self, small_er):
        feats = extract_graph_features(small_er)
        assert isinstance(feats, dict)

    def test_all_feature_names_present(self, small_er):
        feats = extract_graph_features(small_er)
        for name in GRAPH_FEATURE_NAMES:
            assert name in feats, f"Missing feature: {name}"

    def test_no_nan_values_on_connected_graph(self, small_er):
        import math
        feats = extract_graph_features(small_er)
        for key, val in feats.items():
            assert not math.isnan(val), f"NaN in feature {key}"

    def test_node_count_feature(self, small_er):
        feats = extract_graph_features(small_er)
        assert feats["graph_n_nodes"] == float(small_er.number_of_nodes())

    def test_density_in_range(self, small_er):
        feats = extract_graph_features(small_er)
        assert 0.0 <= feats["graph_density"] <= 1.0

    def test_clustering_in_range(self, small_er):
        feats = extract_graph_features(small_er)
        assert 0.0 <= feats["graph_clustering"] <= 1.0

    def test_largest_cc_fraction_in_range(self, small_er):
        feats = extract_graph_features(small_er)
        assert 0.0 <= feats["graph_largest_cc_fraction"] <= 1.0

    def test_mean_degree_non_negative(self, small_er):
        feats = extract_graph_features(small_er)
        assert feats["graph_mean_degree"] >= 0.0

    def test_std_degree_non_negative(self, small_er):
        feats = extract_graph_features(small_er)
        assert feats["graph_std_degree"] >= 0.0

    def test_avg_path_length_positive_connected(self, small_er):
        if nx.is_connected(small_er):
            feats = extract_graph_features(small_er)
            assert feats["graph_avg_path_length"] > 0.0

    def test_empty_graph_returns_zeros(self, empty_graph):
        feats = extract_graph_features(empty_graph)
        for key in GRAPH_FEATURE_NAMES:
            assert feats[key] == 0.0

    def test_graph_feature_names_is_list_of_strings(self):
        assert isinstance(GRAPH_FEATURE_NAMES, list)
        assert all(isinstance(n, str) for n in GRAPH_FEATURE_NAMES)
        assert len(GRAPH_FEATURE_NAMES) == 10

    def test_deterministic_on_same_graph(self, small_lattice):
        random.seed(0)
        f1 = extract_graph_features(small_lattice)
        # avg_path_length is sampled but deterministic with same seed
        random.seed(0)
        f2 = extract_graph_features(small_lattice)
        assert f1["graph_n_nodes"] == f2["graph_n_nodes"]
        assert f1["graph_density"] == f2["graph_density"]


# ─── sweep_seed_fraction (Bootstrap) ─────────────────────────────────────────

class TestSweepSeedFraction:

    def test_returns_list_of_correct_length(self, small_er):
        fracs = [0.05, 0.1, 0.2]
        results = sweep_seed_fraction(small_er, fracs, threshold=2, num_trials=5)
        assert len(results) == len(fracs)

    def test_each_entry_has_required_keys(self, small_er):
        results = sweep_seed_fraction(small_er, [0.1], threshold=2, num_trials=5)
        expected = {"seed_fraction", "seed_size", "cascade_probability",
                    "cascade_size", "time_to_cascade"}
        for entry in results:
            assert expected.issubset(set(entry.keys()))

    def test_cascade_size_in_range(self, small_er):
        results = sweep_seed_fraction(small_er, [0.05, 0.1, 0.2], threshold=2, num_trials=5)
        for entry in results:
            assert 0.0 <= entry["cascade_size"] <= 1.0

    def test_cascade_probability_in_range(self, small_er):
        results = sweep_seed_fraction(small_er, [0.1, 0.3], threshold=2, num_trials=5)
        for entry in results:
            assert 0.0 <= entry["cascade_probability"] <= 1.0

    def test_seed_size_at_least_1(self, small_er):
        results = sweep_seed_fraction(small_er, [0.01, 0.05], threshold=2, num_trials=5)
        for entry in results:
            assert entry["seed_size"] >= 1

    def test_time_to_cascade_non_negative(self, small_er):
        results = sweep_seed_fraction(small_er, [0.1], threshold=2, num_trials=5)
        for entry in results:
            assert entry["time_to_cascade"] >= 0.0


# ─── sweep_er_probability (Bootstrap) ────────────────────────────────────────

class TestSweepERProbability:

    def test_returns_correct_length(self):
        probs = [0.1, 0.2]
        results = sweep_er_probability(15, probs, threshold=2, num_trials=5)
        assert len(results) == len(probs)

    def test_entries_have_required_keys(self):
        results = sweep_er_probability(15, [0.2], threshold=2, num_trials=5)
        expected = {"graph_type", "n", "p", "edges", "cascade_size",
                    "cascade_probability", "critical_seed_size",
                    "time_to_cascade", "percolation_threshold"}
        for entry in results:
            assert expected.issubset(set(entry.keys()))

    def test_cascade_size_in_range(self):
        results = sweep_er_probability(15, [0.15, 0.3], threshold=2, num_trials=5)
        for entry in results:
            assert 0.0 <= entry["cascade_size"] <= 1.0

    def test_percolation_threshold_in_range(self):
        results = sweep_er_probability(15, [0.2], threshold=2, num_trials=5)
        for entry in results:
            assert 0.0 <= entry["percolation_threshold"] <= 1.0


# ─── sweep_lattice_size (Bootstrap) ──────────────────────────────────────────

class TestSweepLatticeSize:

    def test_returns_correct_length(self):
        sizes = [3, 4]
        results = sweep_lattice_size(sizes, threshold=2, num_trials=5)
        assert len(results) == len(sizes)

    def test_entries_have_required_keys(self):
        results = sweep_lattice_size([3], threshold=2, num_trials=5)
        expected = {"graph_type", "grid_size", "nodes", "edges", "cascade_size",
                    "cascade_probability", "critical_seed_size",
                    "time_to_cascade", "percolation_threshold"}
        for entry in results:
            assert expected.issubset(set(entry.keys()))

    def test_node_count_equals_grid_size_squared(self):
        results = sweep_lattice_size([3, 4], threshold=2, num_trials=5)
        for entry in results:
            expected_nodes = entry["grid_size"] ** 2
            assert entry["nodes"] == expected_nodes

    def test_cascade_size_in_range(self):
        results = sweep_lattice_size([3], threshold=2, num_trials=5)
        for entry in results:
            assert 0.0 <= entry["cascade_size"] <= 1.0


# ─── sir_sweep_beta ───────────────────────────────────────────────────────────

class TestSIRSweepBeta:

    def test_returns_correct_length(self, small_er):
        betas = [0.1, 0.2, 0.4]
        results = sir_sweep_beta(small_er, betas, gamma=0.1, num_trials=10)
        assert len(results) == len(betas)

    def test_entries_have_required_keys(self, small_er):
        results = sir_sweep_beta(small_er, [0.2], gamma=0.1, num_trials=10)
        expected = {"beta", "epidemic_size", "epidemic_probability", "time_to_epidemic"}
        for entry in results:
            assert expected.issubset(set(entry.keys()))

    def test_epidemic_size_in_range(self, small_er):
        results = sir_sweep_beta(small_er, [0.1, 0.3], gamma=0.1, num_trials=10)
        for entry in results:
            assert 0.0 <= entry["epidemic_size"] <= 1.0

    def test_epidemic_probability_in_range(self, small_er):
        results = sir_sweep_beta(small_er, [0.2], gamma=0.1, num_trials=10)
        for entry in results:
            assert 0.0 <= entry["epidemic_probability"] <= 1.0

    def test_beta_values_recorded_correctly(self, small_er):
        betas = [0.1, 0.3, 0.5]
        results = sir_sweep_beta(small_er, betas, gamma=0.1, num_trials=5)
        returned_betas = [entry["beta"] for entry in results]
        for b in betas:
            assert round(b, 4) in returned_betas

    def test_higher_beta_tends_to_larger_epidemic(self, small_er):
        """On average, higher beta should produce at least as large an epidemic."""
        betas = [0.05, 0.5]
        results = sir_sweep_beta(small_er, betas, gamma=0.1, num_trials=20)
        low_size  = results[0]["epidemic_size"]
        high_size = results[1]["epidemic_size"]
        assert high_size >= low_size


# ─── sir_sweep_seed_fraction ──────────────────────────────────────────────────

class TestSIRSweepSeedFraction:

    def test_returns_correct_length(self, small_er):
        fracs = [0.05, 0.1, 0.2]
        results = sir_sweep_seed_fraction(small_er, fracs, beta=0.3, gamma=0.1, num_trials=10)
        assert len(results) == len(fracs)

    def test_entries_have_required_keys(self, small_er):
        results = sir_sweep_seed_fraction(
            small_er, [0.1], beta=0.3, gamma=0.1, num_trials=10
        )
        expected = {"seed_fraction", "seed_size", "epidemic_size",
                    "epidemic_probability", "time_to_epidemic"}
        for entry in results:
            assert expected.issubset(set(entry.keys()))

    def test_epidemic_size_in_range(self, small_er):
        results = sir_sweep_seed_fraction(
            small_er, [0.05, 0.2], beta=0.3, gamma=0.1, num_trials=10
        )
        for entry in results:
            assert 0.0 <= entry["epidemic_size"] <= 1.0

    def test_larger_seed_fraction_tends_to_larger_epidemic(self, small_er):
        results = sir_sweep_seed_fraction(
            small_er, [0.05, 0.5], beta=0.3, gamma=0.1, num_trials=20
        )
        assert results[1]["epidemic_size"] >= results[0]["epidemic_size"]


# ─── sir_sweep_er_probability ─────────────────────────────────────────────────

class TestSIRSweepERProbability:

    def test_returns_correct_length(self):
        probs = [0.15, 0.3]
        results = sir_sweep_er_probability(
            15, probs, beta=0.3, gamma=0.1, num_trials=10
        )
        assert len(results) == len(probs)

    def test_entries_have_required_keys(self):
        results = sir_sweep_er_probability(
            15, [0.3], beta=0.3, gamma=0.1, num_trials=10
        )
        expected = {"graph_type", "n", "p", "edges", "epidemic_size",
                    "epidemic_probability", "critical_seed_size",
                    "time_to_epidemic", "epidemic_threshold"}
        for entry in results:
            assert expected.issubset(set(entry.keys()))

    def test_epidemic_size_in_range(self):
        results = sir_sweep_er_probability(
            15, [0.2], beta=0.3, gamma=0.1, num_trials=10
        )
        for entry in results:
            assert 0.0 <= entry["epidemic_size"] <= 1.0

    def test_epidemic_threshold_in_range(self):
        results = sir_sweep_er_probability(
            15, [0.2], beta=0.3, gamma=0.1, num_trials=10
        )
        for entry in results:
            assert 0.0 <= entry["epidemic_threshold"] <= 1.0
