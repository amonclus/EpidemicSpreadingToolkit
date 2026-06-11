import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from input.graph_generator import generate_er_graph, generate_lattice_graph
from input.graph_loader import load_graph_auto
from input.write_graph import write_graph
from simulation.seed_selection import select_seeds, SeedStrategy
from simulation.sir import SIRModel
from simulation.sis import SISModel
from simulation.bootstrap import BootstrapPercolation
from analysis.graph_statistics import compute_graph_statistics
from analysis.graph_features import extract_graph_features, GRAPH_FEATURE_NAMES

N_TRIALS = 20
_STAT_KEYS = {
    "nodes", "edges", "density", "average_degree", "average_clustering",
    "num_components", "largest_component_size",
}


class TestFullPipeline:

    def test_er_random_seeds_sir_statistics(self):
        """ER graph → random seeds → SIR → graph statistics."""
        g = generate_er_graph(50, 0.2)
        seeds = set(select_seeds(g, 5, SeedStrategy.RANDOM))

        result, _ = SIRModel(g, beta=0.3, gamma=0.1).run(seeds)

        assert 0.0 <= result.epidemic_fraction <= 1.0
        assert result.epidemic_size == len(result.infected_nodes)
        assert result.recovered_nodes <= result.infected_nodes
        assert seeds <= result.infected_nodes

        stats = compute_graph_statistics(g)
        assert _STAT_KEYS <= set(stats.keys())
        assert stats["nodes"] == g.number_of_nodes()
        assert stats["edges"] == g.number_of_edges()

    def test_lattice_high_degree_seeds_bootstrap_features(self):
        """Lattice graph → high-degree seeds → Bootstrap Percolation → graph features."""
        g = generate_lattice_graph(5)  # 5×5 = 25 nodes
        seeds = set(select_seeds(g, 3, SeedStrategy.HIGH_DEGREE))

        result, _ = BootstrapPercolation(g, threshold=2).run(seeds)

        assert 0.0 <= result.cascade_fraction <= 1.0
        assert result.cascade_size == len(result.infected_nodes)
        assert seeds <= result.infected_nodes

        # Seeds must be among the highest-degree nodes (no non-seed node should
        # have a higher degree than the lowest-degree seed).
        seed_min_degree = min(g.degree(v) for v in seeds)
        non_seed_max_degree = max(g.degree(v) for v in g.nodes() if v not in seeds)
        assert seed_min_degree >= non_seed_max_degree

        features = extract_graph_features(g)
        assert set(features.keys()) == set(GRAPH_FEATURE_NAMES)
        assert all(isinstance(v, float) for v in features.values())

    def test_er_kcore_seeds_sis_statistics(self):
        """ER graph → high-k-core seeds → SIS → graph statistics."""
        g = generate_er_graph(40, 0.25)
        seeds = set(select_seeds(g, 5, SeedStrategy.HIGH_KCORE))

        result, _ = SISModel(g, beta=0.4, gamma=0.1).run(seeds)

        assert 0.0 <= result.cascade_fraction <= 1.0
        assert result.cascade_size == result.peak_infected
        assert seeds <= result.infected_nodes

        stats = compute_graph_statistics(g)
        assert stats["nodes"] == g.number_of_nodes()
        assert stats["edges"] == g.number_of_edges()

    def test_file_roundtrip_pipeline(self, tmp_path):
        """Generate graph → write DIMACS → reload → Bootstrap Percolation.

        Uses a lattice (no isolated nodes) because the DIMACS loader reconstructs
        nodes only from edge lines, so isolated nodes are not preserved on reload.
        """
        g = generate_lattice_graph(5)  # 25 nodes, fully connected, no isolated nodes
        path = str(tmp_path / "graph.dimacs")
        write_graph(g, path)

        g_loaded = load_graph_auto(path)

        assert g_loaded.number_of_nodes() == g.number_of_nodes()
        assert g_loaded.number_of_edges() == g.number_of_edges()

        seeds = set(select_seeds(g_loaded, 3, SeedStrategy.HIGH_DEGREE))
        result, _ = BootstrapPercolation(g_loaded, threshold=2).run(seeds)

        assert 0.0 <= result.cascade_fraction <= 1.0
        assert result.cascade_size == len(result.infected_nodes)

    def test_supercritical_sir_spreads_beyond_seeds(self):
        """Supercritical SIR must spread beyond the seed set in at least one of N_TRIALS runs."""
        g = generate_er_graph(60, 0.3)
        model = SIRModel(g, beta=0.6, gamma=0.05)
        seeds = set(select_seeds(g, 5, SeedStrategy.HIGH_DEGREE))

        spread_observed = any(
            model.run(seeds)[0].epidemic_size > len(seeds)
            for _ in range(N_TRIALS)
        )

        assert spread_observed, (
            f"Supercritical SIR (β=0.6, γ=0.05) failed to spread beyond {len(seeds)} "
            f"seeds in {N_TRIALS} trials on a {g.number_of_nodes()}-node ER graph"
        )
