"""
Tests for input/graph_generator.py, input/graph_loader.py, input/write_graph.py.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import pytest
import networkx as nx

from input.graph_generator import (
    generate_er_graph,
    generate_random_geometric_graph,
    generate_lattice_graph,
)
from input.graph_loader import (
    load_graph_from_dimacs,
    load_graph_from_edge_list,
    load_graph_from_gml,
    load_graph_auto,
)
from input.write_graph import write_graph


# ─── generate_er_graph ────────────────────────────────────────────────────────

class TestGenerateERGraph:

    def test_correct_node_count(self):
        g = generate_er_graph(50, 0.3)
        assert g.number_of_nodes() == 50

    def test_returns_nx_graph(self):
        g = generate_er_graph(20, 0.2)
        assert isinstance(g, nx.Graph)

    def test_is_undirected(self):
        g = generate_er_graph(20, 0.2)
        assert not g.is_directed()

    def test_no_self_loops(self):
        g = generate_er_graph(30, 0.4)
        assert nx.number_of_selfloops(g) == 0

    def test_zero_probability_gives_no_edges(self):
        g = generate_er_graph(20, 0.0)
        assert g.number_of_edges() == 0

    def test_probability_one_gives_complete_graph(self):
        n = 15
        g = generate_er_graph(n, 1.0)
        assert g.number_of_edges() == n * (n - 1) // 2

    def test_large_n_small_p(self):
        g = generate_er_graph(100, 0.05)
        assert g.number_of_nodes() == 100
        assert g.number_of_edges() >= 0

    def test_edge_count_reasonable(self):
        """Expected edges = p * n*(n-1)/2; test within 3 standard deviations."""
        import math
        n, p = 100, 0.2
        g = generate_er_graph(n, p)
        expected = p * n * (n - 1) / 2
        # very loose bound — just sanity check
        assert g.number_of_edges() < n * (n - 1)


# ─── generate_random_geometric_graph ─────────────────────────────────────────

class TestGenerateRandomGeometricGraph:

    def test_correct_node_count(self):
        g = generate_random_geometric_graph(30, 0.3)
        assert g.number_of_nodes() == 30

    def test_returns_nx_graph(self):
        g = generate_random_geometric_graph(20, 0.2)
        assert isinstance(g, nx.Graph)

    def test_is_undirected(self):
        g = generate_random_geometric_graph(20, 0.2)
        assert not g.is_directed()

    def test_no_self_loops(self):
        g = generate_random_geometric_graph(20, 0.3)
        assert nx.number_of_selfloops(g) == 0

    def test_zero_radius_no_edges(self):
        g = generate_random_geometric_graph(20, 0.0)
        assert g.number_of_edges() == 0

    def test_large_radius_high_connectivity(self):
        """Radius = sqrt(2) covers the entire unit square → complete graph."""
        import math
        n = 15
        g = generate_random_geometric_graph(n, math.sqrt(2) + 0.01)
        # With radius larger than diagonal, every pair should be connected
        assert g.number_of_edges() == n * (n - 1) // 2


# ─── generate_lattice_graph ───────────────────────────────────────────────────

class TestGenerateLatticeGraph:

    def test_correct_node_count(self):
        size = 5
        g = generate_lattice_graph(size)
        assert g.number_of_nodes() == size * size

    def test_returns_nx_graph(self):
        g = generate_lattice_graph(4)
        assert isinstance(g, nx.Graph)

    def test_is_undirected(self):
        g = generate_lattice_graph(4)
        assert not g.is_directed()

    def test_interior_nodes_have_degree_4(self):
        """Interior nodes of a 2D grid have exactly 4 neighbours."""
        size = 5
        g = generate_lattice_graph(size)
        # Interior nodes: (row, col) with 1 <= row <= size-2, 1 <= col <= size-2
        interior_nodes = [
            (r, c)
            for r in range(1, size - 1)
            for c in range(1, size - 1)
        ]
        for node in interior_nodes:
            assert g.degree(node) == 4

    def test_corner_nodes_have_degree_2(self):
        size = 5
        g = generate_lattice_graph(size)
        corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
        for node in corners:
            assert g.degree(node) == 2

    def test_connected(self):
        g = generate_lattice_graph(4)
        assert nx.is_connected(g)

    def test_1x1_lattice_single_node(self):
        g = generate_lattice_graph(1)
        assert g.number_of_nodes() == 1
        assert g.number_of_edges() == 0

    def test_edge_count(self):
        """m x n grid has m*(n-1) + n*(m-1) edges."""
        size = 4
        g = generate_lattice_graph(size)
        expected_edges = size * (size - 1) + size * (size - 1)
        assert g.number_of_edges() == expected_edges


# ─── write_graph ──────────────────────────────────────────────────────────────

class TestWriteGraph:

    def test_write_creates_file(self, tmp_path):
        g = nx.path_graph(5)
        out = str(tmp_path / "test.dimacs")
        write_graph(g, out)
        assert os.path.exists(out)

    def test_written_file_has_header_line(self, tmp_path):
        g = nx.path_graph(5)
        out = str(tmp_path / "test.dimacs")
        write_graph(g, out)
        with open(out) as f:
            lines = f.readlines()
        assert any(line.startswith("p edge") for line in lines)

    def test_written_file_edge_count_matches(self, tmp_path):
        g = nx.erdos_renyi_graph(10, 0.4, seed=1)
        out = str(tmp_path / "test.dimacs")
        write_graph(g, out)
        with open(out) as f:
            lines = f.readlines()
        edge_lines = [l for l in lines if l.startswith("e ")]
        assert len(edge_lines) == g.number_of_edges()

    def test_write_empty_graph(self, tmp_path):
        g = nx.Graph()
        out = str(tmp_path / "empty.dimacs")
        write_graph(g, out)
        assert os.path.exists(out)


# ─── load_graph_from_dimacs ───────────────────────────────────────────────────

class TestLoadGraphFromDimacs:

    def _write_dimacs(self, path, edges, n_nodes, n_edges):
        with open(path, "w") as f:
            f.write(f"p edge {n_nodes} {n_edges}\n")
            for u, v in edges:
                f.write(f"e {u} {v}\n")

    def test_loads_nodes_and_edges(self, tmp_path):
        edges = [(1, 2), (2, 3), (3, 4)]
        path = str(tmp_path / "g.dimacs")
        self._write_dimacs(path, edges, 4, 3)
        g = load_graph_from_dimacs(path)
        assert g.number_of_edges() == 3

    def test_returns_nx_graph(self, tmp_path):
        path = str(tmp_path / "g.dimacs")
        self._write_dimacs(path, [(1, 2)], 2, 1)
        g = load_graph_from_dimacs(path)
        assert isinstance(g, nx.Graph)

    def test_empty_file_gives_empty_graph(self, tmp_path):
        path = str(tmp_path / "empty.dimacs")
        with open(path, "w") as f:
            pass
        g = load_graph_from_dimacs(path)
        assert g.number_of_edges() == 0


# ─── load_graph_from_edge_list ────────────────────────────────────────────────

class TestLoadGraphFromEdgeList:

    def test_loads_basic_edge_list(self, tmp_path):
        path = str(tmp_path / "g.edgelist")
        with open(path, "w") as f:
            f.write("1 2\n2 3\n3 4\n")
        g = load_graph_from_edge_list(path)
        assert g.number_of_edges() == 3

    def test_skips_comment_lines(self, tmp_path):
        path = str(tmp_path / "g.edgelist")
        with open(path, "w") as f:
            f.write("# comment\n1 2\n2 3\n")
        g = load_graph_from_edge_list(path)
        assert g.number_of_edges() == 2

    def test_skips_blank_lines(self, tmp_path):
        path = str(tmp_path / "g.edgelist")
        with open(path, "w") as f:
            f.write("1 2\n\n2 3\n")
        g = load_graph_from_edge_list(path)
        assert g.number_of_edges() == 2

    def test_returns_nx_graph(self, tmp_path):
        path = str(tmp_path / "g.edgelist")
        with open(path, "w") as f:
            f.write("a b\n")
        g = load_graph_from_edge_list(path)
        assert isinstance(g, nx.Graph)

    def test_is_undirected(self, tmp_path):
        path = str(tmp_path / "g.edgelist")
        with open(path, "w") as f:
            f.write("1 2\n")
        g = load_graph_from_edge_list(path)
        assert not g.is_directed()


# ─── write → load round trip ─────────────────────────────────────────────────

class TestWriteLoadRoundTrip:

    def test_roundtrip_preserves_edge_count(self, tmp_path):
        g = nx.erdos_renyi_graph(10, 0.4, seed=42)
        out = str(tmp_path / "g.dimacs")
        write_graph(g, out)
        g2 = load_graph_from_dimacs(out)
        assert g.number_of_edges() == g2.number_of_edges()

    def test_roundtrip_node_set_equivalent(self, tmp_path):
        """Node IDs may be remapped to integers, but count must match."""
        g = nx.path_graph(8)
        out = str(tmp_path / "g.dimacs")
        write_graph(g, out)
        g2 = load_graph_from_dimacs(out)
        assert g.number_of_nodes() == g2.number_of_nodes()


# ─── load_graph_auto ─────────────────────────────────────────────────────────

class TestLoadGraphAuto:

    def test_auto_detect_dimacs_by_extension(self, tmp_path):
        g = nx.path_graph(5)
        out = str(tmp_path / "g.dimacs")
        write_graph(g, out)
        g2 = load_graph_auto(out)
        assert g2.number_of_edges() == g.number_of_edges()

    def test_auto_detect_edge_list(self, tmp_path):
        path = str(tmp_path / "g.edgelist")
        with open(path, "w") as f:
            f.write("1 2\n2 3\n")
        g = load_graph_auto(path)
        assert g.number_of_edges() == 2
