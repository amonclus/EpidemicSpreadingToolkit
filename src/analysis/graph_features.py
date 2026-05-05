"""
Structural graph feature extractor shared between dataset generation and the live app.
"""
from __future__ import annotations

import random

import networkx as nx
import numpy as np

_PATH_SAMPLE = 200  # nodes sampled for avg-path-length on large graphs

GRAPH_FEATURE_NAMES = [
    "graph_n_nodes",
    "graph_density",
    "graph_mean_degree",
    "graph_std_degree",
    "graph_degree_heterogeneity",
    "graph_max_degree",
    "graph_clustering",
    "graph_assortativity",
    "graph_largest_cc_fraction",
    "graph_avg_path_length",
]


def extract_graph_features(graph: nx.Graph) -> dict[str, float]:
    """Return a dict of continuous graph-level features keyed by GRAPH_FEATURE_NAMES."""
    n = graph.number_of_nodes()
    if n == 0:
        return {k: 0.0 for k in GRAPH_FEATURE_NAMES}

    degrees = [d for _, d in graph.degree()]
    mean_deg = float(np.mean(degrees))
    std_deg  = float(np.std(degrees))

    try:
        assortativity = float(nx.degree_assortativity_coefficient(graph))
        if np.isnan(assortativity):
            assortativity = 0.0
    except Exception:
        assortativity = 0.0

    components = list(nx.connected_components(graph))
    largest_cc  = max(components, key=len)
    largest_cc_frac = len(largest_cc) / n

    # Average shortest path length — sampled to keep it fast on large graphs
    lc_sub = graph.subgraph(largest_cc)
    lc_n   = lc_sub.number_of_nodes()
    if lc_n > 1:
        sample = random.sample(list(lc_sub.nodes()), min(_PATH_SAMPLE, lc_n))
        lengths = []
        for src in sample:
            lengths.extend(nx.single_source_shortest_path_length(lc_sub, src).values())
        avg_path_length = float(np.mean(lengths))
    else:
        avg_path_length = 0.0

    return {
        "graph_n_nodes":             float(n),
        "graph_density":             float(nx.density(graph)),
        "graph_mean_degree":         mean_deg,
        "graph_std_degree":          std_deg,
        "graph_degree_heterogeneity": std_deg / (mean_deg + 1e-8),
        "graph_max_degree":          float(max(degrees)),
        "graph_clustering":          float(nx.average_clustering(graph)),
        "graph_assortativity":       assortativity,
        "graph_largest_cc_fraction": largest_cc_frac,
        "graph_avg_path_length":     avg_path_length,
    }
