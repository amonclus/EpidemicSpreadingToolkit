"""
store.py — Graph serialisation helpers for dcc.Store.

NetworkX graphs cannot be stored directly in dcc.Store (must be JSON-serialisable).
We serialise via nx.node_link_data / nx.node_link_graph.

Tuple keys (lattice nodes) are serialised as lists in JSON and must be
converted back to tuples on deserialisation.
"""
from __future__ import annotations

import json
from typing import Any

import networkx as nx


def graph_to_store(g: nx.Graph) -> dict:
    """Convert a NetworkX graph to a JSON-serialisable dict."""
    data = nx.node_link_data(g)
    # Ensure node ids that are tuples become lists (JSON-safe)
    return data


def graph_from_store(data: dict | None) -> nx.Graph | None:
    """Reconstruct a NetworkX graph from stored dict.

    Handles lattice graphs where node ids were serialised as lists and need
    to be converted back to tuples.
    """
    if data is None:
        return None
    g = nx.node_link_graph(data)
    # If nodes are lists (came from tuple keys), convert to tuples
    sample = next(iter(g.nodes()), None)
    if isinstance(sample, list):
        mapping = {n: tuple(n) for n in g.nodes()}
        g = nx.relabel_nodes(g, mapping)
    return g


def config_to_dict(config) -> dict:
    """Serialise a SidebarConfig dataclass to a plain dict."""
    import dataclasses
    return dataclasses.asdict(config)


def config_from_dict(d: dict | None):
    """Deserialise a SidebarConfig from a plain dict."""
    if d is None:
        return None
    from ui.state import SidebarConfig
    return SidebarConfig(**d)
