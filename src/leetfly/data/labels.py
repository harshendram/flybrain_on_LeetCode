"""Technique labels: algorithmic paradigms, not generic data-structure tags.

A problem can use several techniques, so labels are multi-label. Generic tags (Array, String, Hash Table, Math,
Sorting, Matrix, Simulation, Counting, Enumeration, ...) are ignored; problems with no technique tag are excluded
from the technique task.
"""

import numpy as np
import pandas as pd

TECHNIQUES: dict[str, set[str]] = {
    "Dynamic Programming": {"Dynamic Programming", "Memoization"},
    "Greedy": {"Greedy"},
    "Binary Search": {"Binary Search"},
    "DFS / BFS": {"Depth-First Search", "Breadth-First Search"},
    "Graph": {"Graph", "Topological Sort", "Shortest Path", "Minimum Spanning Tree", "Eulerian Circuit",
              "Strongly Connected Component"},
    "Tree": {"Tree", "Binary Tree", "Binary Search Tree"},
    "Two Pointers": {"Two Pointers"},
    "Sliding Window": {"Sliding Window", "Monotonic Queue"},
    "Prefix Sum": {"Prefix Sum"},
    "Heap": {"Heap (Priority Queue)"},
    "Stack": {"Stack", "Monotonic Stack"},
    "Backtracking": {"Backtracking"},
    "Union Find": {"Union Find"},
    "Bit Manipulation": {"Bit Manipulation", "Bitmask"},
    "Segment Tree / BIT": {"Segment Tree", "Binary Indexed Tree"},
    "Trie": {"Trie"},
}


def technique_matrix(tags: pd.Series, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    names = list(TECHNIQUES) if names is None else names
    y = np.array([[bool(TECHNIQUES[n] & set(t)) for n in names] for t in tags], dtype=bool)
    return y, names


def select_techniques(y: np.ndarray, names: list[str], dev_mask: np.ndarray, min_count: int) -> list[str]:
    """Keep techniques with at least `min_count` positive problems in the dev portion."""
    counts = y[dev_mask].sum(axis=0)
    return [n for n, c in zip(names, counts) if c >= min_count]
