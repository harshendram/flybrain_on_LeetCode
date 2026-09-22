"""Null wirings of the KC x glomerulus graph.

N1 degree-preserving: curveball trades (Strona et al. 2014) keep every KC's number of glomerular inputs AND every
   glomerulus's number of KC targets (its coverage). Synapse weights travel with the edges. Only "who connects to
   whom" is destroyed.
N2 uniform: every KC keeps its number of inputs and its own synapse weights, but draws its glomeruli uniformly at
   random. This also destroys the coverage bias (FlyHash-style random projection).
"""

import numpy as np
from scipy import sparse


def _rows(w: sparse.csr_matrix) -> list[dict[int, float]]:
    return [dict(zip(w.indices[w.indptr[i] : w.indptr[i + 1]], w.data[w.indptr[i] : w.indptr[i + 1]]))
            for i in range(w.shape[0])]


def _to_csr(rows: list[dict[int, float]], n_cols: int) -> sparse.csr_matrix:
    r = np.repeat(np.arange(len(rows)), [len(x) for x in rows])
    c = np.fromiter((g for x in rows for g in x), dtype=np.int64, count=len(r))
    v = np.fromiter((w for x in rows for w in x.values()), dtype=np.float64, count=len(r))
    return sparse.csr_matrix((v, (r, c)), shape=(len(rows), n_cols))


def degree_preserving(
    w: sparse.csr_matrix, rng: np.random.Generator, trades_per_row: int = 20, groups: np.ndarray | None = None
) -> sparse.csr_matrix:
    """With `groups` (e.g. KC class per row), trades only happen within a group, so each group's coverage of every
    glomerulus is preserved too (a stricter null)."""
    rows = _rows(sparse.csr_matrix(w))
    n = len(rows)
    members = None
    if groups is not None:
        members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        members = [m for m in members if len(m) >= 2]
        weights = np.array([len(m) for m in members], dtype=float) / sum(len(m) for m in members)
    for _ in range(trades_per_row * n):
        if members is None:
            a, b = rng.choice(n, size=2, replace=False)
        else:
            a, b = rng.choice(members[rng.choice(len(members), p=weights)], size=2, replace=False)
        ra, rb = rows[a], rows[b]
        a_only = [(g, ra[g]) for g in ra if g not in rb]
        b_only = [(g, rb[g]) for g in rb if g not in ra]
        if not a_only or not b_only:
            continue
        pool = a_only + b_only
        order = rng.permutation(len(pool))
        for g, _ in a_only:
            del ra[g]
        for g, _ in b_only:
            del rb[g]
        for pos, idx in enumerate(order):
            g, weight = pool[idx]
            (ra if pos < len(a_only) else rb)[g] = weight
    return _to_csr(rows, w.shape[1])


def uniform(w: sparse.csr_matrix, rng: np.random.Generator) -> sparse.csr_matrix:
    rows = _rows(sparse.csr_matrix(w))
    n_cols = w.shape[1]
    new = []
    for r in rows:
        gloms = rng.choice(n_cols, size=len(r), replace=False)
        new.append(dict(zip(gloms.tolist(), rng.permutation(list(r.values())).tolist())))
    return _to_csr(new, n_cols)
