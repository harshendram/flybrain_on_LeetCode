"""Similar-problem search with the fly's sparse KC codes (FlyHash, Dasgupta et al. 2017) and baselines."""

import numpy as np


def jaccard_neighbors(query: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
    q, p = query.astype(np.float32), pool.astype(np.float32)
    inter = q @ p.T
    union = q.sum(1, keepdims=True) + p.sum(1)[None, :] - inter
    return np.argsort(-(inter / np.maximum(union, 1)), axis=1, kind="stable")[:, :k]


def cosine_neighbors(query, pool, k: int) -> np.ndarray:
    def unit(x):
        x = np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x, dtype=np.float64)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    return np.argsort(-(unit(query) @ unit(pool).T), axis=1, kind="stable")[:, :k]


def simhash_codes(x: np.ndarray, n_bits: int, rng: np.random.Generator) -> np.ndarray:
    """Classic LSH baseline: signs of dense random projections (after centering)."""
    proj = rng.standard_normal((x.shape[1], n_bits))
    return ((x - x.mean(axis=0)) @ proj) > 0


def hamming_neighbors(query: np.ndarray, pool: np.ndarray, k: int) -> np.ndarray:
    q, p = query.astype(np.float32), pool.astype(np.float32)
    dist = q.sum(1, keepdims=True) + p.sum(1)[None, :] - 2 * (q @ p.T)
    return np.argsort(dist, axis=1, kind="stable")[:, :k]


def mean_average_precision(neighbors: np.ndarray, y_query: np.ndarray, y_pool: np.ndarray) -> float:
    """Relevant = shares at least one technique with the query. AP@k averaged over queries."""
    rel = (y_query[:, None, :] & y_pool[neighbors]).any(axis=2).astype(float)  # (n_q, k)
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, rel.shape[1] + 1)
    ap = (prec * rel).sum(axis=1) / np.maximum(rel.sum(axis=1), 1)
    return float(ap.mean())
