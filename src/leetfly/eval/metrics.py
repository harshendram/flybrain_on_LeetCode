"""Ranking metrics for multi-label technique prediction, plus bootstrap confidence intervals."""

import numpy as np
from scipy.stats import rankdata


def hit_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Per-problem 1/0: is at least one of the top-k guesses among the problem's true techniques?"""
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(y, top, axis=1).any(axis=1).astype(float)


def auroc(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-technique AUROC via ranks (NaN where a technique has no positives or no negatives)."""
    out = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        pos = y[:, c]
        n_pos, n_neg = pos.sum(), (~pos).sum()
        if n_pos and n_neg:
            r = rankdata(scores[:, c])
            out[c] = (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def macro_auroc(scores: np.ndarray, y: np.ndarray) -> float:
    return float(np.nanmean(auroc(scores, y)))


def summary(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "hit@1": float(hit_at_k(scores, y, 1).mean()),
        "hit@3": float(hit_at_k(scores, y, 3).mean()),
        "macro_auroc": macro_auroc(scores, y),
    }


def bootstrap(scores: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 0) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    h1, h3 = hit_at_k(scores, y, 1), hit_at_k(scores, y, 3)
    draws = {"hit@1": [], "hit@3": [], "macro_auroc": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        draws["hit@1"].append(h1[i].mean())
        draws["hit@3"].append(h3[i].mean())
        draws["macro_auroc"].append(macro_auroc(scores[i], y[i]))
    return {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in draws.items()}
