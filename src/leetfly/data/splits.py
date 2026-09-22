"""Leakage-safe splits.

Test = the most recent problems (temporal holdout). Dev CV folds are grouped by problem family, so "Two Sum" and
"Two Sum II - Input Array Is Sorted" never end up on opposite sides of a fold.
"""

import numpy as np
from sklearn.model_selection import GroupKFold

_ROMAN = {"ii", "iii", "iv", "vi", "vii", "viii", "ix"}
_ROMAN_LAST_ONLY = {"i", "v", "x"}  # also common variable names ("make-x-and-y-equal"), so only trust them at the end


def _stem(slug: str) -> tuple[str, ...]:
    tokens = slug.split("-")
    for i, t in enumerate(tokens):
        if i > 0 and (t in _ROMAN or (t in _ROMAN_LAST_ONLY and i == len(tokens) - 1)):
            return tuple(tokens[:i])
    return tuple(tokens)


def family_keys(slugs: list[str]) -> np.ndarray:
    """A problem joins the family of the shortest (>= 3 token) stem that prefixes its slug."""
    stems = {_stem(s) for s in slugs}
    keys = []
    for s in slugs:
        tokens = tuple(s.split("-"))
        best = _stem(s)
        for n in range(3, len(tokens) + 1):
            if tokens[:n] in stems:
                best = tokens[:n]
                break
        keys.append("-".join(best))
    return np.array(keys)


def temporal_split(n: int, test_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows must already be sorted by (date, qid). Returns (dev_idx, test_idx)."""
    n_test = int(round(n * test_frac))
    idx = np.arange(n)
    return idx[: n - n_test], idx[n - n_test :]


def grouped_folds(groups: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    # GroupKFold is deterministic; shuffle group identities with the seed first so folds vary across seeds.
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    relabel = dict(zip(uniq, rng.permutation(len(uniq))))
    g = np.array([relabel[x] for x in groups])
    return list(GroupKFold(n_splits=n_folds).split(np.zeros(len(g)), groups=g))
