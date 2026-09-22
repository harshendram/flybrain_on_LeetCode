"""The technique-prediction task: data, labels, leakage-safe splits and the fitted odor featurizer (cached)."""

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from leetfly import paths
from leetfly.data import labels, leetcode, splits
from leetfly.features.odor import OdorFeaturizer


@dataclass
class TechniqueTask:
    df: pd.DataFrame  # problems with >= 1 technique, sorted by (date, qid)
    y: np.ndarray  # (n, C) bool
    names: list[str]
    dev: np.ndarray
    test: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]  # absolute row indices, all inside dev
    featurizer: OdorFeaturizer
    receptors: np.ndarray  # (n, K)
    tfidf: sparse.csr_matrix  # (n, V)


def build(
    n_receptors: int, test_frac: float = 0.2, min_count: int = 60, n_folds: int = 3, seed: int = 0
) -> TechniqueTask:
    cache = (
        paths.PROCESSED
        / f"task_v{leetcode.TEXT_VERSION}_K{n_receptors}_t{test_frac}_m{min_count}_f{n_folds}_s{seed}.pkl"
    )
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    everything = leetcode.load()
    y_all, all_names = labels.technique_matrix(everything["tags"])
    keep = y_all.any(axis=1)
    df = everything[keep].reset_index(drop=True)
    dev, test = splits.temporal_split(len(df), test_frac)
    names = labels.select_techniques(y_all[keep], all_names, np.isin(np.arange(len(df)), dev), min_count)
    y, _ = labels.technique_matrix(df["tags"], names)

    # the featurizer may read any problem published before the test period (it never sees labels)
    cutoff = df["date"].iloc[test[0]]
    fit_texts = everything.loc[everything["date"] < cutoff, "text"].tolist()
    featurizer = OdorFeaturizer(n_receptors, seed=seed).fit(fit_texts)

    families = splits.family_keys(df["slug"].iloc[dev].tolist())
    folds = [(dev[tr], dev[va]) for tr, va in splits.grouped_folds(families, n_folds, seed)]

    task = TechniqueTask(
        df=df,
        y=y,
        names=names,
        dev=dev,
        test=test,
        folds=folds,
        featurizer=featurizer,
        receptors=featurizer.receptors(df["text"].tolist()),
        tfidf=featurizer.tfidf_matrix(df["text"].tolist()),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(task, f)
    return task
