"""Non-fly baselines, evaluated with exactly the same splits and metrics as the fly."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


def frequency_scores(y_train: np.ndarray, n: int) -> np.ndarray:
    """B0: always rank techniques by how common they were in training ("always guess DP")."""
    return np.tile(y_train.mean(axis=0), (n, 1))


def logistic_scores(x_train, y_train, x_eval, c: float, scale: bool) -> np.ndarray:
    if scale:
        s = StandardScaler(with_mean=not hasattr(x_train, "tocsr")).fit(x_train)
        x_train, x_eval = s.transform(x_train), s.transform(x_eval)
    clf = OneVsRestClassifier(LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))
    clf.fit(x_train, y_train.astype(int))
    return clf.decision_function(x_eval)
