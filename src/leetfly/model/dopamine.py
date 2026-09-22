"""KC -> MBON plasticity gated by dopamine. This is the only learning in the fly.

Each technique c has an approach MBON (weights w_plus) and an avoid MBON (w_minus), and every KC synapses onto both
(initial weight 1). On a training problem:
  - c is one of its techniques -> reward DAN (PAM_c) depresses the avoid synapses of the active KCs
  - c is not                   -> punishment DAN (PPL1_c) depresses the approach synapses of the active KCs
Depression is multiplicative, w *= exp(-a) per coincident event, with a = eta / (number of problems of that kind)
when balanced. So each technique's two MBONs see an equal total amount of dopamine however rare the technique is.

Multiplication commutes, so the order of training problems doesn't matter and the result has an exact closed form:
    w_plus[j, c]  = exp(-a_not_c * #{non-c problems where KC j fired})
    w_minus[j, c] = exp(-a_c     * #{c problems where KC j fired})
The closed form is what makes Phase-2 evolution cheap; `online_fit` is the literal rule, used to verify it.
"""

from dataclasses import dataclass

import numpy as np


def _rates(y: np.ndarray, eta: float, balanced: bool) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    n_c = y.sum(axis=0).astype(np.float64)
    if balanced:
        return eta / np.maximum(n_c, 1), eta / np.maximum(n - n_c, 1)
    a = np.full(y.shape[1], eta / n)
    return a, a


@dataclass
class DopamineReadout:
    eta: float = 2.0
    balanced: bool = True

    def fit(self, codes: np.ndarray, y: np.ndarray) -> "DopamineReadout":
        k = codes.astype(np.float32)
        yf = y.astype(np.float32)
        count_c = k.T @ yf  # (n_kc, C) times KC j fired on a problem with technique c
        count_not = k.sum(axis=0)[:, None] - count_c
        a_c, a_not = _rates(y, self.eta, self.balanced)
        self.w_plus = np.exp(-a_not * count_not)
        self.w_minus = np.exp(-a_c * count_c)
        return self

    def scores(self, codes: np.ndarray) -> np.ndarray:
        k = codes.astype(np.float32)
        return (k @ (self.w_plus - self.w_minus)) / np.maximum(k.sum(axis=1, keepdims=True), 1)


def online_fit(codes: np.ndarray, y: np.ndarray, eta: float, balanced: bool) -> tuple[np.ndarray, np.ndarray]:
    """The literal trial-by-trial rule: w *= (1 - delta) on active KCs, delta = 1 - exp(-a)."""
    n_kc, n_cls = codes.shape[1], y.shape[1]
    w_plus, w_minus = np.ones((n_kc, n_cls)), np.ones((n_kc, n_cls))
    a_c, a_not = _rates(y, eta, balanced)
    d_c, d_not = 1 - np.exp(-a_c), 1 - np.exp(-a_not)
    for k, tags in zip(codes, y):
        for c in range(n_cls):
            if tags[c]:
                w_minus[k, c] *= 1 - d_c[c]  # PAM_c: reward
            else:
                w_plus[k, c] *= 1 - d_not[c]  # PPL1_c: punishment
    return w_plus, w_minus


def reward_update(w_plus, w_minus, k, guess: int, correct: bool, delta: float, recovery: float) -> None:
    """The demo's "train your fly": the fly guesses, then gets sugar (PAM) or a shock (PPL1) on that compartment."""
    if correct:
        w_minus[k, guess] *= 1 - delta
    else:
        w_plus[k, guess] *= 1 - delta
    w_plus += recovery * (1 - w_plus)
    w_minus += recovery * (1 - w_minus)
