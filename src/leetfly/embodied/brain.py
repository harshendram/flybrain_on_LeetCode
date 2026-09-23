"""The mushroom body as the fly's high-level controller, learning online from where its body actually lands.

Phase 1 trained the KC->MBON synapses with *full* feedback: every problem sent dopamine to all 14 technique
compartments (reward where the tag is true, punishment elsewhere), which has an exact closed form. A fly in a body
gets far less. It lands on one feeder and tastes sugar or gets shocked **there**, so one compartment learns per
landing: PAM_c depresses the avoid synapses of the active KCs if feeder c was right, and PPL1_c depresses the
approach synapses if it was wrong. The per-event plasticity rate is the same as in Phase 1 (1 - exp(-a), with
Phase 1's balanced a); only the feedback is partial. Where it lands is decided by physics, not by the intent.

KC codes are exactly Phase 1's (same wiring, nose, antennal lobe, homeostasis on the dev odours, APL top-k), so
with full feedback this reproduces the closed form (tests/test_embodied_brain.py).
"""

from dataclasses import dataclass

import numpy as np

from leetfly.evolve.fitness import NoseFitness
from leetfly.features.antennal_lobe import Nose
from leetfly.model.dopamine import _rates

MAX_TRIES = 3


@dataclass
class Choice:
    goal: int  # feeder the brain wants
    margin: float  # (top - second) / (top - lowest), as on the website: small = unsure = cast
    scores: np.ndarray


class OnlineBrain:
    def __init__(self, w_glom, cfg, task, nose: Nose):
        f = NoseFitness(w_glom, cfg, task)
        d_dev = f._drive(nose, task.dev)
        scale = d_dev.mean(axis=0) + 1e-12  # homeostasis: calibrated on the dev odours, no labels
        self.top_dev, self.alive_dev = f._top(d_dev / scale)
        self.top_test, self.alive_test = f._top(f._drive(nose, task.test) / scale)
        self.y_dev, self.y_test = task.y[task.dev], task.y[task.test]
        self.n_kc, self.n_cls = f.n_kc, task.y.shape[1]
        a_c, a_not = _rates(self.y_dev, cfg.eta, cfg.balanced)
        self.d_reward, self.d_punish = 1 - np.exp(-a_c), 1 - np.exp(-a_not)
        self.reset()

    def reset(self) -> None:
        self.w_plus = np.ones((self.n_kc, self.n_cls))
        self.w_minus = np.ones((self.n_kc, self.n_cls))

    def code(self, split: str, i: int) -> tuple[np.ndarray, np.ndarray]:
        return (self.top_dev[i], self.alive_dev[i]) if split == "dev" else (self.top_test[i], self.alive_test[i])

    def scores(self, top: np.ndarray, alive: np.ndarray) -> np.ndarray:
        net = (self.w_plus[top] - self.w_minus[top]) * alive[:, None]
        return net.sum(axis=0) / max(alive.sum(), 1)

    def choose(self, top, alive, tried: set[int]) -> Choice:
        s = self.scores(top, alive)
        order = [c for c in np.argsort(-s, kind="stable") if c not in tried]
        lo = s.min()
        margin = 1.0 if len(order) < 2 else float((s[order[0]] - s[order[1]]) / max(s[order[0]] - lo, 1e-9))
        return Choice(goal=int(order[0]), margin=margin, scores=s)

    def dopamine(self, top, alive, feeder: int, correct: bool) -> None:
        """One landing: sugar (PAM) or a shock (PPL1) in that feeder's compartment only."""
        active = top[alive > 0]
        if correct:
            self.w_minus[active, feeder] *= 1 - self.d_reward[feeder]
        else:
            self.w_plus[active, feeder] *= 1 - self.d_punish[feeder]

    def full_feedback(self, top, alive, tags: np.ndarray) -> None:
        """Phase 1's rule: every compartment learns from every problem (the closed form, applied online)."""
        active = top[alive > 0]
        for c in range(self.n_cls):
            if tags[c]:
                self.w_minus[active, c] *= 1 - self.d_reward[c]
            else:
                self.w_plus[active, c] *= 1 - self.d_punish[c]

    def test_top1(self) -> float:
        """First guess on each future problem, frozen weights; ties give fractional credit."""
        hits = 0.0
        for i in range(len(self.y_test)):
            s = self.scores(self.top_test[i], self.alive_test[i])
            best = np.flatnonzero(s >= s.max() - 1e-12)
            hits += self.y_test[i, best].mean()
        return hits / len(self.y_test)
