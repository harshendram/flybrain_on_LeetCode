import numpy as np
from scipy import sparse

from leetfly.embodied.brain import OnlineBrain
from leetfly.evolve.fitness import NoseFitness
from leetfly.features.antennal_lobe import Nose
from leetfly.fly import FlyConfig


class ToyTask:
    def __init__(self, rng, n=120, g=12, c=3, n_kc=200):
        self.receptors = rng.gamma(0.6, 1.0, size=(n, g))
        self.y = rng.random((n, c)) < 0.35
        self.y[:, 0] |= self.receptors[:, 0] > 1.0
        self.dev, self.test = np.arange(90), np.arange(90, n)
        self.folds = [(self.dev[self.dev % 3 != f], self.dev[self.dev % 3 == f]) for f in range(3)]
        self.w = sparse.csr_matrix((rng.random((n_kc, g)) < 0.3).astype(float))


def test_full_feedback_online_equals_closed_form_scores():
    rng = np.random.default_rng(0)
    t = ToyTask(rng)
    cfg = FlyConfig(sparsity=0.1, eta=4.0)
    nose = Nose.identity(12)
    brain = OnlineBrain(t.w, cfg, t, nose)
    for i in range(len(t.dev)):
        brain.full_feedback(*brain.code("dev", i), t.y[t.dev][i])
    f = NoseFitness(t.w, cfg, t)
    d_dev = f._drive(nose, t.dev)
    scale = d_dev.mean(axis=0) + 1e-12
    top_d, alive_d = f._top(d_dev / scale)
    top_t, alive_t = f._top(f._drive(nose, t.test) / scale)
    closed = f._readout_scores(top_d, alive_d, t.y[t.dev], top_t, alive_t)
    online = np.array([brain.scores(*brain.code("test", i)) for i in range(len(t.test))])
    assert np.allclose(online, closed, atol=1e-9)


def test_bandit_dopamine_touches_one_compartment_and_steers_away():
    rng = np.random.default_rng(1)
    t = ToyTask(rng)
    brain = OnlineBrain(t.w, FlyConfig(sparsity=0.1, eta=4.0), t, Nose.identity(12))
    top, alive = brain.code("dev", 0)
    first = brain.choose(top, alive, set())
    before = brain.w_plus.copy(), brain.w_minus.copy()
    brain.dopamine(top, alive, first.goal, correct=False)
    changed = np.argwhere(brain.w_plus != before[0])
    assert set(changed[:, 1]) == {first.goal} and np.array_equal(brain.w_minus, before[1])
    assert brain.scores(top, alive)[first.goal] < first.scores[first.goal]
    second = brain.choose(top, alive, {first.goal})
    assert second.goal != first.goal and 0.0 <= second.margin <= 1.0
