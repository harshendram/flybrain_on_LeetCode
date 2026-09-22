from types import SimpleNamespace

import numpy as np
from scipy import sparse

from leetfly.evolve.fitness import NoseFitness, reference_fitness
from leetfly.evolve.strategy import ESConfig, GAIN_MAX, GAIN_MIN, evolve, gains_from
from leetfly.features.antennal_lobe import Nose
from leetfly.fly import FlyConfig


def toy_task(rng, n=240, n_glom=12, n_cls=3):
    receptors = rng.gamma(0.6, 1.0, size=(n, n_glom))
    y = rng.random((n, n_cls)) < 0.3
    y[:, 0] |= receptors[:, 0] > 1.2  # make technique 0 depend on receptor 0
    dev, test = np.arange(180), np.arange(180, n)
    folds = [(dev[dev % 3 != f], dev[dev % 3 == f]) for f in range(3)]
    return SimpleNamespace(receptors=receptors, y=y, dev=dev, test=test, folds=folds)


def toy_wiring(rng, n_kc=300, n_glom=12, claws=4):
    rows = np.repeat(np.arange(n_kc), claws)
    cols = np.concatenate([rng.choice(n_glom, claws, replace=False) for _ in range(n_kc)])
    return sparse.csr_matrix((rng.integers(5, 30, len(rows)).astype(float), (rows, cols)), shape=(n_kc, n_glom))


def test_fast_fitness_matches_reference():
    rng = np.random.default_rng(0)
    t, w = toy_task(rng), toy_wiring(rng)
    cfg = FlyConfig(sigma=1.0, m=0.05, sparsity=0.05, eta=8, balanced=True)
    f = NoseFitness(w, cfg, t)
    for seed in range(4):
        nose = Nose.born(12, np.random.default_rng(seed))
        nose.gain = gains_from(np.random.default_rng(seed + 9).normal(0, 0.7, 12))
        assert abs(f(nose) - reference_fitness(w, cfg, t, nose)) < 1e-6


def test_gains_respect_budget_and_bounds():
    for seed in range(20):
        g = gains_from(np.random.default_rng(seed).normal(0, 3, 50))
        assert abs(g.mean() - 1) < 1e-9
        assert g.min() >= GAIN_MIN * (1 - 1e-6) and g.max() <= GAIN_MAX * (1 + 1e-6)


def test_evolution_climbs_a_toy_landscape():
    target = np.random.default_rng(2).permutation(10)

    def fitness(nose):
        return float((nose.perm == target).mean())

    best, hist = evolve(fitness, 10, ESConfig(mu=4, lam=12, generations=60), np.random.default_rng(3))
    assert hist.best_fitness[-1] >= hist.best_fitness[0]
    assert best.fitness >= 0.5
    assert all(b2 >= b1 for b1, b2 in zip(hist.best_fitness, hist.best_fitness[1:]))  # elitist
