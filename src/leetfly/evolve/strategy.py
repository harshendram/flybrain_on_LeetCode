"""Evolving the nose: a (mu + lambda) evolution strategy over receptor->glomerulus maps and ORN counts.

Genotype: a permutation (which receptor each glomerulus expresses) and log ORN counts z. The phenotype's gains are
exp(z) rescaled to mean 1 (a fixed total ORN budget) and clipped to [0.25, 4], the order of the changes seen in
host-specialist Drosophila (e.g. ~2x Or22a neurons in D. sechellia). Mutation swaps receptors between glomeruli and
jitters z; the jitter size follows the 1/5 success rule.
"""

from dataclasses import dataclass, field

import numpy as np

from leetfly.features.antennal_lobe import Nose

GAIN_MIN, GAIN_MAX = 0.25, 4.0


def gains_from(z: np.ndarray) -> np.ndarray:
    """exp(z) projected onto {mean = 1, GAIN_MIN <= g <= GAIN_MAX} by alternating clip and rescale."""
    g = np.exp(z - z.mean())
    for _ in range(50):
        g = np.clip(g / g.mean(), GAIN_MIN, GAIN_MAX)
        if abs(g.mean() - 1) < 1e-9:
            break
    return g / g.mean()


@dataclass
class Genome:
    perm: np.ndarray
    z: np.ndarray
    fitness: float = -np.inf

    def nose(self) -> Nose:
        return Nose(perm=self.perm.copy(), gain=gains_from(self.z))


@dataclass
class ESConfig:
    mu: int = 8
    lam: int = 24
    generations: int = 150
    sigma0: float = 0.3
    p_swap: float = 0.8
    max_swaps: int = 3


@dataclass
class History:
    best_fitness: list[float] = field(default_factory=list)
    mean_fitness: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    best_perm: list[list[int]] = field(default_factory=list)
    best_gain: list[list[float]] = field(default_factory=list)


def evolve(fitness, n: int, cfg: ESConfig, rng: np.random.Generator, log_every: int = 0) -> tuple[Genome, History]:
    parents = [Genome(rng.permutation(n), np.zeros(n)) for _ in range(cfg.mu)]
    for g in parents:
        g.fitness = fitness(g.nose())
    sigma = cfg.sigma0
    hist = History()
    for gen in range(cfg.generations):
        children = []
        for _ in range(cfg.lam):
            parent = parents[rng.integers(cfg.mu)]
            perm = parent.perm.copy()
            if rng.random() < cfg.p_swap:
                for _ in range(rng.integers(1, cfg.max_swaps + 1)):
                    i, j = rng.choice(n, 2, replace=False)
                    perm[i], perm[j] = perm[j], perm[i]
            child = Genome(perm, parent.z + sigma * rng.standard_normal(n))
            child.fitness = fitness(child.nose())
            children.append((child, child.fitness > parent.fitness))
        success = np.mean([s for _, s in children])
        sigma *= np.exp((success - 0.2) / 0.8)  # 1/5 success rule
        sigma = float(np.clip(sigma, 0.02, 1.0))
        pool = parents + [c for c, _ in children]
        parents = sorted(pool, key=lambda g: -g.fitness)[: cfg.mu]
        best = parents[0]
        hist.best_fitness.append(best.fitness)
        hist.mean_fitness.append(float(np.mean([g.fitness for g in parents])))
        hist.sigma.append(sigma)
        hist.best_perm.append(best.perm.tolist())
        hist.best_gain.append(np.round(gains_from(best.z), 4).tolist())
        if log_every and (gen + 1) % log_every == 0:
            print(f"  gen {gen + 1:4d}  best {best.fitness:.4f}  mean {hist.mean_fitness[-1]:.4f}  sigma {sigma:.3f}")
    return parents[0], hist
