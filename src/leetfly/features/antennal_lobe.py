"""The nose (receptor -> glomerulus wiring plus ORN counts) and the antennal lobe's divisive normalization.

The nose is what evolution changes in Phase 2: which receptor type is expressed in which glomerulus (`perm`) and how
many sensory neurons each glomerulus gets (`gain`, with a fixed total budget).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Nose:
    perm: np.ndarray  # (K,) receptor r feeds glomerulus perm[r]
    gain: np.ndarray  # (G,) ORN count per glomerulus relative to the average; sums to G

    @classmethod
    def born(cls, n: int, rng: np.random.Generator) -> "Nose":
        """A random receptor-to-glomerulus map with equal ORN counts."""
        return cls(perm=rng.permutation(n), gain=np.ones(n))

    @classmethod
    def identity(cls, n: int) -> "Nose":
        return cls(perm=np.arange(n), gain=np.ones(n))

    def glomerular_input(self, receptors: np.ndarray) -> np.ndarray:
        x = np.empty_like(receptors)
        x[:, self.perm] = receptors
        return x * self.gain


def divisive_normalization(x: np.ndarray, sigma: float, m: float, n: float = 1.5) -> np.ndarray:
    """PN firing rate from ORN input (Olsen, Bhandawat & Wilson 2010): x^n / (sigma^n + x^n + (m * sum x)^n)."""
    if sigma <= 0 and m <= 0:
        return x
    xn = x**n
    lateral = (m * x.sum(axis=1, keepdims=True)) ** n
    return xn / (sigma**n + xn + lateral)
