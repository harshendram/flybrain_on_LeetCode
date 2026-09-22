"""PN -> KC expansion with APL-enforced sparseness.

KC drive is the glomerulus-level synapse-count wiring applied to PN rates. Homeostatic intrinsic plasticity
(Apostolopoulou & Lin 2020) scales each KC so that its mean drive over the odors it has experienced is 1. Without it,
KCs with many claws win every competition regardless of the odor. The APL neuron is modelled as k-winners-take-all.
"""

import numpy as np
from scipy import sparse


def kwta(drive: np.ndarray, k: int) -> np.ndarray:
    """Binary code with the k most driven KCs active (only KCs with positive drive can fire)."""
    top = np.argpartition(-drive, k - 1, axis=1)[:, :k]
    codes = np.zeros(drive.shape, dtype=bool)
    np.put_along_axis(codes, top, True, axis=1)
    return codes & (drive > 0)


class MushroomBody:
    def __init__(self, w_glom: sparse.spmatrix, sparsity: float, homeostasis: bool = True):
        self.w = sparse.csr_matrix(w_glom, dtype=np.float64)  # (n_kc, G)
        self.n_kc = self.w.shape[0]
        self.k = max(1, int(round(sparsity * self.n_kc)))
        self.homeostasis = homeostasis
        self.kc_scale = np.ones(self.n_kc)

    def raw_drive(self, pn: np.ndarray) -> np.ndarray:
        return np.asarray(self.w @ pn.T).T  # (n, n_kc)

    def calibrate(self, pn_pool: np.ndarray) -> "MushroomBody":
        if self.homeostasis:
            self.kc_scale = self.raw_drive(pn_pool).mean(axis=0) + 1e-12
        return self

    def drive(self, pn: np.ndarray) -> np.ndarray:
        return self.raw_drive(pn) / self.kc_scale

    def codes(self, pn: np.ndarray) -> np.ndarray:
        return kwta(self.drive(pn), self.k)
