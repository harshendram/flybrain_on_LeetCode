"""One fly: nose -> antennal lobe -> mushroom body. The dopamine readout is fit separately (model/dopamine.py)."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import sparse

from leetfly.features.antennal_lobe import Nose, divisive_normalization
from leetfly.model.mushroom_body import MushroomBody


@dataclass(frozen=True)
class FlyConfig:
    sigma: float = 1.0  # antennal-lobe semi-saturation
    m: float = 0.05  # lateral (divisive) inhibition strength
    sparsity: float = 0.05  # fraction of KCs the APL lets fire
    eta: float = 2.0  # total dopamine depression per technique
    balanced: bool = True
    homeostasis: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class Fly:
    def __init__(self, w_glom: sparse.spmatrix, config: FlyConfig, nose: Nose):
        self.config = config
        self.nose = nose
        self.mb = MushroomBody(w_glom, config.sparsity, config.homeostasis)

    def pn(self, receptors: np.ndarray) -> np.ndarray:
        return divisive_normalization(self.nose.glomerular_input(receptors), self.config.sigma, self.config.m)

    def calibrate(self, receptors_pool: np.ndarray) -> "Fly":
        """Homeostasis: KCs adapt their excitability to the odors this fly has experienced (no labels)."""
        self.mb.calibrate(self.pn(receptors_pool))
        return self

    def codes(self, receptors: np.ndarray) -> np.ndarray:
        return self.mb.codes(self.pn(receptors))
