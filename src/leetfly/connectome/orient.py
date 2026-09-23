"""One display orientation for every fly on the site.

The 3D scene uses the JRC2018U convention: x across the brain, y dorsal -> ventral, z anterior -> posterior.
MaleCNS EM space (where both the skeletons and the 141k soma positions live) has its own axes. MaleCNS publishes
each skeleton in EM space and warped into JRC2018U with the same node order, so the orthogonal map between the two
frames is a Procrustes fit over matching nodes (the non-linear part of the warp stays in the residual).
"""

import json

import numpy as np

from leetfly import paths

EM_VOXEL_UM = 0.008  # MaleCNS EM voxels are 8 nm
ROTATION_FILE = paths.PROCESSED / "malecns_em_to_display.json"


def procrustes_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||(a - mean a) R - (b - mean b)||; reflections allowed (axis conventions differ)."""
    a0, b0 = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a0.T @ b0)
    return u @ vt


def save_rotation(r: np.ndarray, residual_um: float, n_pairs: int) -> None:
    ROTATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    ROTATION_FILE.write_text(json.dumps({"R": r.tolist(), "residual_um": residual_um, "n_pairs": n_pairs}))


def load_rotation() -> np.ndarray:
    return np.array(json.loads(ROTATION_FILE.read_text())["R"])


def em_to_display(xyz_voxels: np.ndarray, r: np.ndarray) -> np.ndarray:
    return (np.asarray(xyz_voxels, dtype=np.float64) * EM_VOXEL_UM) @ r
