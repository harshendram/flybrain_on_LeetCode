"""How well does a fly with a given nose learn LeetCode techniques? Fast enough to evaluate thousands of noses.

Everything downstream of the nose is fixed: antennal-lobe normalization, the fly's PN->KC wiring (real or null),
homeostasis, APL sparseness and the dopamine rule (Phase-1 settings, frozen). Fitness = mean macro-AUROC over the
grouped dev folds, using the exact closed form of the dopamine rule. Test scores are only computed by `test_score`.

Speed matters (evolution needs ~3,600 evaluations per run), so KC codes are kept as top-k index lists and the
dopamine counts are bincounts. `reference_fitness` is the plain implementation (model/*.py), used by the tests.
"""

import numpy as np
from scipy import sparse

from leetfly.eval import metrics
from leetfly.features.antennal_lobe import Nose, divisive_normalization
from leetfly.fly import FlyConfig
from leetfly.model.dopamine import DopamineReadout, _rates
from leetfly.model.mushroom_body import kwta


class NoseFitness:
    def __init__(self, w_glom: sparse.spmatrix, cfg: FlyConfig, task):
        self.cfg = cfg
        self.w_t = np.asarray(sparse.csr_matrix(w_glom).todense(), dtype=np.float32).T  # (G, n_kc)
        self.n_kc = self.w_t.shape[1]
        self.k = max(1, int(round(cfg.sparsity * self.n_kc)))
        self.task = task
        self.receptors = task.receptors.astype(np.float32)
        pos = {r: i for i, r in enumerate(task.dev)}
        self.folds = [(np.array([pos[r] for r in tr]), np.array([pos[r] for r in va])) for tr, va in task.folds]
        self.y_dev = task.y[task.dev]

    def _drive(self, nose: Nose, idx: np.ndarray) -> np.ndarray:
        x = nose.glomerular_input(self.receptors[idx])
        pn = divisive_normalization(x, self.cfg.sigma, self.cfg.m).astype(np.float32)
        pn[pn < 1e-30] = 0  # float32 denormals (from ~1e-45 NMF codes ** 1.5) make the matmul ~10x slower
        return pn @ self.w_t

    def _top(self, drive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Indices of the k firing KCs per row, and a 0/1 weight (0 where a 'winner' had no input at all)."""
        top = np.argpartition(drive, self.n_kc - self.k, axis=1)[:, self.n_kc - self.k :]
        alive = (np.take_along_axis(drive, top, axis=1) > 0).astype(np.float32)
        return top, alive

    def _readout_scores(self, top_tr, alive_tr, y_tr, top_va, alive_va) -> np.ndarray:
        n_cls = y_tr.shape[1]
        count_c = np.empty((self.n_kc, n_cls), dtype=np.float64)
        for c in range(n_cls):
            rows = y_tr[:, c]
            count_c[:, c] = np.bincount(top_tr[rows].ravel(), weights=alive_tr[rows].ravel(), minlength=self.n_kc)
        count_all = np.bincount(top_tr.ravel(), weights=alive_tr.ravel(), minlength=self.n_kc)
        a_c, a_not = _rates(y_tr, self.cfg.eta, self.cfg.balanced)
        net = np.exp(-a_not * (count_all[:, None] - count_c)) - np.exp(-a_c * count_c)  # w_plus - w_minus
        s = (net[top_va] * alive_va[:, :, None]).sum(axis=1)
        return s / np.maximum(alive_va.sum(axis=1, keepdims=True), 1)

    def __call__(self, nose: Nose) -> float:
        drive = self._drive(nose, self.task.dev)
        drive /= drive.mean(axis=0) + 1e-12  # homeostasis on the dev odors (label-free)
        top, alive = self._top(drive)
        vals = []
        for tr, va in self.folds:
            s = self._readout_scores(top[tr], alive[tr], self.y_dev[tr], top[va], alive[va])
            vals.append(metrics.macro_auroc(s, self.y_dev[va]))
        return float(np.mean(vals))

    def test_score(self, nose: Nose) -> dict[str, float]:
        t = self.task
        d_dev = self._drive(nose, t.dev)
        scale = d_dev.mean(axis=0) + 1e-12
        top_d, alive_d = self._top(d_dev / scale)
        top_t, alive_t = self._top(self._drive(nose, t.test) / scale)
        s = self._readout_scores(top_d, alive_d, t.y[t.dev], top_t, alive_t)
        return metrics.summary(s, t.y[t.test])


def reference_fitness(w_glom, cfg: FlyConfig, task, nose: Nose) -> float:
    """Slow, plain version of NoseFitness.__call__ built from the model classes (for tests)."""
    from leetfly.fly import Fly

    fly = Fly(w_glom, cfg, nose).calibrate(task.receptors[task.dev])
    codes = kwta(fly.mb.drive(fly.pn(task.receptors)), fly.mb.k)
    vals = []
    for tr, va in task.folds:
        r = DopamineReadout(cfg.eta, cfg.balanced).fit(codes[tr], task.y[tr])
        vals.append(metrics.macro_auroc(r.scores(codes[va]), task.y[va]))
    return float(np.mean(vals))
