"""Phase 2, descriptive: how much of the PN->KC wiring is shared between flies?

- Coverage: how many KCs each glomerulus reaches. Correlated across mushroom bodies => an inherited bias.
- Co-convergence beyond coverage: for each glomerulus pair, the number of KCs sampling both, z-scored against
  degree-preserving rewirings (which keep coverage). Correlated residuals across MBs => inherited higher-order structure.
- Sexual dimorphism: coverage of pheromone glomeruli (DA1: cVA; VA1v, VL2a: fruit/food-pheromone related) by sex.
"""

import itertools
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

from leetfly import paths
from leetfly.connectome import nulls
from leetfly.connectome.extract_mb import MBCircuit, circuit_path
from leetfly.connectome.stats import kc_class

MBS = ["malecns_R", "malecns_L", "flywire_R", "flywire_L", "hemibrain_R"]
SEX = {"malecns": "male", "flywire": "female", "hemibrain": "female"}
PHEROMONE = ["DA1", "VA1v", "VL2a"]


def coconvergence(b) -> np.ndarray:
    """(G, G) number of KCs that sample both glomeruli."""
    b = b.astype(np.float64)
    return np.asarray((b.T @ b).todense())


def residual_coconvergence(w, rng, n_null: int = 20, groups=None) -> np.ndarray:
    real = coconvergence(w > 0)
    null = np.stack([coconvergence(nulls.degree_preserving(w, rng, groups=groups) > 0) for _ in range(n_null)])
    return (real - null.mean(0)) / (null.std(0) + 1e-9)


def main(n_null: int) -> None:
    rng = np.random.default_rng(7)
    circuits = {m: MBCircuit.load(circuit_path(m, 5)) for m in MBS if circuit_path(m, 5).exists()}
    names = list(circuits)
    glom = circuits[names[0]].glomeruli
    assert all((c.glomeruli == glom).all() for c in circuits.values()), "glomerulus order must match for transplants"

    cov = {m: np.asarray((c.w_glom() > 0).sum(axis=0)).ravel() for m, c in circuits.items()}
    frac = {m: v / circuits[m].n_kc for m, v in cov.items()}
    res = {m: residual_coconvergence(c.w_glom(), rng, n_null) for m, c in circuits.items()}
    # stricter null: rewire only within KC class (gamma / alpha'beta' / alphabeta), keeping each class's coverage
    classes = {m: np.array([kc_class(t) for t in c.kc_types]) for m, c in circuits.items()}
    res_cls = {m: residual_coconvergence(c.w_glom(), rng, n_null, classes[m]) for m, c in circuits.items()}
    iu = np.triu_indices(len(glom), 1)

    pairs = []
    for a, b in itertools.combinations(names, 2):
        kind = (
            "within-fly" if a.split("_")[0] == b.split("_")[0]
            else "same-sex" if SEX[a.split("_")[0]] == SEX[b.split("_")[0]]
            else "cross-sex"
        )
        pairs.append({
            "a": a, "b": b, "kind": kind,
            "coverage_spearman": float(spearmanr(frac[a], frac[b])[0]),
            "coconv_residual_pearson": float(pearsonr(res[a][iu], res[b][iu])[0]),
            "coconv_residual_within_class_pearson": float(pearsonr(res_cls[a][iu], res_cls[b][iu])[0]),
        })
    # reference: residuals of a null graph vs a real MB should not correlate
    null_ref = [
        float(pearsonr(residual_coconvergence(nulls.degree_preserving(circuits[m].w_glom(), rng), rng, 5)[iu],
                       res[m][iu])[0])
        for m in names
    ]
    out = {
        "glomeruli": glom.tolist(),
        "coverage_fraction": {m: v.round(4).tolist() for m, v in frac.items()},
        "coverage_ratio_max_min": {m: float(v.max() / max(v.min(), 1)) for m, v in cov.items()},
        "pairs": pairs,
        "null_residual_vs_real_pearson": null_ref,
        "pheromone_coverage_fraction": {
            m: {g: float(frac[m][list(glom).index(g)]) for g in PHEROMONE} for m in names
        },
    }
    paths.RESULTS.mkdir(exist_ok=True)
    (paths.RESULTS / "describe.json").write_text(json.dumps(out, indent=1))

    print("coverage max/min:", {m: round(v, 1) for m, v in out["coverage_ratio_max_min"].items()})
    print(f"{'pair':28s} {'kind':11s} coverage rho   co-conv. residual r   (within-class null)")
    for p in pairs:
        print(f"{p['a'] + ' ~ ' + p['b']:28s} {p['kind']:11s} {p['coverage_spearman']:8.3f}   "
              f"{p['coconv_residual_pearson']:8.3f}            {p['coconv_residual_within_class_pearson']:8.3f}")
    print("null-vs-real residual r (should be ~0):", np.round(null_ref, 3))
    print("pheromone glomeruli, fraction of KCs reached:")
    for m, d in out["pheromone_coverage_fraction"].items():
        print(f"  {m:12s} " + "  ".join(f"{g} {v:.3f}" for g, v in d.items()))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=20)
    main(ap.parse_args().n_null)
