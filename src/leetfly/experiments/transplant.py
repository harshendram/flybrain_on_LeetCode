"""Phase 2b: the pre-registered analysis (README, "Phase 2 — pre-registration").

1. Born-nose baselines B(t, w): random noses on every wiring used in evolution.
2. Evolvability E = test AUROC of an evolved nose minus the born baseline of the wiring it evolved on.  (H1, H2, H6)
3. Transplants: every nose evolved on a real MB, and every nose evolved on a uniform null (periphery-only
   control), is given to every real MB, which learns with its own dopamine and homeostasis.        (H4, H5)
4. Mechanism: receptor informativeness vs the KC coverage of the glomerulus evolution gave it.       (H3)
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import zlib
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from leetfly import paths
from leetfly import task as task_mod
from leetfly.eval import metrics
from leetfly.evolve.fitness import NoseFitness
from leetfly.experiments.evolve import OUT as EVOLVE_DIR
from leetfly.experiments.evolve import fly_config, wiring_for
from leetfly.features.antennal_lobe import Nose

MBS = ["malecns_R", "malecns_L", "flywire_R", "flywire_L", "hemibrain_R"]
FLY = {"malecns": "male", "flywire": "female#1", "hemibrain": "female#2"}


def pair_kind(s: str, t: str) -> str:
    fs, ft = s.split("_")[0], t.split("_")[0]
    if fs == ft:
        return "within-fly"
    return "same-sex" if FLY[fs].startswith("female") and FLY[ft].startswith("female") else "cross-sex"


def score_noses(mb: str, kind: str, sample: int, noses: list[tuple[list[int], list[float]]]) -> list[dict]:
    task = task_mod.build(n_receptors=51)
    f = NoseFitness(wiring_for(mb, kind, sample), fly_config(), task)
    return [f.test_score(Nose(np.array(p), np.array(g))) for p, g in noses]


def dev_fitness_noses(mb: str, kind: str, sample: int, noses: list[tuple[list[int], list[float]]]) -> list[float]:
    """Exploratory: the evolution fitness (dev folds) of each nose inside this fly's wiring."""
    task = task_mod.build(n_receptors=51)
    f = NoseFitness(wiring_for(mb, kind, sample), fly_config(), task)
    return [f(Nose(np.array(p), np.array(g))) for p, g in noses]


def born_noses(n: int, key: str) -> list[tuple[list[int], list[float]]]:
    """Random born noses, seeded by a stable hash of `key` (Python's hash() differs between processes)."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return [(rng.permutation(51).tolist(), [1.0] * 51) for _ in range(n)]


def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def informativeness(task) -> np.ndarray:
    r, y = task.receptors[task.dev], task.y[task.dev]
    return np.array([np.nanmean(np.abs(metrics.auroc(np.repeat(r[:, [i]], y.shape[1], 1), y) - 0.5))
                     for i in range(r.shape[1])])


def main(jobs: int, n_born_real: int, n_born_null: int) -> None:
    runs = [json.loads(p.read_text()) for p in sorted(EVOLVE_DIR.glob("*.json"))]
    mbs = [m for m in MBS if any(r["mb"] == m for r in runs)]
    wirings = sorted({(r["mb"], r["wiring"], r["sample"]) for r in runs})
    print(f"{len(runs)} evolution runs over {len(mbs)} mushroom bodies")

    # ---- 1 + 3. one parallel batch: born baselines, and transplants onto every real MB ----
    jobs_list, keys = [], []
    for mb, kind, sample in wirings:
        n = n_born_real if kind == "real" else n_born_null
        jobs_list.append((mb, kind, sample, born_noses(n, key=f"born|{mb}|{kind}|{sample}")))
        keys.append(("born", mb, kind, sample))
    donors = [r for r in runs if r["wiring"] in ("real", "uniform")]
    for target in mbs:
        jobs_list.append((target, "real", 0, [(r["perm"], r["gain"]) for r in donors]))
        keys.append(("transplant", target))
    results = Parallel(n_jobs=jobs)(delayed(score_noses)(*j) for j in jobs_list)
    born, transplanted = {}, {}
    for key, res in zip(keys, results):
        if key[0] == "born":
            born[key[1:]] = res
        else:
            transplanted[key[1]] = res

    B = {k: float(np.mean([s["macro_auroc"] for s in v])) for k, v in born.items()}
    B_hit = {k: float(np.mean([s["hit@1"] for s in v])) for k, v in born.items()}

    # ---- 2. evolvability ----
    for r in runs:
        r["E"] = r["test"]["macro_auroc"] - B[(r["mb"], r["wiring"], r["sample"])]
        r["E_hit1"] = r["test"]["hit@1"] - B_hit[(r["mb"], r["wiring"], r["sample"])]
    E = defaultdict(list)  # (mb, kind, seed) -> values over null samples
    for r in runs:
        E[(r["mb"], r["wiring"], r["seed"])].append(r["E"])
    seeds = sorted({r["seed"] for r in runs})
    units = [(m, s) for m in mbs for s in seeds if (m, "real", s) in E]

    def e(m, kind, s):
        return float(np.mean(E[(m, kind, s)]))

    h1 = {m: [e(m, "real", s) for s in seeds if (m, "real", s) in E] for m in mbs}
    diff_uni = np.array([e(m, "real", s) - e(m, "uniform", s) for m, s in units])
    diff_dp = np.array([e(m, "real", s) - e(m, "dp", s) for m, s in units])
    mean_E = {k: float(np.mean([e(m, k, s) for m, s in units])) for k in ("real", "dp", "uniform")}

    # ---- 3. transplants ----
    gain = defaultdict(list)  # (source, target, donor_kind) -> gains over seeds/samples
    for target in mbs:
        for donor, sc in zip(donors, transplanted[target]):
            gain[(donor["mb"], target, donor["wiring"])].append(sc["macro_auroc"] - B[(target, "real", 0)])
    g_self = {t: float(np.mean(h1[t])) for t in mbs}
    tr_rows = []
    for s, t in itertools.product(mbs, mbs):
        g = float(np.mean(gain[(s, t, "real")]))
        p = float(np.mean(gain[(s, t, "uniform")]))
        tr_rows.append({
            "source": s, "target": t, "kind": "self" if s == t else pair_kind(s, t),
            "gain": g, "periphery_only": p, "inherited_wiring": g - p,
            "idiosyncratic": g_self[t] - g, "TR": g / g_self[t] if g_self[t] > 0 else float("nan"),
        })
    cross = [r for r in tr_rows if r["kind"] != "self"]
    tr_by_kind = {k: float(np.median([r["TR"] for r in cross if r["kind"] == k]))
                  for k in ("within-fly", "same-sex", "cross-sex") if any(r["kind"] == k for r in cross)}

    # ---- 4. mechanism ----
    task = task_mod.build(n_receptors=51)
    info = informativeness(task)
    cov_cache = {}
    rho = defaultdict(list)
    for r in runs:
        key = (r["mb"], r["wiring"], r["sample"])
        if key not in cov_cache:
            cov_cache[key] = np.asarray((wiring_for(*key) > 0).sum(axis=0)).ravel()
        cov = cov_cache[key]
        rho[r["wiring"]].append(float(spearmanr(info, cov[np.array(r["perm"])])[0]))

    report = {
        "born_baseline_auroc": {"|".join(map(str, k)): v for k, v in B.items()},
        "H1_gain_self_by_mb": h1,
        "H1_supported": all(len(v) > 0 and min(v) > 0 for v in h1.values()),
        "mean_evolvability": mean_E,
        "H2a_real_minus_uniform": {"mean": float(diff_uni.mean()), "ci95": boot_ci(diff_uni)},
        "H2b_real_minus_dp_abs": float(abs(diff_dp.mean())),
        "H2_supported": bool(boot_ci(diff_uni)[0] > 0 and abs(diff_dp.mean()) < 0.5 * diff_uni.mean()),
        "H3_rho_by_wiring": {k: {"mean": float(np.mean(v)), "ci95": boot_ci(np.array(v)), "n": len(v)}
                             for k, v in rho.items()},
        "H4_median_TR_all": float(np.median([r["TR"] for r in cross])),
        "H4_median_TR_by_kind": tr_by_kind,
        "H4_supported": bool(np.median([r["TR"] for r in cross]) >= 0.7 and all(v >= 0.7 for v in tr_by_kind.values())),
        "H6_real_minus_dp": {"mean": float(diff_dp.mean()), "ci95": boot_ci(diff_dp)},
        "transplants": tr_rows,
        "evolution_runs": [{k: r[k] for k in ("mb", "wiring", "sample", "seed", "dev_fitness", "test", "E", "E_hit1")}
                           for r in runs],
    }
    report["exploratory_dev_transplants"] = exploratory_dev(runs, mbs, jobs)
    (paths.RESULTS / "phase2.json").write_text(json.dumps(report, indent=1))
    print_report(report, mbs)


def exploratory_dev(runs: list[dict], mbs: list[str], jobs: int, n_born: int = 100) -> dict:
    """NOT pre-registered. Transplants scored on the dev problems the noses were evolved for.

    Test-set gains mix two things: does an evolved nose work in another fly's wiring (the question), and do dev-period
    gains survive the drift to newer problems. Scoring on dev removes the drift, so what's left is wiring transfer.
    Self gains are in-sample (the nose was optimised on exactly this score), so the idiosyncratic part is an upper
    bound. Noses evolved on uniform nulls carry only wiring-independent (periphery + problem-set) gains; noses
    evolved on degree-preserving nulls carry coverage-level structure but none of the real partner choices.
    """
    donors = [r for r in runs if r["wiring"] in ("real", "dp", "uniform")]
    batch = [(t, "real", 0, born_noses(n_born, key=f"dev|{t}")) for t in mbs]
    batch += [(t, "real", 0, [(r["perm"], r["gain"]) for r in donors]) for t in mbs]
    out = Parallel(n_jobs=jobs)(delayed(dev_fitness_noses)(*b) for b in batch)
    born = {t: float(np.mean(v)) for t, v in zip(mbs, out[: len(mbs)])}
    gain = defaultdict(list)
    for t, scores in zip(mbs, out[len(mbs):]):
        for r, s in zip(donors, scores):
            gain[(r["mb"], t, r["wiring"])].append(s - born[t])
    rows = []
    for s, t in itertools.product(mbs, mbs):
        g = {k: float(np.mean(gain[(s, t, k)])) for k in ("real", "dp", "uniform")}
        rows.append({"source": s, "target": t, "kind": "self" if s == t else pair_kind(s, t),
                     "gain_real_nose": g["real"], "gain_dp_nose": g["dp"], "gain_uniform_nose": g["uniform"]})
    self_gain = {r["target"]: r["gain_real_nose"] for r in rows if r["kind"] == "self"}
    for r in rows:
        r["TR"] = r["gain_real_nose"] / self_gain[r["target"]] if self_gain[r["target"]] > 0 else float("nan")
        r["inherited_wiring"] = r["gain_real_nose"] - r["gain_uniform_nose"]
    by_kind = {k: {"median_TR": float(np.median([r["TR"] for r in rows if r["kind"] == k])),
                   "mean_gain_real_nose": float(np.mean([r["gain_real_nose"] for r in rows if r["kind"] == k])),
                   "mean_gain_dp_nose": float(np.mean([r["gain_dp_nose"] for r in rows if r["kind"] == k])),
                   "mean_gain_uniform_nose": float(np.mean([r["gain_uniform_nose"] for r in rows if r["kind"] == k]))}
               for k in ("self", "within-fly", "same-sex", "cross-sex") if any(r["kind"] == k for r in rows)}
    return {"born_dev_fitness": born, "rows": rows, "by_kind": by_kind}


def print_report(r: dict, mbs: list[str]) -> None:
    print("\nH1 evolution beats born noses (test AUROC gain, per seed):")
    for m, v in r["H1_gain_self_by_mb"].items():
        print(f"  {m:12s} " + " ".join(f"{x:+.4f}" for x in v))
    print(f"  -> supported: {r['H1_supported']}")
    print("mean evolvability E:", {k: round(v, 4) for k, v in r["mean_evolvability"].items()})
    print(f"H2a real-uniform {r['H2a_real_minus_uniform']['mean']:+.4f} CI {np.round(r['H2a_real_minus_uniform']['ci95'], 4)}"
          f" | H2b |real-dp| {r['H2b_real_minus_dp_abs']:.4f} -> supported: {r['H2_supported']}")
    print("H3 spearman(informativeness, coverage of assigned glomerulus):")
    for k, v in r["H3_rho_by_wiring"].items():
        print(f"  {k:8s} mean {v['mean']:+.3f} CI {np.round(v['ci95'], 3)} (n={v['n']})")
    print(f"H4 median TR all {r['H4_median_TR_all']:.3f}; by kind {({k: round(v, 3) for k, v in r['H4_median_TR_by_kind'].items()})}"
          f" -> supported: {r['H4_supported']}")
    print(f"H6 real-dp {r['H6_real_minus_dp']['mean']:+.4f} CI {np.round(r['H6_real_minus_dp']['ci95'], 4)}")
    print("\ntransplant gains (rows = source nose, cols = target fly), AUROC above born baseline:")
    print(" " * 13 + "".join(f"{t:>13s}" for t in mbs))
    rows = {(x["source"], x["target"]): x for x in r["transplants"]}
    for s in mbs:
        print(f"{s:13s}" + "".join(f"{rows[(s, t)]['gain']:+13.4f}" for t in mbs))
    print("periphery-only (nose evolved on the source's uniform null):")
    for s in mbs:
        print(f"{s:13s}" + "".join(f"{rows[(s, t)]['periphery_only']:+13.4f}" for t in mbs))

    ex = r["exploratory_dev_transplants"]
    print("\nEXPLORATORY (not pre-registered): transplants scored on the dev problems (no temporal drift)")
    print(f"{'pair kind':12s} {'median TR':>10s} {'real nose':>10s} {'dp nose':>10s} {'uniform nose':>13s}   (mean gain in dev AUROC)")
    for k, v in ex["by_kind"].items():
        print(f"{k:12s} {v['median_TR']:10.3f} {v['mean_gain_real_nose']:+10.4f} {v['mean_gain_dp_nose']:+10.4f} "
              f"{v['mean_gain_uniform_nose']:+13.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--n-born-real", type=int, default=200)
    ap.add_argument("--n-born-null", type=int, default=100)
    a = ap.parse_args()
    main(a.jobs, a.n_born_real, a.n_born_null)
