"""Phase 2a: evolve a nose on each mushroom body, on its real wiring and on null rewirings of it.

A run = (mushroom body, wiring, null sample, seed). Null wirings are fixed per (body, kind, sample) so that seeds
differ only in evolution. Everything downstream of the nose uses the frozen Phase-1 settings. Each run saves its
evolved nose, dev fitness, test scores and the per-generation history (for the 3D replay).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker; we parallelise over runs instead

import argparse
import itertools
import json
import pickle
import platform
import time
import zlib

import numpy as np
import yaml
from joblib import Parallel, delayed

from leetfly import paths
from leetfly import task as task_mod
from leetfly.connectome import nulls
from leetfly.connectome.extract_mb import MBCircuit, circuit_path
from leetfly.evolve.fitness import NoseFitness
from leetfly.evolve.strategy import ESConfig, evolve

OUT = paths.RESULTS / "evolve"


def wiring_for(mb: str, kind: str, sample: int, min_syn: int = 5):
    w = MBCircuit.load(circuit_path(mb, min_syn)).w_glom()
    if kind == "real":
        return w
    rng = np.random.default_rng(zlib.crc32(f"{mb}|{kind}|{sample}".encode()))
    return nulls.degree_preserving(w, rng) if kind == "dp" else nulls.uniform(w, rng)


def run_name(mb: str, kind: str, sample: int, seed: int) -> str:
    return f"{mb}__{kind}{sample}__s{seed}"


def fly_config():
    with open(paths.RESULTS / "cache" / "phase1_model.pkl", "rb") as f:
        return pickle.load(f)["config"]


def run_one(mb: str, kind: str, sample: int, seed: int, es: dict, replay_every: int) -> str:
    name = run_name(mb, kind, sample, seed)
    out_path = OUT / f"{name}.json"
    if out_path.exists():
        return f"{name}: cached"
    t0 = time.time()
    task = task_mod.build(n_receptors=51)
    fitness = NoseFitness(wiring_for(mb, kind, sample), fly_config(), task)
    best, hist = evolve(fitness, task.receptors.shape[1], ESConfig(**es), np.random.default_rng(seed))
    nose = best.nose()
    result = {
        "mb": mb,
        "wiring": kind,
        "sample": sample,
        "seed": seed,
        "dev_fitness": best.fitness,
        "test": fitness.test_score(nose),
        "perm": nose.perm.tolist(),
        "gain": nose.gain.tolist(),
        "history": {
            "best_fitness": hist.best_fitness,
            "mean_fitness": hist.mean_fitness,
            "sigma": hist.sigma,
            "replay_every": replay_every,
            "best_perm": hist.best_perm[::replay_every],
            "best_gain": hist.best_gain[::replay_every],
        },
        "minutes": (time.time() - t0) / 60,
        "host": platform.node(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result))
    return f"{name}: dev {best.fitness:.4f} test AUROC {result['test']['macro_auroc']:.4f} ({result['minutes']:.1f} min)"


def main(config_path: str, jobs: int, only_mbs: list[str] | None = None) -> None:
    conf = yaml.safe_load(open(config_path))
    if only_mbs:
        conf["mbs"] = [m for m in conf["mbs"] if m in only_mbs]
    runs = []
    for mb, seed in itertools.product(conf["mbs"], conf["seeds"]):
        runs.append((mb, "real", 0, seed))
        runs += [(mb, "dp", s, seed) for s in range(conf["null_samples"])]
        runs += [(mb, "uniform", s, seed) for s in range(conf["null_samples"])]
    task_mod.build(n_receptors=51)  # build the cache once before forking
    print(f"{len(runs)} runs on {jobs} workers")
    for msg in Parallel(n_jobs=jobs, verbose=0, return_as="generator_unordered")(
        delayed(run_one)(*r, conf["es"], conf["replay_every"]) for r in runs
    ):
        print(msg, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.ROOT / "configs" / "evolve.yaml"))
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--mbs", nargs="*", help="subset of the configured mushroom bodies to run now")
    args = ap.parse_args()
    main(args.config, args.jobs, args.mbs)
