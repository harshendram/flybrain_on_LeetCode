"""Phase 3 streams: the fly learns the 1,658 dev problems one at a time, then is tested on the 415 future ones.

    python -m leetfly.embodied.run --feedback full bandit            (laptop: no physics)
    python -m leetfly.embodied.run --feedback embodied --jobs 6      (AWS: flybody physics)

Feedback conditions (everything else identical: wiring, nose, KC codes, plasticity rate, problem order):
  full      Phase 1's rule, every compartment learns every problem (the closed form, online).
  bandit    The fly lands where it intended; dopamine only in that feeder's compartment; up to 3 tries.
  embodied  The mushroom body's choice becomes a flight path (plan.py) flown by the pretrained flybody controller
            in MuJoCo; dopamine goes to whichever feeder the body actually reached.
Seeds shuffle the training order (seed 0 keeps LeetCode's chronological order).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np

from leetfly import paths
from leetfly import task as task_mod
from leetfly.embodied.brain import MAX_TRIES, OnlineBrain
from leetfly.experiments.evolve import wiring_for

OUT = paths.RESULTS / "embodied"
MB = "malecns_R"
WIRINGS = {"real": ("real", 0), "dp": ("dp", 0), "uni": ("uni", 0)}
CURVE_EVERY = 100
KEEP_EVERY = 20  # keep the flown path of every 20th training problem's first flight (for the website replay)
KEEP_TEST = 30


ARRAYS = paths.PROCESSED / "embodied_task_K51.npz"  # pandas-free copy of what the brain needs (for AWS)


def save_arrays() -> Path:
    """The pickled task and Phase 1 model hold pandas objects; flybody's TF 2.8 stack pins an older pandas that
    can't unpickle them. The brain needs only arrays, so ship those."""
    with open(paths.RESULTS / "cache" / "phase1_model.pkl", "rb") as f:
        p1 = pickle.load(f)
    task = task_mod.build(n_receptors=51)
    folds = {f"fold{k}_{part}": idx for k, fold in enumerate(task.folds) for part, idx in zip(("tr", "va"), fold)}
    cfg = p1["config"]
    np.savez_compressed(
        ARRAYS, receptors=task.receptors, y=task.y, dev=task.dev, test=task.test, n_folds=len(task.folds),
        nose_perm=p1["nose"].perm, nose_gain=p1["nose"].gain,
        config=json.dumps(cfg.to_dict()), **folds,
    )
    return ARRAYS


def load_arrays():
    from types import SimpleNamespace

    from leetfly.features.antennal_lobe import Nose
    from leetfly.fly import FlyConfig

    z = np.load(ARRAYS)
    folds = [(z[f"fold{k}_tr"], z[f"fold{k}_va"]) for k in range(int(z["n_folds"]))]
    task = SimpleNamespace(receptors=z["receptors"], y=z["y"], dev=z["dev"], test=z["test"], folds=folds)
    return task, FlyConfig(**json.loads(str(z["config"]))), Nose(perm=z["nose_perm"], gain=z["nose_gain"])


def make_brain(wiring: str) -> tuple[OnlineBrain, object]:
    if not ARRAYS.exists():
        save_arrays()
    task, cfg, nose = load_arrays()
    kind, sample = WIRINGS[wiring]
    return OnlineBrain(wiring_for(MB, kind, sample), cfg, task, nose), task


def order_for(n: int, seed: int) -> np.ndarray:
    return np.arange(n) if seed == 0 else np.random.default_rng(seed).permutation(n)


class PerfectBody:
    """Disembodied control: the fly always lands exactly where its mushroom body chose."""

    def fly(self, choice, split: str, i: int, keep_path: bool = False) -> dict:
        return {"reached": choice.goal}


def run_stream(wiring: str, feedback: str, seed: int, body=None, log_every: int = 200, limit_dev: int = 0,
               limit_test: int = 0) -> dict:
    brain, task = make_brain(wiring)
    body = body or PerfectBody()
    y_dev = brain.y_dev
    order = order_for(len(y_dev), seed)
    if limit_dev:
        order = order[:limit_dev]
    curve, trials, t0 = [], [], time.time()
    first_right = 0
    for n, i in enumerate(order, 1):
        top, alive = brain.code("dev", i)
        if feedback == "full":
            first = brain.choose(top, alive, set())
            first_right += bool(y_dev[i, first.goal])
            brain.full_feedback(top, alive, y_dev[i])
        else:
            tried: set[int] = set()
            for attempt in range(MAX_TRIES):
                choice = brain.choose(top, alive, tried)
                flight = body.fly(choice, "dev", int(i), keep_path=(n % KEEP_EVERY == 0 and attempt == 0))
                reached = flight["reached"]
                tried.add(choice.goal)
                ok = reached >= 0 and bool(y_dev[i, reached])
                trials.append({"split": "dev", "i": int(i), "try": attempt, "goal": choice.goal, "margin": round(choice.margin, 4),
                               **{k: v for k, v in flight.items() if k != "reached"}, "reached": reached, "correct": ok})
                if attempt == 0:
                    first_right += ok
                if reached >= 0:
                    tried.add(reached)
                    brain.dopamine(top, alive, reached, ok)
                if ok:
                    break
        if n % CURVE_EVERY == 0 or n == len(order):
            curve.append({"n": n, "test_top1": round(brain.test_top1(), 4)})
        if n % log_every == 0:
            top1 = f"test top-1 {curve[-1]['test_top1']:.3f} " if curve else ""
            print(f"{wiring}/{feedback}/s{seed}: {n}/{len(order)} {top1}({(time.time() - t0) / 60:.1f} min)", flush=True)
    # test: one flight per future problem, frozen weights
    test_hits, test_trials = 0.0, []
    n_test = limit_test or len(brain.y_test)
    for i in range(n_test):
        top, alive = brain.code("test", i)
        choice = brain.choose(top, alive, set())
        flight = body.fly(choice, "test", i, keep_path=i < KEEP_TEST) if feedback == "embodied" else {"reached": choice.goal}
        reached = flight["reached"]
        ok = reached >= 0 and bool(brain.y_test[i, reached])
        test_hits += ok
        test_trials.append({"split": "test", "i": i, "goal": choice.goal, "margin": round(choice.margin, 4),
                            **{k: v for k, v in flight.items() if k != "reached"}, "reached": reached, "correct": ok})
    return {
        "mb": MB, "wiring": wiring, "feedback": feedback, "seed": seed, "limits": [limit_dev, limit_test],
        "test_top1_frozen": round(brain.test_top1(), 4),  # argmax of the learned weights, ties fractional
        "test_first_landing": round(test_hits / n_test, 4),  # where the fly actually landed first
        "train_first_try": round(first_right / len(order), 4),
        "curve": curve,
        "trials": trials + test_trials if feedback == "embodied" else [],  # per-flight logs only matter with a body
        "minutes": round((time.time() - t0) / 60, 2),
        "host": platform.node(),
    }


def save(result: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "__smoke" if any(result.get("limits", [0, 0])) else ""
    path = OUT / f"{result['mb']}__{result['wiring']}__{result['feedback']}__s{result['seed']}{tag}.json"
    path.write_text(json.dumps(result))
    return path


def main(feedbacks: list[str], wirings: list[str], seeds: list[int], jobs: int, limit_dev: int = 0, limit_test: int = 0) -> None:
    tag = "__smoke" if (limit_dev or limit_test) else ""
    runs = [r for r in itertools.product(wirings, feedbacks, seeds)
            if not (OUT / f"{MB}__{r[0]}__{r[1]}__s{r[2]}{tag}.json").exists()]
    print(f"{len(runs)} streams to run", flush=True)
    if "embodied" in feedbacks:
        from leetfly.embodied import body as body_mod

        body_mod.download()  # unpack once, before the workers start

    def one(w, fb, s):
        body = None
        if fb == "embodied":
            from leetfly.embodied.physics import PhysicsBody

            body = PhysicsBody(seed=s)
        r = run_stream(w, fb, s, body, log_every=20 if limit_dev else 200, limit_dev=limit_dev, limit_test=limit_test)
        save(r)
        return f"{w}/{fb}/s{s}: test first landing {r['test_first_landing']:.3f}, frozen top-1 {r['test_top1_frozen']:.3f} ({r['minutes']} min)"

    if jobs > 1:
        from joblib import Parallel, delayed

        for msg in Parallel(n_jobs=jobs, verbose=0, return_as="generator")(delayed(one)(*r) for r in runs):
            print(msg, flush=True)
    else:
        for r in runs:
            print(one(*r), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback", nargs="+", default=["full", "bandit"])
    ap.add_argument("--wirings", nargs="+", default=list(WIRINGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit-dev", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    a = ap.parse_args()
    main(a.feedback, a.wirings, a.seeds, a.jobs, a.limit_dev, a.limit_test)
