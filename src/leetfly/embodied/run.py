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


def make_brain(wiring: str) -> tuple[OnlineBrain, object]:
    with open(paths.RESULTS / "cache" / "phase1_model.pkl", "rb") as f:
        p1 = pickle.load(f)
    task = task_mod.build(n_receptors=51)
    kind, sample = WIRINGS[wiring]
    return OnlineBrain(wiring_for(MB, kind, sample), p1["config"], task, p1["nose"]), task


def order_for(n: int, seed: int) -> np.ndarray:
    return np.arange(n) if seed == 0 else np.random.default_rng(seed).permutation(n)


class PerfectBody:
    """Disembodied control: the fly always lands exactly where its mushroom body chose."""

    def fly(self, choice, split: str, i: int) -> dict:
        return {"reached": choice.goal}


def run_stream(wiring: str, feedback: str, seed: int, body=None, log_every: int = 200) -> dict:
    brain, task = make_brain(wiring)
    body = body or PerfectBody()
    y_dev = brain.y_dev
    order = order_for(len(y_dev), seed)
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
                flight = body.fly(choice, "dev", int(i))
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
            print(f"{wiring}/{feedback}/s{seed}: {n}/{len(order)} test top-1 {curve[-1]['test_top1']:.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    # test: one flight per future problem, frozen weights
    test_hits, test_trials = 0.0, []
    for i in range(len(brain.y_test)):
        top, alive = brain.code("test", i)
        choice = brain.choose(top, alive, set())
        flight = body.fly(choice, "test", i) if feedback == "embodied" else {"reached": choice.goal}
        reached = flight["reached"]
        ok = reached >= 0 and bool(brain.y_test[i, reached])
        test_hits += ok
        test_trials.append({"split": "test", "i": i, "goal": choice.goal, "margin": round(choice.margin, 4),
                            **{k: v for k, v in flight.items() if k != "reached"}, "reached": reached, "correct": ok})
    return {
        "mb": MB, "wiring": wiring, "feedback": feedback, "seed": seed,
        "test_top1_frozen": round(brain.test_top1(), 4),  # argmax of the learned weights, ties fractional
        "test_first_landing": round(test_hits / len(brain.y_test), 4),  # where the fly actually landed first
        "train_first_try": round(first_right / len(order), 4),
        "curve": curve,
        "trials": trials + test_trials,
        "minutes": round((time.time() - t0) / 60, 2),
        "host": platform.node(),
    }


def save(result: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{result['mb']}__{result['wiring']}__{result['feedback']}__s{result['seed']}.json"
    path.write_text(json.dumps(result))
    return path


def main(feedbacks: list[str], wirings: list[str], seeds: list[int], jobs: int) -> None:
    runs = [r for r in itertools.product(wirings, feedbacks, seeds)
            if not (OUT / f"{MB}__{r[0]}__{r[1]}__s{r[2]}.json").exists()]
    print(f"{len(runs)} streams to run", flush=True)

    def one(w, fb, s):
        body = None
        if fb == "embodied":
            from leetfly.embodied.physics import PhysicsBody

            body = PhysicsBody(seed=s)
        r = run_stream(w, fb, s, body)
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
    a = ap.parse_args()
    main(a.feedback, a.wirings, a.seeds, a.jobs)
