"""A bank of real MuJoCo flights, simulated ahead of time and in parallel.

    python -m leetfly.embodied.bank --jobs 8 --repeats 18      (AWS; writes results/embodied/bank/flights.jsonl)

Every embodied flight starts from the same airborne state at the arena centre, so its physics depends only on the
goal feeder, the (quantized) cast level and the random wing-beat phase, never on what the brain has learned. So
instead of flying each learning trial live (a sequential chain, ~9 h per stream), all 14 feeders x 11 cast levels x
N repeats are flown once, all cores busy, and the learning streams draw each landing from this bank (run.BankBody).

Each flight gets its own seed (for the wing phase), so the bank is reproducible. A dry run (feeder 0, every level,
2 repeats) goes first and the job stops if any of those flights crashes.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from leetfly.embodied import plan

OUT = Path("results/embodied/bank")
_body = None


def _get_body():
    global _body
    if _body is None:
        from leetfly.embodied.physics import PhysicsBody

        _body = PhysicsBody(seed=0)
    return _body


def fly_one(feeder: int, level: int, rep: int) -> dict:
    pb = _get_body()
    seed = (feeder * 1000 + level * 50 + rep) * 7919 + 17
    pb.env._env._random_state.seed(seed)  # the wing-beat phase at takeoff comes from the environment's RandomState
    choice = SimpleNamespace(goal=feeder, margin=plan.level_margin(level))
    out = pb.fly(choice, "bank", 0, keep_path=True)
    return {"feeder": feeder, "level": level, "rep": rep, "seed": seed, **out}


def profile() -> dict:
    """Where does a flight's time go: the TF policy call or the physics step?"""
    pb = _get_body()
    ts = pb.env.reset()
    t0 = time.perf_counter()
    for _ in range(300):
        a = pb.policy(ts.observation)
    t_policy = (time.perf_counter() - t0) / 300
    t0 = time.perf_counter()
    for _ in range(300):
        ts = pb.env.step(a)
    t_step = (time.perf_counter() - t0) / 300
    return {"policy_ms": round(1e3 * t_policy, 3), "env_step_ms": round(1e3 * t_step, 3),
            "per_sim_second_s": round((t_policy + t_step) / plan.CONTROL_DT, 1)}


def main(jobs: int, repeats: int) -> None:
    from joblib import Parallel, delayed

    from leetfly.embodied import body

    OUT.mkdir(parents=True, exist_ok=True)
    body.download()
    t0 = time.time()
    prof = profile()
    print("profile", prof, flush=True)
    (OUT / "meta.json").write_text(json.dumps({"host": platform.node(), "jobs": jobs, "repeats": repeats,
                                               "levels": plan.CAST_LEVELS, "profile": prof}))
    path = OUT / "flights.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["feeder"], r["level"], r["rep"]))

    dry = [(0, lv, rep) for lv in range(plan.CAST_LEVELS) for rep in range(2)]
    full = [(f, lv, rep) for rep in range(repeats) for lv in range(plan.CAST_LEVELS) for f in range(14)]
    todo = [k for k in dict.fromkeys(dry + full) if k not in done]
    print(f"{len(todo)} flights to simulate on {jobs} cores (dry run first)", flush=True)

    with open(path, "a") as out:
        def run(keys):
            n = 0
            for r in Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(fly_one)(*k) for k in keys):
                out.write(json.dumps(r) + "\n")
                out.flush()
                n += 1
                yield r, n

        dry_todo = [k for k in dry if k in todo]
        crashed = 0
        for r, n in run(dry_todo):
            crashed += r["reached"] < 0
        print(f"dry run: {len(dry_todo)} flights, {crashed} crashed ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if crashed:
            raise SystemExit("dry run had crashes: stopping before the full bank")
        rest = [k for k in todo if k not in set(dry)]
        for r, n in run(rest):
            if n % 50 == 0 or n == len(rest):
                print(f"bank {n}/{len(rest)} flights ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"bank done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=18)
    a = ap.parse_args()
    main(a.jobs, a.repeats)
