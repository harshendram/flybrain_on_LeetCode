"""Phase 3, step 1: can the pretrained flight controller fly *our* paths, and what does a flight cost?

    python -m leetfly.embodied.probe            (on AWS; writes results/embodied/probe.json)

1. What's in the flybody data (trees, sizes).
2. The real flights the controller learned from: speed, yaw rate, height, body pitch, duration.
3. The controller on its own dataset (baseline tracking error), for two joint-filter settings.
4. The controller on synthetic paths from plan.py: straight at several speeds, steady turns, and casts, each for
   1.0 s, i.e. longer than the 0.6 s training clips. Error vs time tells us how long and how fast we may fly.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np

from leetfly.embodied import body, plan

OUT = Path("results/embodied")


def tree(root: Path, depth: int = 4) -> list[str]:
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if len(rel.parts) <= depth:
            size = p.stat().st_size if p.is_file() else 0
            out.append(f"{rel.as_posix()}{'/' if p.is_dir() else ''} {size / 1e6:.1f} MB" if size else f"{rel.as_posix()}/")
    return out


def dataset_stats(path: Path) -> dict:
    import h5py

    speeds, yaw_rates, heights, pitches, durations = [], [], [], [], []
    with h5py.File(path, "r") as f:
        keys = []
        f.visit(keys.append)
        trajs = [k for k in keys if isinstance(f[k], h5py.Dataset) and f[k].ndim == 2 and f[k].shape[1] in (7, 13)]
        for k in trajs[:400]:
            q = f[k][()]
            if q.shape[1] < 7 or len(q) < 10:
                continue
            dt = plan.CONTROL_DT
            v = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1) / dt
            w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
            yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
            speeds.append(v)
            yaw_rates.append(np.abs(np.diff(yaw)) / dt)
            heights.append(q[:, 2])
            pitches.append(pitch)
            durations.append(len(q) * dt)
    pct = lambda a: {p: round(float(np.percentile(np.concatenate(a), p)), 3) for p in (5, 25, 50, 75, 95)}
    return {
        "keys_sample": keys[:20],
        "n_trajectories": len(durations),
        "duration_s": pct([np.array(durations)]),
        "speed_cm_s": pct(speeds),
        "yaw_rate_rad_s": pct(yaw_rates),
        "height_cm": pct(heights),
        "pitch_deg": pct(pitches),
    }


def summarize(run: dict) -> dict:
    e = run["err"]
    q = lambda frac: round(float(e[: max(1, int(len(e) * frac))].mean()), 4) if len(e) else None
    return {
        "steps": run["steps"],
        "sim_s": round(run["sim_s"], 3),
        "wall_s": round(run["wall_s"], 2),
        "wall_per_sim_s": round(run["wall_s"] / max(run["sim_s"], 1e-9), 1),
        "err_mean_cm": round(float(e.mean()), 4) if len(e) else None,
        "err_first_40pct": q(0.4),
        "err_last_cm": round(float(e[-1]), 4) if len(e) else None,
        "err_max_cm": round(float(e.max()), 4) if len(e) else None,
        "height_ok": run["height_ok"],
        "end_xyz": run["com"][-1].round(3).tolist() if len(run["com"]) else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"host": platform.node(), "machine": platform.machine(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    body.download()
    report["data_tree"] = tree(body.DATA)
    ref = body.find("*.hdf5") if list(body.DATA.rglob("*.hdf5")) else body.find("*.h5")
    report["dataset"] = str(ref)
    try:
        report["dataset_stats"] = dataset_stats(ref)
    except Exception as e:  # the HDF5 layout is not documented; keep going with defaults
        import traceback

        report["dataset_stats"] = {"error": repr(e), "trace": traceback.format_exc(), "height_cm": {50: 1.0}}
    print(json.dumps(report["dataset_stats"], indent=1, default=str), flush=True)
    (OUT / "probe.json").write_text(json.dumps(report, indent=1, default=str))
    policy_dir = body.flight_policy_dir()
    report["policy"] = str(policy_dir)
    policy = body.load_policy(policy_dir)

    # 3. the controller on its own data
    report["dataset_tracking"] = {}
    for jf in (0.0, 0.0002):
        env = body.make_env(time_limit=0.6, ref_path=str(ref), joint_filter=jf)
        runs = []
        for i in range(8):
            env.task.set_next_trajectory_index(i)
            ts = env.reset()
            err, t0, steps = [], time.perf_counter(), 0
            while not ts.last():
                ts = env.step(policy(ts.observation))
                steps += 1
                if steps % 20 == 0:
                    err.append(float(np.linalg.norm(np.asarray(ts.observation["walker/ref_displacement"]).reshape(-1, 3)[0])))
            runs.append({"steps": steps, "err_mean_cm": round(float(np.mean(err)), 4), "wall_s": round(time.perf_counter() - t0, 2)})
        report["dataset_tracking"][str(jf)] = runs
        print("dataset tracking jf", jf, runs, flush=True)
        (OUT / "probe.json").write_text(json.dumps(report, indent=1, default=str))
    best_jf = min(report["dataset_tracking"], key=lambda k: np.mean([r["err_mean_cm"] for r in report["dataset_tracking"][k]]))
    report["joint_filter"] = float(best_jf)

    # 4. our paths, 1.0 s each
    env = body.make_env(time_limit=1.0, joint_filter=float(best_jf))
    z = float(report["dataset_stats"]["height_cm"][50])
    cases = []
    for speed in (10, 20, 30, 40):
        cases.append((f"straight_{speed}", plan.Envelope(speed=speed), 1.0))
    for margin in (0.0, 0.15, 0.3):
        cases.append((f"cast_margin_{margin}", plan.Envelope(speed=20), margin))
    synth = {}
    for name, envl, margin in cases:
        length = envl.speed * 0.9
        xy = plan.path_xy([0, 0], [length, 0], margin if name.startswith("cast") else 1.0, envl, overshoot=0.3)
        qpos, qvel = plan.reference(xy, z, env=envl)
        run = body.fly(env, policy, qpos, qvel)
        synth[name] = summarize(run)
        print(name, synth[name], flush=True)
        report["synthetic"] = synth
        (OUT / "probe.json").write_text(json.dumps(report, indent=1, default=str))
    # a turn: fly to a goal 90 degrees off the initial heading via re-planning at 10 Hz
    xy = plan.path_xy([0, 0], [15, 0], 1.0, plan.Envelope(speed=20), overshoot=0.5)
    qpos, qvel = plan.reference(xy, z)
    goal = np.array([10.0, 10.0])

    def replan(step, here):
        path = plan.path_xy(here[:2], goal, 1.0, plan.Envelope(speed=20), overshoot=0.5)
        q, _ = plan.reference(path, z)
        return q

    run = body.fly(env, policy, qpos, qvel, replan=replan)
    synth["replan_turn_to_10_10"] = summarize(run) | {"closest_to_goal_cm": round(float(np.min(np.linalg.norm(run["com"][:, :2] - goal, axis=1))), 3)}
    print("replan", synth["replan_turn_to_10_10"], flush=True)
    report["synthetic"] = synth
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "probe.json").write_text(json.dumps(report, indent=1, default=str))
    print("wrote", OUT / "probe.json")


if __name__ == "__main__":
    main()
