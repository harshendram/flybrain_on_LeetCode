"""The physics body: flybody's fruit fly in MuJoCo, flown by the pretrained flight-imitation policy.

Needs Linux + `flybody[tf]` (TensorFlow 2.8, Python 3.10), so it runs on AWS (scripts/aws_embodied.py); nothing in
this module is imported by the laptop-side code or tests. Data (policies, wing pattern, flight dataset) comes from
the flybody figshare record, doi:10.25378/janelia.25309105.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

DATA = Path(os.environ.get("FLYBODY_DATA", "flybody-data"))


def download() -> None:
    """Unpack the figshare archives shipped in the job payload. figshare answers scripted downloads with a JS
    challenge (HTTP 202, empty body), so the zips are fetched once through a real browser and travel with the code."""
    import zipfile

    for z in sorted(DATA.glob("*.zip")):
        target = DATA / z.stem
        if not target.exists():
            try:
                with zipfile.ZipFile(z) as f:
                    f.extractall(target)
            except FileExistsError:  # a parallel worker got there first
                pass


def find(pattern: str) -> Path:
    hits = sorted(DATA.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"{pattern} not under {DATA}")
    return hits[0]


def flight_policy_dir() -> Path:
    """The flight-imitation policy snapshot (a TF SavedModel directory) inside the trained-policies archive."""
    for pb in sorted(DATA.rglob("saved_model.pb")):
        if pb.parent.name == "flight":  # trained-fly-policies/flight (not the controller-reuse snapshots)
            return pb.parent
    raise FileNotFoundError("no flight saved_model.pb under " + str(DATA))


def make_env(time_limit: float, ref_path: str | None = None, joint_filter: float = 0.0, future_steps: int = 5,
             seed: int = 0):
    """flybody's `flight_imitation` environment, with our own time limit and no distance termination (we measure
    tracking ourselves). With ref_path=None it flies whatever trajectory we hand it."""
    from dm_control import composer
    from dm_control.locomotion.arenas import floors
    from flybody.fruitfly import fruitfly
    from flybody.tasks.flight_imitation import FlightImitationWBPG
    from flybody.tasks.pattern_generators import WingBeatPatternGenerator
    from flybody.tasks.trajectory_loaders import HDF5FlightTrajectoryLoader, InferenceFlightTrajectoryLoader

    wbpg = WingBeatPatternGenerator(base_pattern_path=str(find("wing_pattern_fmech.npy")))
    rs = np.random.RandomState(seed)
    if ref_path:
        traj = HDF5FlightTrajectoryLoader(path=ref_path, randomize_start_step=False, random_state=rs)
    else:
        traj = InferenceFlightTrajectoryLoader()
    task = FlightImitationWBPG(
        walker=fruitfly.FruitFly,
        arena=floors.Floor(),
        wbpg=wbpg,
        traj_generator=traj,
        terminal_com_dist=float("inf"),
        trajectory_sites=False,
        initialize_qvel=True,
        force_actuators=False,
        disable_legs=True,
        time_limit=time_limit,
        joint_filter=joint_filter,
        future_steps=future_steps,
    )
    env = composer.Environment(time_limit=time_limit, task=task, random_state=rs, strip_singleton_obs_buffer_dim=True)
    return CanonicalEnv(env)


class CanonicalEnv:
    """What acme's SinglePrecisionWrapper + CanonicalSpecWrapper(clip=True) do, without importing acme: the policy
    acts in [-1, 1] and sees float32 observations. (acme pulls in dm-reverb, whose binary needs a shared libpython
    that uv's Python builds don't ship.)"""

    def __init__(self, env):
        self._env = env
        spec = env.action_spec()
        self._lo, self._hi = np.asarray(spec.minimum, np.float64), np.asarray(spec.maximum, np.float64)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _f32(self, ts):
        obs = {k: (np.asarray(v, np.float32) if np.issubdtype(np.asarray(v).dtype, np.floating) else v)
               for k, v in ts.observation.items()}
        return ts._replace(observation=obs)

    def reset(self):
        return self._f32(self._env.reset())

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        return self._f32(self._env.step(self._lo + (a + 1.0) / 2.0 * (self._hi - self._lo)))


class Policy:
    """flybody's TestPolicyWrapper, minus its acme import: batch the observation, take the distribution's mean."""

    def __init__(self, path: Path):
        import tensorflow as tf
        import tensorflow_probability  # noqa: F401  (registers the distribution types the snapshot returns)

        self._tf = tf
        self._net = tf.saved_model.load(str(path))

    # The published flight snapshot was trained when flybody reported no actuator activations; current flybody
    # reports 11. The network never saw them, so they are passed as the empty vector it was traced with.
    EMPTY_INPUTS = ("walker/actuator_activation",)

    def __call__(self, observation) -> np.ndarray:
        tf = self._tf
        obs = dict(observation)
        for k in self.EMPTY_INPUTS:
            if k in obs:
                obs[k] = np.zeros(0, np.float32)
        batched = tf.nest.map_structure(lambda x: tf.expand_dims(tf.convert_to_tensor(x), 0), obs)
        return self._net(batched).mean()[0].numpy()


def load_policy(path: Path | None = None) -> Policy:
    return Policy(path or flight_policy_dir())


def set_reference_rows(env, com_qpos: np.ndarray, start: int) -> None:
    """Re-plan mid-flight: overwrite the task's reference from control step `start` on (root-joint frame, as the
    task stores it). Rows before `start` are history and are left alone."""
    from flybody.tasks.task_utils import com2root

    task = env.task
    rows = min(len(com_qpos), len(task._ref_qpos) - start)
    if rows <= 0:
        return
    root = com2root(com_qpos[:rows, :3], com_qpos[:rows, 3:7])
    task._ref_qpos[start : start + rows, :3] = root
    task._ref_qpos[start : start + rows, 3:7] = com_qpos[:rows, 3:7]


def fly(env, policy, com_qpos: np.ndarray, com_qvel: np.ndarray, record_every: int = 20,
        replan: Callable[[int, np.ndarray, float], np.ndarray | None] | None = None, replan_every: int = 500,
        stop: Callable[[np.ndarray], bool] | None = None) -> dict:
    """Fly one reference. Every `replan_every` control steps (0.1 s), `replan(step, com_xyz)` may return a new
    reference (com_qpos rows from `step` on). Records the true CoM, root pose and the tracking error."""
    env.task._traj_generator.set_next_trajectory(com_qpos, com_qvel)
    ts = env.reset()
    physics = env.physics
    origin = com_qpos[0, :2].copy()  # the loader re-centres x, y at the first point; we report world coordinates
    com, root, err, t_wall = [], [], [], time.perf_counter()
    step = 0
    while not ts.last():
        if replan is not None and step and step % replan_every == 0:
            here = physics.named.data.subtree_com["walker/"].copy()
            here[:2] += origin
            w, x, y, z = physics.named.data.qpos["walker/"][3:7]
            yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            new = replan(step, here, yaw)
            if new is not None:
                new = new.copy()
                new[:, :2] -= origin
                set_reference_rows(env, new, step)
        ts = env.step(policy(ts.observation))
        step += 1
        if step % record_every == 0:
            c = physics.named.data.subtree_com["walker/"].copy()
            ref = np.asarray(ts.observation["walker/ref_displacement"])
            com.append(c)
            root.append(physics.named.data.qpos["walker/"][:7].copy())
            err.append(float(np.linalg.norm(ref.reshape(-1, 3)[0])))
            if stop is not None:
                here = c.copy()
                here[:2] += origin
                if stop(here):  # e.g. a feeder was reached: the rest of the flight changes nothing
                    break
    com = np.array(com)
    if len(com):
        com[:, :2] += origin
    return {
        "com": com,
        "root": np.array(root),
        "err": np.array(err),
        "steps": step,
        "sim_s": step * env.task.control_timestep,
        "wall_s": time.perf_counter() - t_wall,
        "height_ok": bool(len(com) and com[:, 2].min() > 0.05),
    }
