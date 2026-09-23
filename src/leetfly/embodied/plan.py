"""From a decision to a flight path: the reference trajectory the pretrained flight controller tracks.

flybody's low-level flight controller (Vaxenburg et al., Nature 2025) was trained to follow reference centre-of-mass
trajectories from real flies: at every control step it sees the next few reference positions and orientations. So the
mushroom body never touches a wing. It picks a feeder and a flight style, and this module turns that into a
reference path (flybody conventions: centimetres, seconds, quaternions (w, x, y, z), z up, heading 0 = +x).

A confident fly surges straight; an unsure one casts, sweeping side to side with an amplitude set by the margin
between its top two votes, as flies do when they lose an odour plume. Pure numpy: no MuJoCo or TensorFlow needed.
"""

from dataclasses import dataclass

import numpy as np

CONTROL_DT = 2e-4  # flybody flight control timestep (s)
BODY_PITCH_DEG = -47.5  # flybody's hovering body angle (nose up)


@dataclass(frozen=True)
class Envelope:
    """Limits that keep our paths inside the real flights the controller learned from (set from the probe)."""

    # from the controller's own training flights (272 saccade/evasion clips): speed 5-95% 4-32 cm/s, median 12.9;
    # yaw rate median 4.2 rad/s, 75% 8.5
    speed: float = 15.0  # cm/s cruising speed
    max_speed: float = 32.0  # cm/s
    max_yaw_rate: float = 8.0  # rad/s
    max_cast: float = 1.5  # cm lateral amplitude
    cast_hz: float = 1.5  # cast sweeps per second
    cast_swing: float = 0.45  # rad of heading swing at full cast: peak yaw rate 2*pi*1.5*0.45 = 4.2 rad/s (the flights' median)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (…, 4) quaternions in (w, x, y, z) order."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def cast_amplitude(margin: float, env: Envelope = Envelope()) -> float:
    """Top-2 margin (0..1, as on the website) -> lateral cast amplitude: sure flies surge, unsure ones cast."""
    return env.max_cast * (1.0 - min(1.0, max(0.0, margin) * 2.2))


def path_xy(start: np.ndarray, goal: np.ndarray, margin: float, env: Envelope = Envelope(),
            dt: float = CONTROL_DT, overshoot: float = 0.2) -> np.ndarray:
    """Positions every control step from start to goal (plus `overshoot` s of the same heading, so the controller's
    preview window never runs off the end). Casting tapers off as the fly closes in on the goal."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    d = goal - start
    length = float(np.linalg.norm(d))
    u = d / max(length, 1e-9)
    v = np.array([-u[1], u[0]])
    amp = cast_amplitude(margin, env)
    # a sweep's peak lateral speed is 2*pi*f*A; keep it inside the envelope by slowing the sweep, not the fly
    hz = env.cast_hz if amp == 0 else min(env.cast_hz, 0.8 * env.speed / (2 * np.pi * amp))
    t_arrive = length / env.speed
    t = np.arange(0.0, t_arrive + overshoot, dt)
    s = env.speed * t
    taper = np.clip(1.0 - s / max(length, 1e-9), 0.0, 1.0) ** 1.2
    lateral = amp * np.sin(2 * np.pi * hz * t) * taper
    return start + np.outer(s, u) + np.outer(lateral, v)


def pursuit(start: np.ndarray, heading0: float, goal: np.ndarray, margin: float, env: Envelope = Envelope(),
            dt: float = CONTROL_DT, overshoot: float = 0.2, max_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """A turn-rate-limited pursuit of the goal from the fly's *current* heading: what re-planning in flight needs
    (a path that snaps toward a new bearing makes the controller crash). Casting is a heading oscillation around the
    bearing, fading as the fly closes in. Returns positions (n, 2) and headings (n,)."""
    pos = np.array(start, float)
    goal = np.asarray(goal, float)
    h = float(heading0)
    length0 = max(float(np.linalg.norm(goal - pos)), 1e-9)
    # heading swing in radians. The smoke test showed a 0.9 rad (52 degree) swing at 2 Hz (peak 11 rad/s) crashes the
    # controller in every full cast; the controller's own flights mostly turn at 4-8.5 rad/s.
    amp = cast_amplitude(margin, env) / env.max_cast * env.cast_swing
    max_step = env.max_yaw_rate * dt
    xy, hd = [pos.copy()], [h]
    t, t_after = 0.0, None
    while t < max_s:
        d = goal - pos
        dist = float(np.linalg.norm(d))
        if t_after is None and dist < 0.3:  # within 3 mm: hold the heading (the bearing is ill-defined at the goal)
            t_after = t
        if t_after is not None and t - t_after >= overshoot:
            break
        bearing = np.arctan2(d[1], d[0]) if t_after is None else h
        taper = 0.0 if t_after is not None else min(1.0, max(0.0, (dist - 2.0) / max(length0 - 2.0, 1e-9))) ** 1.2
        # (no casting in the last 2 cm, nor after passing the goal)
        want = bearing + amp * taper * np.sin(2 * np.pi * env.cast_hz * t)
        dh = (want - h + np.pi) % (2 * np.pi) - np.pi
        h += float(np.clip(dh, -max_step, max_step))
        pos = pos + env.speed * dt * np.array([np.cos(h), np.sin(h)])
        t += dt
        xy.append(pos.copy())
        hd.append(h)
    return np.array(xy), np.array(hd)


def yaw_of(quat: np.ndarray) -> float:
    """Heading (rotation about z) of a (w, x, y, z) body quaternion, pitch notwithstanding."""
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def reference(xy: np.ndarray, z: float, dt: float = CONTROL_DT, pitch_deg: float = BODY_PITCH_DEG,
              env: Envelope = Envelope(), heading: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Planar path -> flybody com_qpos (n, 7) and com_qvel (n, 6), with the body facing its direction of travel.

    The heading is smoothed so its rate never exceeds the envelope's yaw rate."""
    n = len(xy)
    vel = np.gradient(xy, dt, axis=0)
    if heading is None:
        heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
        max_step = env.max_yaw_rate * dt
        for i in range(1, n):  # rate-limit the turn, as a real fly can't snap round
            heading[i] = heading[i - 1] + np.clip(heading[i] - heading[i - 1], -max_step, max_step)
    else:
        heading = np.unwrap(np.asarray(heading, float))
    half_p = np.deg2rad(pitch_deg) / 2
    q_pitch = np.array([np.cos(half_p), 0.0, np.sin(half_p), 0.0])
    q_yaw = np.stack([np.cos(heading / 2), np.zeros(n), np.zeros(n), np.sin(heading / 2)], axis=1)
    quat = quat_mul(q_yaw, np.broadcast_to(q_pitch, (n, 4)))
    qpos = np.concatenate([xy, np.full((n, 1), z), quat], axis=1)
    qvel = np.zeros((n, 6))
    qvel[:, :2] = vel
    qvel[:, 5] = np.gradient(heading, dt)
    return qpos, qvel


def feeder_ring(n: int, radius: float) -> np.ndarray:
    """Feeder positions (n, 2) on a ring round the start, the same layout as the website's arena (feeder 0 at -y)."""
    ang = np.arange(n) / n * 2 * np.pi - np.pi / 2
    return radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def reached(xy: np.ndarray, feeders: np.ndarray, radius: float) -> tuple[int, int]:
    """First feeder the flown path comes within `radius` of, and the step it happened at; (-1, -1) if none."""
    d = np.linalg.norm(xy[:, None, :] - feeders[None, :, :], axis=2)
    hit = np.argwhere(d < radius)
    if len(hit) == 0:
        return -1, -1
    step, feeder = hit[np.argmin(hit[:, 0])]
    return int(feeder), int(step)
