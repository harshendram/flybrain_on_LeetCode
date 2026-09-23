import numpy as np

from leetfly.embodied import plan


def test_surge_is_straight_and_reaches_goal():
    xy = plan.path_xy([0, 0], [10, 0], margin=1.0)
    assert np.allclose(xy[:, 1], 0, atol=1e-9)  # a sure fly does not cast
    t_goal = 10 / plan.Envelope().speed
    i = int(round(t_goal / plan.CONTROL_DT))
    assert np.allclose(xy[i], [10, 0], atol=0.01)


def test_cast_amplitude_shrinks_with_margin_and_tapers_to_goal():
    amps = [plan.cast_amplitude(m) for m in (0.0, 0.15, 0.3, 0.6)]
    assert amps[0] == plan.Envelope().max_cast and amps == sorted(amps, reverse=True) and amps[-1] == 0
    xy = plan.path_xy([0, 0], [10, 0], margin=0.0)
    lateral = np.abs(xy[:, 1])
    assert lateral.max() <= plan.Envelope().max_cast + 1e-9
    i = int(round(10 / plan.Envelope().speed / plan.CONTROL_DT))
    assert lateral[i] < 1e-3  # casting has died out on arrival (within 10 um; i is rounded to a control step)


def test_reference_stays_in_envelope_and_faces_travel():
    env = plan.Envelope()
    xy = plan.path_xy([0, 0], [8, 6], margin=0.0, env=env)
    qpos, qvel = plan.reference(xy, z=1.0, env=env)
    assert qpos.shape == (len(xy), 7) and qvel.shape == (len(xy), 6)
    assert np.allclose(np.linalg.norm(qpos[:, 3:], axis=1), 1)
    assert np.abs(qvel[:, 5]).max() <= env.max_yaw_rate * 1.01
    speed = np.linalg.norm(qvel[:, :2], axis=1)
    assert speed.max() <= env.max_speed
    # the nose points along the direction of travel (yaw from the quaternion vs velocity heading), once turns settle
    w, x, y, zq = qpos[:, 3:].T
    fwd = np.stack([1 - 2 * (y * y + zq * zq), 2 * (x * y + w * zq)], axis=1)  # body x-axis in the plane
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    vel = qvel[:, :2] / np.maximum(speed[:, None], 1e-9)
    assert np.median(np.sum(fwd * vel, axis=1)) > 0.95


def test_feeder_ring_and_reached():
    feeders = plan.feeder_ring(14, 8.0)
    assert np.allclose(feeders[0], [0, -8]) and np.allclose(np.linalg.norm(feeders, axis=1), 8)
    xy = plan.path_xy([0, 0], feeders[3], margin=1.0)
    f, step = plan.reached(xy, feeders, radius=0.5)
    assert f == 3 and step > 0
    assert plan.reached(np.zeros((5, 2)), feeders, radius=0.5) == (-1, -1)


def test_pursuit_turns_at_bounded_rate_from_a_wrong_heading_and_arrives():
    env = plan.Envelope()
    goal = np.array([0.0, 6.0])
    xy, hd = plan.pursuit([0, 0], heading0=0.0, goal=goal, margin=1.0, env=env)  # starts 90 degrees off
    assert np.linalg.norm(xy - goal, axis=1).min() < env.speed * plan.CONTROL_DT * 2
    rate = np.abs(np.diff(hd)) / plan.CONTROL_DT
    assert rate.max() <= env.max_yaw_rate + 1e-6
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1) / plan.CONTROL_DT
    assert np.allclose(step, env.speed)


def test_pursuit_casts_wider_when_unsure_and_reference_uses_given_heading():
    goal = np.array([6.0, 0.0])
    sure, _ = plan.pursuit([0, 0], 0.0, goal, margin=1.0)
    unsure, hd = plan.pursuit([0, 0], 0.0, goal, margin=0.0)
    assert np.abs(sure[:, 1]).max() < 1e-6 < np.abs(unsure[:, 1]).max()
    assert len(unsure) > len(sure)  # sweeping costs time
    qpos, _ = plan.reference(unsure, 0.74, heading=hd)
    assert np.allclose([plan.yaw_of(q) for q in qpos[::500, 3:]], np.unwrap(hd)[::500] - 2 * np.pi * np.round(np.unwrap(hd)[::500] / (2 * np.pi)), atol=1e-6)


def test_full_cast_stays_within_the_controllers_turn_rates():
    env = plan.Envelope()
    xy, hd = plan.pursuit([0, 0], 0.0, [6.0, 0.0], margin=0.0, env=env)
    d = np.linalg.norm(xy[1:] - [6.0, 0.0], axis=1)
    arrive = int(np.argmax(d < 0.6))  # a flight ends at the 6 mm arrival radius
    assert d[arrive] < 0.6
    rate = np.abs(np.diff(hd))[:arrive] / plan.CONTROL_DT
    assert rate.max() <= 2 * np.pi * env.cast_hz * env.cast_swing + 0.5  # ~4.2 rad/s, the training flights' median
