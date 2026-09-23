"""The physics side of an embodied trial: one flight from the centre of the arena to the feeder the brain chose.

Every flight starts airborne at the arena centre (flybody's imitation episodes start in flight), at the median height
of the controller's training flights. The path comes from plan.py (surge or cast, from the brain's top-2 margin) and
is re-planned from the fly's true position every 0.1 s, so the controller always steers from where the body is, not
where it was meant to be. The trial's outcome is the first feeder the body's centre of mass comes within ARRIVE_CM
of, or none. Touchdown and drinking are not simulated (the imitation data has no landings).
"""

import numpy as np

from leetfly.embodied import body, plan

N_FEEDERS = 14
RING_CM = 6.0  # feeder ring radius: about 0.4 s of straight flight at the envelope's 15 cm/s
ARRIVE_CM = 0.6  # a feeder is "reached" when the fly's centre of mass comes this close (about 2 body lengths)
HEIGHT_CM = 0.74  # median height of the controller's training flights
OVERSHOOT_S = 0.3


class PhysicsBody:
    def __init__(self, seed: int = 0, envelope: plan.Envelope = plan.Envelope()):
        body.download()
        self.envl = envelope
        self.feeders = plan.feeder_ring(N_FEEDERS, RING_CM)
        self.policy = body.load_policy()
        self.time_limit = RING_CM / envelope.speed * 1.6 + OVERSHOOT_S  # casting lengthens the route
        self.env = body.make_env(time_limit=self.time_limit, joint_filter=0.0, seed=seed)

    def fly(self, choice, split: str, i: int, keep_path: bool = False) -> dict:
        goal = self.feeders[choice.goal]
        start = np.zeros(2)
        heading0 = float(np.arctan2(goal[1], goal[0]))  # it starts facing its choice
        xy, hd = plan.pursuit(start, heading0, goal, choice.margin, self.envl, overshoot=OVERSHOOT_S + 0.1)
        qpos, qvel = plan.reference(xy, HEIGHT_CM, env=self.envl, heading=hd)

        def replan(step, here, yaw):
            # steer from the true position *and heading*, at a bounded turn rate
            path, heads = plan.pursuit(here[:2], yaw, goal, choice.margin, self.envl, overshoot=OVERSHOOT_S + 0.1)
            q, _ = plan.reference(path, HEIGHT_CM, env=self.envl, heading=heads)
            return q

        arrived = lambda here: bool((np.linalg.norm(self.feeders - here[:2], axis=1) < ARRIVE_CM).any())
        run = body.fly(self.env, self.policy, qpos, qvel, replan=replan, stop=arrived)
        com = run["com"]
        reached, step = plan.reached(com[:, :2], self.feeders, ARRIVE_CM) if len(com) else (-1, -1)
        dt_rec = 20 * plan.CONTROL_DT  # body.fly records every 20 control steps
        out = {
            "reached": reached,
            "t_arrive": None if reached < 0 else round(step * dt_rec, 4),
            "err_mean_cm": round(float(run["err"].mean()), 4) if len(run["err"]) else None,
            "min_height_cm": round(float(com[:, 2].min()), 3) if len(com) else None,
            "wall_s": round(run["wall_s"], 2),
        }
        if keep_path:
            out["path"] = com[:, :3].round(4).tolist()
            out["root"] = run["root"].round(5).tolist()
        return out
