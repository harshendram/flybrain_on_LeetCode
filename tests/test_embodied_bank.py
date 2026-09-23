import json

import numpy as np

from leetfly.embodied import plan
from leetfly.embodied.brain import Choice
from leetfly.embodied.run import BankBody


def fake_bank(tmp_path):
    lines = []
    for f in range(14):
        for lv in range(plan.CAST_LEVELS):
            for rep in range(3):
                reached = f if rep < 2 else -1
                lines.append(json.dumps({"feeder": f, "level": lv, "rep": rep, "reached": reached, "t_arrive": 0.3 + 0.01 * lv,
                                         "err_mean_cm": 0.02, "min_height_cm": 0.7}))
    path = tmp_path / "flights.jsonl"
    path.write_text("\n".join(lines))
    return path


def test_bank_draws_only_matching_goal_and_level_and_is_seeded(tmp_path):
    path = fake_bank(tmp_path)
    a, b = BankBody(seed=4, path=path), BankBody(seed=4, path=path)
    for goal, margin in [(3, 0.0), (7, 0.12), (13, 0.9)]:
        c = Choice(goal=goal, margin=margin, scores=np.zeros(14))
        for _ in range(20):
            fa, fb = a.fly(c, "dev", 0), b.fly(c, "dev", 0)
            assert fa == fb  # same seed, same draws
            assert fa["level"] == plan.cast_level(margin)
            assert fa["reached"] in (goal, -1)
            assert abs(fa["t_arrive"] - (0.3 + 0.01 * fa["level"])) < 1e-9
