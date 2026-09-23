"""Pack real MuJoCo flights from the Phase 3 embodied streams for the fly page's "Physics" mode.

    python scripts/export_embodied.py      -> web/public/data/embodied_flights.json

Paths come from the flight bank (results/embodied/bank/flights.jsonl): every trial of the pilot-seed streams records
the index of the real MuJoCo flight it drew. Picks a spread: early vs late in training, test flights, surges and casts, right and wrong,
and any motor miss (reached a feeder it didn't aim for). The simulation's feeder ring runs the other way round from
the website's, so each flight is mirrored across the fly's own symmetry plane (y -> -y, q -> (w, -x, y, -z)) to put
feeder i where the site draws technique i. Ships titles and slugs only, never problem text.
"""

import json

import numpy as np

from leetfly import paths
from leetfly import task as task_mod
from leetfly.embodied import physics

RUNS = paths.RESULTS / "embodied"
OUT = paths.WEB_DATA / "embodied_flights.json"
SLOW = 8.0


def mirrored(path: list, quat: list) -> tuple[list, list]:
    p = np.array(path, float)
    p[:, 1] *= -1
    q = np.array(quat, float)
    q[:, 1] *= -1
    q[:, 3] *= -1
    return p.round(4).tolist(), q.round(5).tolist()


def main() -> None:
    task = task_mod.build(n_receptors=51)
    titles = task.df["title"].tolist()
    slugs = task.df["slug"].tolist()
    bank = [json.loads(line) for line in (RUNS / "bank" / "flights.jsonl").read_text().splitlines()]
    streams = [json.loads((RUNS / f"malecns_R__real__embodied__s{s}.json").read_text()) for s in (0,)]
    kept = []
    for st in streams:
        n_dev = len([t for t in st["trials"] if t["split"] == "dev"])
        seen = 0
        for t in st["trials"]:
            if t["split"] == "dev" and t.get("try", 0) == 0:
                seen += 1
            b = bank[t["bank"]] if "bank" in t else None
            if b is None or not b.get("path"):
                continue
            t = t | {"path": b["path"], "root": [r[3:7] for r in b["root"]]}
            row = task.dev[t["i"]] if t["split"] == "dev" else task.test[t["i"]]
            kept.append(t | {"row": int(row), "n_seen": seen if t["split"] == "dev" else n_dev, "seed": st["seed"]})
    if not kept:
        raise SystemExit("no flights with kept paths yet: run the embodied stream first")

    def pick(pred, k):
        pool = [t for t in kept if pred(t) and t not in chosen]
        idx = np.linspace(0, len(pool) - 1, min(k, len(pool))).round().astype(int) if pool else []
        return [pool[i] for i in sorted(set(idx))]

    chosen: list = []
    dev = [t for t in kept if t["split"] == "dev"]
    cut = np.median([t["n_seen"] for t in dev]) if dev else 0
    chosen += pick(lambda t: t["split"] == "dev" and t["n_seen"] <= cut, 5)
    chosen += pick(lambda t: t["split"] == "dev" and t["n_seen"] > cut, 5)
    chosen += pick(lambda t: t["split"] == "test" and t["correct"], 4)
    chosen += pick(lambda t: t["split"] == "test" and not t["correct"], 3)
    chosen += pick(lambda t: t["margin"] < 0.2, 2)  # casts
    chosen += pick(lambda t: t["reached"] >= 0 and t["reached"] != t["goal"], 2)  # motor misses, if any

    flights = []
    for t in chosen:
        path, quat = mirrored(t["path"], t["root"][: len(t["path"])])
        stage = "test" if t["split"] == "test" else ("early" if t["n_seen"] <= cut else "late")
        label = {"early": f"learning #{t['n_seen']}", "late": f"learning #{t['n_seen']}", "test": "new problem"}[stage]
        flights.append({
            "label": label, "stage": stage, "n_seen": int(t["n_seen"]),
            "title": titles[t["row"]].title(), "slug": slugs[t["row"]],
            "goal": int(t["goal"]), "reached": int(t["reached"]), "correct": bool(t["correct"]),
            "margin": float(t["margin"]), "t_arrive": t.get("t_arrive"),
            "err_mm": None if t.get("err_mean_cm") is None else round(10 * t["err_mean_cm"], 2),
            "dt": 20 * 2e-4, "path": path, "quat": quat,
        })
    order = {"early": 0, "late": 1, "test": 2}
    flights.sort(key=lambda f: (order[f["stage"]], f["n_seen"]))

    summary = {}
    p3 = paths.RESULTS / "phase3.json"
    if p3.exists():
        r = json.loads(p3.read_text())
        emb = r.get("embodied", {}).get("real")
        if emb:
            summary["reached the feeder it aimed for"] = f"{100 * emb['E1_fidelity']:.1f}%"
            summary["first landings right (new problems)"] = f"{100 * emb['embodied_mean']:.1f}%"
        t = r["table"]
        summary["same brain, perfect body"] = f"{100 * t['real/bandit']['mean']:.1f}%"
        summary["told every answer"] = f"{100 * t['real/full']['mean']:.1f}%"
    summary["real MuJoCo flights in the bank"] = f"{len(bank):,}"
    OUT.write_text(json.dumps({"ring_cm": physics.RING_CM, "arrive_cm": physics.ARRIVE_CM, "slow": SLOW,
                               "summary": summary, "flights": flights}, separators=(",", ":")))
    print(f"{len(flights)} flights -> {OUT} ({OUT.stat().st_size / 1e3:.0f} kB); summary {summary}")


if __name__ == "__main__":
    main()
