"""Phase 3 analysis, exactly as pre-registered in the README (branch research/embodied).

    python -m leetfly.experiments.embodied    -> results/phase3.json, results/phase3_report.txt

Confirmatory disembodied seeds are 3-52; seeds 0-2 were the pilot and are reported separately. Bootstrap CIs
resample seeds (paired: the same seed means the same problem order in every condition).
"""

import json

import numpy as np

from leetfly import paths

RUNS = paths.RESULTS / "embodied"
CONFIRM = range(3, 53)
PILOT = range(0, 3)
B = 20000


def load() -> dict:
    out = {}
    for f in RUNS.glob("malecns_R__*.json"):
        r = json.loads(f.read_text())
        out[(r["wiring"], r["feedback"], r["seed"])] = r
    return out


def boot_mean_ci(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def acc(runs, wiring, feedback, seeds) -> np.ndarray:
    return np.array([runs[(wiring, feedback, s)]["test_first_landing"] for s in seeds if (wiring, feedback, s) in runs])


def main() -> None:
    runs = load()
    rng = np.random.default_rng(0)
    seeds = [s for s in CONFIRM if all((w, fb, s) in runs for w in ("real", "dp", "uni") for fb in ("full", "bandit"))]
    res: dict = {"n_confirmatory_seeds": len(seeds)}
    table = {}
    for w in ("real", "dp", "uni"):
        for fb in ("full", "bandit"):
            a = acc(runs, w, fb, seeds)
            table[f"{w}/{fb}"] = {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4),
                                  "ci": boot_mean_ci(a, rng), "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}
    res["table"] = table

    # B1: partial feedback costs little on real wiring
    d = acc(runs, "real", "full", seeds) - acc(runs, "real", "bandit", seeds)
    ci = boot_mean_ci(d, rng)
    res["B1"] = {"mean_full_minus_bandit": round(float(d.mean()), 4), "ci": ci,
                 "supported": bool(d.mean() < 0.05 and ci[1] < 0.05)}
    # B2: wiring matters under partial feedback (real - uni, paired by seed)
    d = acc(runs, "real", "bandit", seeds) - acc(runs, "uni", "bandit", seeds)
    ci = boot_mean_ci(d, rng)
    full_diff = table["real/full"]["mean"] - table["uni/full"]["mean"]
    res["B2"] = {"bandit_real_minus_uni": round(float(d.mean()), 4), "ci": ci, "supported": bool(ci[0] > 0),
                 "full_real_minus_uni_descriptive": round(full_diff, 4),
                 "real_beats_uni_seeds": int((d > 0).sum()), "ties": int((d == 0).sum())}
    # B3: exploratory, real - dp under bandit
    d = acc(runs, "real", "bandit", seeds) - acc(runs, "dp", "bandit", seeds)
    res["B3"] = {"bandit_real_minus_dp": round(float(d.mean()), 4), "ci": boot_mean_ci(d, rng)}
    # learning curves (mean over confirmatory seeds)
    curves = {}
    for w in ("real", "dp", "uni"):
        for fb in ("full", "bandit"):
            cs = [runs[(w, fb, s)]["curve"] for s in seeds]
            curves[f"{w}/{fb}"] = {"n": [c["n"] for c in cs[0]],
                                    "mean": np.round(np.mean([[c["test_top1"] for c in cc] for cc in cs], axis=0), 4).tolist()}
    res["curves"] = curves
    res["pilot"] = {f"{w}/{fb}": acc(runs, w, fb, PILOT).round(4).tolist() for w in ("real", "dp", "uni") for fb in ("full", "bandit")}

    # embodied (if present)
    emb = {k: v for k, v in runs.items() if k[1] == "embodied"}
    if emb:
        res["embodied"] = embodied_summary(runs, emb, rng)

    (paths.RESULTS / "phase3.json").write_text(json.dumps(res, indent=1))
    lines = [f"Phase 3 ({len(seeds)} confirmatory seeds)", ""]
    for k, v in table.items():
        lines.append(f"  {k:12s} test first-landing {v['mean']:.4f} (sd {v['sd']:.4f}, 95% CI {v['ci']})")
    for h in ("B1", "B2", "B3"):
        lines.append(f"{h}: {res[h]}")
    if "embodied" in res:
        lines.append(f"embodied: {json.dumps(res['embodied'])}")
    (paths.RESULTS / "phase3_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def embodied_summary(runs, emb, rng) -> dict:
    from scipy.stats import spearmanr

    out = {}
    for w in sorted({k[0] for k in emb}):
        ss = sorted(k[2] for k in emb if k[0] == w)
        trials = [t for s in ss for t in runs[(w, "embodied", s)]["trials"]]
        flights = [t for t in trials if "t_arrive" in t or "reached" in t]
        fidelity = np.mean([t["reached"] == t["goal"] for t in flights])
        e_acc = acc(runs, w, "embodied", ss)
        b_acc = acc(runs, w, "bandit", ss)
        first = [t for t in trials if t.get("t_arrive") is not None and t["split"] == "test"]
        rho, ci = None, None
        if len(first) > 10:
            m = np.array([t["margin"] for t in first])
            tt = np.array([t["t_arrive"] for t in first])
            rho = float(spearmanr(m, tt).statistic)
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(m), len(m))
                bs.append(spearmanr(m[i], tt[i]).statistic)
            ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        out[w] = {
            "seeds": ss,
            "flights": len(flights),
            "E1_fidelity": round(float(fidelity), 4),
            "E1_supported": bool(fidelity >= 0.9),
            "embodied_acc": e_acc.round(4).tolist(),
            "bandit_acc_same_seeds": b_acc.round(4).tolist(),
            "E2_diff": round(float(e_acc.mean() - b_acc.mean()), 4),
            "E2_supported": bool(abs(e_acc.mean() - b_acc.mean()) <= 0.03),
            "E3_rho_margin_vs_time": None if rho is None else round(rho, 4),
            "E3_ci": ci,
            "E3_supported": bool(ci is not None and ci[1] < 0),
            "missed_flights": int(sum(t["reached"] < 0 for t in flights)),
        }
    return out


if __name__ == "__main__":
    main()
