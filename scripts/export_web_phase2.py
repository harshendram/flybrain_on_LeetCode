"""Export Phase 2 (evolve + transplant) for the 3D site: web/public/data/phase2.json + phase2.bin.

Two flies: male = MaleCNS right MB (already in model.bin as "real"), female = FlyWire right MB.
Noses per fly: "born" (the Phase-1 random nose, identical for both flies), "evolved" (the fly's own best evolved
nose, by dev fitness), and for the female "transplant" (the male's evolved nose). For every (fly, nose) the fly
learns with its own homeostasis and dopamine rule on the dev problems, exactly as in the analysis.
"""

import json
import pickle

import numpy as np

from leetfly import paths
from leetfly import task as task_mod
from leetfly.eval import metrics
from leetfly.evolve.fitness import NoseFitness
from leetfly.experiments.evolve import OUT as EVOLVE_DIR
from leetfly.experiments.transplant import informativeness
from leetfly.features.antennal_lobe import Nose
from leetfly.fly import Fly
from leetfly.connectome.extract_mb import MBCircuit, circuit_path
from leetfly.model.dopamine import DopamineReadout
from leetfly.webexport import Blob

MALE, FEMALE = "malecns_R", "flywire_R"


def best_run(mb: str) -> dict:
    runs = [json.loads(p.read_text()) for p in EVOLVE_DIR.glob(f"{mb}__real0__s*.json")]
    return max(runs, key=lambda r: r["dev_fitness"])


def trained(w_glom, cfg, nose: Nose, t) -> tuple[Fly, DopamineReadout]:
    fly = Fly(w_glom, cfg, nose).calibrate(t.receptors[t.dev])
    return fly, DopamineReadout(cfg.eta, cfg.balanced).fit(fly.codes(t.receptors[t.dev]), t.y[t.dev])


def headline(r: dict) -> dict:
    """The numbers the site quotes, all computed from results/phase2.json."""
    real = [x for x in r["evolution_runs"] if x["wiring"] == "real"]
    off = [x for x in r["transplants"] if x["kind"] != "self"]
    self_gain = float(np.mean([x["gain"] for x in r["transplants"] if x["kind"] == "self"]))
    off_gain = float(np.mean([x["gain"] for x in off]))
    return {
        "H1": r["H1_supported"], "H2": r["H2_supported"], "H4": r["H4_supported"],
        "runs_positive": sum(x["E"] > 0 for x in real), "runs_total": len(real),
        "E": r["mean_evolvability"],
        "H2_diff": r["H2a_real_minus_uniform"],
        "H3_rho": {k: v["mean"] for k, v in r["H3_rho_by_wiring"].items()},
        "H6_diff": r["H6_real_minus_dp"],
        "transplant_ratio_of_means": off_gain / self_gain,
        "transplants_positive": sum(x["gain"] > 0 for x in off), "transplants_total": len(off),
        "transplant_gain_by_kind": {k: float(np.mean([x["gain"] for x in off if x["kind"] == k]))
                                    for k in ("within-fly", "same-sex", "cross-sex")},
    }


def main() -> None:
    with open(paths.RESULTS / "cache" / "phase1_model.pkl", "rb") as f:
        p1 = pickle.load(f)
    cfg, born = p1["config"], p1["nose"]
    t = task_mod.build(n_receptors=51)
    circuits = {mb: MBCircuit.load(circuit_path(mb, 5)) for mb in (MALE, FEMALE)}
    runs = {mb: best_run(mb) for mb in (MALE, FEMALE)}
    noses = {
        (MALE, "born"): born,
        (MALE, "evolved"): Nose(np.array(runs[MALE]["perm"]), np.array(runs[MALE]["gain"])),
        (FEMALE, "born"): born,
        (FEMALE, "evolved"): Nose(np.array(runs[FEMALE]["perm"]), np.array(runs[FEMALE]["gain"])),
        (FEMALE, "transplant"): Nose(np.array(runs[MALE]["perm"]), np.array(runs[MALE]["gain"])),
    }

    blob = Blob()
    w = circuits[FEMALE].w_glom().tocsr()
    blob.add("female.w_indptr", w.indptr, "<u4")
    blob.add("female.w_indices", w.indices, "<u2")
    blob.add("female.w_data", w.data, "<f4")
    variants, scores = {}, {}
    for (mb, name), nose in noses.items():
        fly, ro = trained(circuits[mb].w_glom(), cfg, nose, t)
        key = f"{'male' if mb == MALE else 'female'}:{name}"
        blob.add(f"{key}.kc_scale", fly.mb.kc_scale, "<f8")
        blob.add(f"{key}.w_plus", ro.w_plus, "<f4")
        blob.add(f"{key}.w_minus", ro.w_minus, "<f4")
        blob.add(f"{key}.perm", nose.perm, "<u2")
        blob.add(f"{key}.gain", nose.gain, "<f8")
        test = metrics.summary(ro.scores(fly.codes(t.receptors[t.test])), t.y[t.test])
        dev = NoseFitness(circuits[mb].w_glom(), cfg, t)(nose)
        scores[key] = {"dev_auroc": round(dev, 4), "test_auroc": round(test["macro_auroc"], 4),
                       "test_hit1": round(test["hit@1"], 4), "test_hit3": round(test["hit@3"], 4)}
        variants[key] = {"fly": "male" if mb == MALE else "female", "nose": name}

    info = informativeness(t)
    auc = np.stack([metrics.auroc(np.repeat(t.receptors[t.dev][:, [r]], t.y.shape[1], 1), t.y[t.dev])
                    for r in range(t.receptors.shape[1])])  # (K, C)
    receptor_technique = np.nanargmax(np.abs(auc - 0.5), axis=1)
    h = runs[MALE]["history"]
    n_gen = len(h["best_fitness"])
    gens = list(range(0, n_gen, h["replay_every"]))  # indices of the saved frames
    frames_perm, frames_gain = list(h["best_perm"]), list(h["best_gain"])
    if gens[-1] != n_gen - 1:  # always end the replay on the final evolved nose
        gens.append(n_gen - 1)
        frames_perm.append(runs[MALE]["perm"])
        frames_gain.append(runs[MALE]["gain"])
    phase2 = json.loads((paths.RESULTS / "phase2.json").read_text()) if (paths.RESULTS / "phase2.json").exists() else {}
    summary = headline(phase2) if phase2 else {}
    meta = {
        "flies": {"male": {"circuit": MALE, "label": "male fly (MaleCNS v1.0, right)"},
                  "female": {"circuit": FEMALE, "label": "female fly (FlyWire v783, right)"}},
        "variants": variants,
        "scores": scores,
        "receptor_informativeness": np.round(info, 4).tolist(),
        "receptor_technique": receptor_technique.tolist(),
        "replay": {"generations": [g + 1 for g in gens], "perm": frames_perm, "gain": np.round(frames_gain, 4).tolist(),
                   "best_fitness": np.round(h["best_fitness"], 5).tolist(), "seed": runs[MALE]["seed"]},
        "summary": summary,
        "exploratory": phase2.get("exploratory_dev_transplants", {}).get("by_kind") if phase2 else None,
        "arrays": blob.manifest,
    }
    blob.write(paths.WEB_DATA / "phase2.bin")
    (paths.WEB_DATA / "phase2.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(json.dumps(scores, indent=1))
    print(f"phase2.bin {blob.offset / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
