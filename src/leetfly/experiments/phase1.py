"""Phase 1: can the male fly's right mushroom body tell which technique a LeetCode problem needs?

Protocol:
  1. Hyperparameters (fly and baselines) tuned by grouped 3-fold CV on the dev period only.
  2. Refit on all of dev, score the held-out future problems (test) once.
  3. Controls on test: frequency baseline, logistic regression on the same 51 channels, LR on the full TF-IDF,
     LR on the fly's own KC codes, 20 degree-preserving and 20 uniform rewirings, and 20 different born noses.
  4. Similar-problem search (FlyHash) vs SimHash, receptor cosine and TF-IDF cosine.
"""

import argparse
import itertools
import json
import pickle
import time

import numpy as np
import yaml

from leetfly import baselines, paths, search
from leetfly.connectome import nulls
from leetfly.connectome.extract_mb import MBCircuit, circuit_path
from leetfly.eval import metrics
from leetfly.features.antennal_lobe import Nose
from leetfly.fly import Fly, FlyConfig
from leetfly.model.dopamine import DopamineReadout
from leetfly import task as task_mod


def fly_scores(w_glom, cfg: FlyConfig, nose: Nose, t, train, evaluate) -> tuple[np.ndarray, Fly, DopamineReadout]:
    fly = Fly(w_glom, cfg, nose).calibrate(t.receptors[t.dev])
    codes_train, codes_eval = fly.codes(t.receptors[train]), fly.codes(t.receptors[evaluate])
    readout = DopamineReadout(cfg.eta, cfg.balanced).fit(codes_train, t.y[train])
    return readout.scores(codes_eval), fly, readout


def cv(score_fn, t, metric: str) -> float:
    return float(np.mean([metrics.summary(score_fn(tr, va), t.y[va])[metric] for tr, va in t.folds]))


def tune_fly(w_glom, nose: Nose, t, grid: dict, metric: str) -> tuple[FlyConfig, list[dict]]:
    """Codes depend only on (sigma, m, sparsity); readouts are cheap, so eta/balanced are swept on cached codes."""
    log = []
    best, best_cfg = -np.inf, None
    for sigma_m, sparsity in itertools.product(grid["sigma_m"], grid["sparsity"]):
        sigma, m = sigma_m
        base = FlyConfig(sigma=sigma, m=m, sparsity=sparsity)
        codes = Fly(w_glom, base, nose).calibrate(t.receptors[t.dev]).codes(t.receptors)
        for eta, balanced in itertools.product(grid["eta"], grid["balanced"]):
            vals = []
            for tr, va in t.folds:
                r = DopamineReadout(eta, balanced).fit(codes[tr], t.y[tr])
                vals.append(metrics.summary(r.scores(codes[va]), t.y[va])[metric])
            score = float(np.mean(vals))
            cfg = FlyConfig(sigma=sigma, m=m, sparsity=sparsity, eta=eta, balanced=balanced)
            log.append({**cfg.to_dict(), metric: score})
            if score > best:
                best, best_cfg = score, cfg
    return best_cfg, log


def main(config_path: str) -> None:
    conf = yaml.safe_load(open(config_path))
    t0 = time.time()
    circuit = MBCircuit.load(circuit_path(conf["circuit"], conf["min_syn"]))
    w_glom = circuit.w_glom()
    t = task_mod.build(n_receptors=circuit.n_glom, **conf["task"])
    rng = np.random.default_rng(conf["seed"])
    nose = Nose.born(circuit.n_glom, np.random.default_rng(conf["nose_seed"]))
    metric = conf["tune_metric"]
    out: dict = {"circuit": circuit.name, "techniques": t.names, "n_dev": len(t.dev), "n_test": len(t.test)}

    # ---- 1. tune on dev ----
    cfg, log = tune_fly(w_glom, nose, t, conf["grid"], metric)
    out["fly_config"] = cfg.to_dict()
    out["tuning_top5"] = sorted(log, key=lambda r: -r[metric])[:5]
    print(f"best fly config {cfg} (dev {metric} {max(r[metric] for r in log):.3f})  [{time.time() - t0:.0f}s]")

    # ---- 2. the fly on test ----
    s_fly, fly, readout = fly_scores(w_glom, cfg, nose, t, t.dev, t.test)
    y_test = t.y[t.test]
    results = {"fly (real wiring)": s_fly}

    # ---- 3. baselines (C tuned on dev folds) ----
    results["B0 frequency"] = baselines.frequency_scores(t.y[t.dev], len(t.test))
    lr_inputs = {
        "B1 LR on same 51 channels": (t.receptors, True),
        "B2 LR on full TF-IDF": (t.tfidf, False),
        "B3 LR on fly KC codes": (fly.codes(t.receptors).astype(np.float32), False),
    }
    for name, (x, scale) in lr_inputs.items():
        best_c = max(
            conf["lr_c"],
            key=lambda c: cv(lambda tr, va: baselines.logistic_scores(x[tr], t.y[tr], x[va], c, scale), t, metric),
        )
        results[name] = baselines.logistic_scores(x[t.dev], t.y[t.dev], x[t.test], best_c, scale)
        out.setdefault("lr_c", {})[name] = best_c

    table = {}
    for name, s in results.items():
        table[name] = {**metrics.summary(s, y_test), "ci95": metrics.bootstrap(s, y_test, n=conf["bootstrap"])}
    out["test"] = table

    # ---- 4. null wirings and born-nose variability (same config, homeostasis recalibrated) ----
    def test_summary(w, n):
        return metrics.summary(fly_scores(w, cfg, n, t, t.dev, t.test)[0], y_test)

    out["null_degree_preserving"] = [test_summary(nulls.degree_preserving(w_glom, rng), nose) for _ in range(conf["n_nulls"])]
    out["null_uniform"] = [test_summary(nulls.uniform(w_glom, rng), nose) for _ in range(conf["n_nulls"])]
    out["born_noses"] = [
        test_summary(w_glom, Nose.born(circuit.n_glom, np.random.default_rng(1000 + i))) for i in range(conf["n_noses"])
    ]
    print(f"nulls done [{time.time() - t0:.0f}s]")

    # ---- 5. similar-problem search: test queries against the dev pool ----
    k = conf["search_k"]
    y_dev = t.y[t.dev]

    def flyhash_map(w, n=nose) -> float:
        codes = Fly(w, cfg, n).calibrate(t.receptors[t.dev]).codes(t.receptors)
        return search.mean_average_precision(search.jaccard_neighbors(codes[t.test], codes[t.dev], k), y_test, y_dev)

    codes_all = fly.codes(t.receptors)
    sim = search.simhash_codes(fly.pn(t.receptors), int(np.round(codes_all.sum(1).mean())), rng)
    retrieval = {
        "FlyHash (real wiring)": search.jaccard_neighbors(codes_all[t.test], codes_all[t.dev], k),
        "SimHash (same #bits)": search.hamming_neighbors(sim[t.test], sim[t.dev], k),
        "cosine on 51 receptors": search.cosine_neighbors(t.receptors[t.test], t.receptors[t.dev], k),
        "cosine on full TF-IDF": search.cosine_neighbors(t.tfidf[t.test], t.tfidf[t.dev], k),
    }
    out["search_map@10"] = {n: search.mean_average_precision(nb, y_test, y_dev) for n, nb in retrieval.items()}
    out["search_null_degree_preserving"] = [flyhash_map(nulls.degree_preserving(w_glom, rng)) for _ in range(conf["n_nulls"])]
    out["search_null_uniform"] = [flyhash_map(nulls.uniform(w_glom, rng)) for _ in range(conf["n_nulls"])]

    # ---- save ----
    paths.RESULTS.mkdir(exist_ok=True)
    with open(paths.RESULTS / "phase1.json", "w") as f:
        json.dump(out, f, indent=2)
    (paths.RESULTS / "cache").mkdir(parents=True, exist_ok=True)
    with open(paths.RESULTS / "cache" / "phase1_model.pkl", "wb") as f:
        pickle.dump({"circuit": circuit, "config": cfg, "nose": nose, "fly": fly, "readout": readout}, f)
    report(out)
    print(f"total {time.time() - t0:.0f}s")


def report(out: dict) -> None:
    print(f"\n== technique prediction, test = {out['n_test']} future problems ==")
    for name, r in out["test"].items():
        ci = r["ci95"]["hit@1"]
        print(f"{name:28s} hit@1 {r['hit@1']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  hit@3 {r['hit@3']:.3f}  "
              f"macroAUROC {r['macro_auroc']:.3f}")
    for key in ("null_degree_preserving", "null_uniform", "born_noses"):
        h = np.array([r["hit@1"] for r in out[key]])
        a = np.array([r["macro_auroc"] for r in out[key]])
        print(f"{key:28s} hit@1 {h.mean():.3f} +- {h.std():.3f}   macroAUROC {a.mean():.3f} +- {a.std():.3f}")
    print("\n== similar-problem search (mAP@10, relevant = shares a technique) ==")
    for name, v in out["search_map@10"].items():
        print(f"{name:34s} {v:.3f}")
    for key in ("search_null_degree_preserving", "search_null_uniform"):
        v = np.array(out[key])
        print(f"FlyHash {key[12:]:26s} {v.mean():.3f} +- {v.std():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.ROOT / "configs" / "phase1.yaml"))
    main(ap.parse_args().config)
