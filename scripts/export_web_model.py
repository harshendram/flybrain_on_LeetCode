"""Export the Phase-1 fly for the browser: web/public/data/model.json + model.bin.

The browser re-implements the whole pipeline (web/src/fly.ts): cleaning -> tokens -> TF-IDF -> NMF receptors -> nose
-> antennal lobe -> KC drive (homeostasis) -> APL k-WTA -> dopamine readout, plus FlyHash search over all problems.
Only titles, slugs and tags of LeetCode problems are shipped, never their text.
"""

import json
import pickle

import numpy as np

from leetfly import paths
from leetfly import task as task_mod
from leetfly.connectome import nulls
from leetfly.data import leetcode
from leetfly.features.odor import STOP_WORDS
from leetfly.fly import Fly
from leetfly.model.dopamine import DopamineReadout
from leetfly.webexport import Blob


def fly_arrays(blob: Blob, prefix: str, fly: Fly, readout: DopamineReadout) -> None:
    w = fly.mb.w.tocsr()
    blob.add(f"{prefix}.w_indptr", w.indptr, "<u4")
    blob.add(f"{prefix}.w_indices", w.indices, "<u2")
    blob.add(f"{prefix}.w_data", w.data, "<f4")
    blob.add(f"{prefix}.kc_scale", fly.mb.kc_scale, "<f8")
    blob.add(f"{prefix}.w_plus", readout.w_plus, "<f4")
    blob.add(f"{prefix}.w_minus", readout.w_minus, "<f4")


def main() -> None:
    with open(paths.RESULTS / "cache" / "phase1_model.pkl", "rb") as f:
        model = pickle.load(f)
    circuit, cfg, nose, fly, readout = (model[k] for k in ("circuit", "config", "nose", "fly", "readout"))
    t = task_mod.build(n_receptors=circuit.n_glom)
    feat = t.featurizer
    # ship receptors as float32; the exported model (and the golden parity file) uses exactly these rounded values
    feat.components = feat.components.astype(np.float32).astype(np.float64)

    # scrambled twin: same nose and settings, degree-preserving rewiring, trained the same way
    rng = np.random.default_rng(12345)
    twin = Fly(nulls.degree_preserving(circuit.w_glom(), rng), cfg, nose).calibrate(t.receptors[t.dev])
    twin_readout = DopamineReadout(cfg.eta, cfg.balanced).fit(twin.codes(t.receptors[t.dev]), t.y[t.dev])

    # search pool = every problem in the dataset (titles/tags only)
    everything = leetcode.load()
    pool_codes = fly.codes(feat.receptors(everything["text"].tolist()))
    counts = pool_codes.sum(axis=1)
    active = np.concatenate([np.flatnonzero(row) for row in pool_codes])

    blob = Blob()
    blob.add("idf", feat.tfidf.idf_, "<f8")
    blob.add("components", feat.components, "<f4")
    blob.add("components_gram", feat.components @ feat.components.T, "<f8")
    blob.add("receptor_scale", feat.scale, "<f8")
    blob.add("nose_perm", nose.perm, "<u2")
    blob.add("nose_gain", nose.gain, "<f8")
    fly_arrays(blob, "real", fly, readout)
    fly_arrays(blob, "scrambled", twin, twin_readout)
    blob.add("pool_counts", counts, "<u2")
    blob.add("pool_active", active, "<u2")

    vocab = feat.tfidf.vocabulary_
    terms = [None] * len(vocab)
    for term, i in vocab.items():
        terms[i] = term
    phase1 = json.loads((paths.RESULTS / "phase1.json").read_text())
    nulls_dp = phase1["null_degree_preserving"]
    phase1["test"]["scrambled wiring (mean of 20)"] = {
        m: float(np.mean([r[m] for r in nulls_dp])) for m in ("hit@1", "hit@3", "macro_auroc")
    }
    meta = {
        "circuit": circuit.name,
        "glomeruli": circuit.glomeruli.tolist(),
        "techniques": t.names,
        "config": cfg.to_dict() | {"k": fly.mb.k, "n_exponent": 1.5, "nmf_iters": 200},
        "n_kc": int(circuit.n_kc),
        "vocab": terms,
        "stop_words": sorted(STOP_WORDS),
        "receptor_terms": feat.top_terms(4),
        "problems": [
            {"slug": s, "title": ti, "difficulty": d, "tags": list(tg)}
            for s, ti, d, tg in zip(everything["slug"], everything["title"], everything["difficulty"], everything["tags"])
        ],
        "scores_on_future_problems": {
            name: {k: round(v, 3) for k, v in r.items() if k != "ci95"} for name, r in phase1["test"].items()
        },
        "arrays": blob.manifest,
    }
    paths.WEB_DATA.mkdir(parents=True, exist_ok=True)
    blob.write(paths.WEB_DATA / "model.bin")
    (paths.WEB_DATA / "model.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"model.bin {blob.offset / 1e6:.2f} MB, model.json {(paths.WEB_DATA / 'model.json').stat().st_size / 1e6:.2f} MB")
    write_golden(feat, fly, readout, twin, twin_readout)


def write_golden(feat, fly, readout, twin, twin_readout) -> None:
    """Reference outputs of the Python pipeline on our own example problems, for the JS parity test."""
    examples = json.loads((paths.ROOT / "web" / "src" / "examples.json").read_text())
    texts = [leetcode.problem_text(e["title"], e["description"]) for e in examples]
    r = feat.receptors(texts)
    golden = []
    for i, e in enumerate(examples):
        entry = {"title": e["title"], "text": texts[i], "receptors": r[i].tolist()}
        for name, f, ro in (("real", fly, readout), ("scrambled", twin, twin_readout)):
            codes = f.codes(r[i : i + 1])
            entry[name] = {
                "pn": f.pn(r[i : i + 1])[0].tolist(),
                "active": np.flatnonzero(codes[0]).tolist(),
                "scores": ro.scores(codes)[0].tolist(),
            }
        golden.append(entry)
    out = paths.ROOT / "web" / "tests" / "golden.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden))
    print(f"golden parity file: {len(golden)} problems")


if __name__ == "__main__":
    main()
