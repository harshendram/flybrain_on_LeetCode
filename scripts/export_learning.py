"""Data for "Watch it learn" (brain page) and the fly page's real-problem draws: web/public/data/learning.{json,bin}.

Per problem: its index in model.json's problem list (titles/slugs only; no problem text is shipped), a technique
bitmask, its 51 receptor activities (float32, for the glomeruli/plume visuals), and the exact KCs that fire in the
Phase-1 fly (male right MB, born nose). The browser replays the trial-by-trial dopamine rule over the dev problems in
date order; streaming all of them must reproduce model.bin's trained weights (web/tests/learn.test.ts).
"""

import json
import pickle

import numpy as np

from leetfly import paths
from leetfly import task as task_mod
from leetfly.data import leetcode
from leetfly.webexport import Blob


def main() -> None:
    with open(paths.RESULTS / "cache" / "phase1_model.pkl", "rb") as f:
        p1 = pickle.load(f)
    fly, cfg = p1["fly"], p1["config"]
    t = task_mod.build(n_receptors=51)
    everything = leetcode.load()
    slug_index = {s: i for i, s in enumerate(everything["slug"])}

    blob = Blob()
    meta = {"techniques": t.names, "eta": cfg.eta, "balanced": cfg.balanced, "k": fly.mb.k, "sets": {}}
    for name, rows in (("dev", t.dev), ("future", t.test)):
        codes = fly.codes(t.receptors[rows])
        k_max = int(codes.sum(axis=1).max())
        active = np.full((len(rows), k_max), 65535, dtype=np.uint16)  # padded with 65535
        for i, row in enumerate(codes):
            idx = np.flatnonzero(row)
            active[i, : len(idx)] = idx
        bits = (t.y[rows].astype(np.uint16) << np.arange(len(t.names), dtype=np.uint16)).sum(axis=1).astype(np.uint16)
        blob.add(f"{name}.problem", [slug_index[s] for s in t.df["slug"].iloc[rows]], "<u2")
        blob.add(f"{name}.techniques", bits, "<u2")
        blob.add(f"{name}.receptors", t.receptors[rows], "<f4")
        blob.add(f"{name}.active", active, "<u2")
        meta["sets"][name] = {"n": int(len(rows)), "k_max": k_max,
                              "class_counts": t.y[rows].sum(axis=0).astype(int).tolist()}
    meta["arrays"] = blob.manifest
    blob.write(paths.WEB_DATA / "learning.bin")
    (paths.WEB_DATA / "learning.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"learning.bin {blob.offset / 1e6:.2f} MB; dev {meta['sets']['dev']['n']}, future {meta['sets']['future']['n']}")


if __name__ == "__main__":
    main()
