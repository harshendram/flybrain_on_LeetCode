"""Pack the neurons the claim names: the left mushroom body, the lateral horn, and one-synapse partners.

Skeletons are the public MaleCNS SWCs, rotated with the same matrix as the circuit already on the page.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from leetfly import paths
from leetfly.connectome import orient, skeletons
from leetfly.connectome.context import ROLE_STEP, TWIG_UM, context_specs

EM_SWC = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{body_id}.swc"
BATCH = 250


def fetch_batch(body_ids: list[int], cache) -> tuple[dict[int, bytes], list[int]]:
    cache.mkdir(parents=True, exist_ok=True)
    missing: list[int] = []

    def get(bid: int) -> tuple[int, bytes | None]:
        path = cache / f"{bid}.swc"
        if not path.exists():
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(EM_SWC.format(body_id=bid), timeout=60) as r:
                        path.write_bytes(r.read())
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        return bid, None
                    if attempt == 3:
                        return bid, None
                except OSError:
                    if attempt == 3:
                        return bid, None
        return bid, path.read_bytes()

    out: dict[int, bytes] = {}
    with ThreadPoolExecutor(8) as pool:
        for bid, data in pool.map(get, body_ids):
            if data is None:
                missing.append(bid)
            else:
                out[bid] = data
    return out, missing


def main() -> None:
    specs = context_specs()
    print(f"context neurons {len(specs)}")
    by = pd.Series([s["group"] for s in specs]).value_counts().to_dict()
    print(by)
    ann = pd.read_feather(paths.MALECNS_ANNOTATIONS, columns=["bodyId", "somaLocation"])
    ann = ann[ann["bodyId"].isin([s["id"] for s in specs])].drop_duplicates("bodyId").set_index("bodyId")
    rotation = orient.load_rotation()
    cache = paths.RAW / "malecns" / "skeletons_em"
    neurons = []
    missing: list[int] = []
    for start in range(0, len(specs), BATCH):
        batch = specs[start : start + BATCH]
        raw, gone = fetch_batch([s["id"] for s in batch], cache)
        missing += gone
        for spec in batch:
            data = raw.get(spec["id"])
            if data is None:
                continue
            ids, xyz, parents = skeletons.read_swc(data.decode())
            xyz = orient.em_to_display(xyz, rotation)
            soma = ann["somaLocation"].get(spec["id"]) if spec["id"] in ann.index else None
            near = None
            if soma is not None and hasattr(soma, "__len__") and len(soma) == 3:
                near = orient.em_to_display(np.array(soma, dtype=float)[None], rotation)[0]
            twig = TWIG_UM.get(spec["role"], TWIG_UM["partner"] if spec["group"] == "partner" else 0.0)
            if twig:
                ids, xyz, parents = skeletons.prune_twigs(ids, xyz, parents, min_length=twig)
            parents = skeletons.reroot(ids, xyz, parents, near)
            step = ROLE_STEP.get(spec["role"], ROLE_STEP["partner"])
            xyz_ds, par_ds = skeletons.downsample(ids, xyz, parents, step)
            neurons.append(
                spec
                | {
                    "xyz": xyz_ds,
                    "parents": par_ds,
                    "dist": skeletons.path_distance(xyz_ds, par_ds),
                }
            )
        del raw
        print(f"packed {len(neurons)} missing {len(missing)} / {start + len(batch)}")
    paths.WEB_DATA.mkdir(parents=True, exist_ok=True)
    out = paths.WEB_DATA / "malecns_context"
    skeletons.pack(
        neurons,
        out.with_suffix(".bin"),
        out.with_suffix(".json"),
        extra={"missing": missing, "space": "MaleCNS EM space, rotated to display axes (microns)"},
    )
    print(
        f"context {len(neurons)} neurons, nodes {sum(len(n['xyz']) for n in neurons)}, "
        f"missing {len(missing)}, {out.with_suffix('.bin').stat().st_size / 1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
