"""Whole-brain point cloud for the 3D site: one dot per neuron at its measured soma, in the same display frame as the
circuit skeletons (see export_skeletons.py).

Binary layout: uint16[3 * n] positions quantized as round((p - origin) * scale) microns, then uint8[n] region codes.
The JSON manifest also maps every circuit neuron (skeleton manifest order) to its dot, so firing KCs can flash it.
"""

import argparse
import json

import numpy as np
import pandas as pd

from leetfly import paths
from leetfly.connectome import orient
from leetfly.connectome.extract_mb import MBCircuit, circuit_path

REGIONS = ["optic lobe", "central brain", "descending / ascending", "ventral nerve cord", "other"]
SCALE = 32.0  # 1/32 micron steps: uint16 covers 2 mm, enough for the whole male CNS


def region_code(superclass: str) -> int:
    s = str(superclass)
    if s.startswith(("ol_", "visual", "optic")):
        return 0
    if s.startswith(("cb_", "central", "endocrine", "motor")):
        return 1
    if s.startswith(("descending", "ascending", "sensory_ascending", "sensory_descending", "efferent")):
        return 2
    if s.startswith("vnc"):
        return 3
    return 4


def malecns_cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ann = pd.read_feather(paths.MALECNS_ANNOTATIONS, columns=["bodyId", "superclass", "somaLocation"])
    ann = ann[ann["somaLocation"].notna()]
    xyz = orient.em_to_display(np.stack(ann["somaLocation"].to_numpy()), orient.load_rotation())
    return ann["bodyId"].to_numpy(np.int64), xyz, ann["superclass"].map(region_code).to_numpy(np.uint8)


def flywire_cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ann = pd.read_csv(paths.FLYWIRE_ANNOTATIONS, sep="\t", usecols=["root_id", "super_class", "soma_x", "soma_y", "soma_z"])
    ann = ann[ann["soma_x"].notna()]
    xyz = ann[["soma_x", "soma_y", "soma_z"]].to_numpy(np.float64) * np.array([0.004, 0.004, 0.040])
    xyz[:, 0] *= -1  # same mirroring as the FlyWire skeletons
    return ann["root_id"].to_numpy(np.int64), xyz, ann["super_class"].map(region_code).to_numpy(np.uint8)


def main(circuit_name: str) -> None:
    dataset = circuit_name.split("_")[0]
    ids, xyz, region = malecns_cloud() if dataset == "malecns" else flywire_cloud()
    circuit = MBCircuit.load(circuit_path(circuit_name, 5))
    manifest = json.loads((paths.WEB_DATA / f"{circuit_name}_skeletons.json").read_text())
    pos = pd.Series(np.arange(len(ids)), index=ids)
    circuit_dot = [int(pos.get(n["id"], -1)) for n in manifest["neurons"]]

    origin = np.floor(xyz.min(axis=0))
    q = np.round((xyz - origin) * SCALE)
    assert q.max() < 65535
    out = paths.WEB_DATA / f"{dataset}_cloud"
    with open(out.with_suffix(".bin"), "wb") as f:
        f.write(q.astype("<u2").tobytes())
        f.write(region.astype(np.uint8).tobytes())
    meta = {
        "n": int(len(ids)),
        "origin": origin.tolist(),
        "scale": SCALE,
        "regions": REGIONS,
        "region_counts": np.bincount(region, minlength=len(REGIONS)).tolist(),
        "circuit": circuit.name,
        "circuit_dot": circuit_dot,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, separators=(",", ":")))
    found = sum(d >= 0 for d in circuit_dot)
    print(f"{dataset}: {len(ids)} somas, regions {dict(zip(REGIONS, meta['region_counts']))}, "
          f"{found}/{len(circuit_dot)} circuit neurons located | {out.with_suffix('.bin').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="malecns_R")
    main(ap.parse_args().circuit)
