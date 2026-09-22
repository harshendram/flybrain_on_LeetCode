"""Download + pack the skeletons of one mushroom body circuit (default: MaleCNS right) for the 3D site."""

import argparse

import numpy as np
import pandas as pd

from leetfly import paths
from leetfly.connectome import skeletons
from leetfly.connectome.extract_mb import MBCircuit, circuit_path

MALECNS_SWC = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
    "skeletons-unisex-template/skeletons-swc/{body_id}.swc"
)
STEP = {"KC": 4.0, "PN": 4.0, "MBON": 3.0, "DAN": 4.0, "APL": 3.0}


def role_of(cell_type: str) -> str:
    if cell_type.startswith("MBON"):
        return "MBON"
    if cell_type.startswith(("PAM", "PPL1")):
        return "DAN"
    return cell_type  # APL


def glomerulus_positions(neurons: list[dict], glomeruli: np.ndarray) -> list[list[float]]:
    """Each glomerulus sits where its PNs' dendrites are: the most anterior (lowest z) 25 um of each PN's arbor."""
    pos = []
    for g in range(len(glomeruli)):
        pts = []
        for n in neurons:
            if n["role"] == "PN" and n["glom"] == g:
                z = n["xyz"][:, 2]
                pts.append(n["xyz"][z < z.min() + 25.0])
        pts = np.concatenate(pts)
        pos.append(np.median(pts, axis=0).round(2).tolist())
    return pos


def main(circuit_name: str) -> None:
    circuit = MBCircuit.load(circuit_path(circuit_name, 5))
    ann = pd.read_feather(paths.MALECNS_ANNOTATIONS, columns=["bodyId", "type"]).set_index("bodyId")["type"]
    specs = (
        [{"id": int(b), "role": "PN", "type": str(ann[b]), "glom": int(g)} for b, g in zip(circuit.pn_ids, circuit.pn_glom)]
        + [{"id": int(b), "role": "KC", "type": str(t), "kc_index": i}
           for i, (b, t) in enumerate(zip(circuit.kc_ids, circuit.kc_types))]
        + [{"id": int(b), "role": role_of(str(t)), "type": str(t)} for b, t in zip(circuit.other_ids, circuit.other_types)]
    )
    texts = skeletons.fetch_swcs(MALECNS_SWC, [s["id"] for s in specs], paths.RAW / "malecns" / "swc_jrc2018u")
    neurons = []
    for s in specs:
        ids, xyz, parents = skeletons.read_swc(texts[s["id"]])
        xyz_ds, par_ds = skeletons.downsample(ids, xyz, parents, STEP[s["role"]])
        neurons.append(s | {"xyz": xyz_ds, "parents": par_ds})
    extra = {
        "circuit": circuit.name,
        "space": "JRC2018U (microns)",
        "glomeruli": circuit.glomeruli.tolist(),
        "glomerulus_xyz": glomerulus_positions(neurons, circuit.glomeruli),
    }
    skeletons.pack(neurons, paths.WEB_DATA / f"{circuit.name}_skeletons.bin",
                   paths.WEB_DATA / f"{circuit.name}_skeletons.json", extra=extra)
    counts = pd.Series([n["role"] for n in neurons]).value_counts().to_dict()
    total = sum(len(n["xyz"]) for n in neurons)
    raw = sum(len(skeletons.read_swc(texts[s["id"]])[0]) for s in specs)
    print(f"{circuit.name}: {counts} | nodes {raw} -> {total} | "
          f"{(paths.WEB_DATA / f'{circuit.name}_skeletons.bin').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="malecns_R")
    main(ap.parse_args().circuit)
