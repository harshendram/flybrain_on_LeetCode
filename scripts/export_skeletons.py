"""Download + pack the skeletons of one mushroom-body circuit for the 3D site.

malecns_*: SWC skeletons already registered to the JRC2018U template (microns).
flywire_*: neuroglancer precomputed skeletons in FlyWire space (nm -> microns). FlyWire's x axis runs the other way
round relative to JRC2018U, so x is mirrored for display to give both flies the same handedness on screen.
"""

import argparse

import numpy as np
import pandas as pd

from leetfly import paths
from leetfly.connectome import skeletons
from leetfly.connectome.extract_mb import MBCircuit, circuit_path

SOURCES = {
    "malecns": {
        "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
               "skeletons-unisex-template/skeletons-swc/{body_id}.swc",
        "suffix": ".swc",
        "space": "JRC2018U (microns)",
    },
    "flywire": {
        "url": "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/{body_id}",
        "suffix": ".precomputed",
        "space": "FlyWire v783 (microns, x mirrored for display)",
    },
}
STEP = {"KC": 4.0, "PN": 4.0, "MBON": 3.0, "DAN": 4.0, "APL": 3.0}


def role_of(cell_type: str) -> str:
    if cell_type.startswith("MBON"):
        return "MBON"
    if cell_type.startswith(("PAM", "PPL1")):
        return "DAN"
    return cell_type  # APL


def pn_types(dataset: str) -> pd.Series:
    if dataset == "malecns":
        return pd.read_feather(paths.MALECNS_ANNOTATIONS, columns=["bodyId", "type"]).set_index("bodyId")["type"]
    raw = pd.read_csv(paths.FLYWIRE_ANNOTATIONS, sep="\t", usecols=["root_id", "cell_type"], dtype=str)
    return pd.Series(raw["cell_type"].to_numpy(), index=raw["root_id"].astype(np.int64))


def glomerulus_positions(neurons: list[dict], n_glom: int) -> list[list[float]]:
    """Each glomerulus sits where its PNs' dendrites are: the most anterior (lowest z) 25 um of each PN's arbor."""
    pos = []
    for g in range(n_glom):
        pts = [n["xyz"][n["xyz"][:, 2] < n["xyz"][:, 2].min() + 25.0] for n in neurons if n["role"] == "PN" and n["glom"] == g]
        pos.append(np.median(np.concatenate(pts), axis=0).round(2).tolist())
    return pos


def main(circuit_name: str) -> None:
    dataset = circuit_name.split("_")[0]
    src = SOURCES[dataset]
    circuit = MBCircuit.load(circuit_path(circuit_name, 5))
    types = pn_types(dataset)
    specs = (
        [{"id": int(b), "role": "PN", "type": str(types.get(b, "")), "glom": int(g)}
         for b, g in zip(circuit.pn_ids, circuit.pn_glom)]
        + [{"id": int(b), "role": "KC", "type": str(t), "kc_index": i}
           for i, (b, t) in enumerate(zip(circuit.kc_ids, circuit.kc_types))]
        + [{"id": int(b), "role": role_of(str(t)), "type": str(t)} for b, t in zip(circuit.other_ids, circuit.other_types)]
    )
    raw = skeletons.fetch_all(src["url"], [s["id"] for s in specs], paths.RAW / dataset / "skeletons", src["suffix"])
    neurons = []
    for s in specs:
        if dataset == "malecns":
            ids, xyz, parents = skeletons.read_swc(raw[s["id"]].decode())
        else:
            ids, xyz, parents = skeletons.read_precomputed(raw[s["id"]])
            xyz[:, 0] *= -1
            ids, xyz, parents = skeletons.prune_twigs(ids, xyz, parents, min_length=8.0)
        xyz_ds, par_ds = skeletons.downsample(ids, xyz, parents, STEP[s["role"]])
        neurons.append(s | {"xyz": xyz_ds, "parents": par_ds})
    extra = {
        "circuit": circuit.name,
        "space": src["space"],
        "glomeruli": circuit.glomeruli.tolist(),
        "glomerulus_xyz": glomerulus_positions(neurons, circuit.n_glom),
    }
    out = paths.WEB_DATA / f"{circuit.name}_skeletons"
    skeletons.pack(neurons, out.with_suffix(".bin"), out.with_suffix(".json"), extra=extra)
    counts = pd.Series([n["role"] for n in neurons]).value_counts().to_dict()
    print(f"{circuit.name}: {counts} | nodes {sum(len(n['xyz']) for n in neurons)} | "
          f"{out.with_suffix('.bin').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="malecns_R")
    main(ap.parse_args().circuit)
