"""Download + pack the skeletons of one mushroom-body circuit for the 3D site.

malecns_*: SWC skeletons in MaleCNS EM space (8 nm voxels), rotated into the display convention (see
           leetfly.connectome.orient). EM space is also where the 141k soma positions live, so the circuit sits exactly
           inside the whole-brain point cloud.
flywire_*: neuroglancer precomputed skeletons in FlyWire space (nm -> microns). FlyWire's x axis runs the other way
           round relative to the display convention, so x is mirrored.
Every neuron is re-rooted at its soma and carries its cable distance from the soma, which drives the spike pulses.
"""

import argparse

import numpy as np
import pandas as pd

from leetfly import paths
from leetfly.connectome import orient, skeletons
from leetfly.connectome.extract_mb import MBCircuit, circuit_path

EM_SWC = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{body_id}.swc"
TEMPLATE_SWC_CACHE = paths.RAW / "malecns" / "swc_jrc2018u"  # JRC2018U versions, same node order (Phase 1 export)
FLYWIRE_URL = "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/{body_id}"
FLYWIRE_VOXEL_UM = np.array([0.004, 0.004, 0.040])
STEP = {"KC": 4.0, "PN": 4.0, "MBON": 3.0, "DAN": 4.0, "APL": 3.0}


def role_of(cell_type: str) -> str:
    if cell_type.startswith("MBON"):
        return "MBON"
    if cell_type.startswith(("PAM", "PPL1")):
        return "DAN"
    return cell_type  # APL


def circuit_specs(circuit: MBCircuit, types: pd.Series) -> list[dict]:
    """Neurons in manifest order: PNs, KCs (model order), then MBONs / DANs / APL. Shared with the cloud export."""
    return (
        [{"id": int(b), "role": "PN", "type": str(types.get(b, "")), "glom": int(g)}
         for b, g in zip(circuit.pn_ids, circuit.pn_glom)]
        + [{"id": int(b), "role": "KC", "type": str(t), "kc_index": i}
           for i, (b, t) in enumerate(zip(circuit.kc_ids, circuit.kc_types))]
        + [{"id": int(b), "role": role_of(str(t)), "type": str(t)} for b, t in zip(circuit.other_ids, circuit.other_types)]
    )


def malecns_annotations() -> pd.DataFrame:
    return pd.read_feather(paths.MALECNS_ANNOTATIONS, columns=["bodyId", "type", "somaLocation"]).set_index("bodyId")


def flywire_annotations() -> pd.DataFrame:
    raw = pd.read_csv(paths.FLYWIRE_ANNOTATIONS, sep="\t", usecols=["root_id", "cell_type", "soma_x", "soma_y", "soma_z"])
    return raw.set_index(raw["root_id"].astype(np.int64))


def flywire_um(xyz_vox: np.ndarray) -> np.ndarray:
    um = np.asarray(xyz_vox, dtype=np.float64) * FLYWIRE_VOXEL_UM
    um[..., 0] *= -1
    return um


def fit_malecns_rotation(texts: dict[int, bytes], n_neurons: int = 300) -> np.ndarray:
    a, b = [], []
    for bid in list(texts)[:n_neurons]:
        tmpl = TEMPLATE_SWC_CACHE / f"{bid}.swc"
        if not tmpl.exists():
            continue
        _, em, _ = skeletons.read_swc(texts[bid].decode())
        _, tp, _ = skeletons.read_swc(tmpl.read_text())
        if len(em) == len(tp):
            a.append(em[::5] * orient.EM_VOXEL_UM)
            b.append(tp[::5])
    a, b = np.concatenate(a), np.concatenate(b)
    r = orient.procrustes_rotation(a, b)
    resid = float(np.sqrt((((a - a.mean(0)) @ r - (b - b.mean(0))) ** 2).sum(axis=1).mean()))
    orient.save_rotation(r, resid, len(a))
    print(f"EM -> display rotation fitted on {len(a)} node pairs, rms residual {resid:.1f} um, det {np.linalg.det(r):+.0f}")
    return r


def glomerulus_positions(neurons: list[dict], n_glom: int) -> list[list[float]]:
    """Each glomerulus sits where its PNs' dendrites are: the most anterior (lowest z) 25 um of each PN's arbor."""
    pos = []
    for g in range(n_glom):
        pts = [n["xyz"][n["xyz"][:, 2] < n["xyz"][:, 2].min() + 25.0] for n in neurons if n["role"] == "PN" and n["glom"] == g]
        pos.append(np.median(np.concatenate(pts), axis=0).round(2).tolist())
    return pos


def main(circuit_name: str) -> None:
    dataset = circuit_name.split("_")[0]
    circuit = MBCircuit.load(circuit_path(circuit_name, 5))
    if dataset == "malecns":
        ann = malecns_annotations()
        specs = circuit_specs(circuit, ann["type"])
        raw = skeletons.fetch_all(EM_SWC, [s["id"] for s in specs], paths.RAW / "malecns" / "skeletons_em", ".swc")
        r = fit_malecns_rotation(raw)
        space = "MaleCNS EM space, rotated to display axes (microns)"
    else:
        ann = flywire_annotations()
        specs = circuit_specs(circuit, ann["cell_type"])
        raw = skeletons.fetch_all(FLYWIRE_URL, [s["id"] for s in specs], paths.RAW / "flywire" / "skeletons", ".precomputed")
        space = "FlyWire v783 (microns, x mirrored for display)"

    neurons = []
    for s in specs:
        bid = s["id"]
        if dataset == "malecns":
            ids, xyz, parents = skeletons.read_swc(raw[bid].decode())
            xyz = orient.em_to_display(xyz, r)
            soma = ann["somaLocation"].get(bid)
            near = None if soma is None or not hasattr(soma, "__len__") else orient.em_to_display(np.array(soma)[None], r)[0]
        else:
            ids, xyz, parents = skeletons.read_precomputed(raw[bid])
            xyz[:, 0] *= -1
            ids, xyz, parents = skeletons.prune_twigs(ids, xyz, parents, min_length=8.0)
            row = ann.loc[bid] if bid in ann.index else None
            near = None if row is None or pd.isna(row["soma_x"]) else flywire_um(row[["soma_x", "soma_y", "soma_z"]].to_numpy())
        parents = skeletons.reroot(ids, xyz, parents, near)
        xyz_ds, par_ds = skeletons.downsample(ids, xyz, parents, STEP[s["role"]])
        neurons.append(s | {"xyz": xyz_ds, "parents": par_ds, "dist": skeletons.path_distance(xyz_ds, par_ds)})

    extra = {
        "circuit": circuit.name,
        "space": space,
        "glomeruli": circuit.glomeruli.tolist(),
        "glomerulus_xyz": glomerulus_positions(neurons, circuit.n_glom),
    }
    out = paths.WEB_DATA / f"{circuit.name}_skeletons"
    skeletons.pack(neurons, out.with_suffix(".bin"), out.with_suffix(".json"), extra=extra)
    counts = pd.Series([n["role"] for n in neurons]).value_counts().to_dict()
    maxd = max(float(n["dist"].max()) for n in neurons)
    print(f"{circuit.name}: {counts} | nodes {sum(len(n['xyz']) for n in neurons)} | longest cable {maxd:.0f} um | "
          f"{out.with_suffix('.bin').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="malecns_R")
    main(ap.parse_args().circuit)
