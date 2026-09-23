"""Pack the FlyWire brain's neuropil surfaces and a few real neurons per region for the anatomy page.

Meshes: fafbseg's JFRC2NP.surf.fw.zip (Ito et al. 2014 surfaces, already in FlyWire nm space). The code of
fafbseg is GPL and is not imported; only this public data file is downloaded.
Outline: FlyWire whole-brain legacy mesh. Skeletons: the same precomputed server and packing as export_skeletons.py.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from leetfly import paths
from leetfly.connectome import skeletons

FLYWIRE_URL = "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/{body_id}"
FLYWIRE_VOXEL_UM = np.array([0.004, 0.004, 0.040])


def flywire_um(xyz_vox: np.ndarray) -> np.ndarray:
    um = np.asarray(xyz_vox, dtype=np.float64) * FLYWIRE_VOXEL_UM
    um[..., 0] *= -1
    return um

ZIP_URL = "https://raw.githubusercontent.com/flyconnectome/fafbseg-py/master/fafbseg/data/JFRC2NP.surf.fw.zip"
OUTLINE_MESH = "https://storage.googleapis.com/flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3/mesh/"
NTS = ("gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg")
NT_NAME = {
    "gaba_avg": "GABA",
    "ach_avg": "acetylcholine",
    "glut_avg": "glutamate",
    "oct_avg": "octopamine",
    "ser_avg": "serotonin",
    "da_avg": "dopamine",
}
MIDLINE = {"EB", "FB", "NO", "PB", "GNG", "OCG", "PRW", "SAD"}


def _get(url: str) -> bytes:
    dest_parent = paths.RAW / "flywire"
    dest_parent.mkdir(parents=True, exist_ok=True)
    print("GET", url[:90])
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def read_ply(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Binary little-endian PLY: float xyz, faces as `list int int` (count is int32, not uchar)."""
    head, _, body = data.partition(b"end_header\n")
    lines = head.decode("latin1").splitlines()
    n_v = n_f = 0
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            n_v = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "element" and parts[1] == "face":
            n_f = int(parts[2])
    v = np.frombuffer(body, "<f4", n_v * 3).reshape(n_v, 3).astype(np.float64)
    raw = np.frombuffer(body, "<i4", offset=n_v * 12)
    faces = np.empty((n_f, 3), dtype=np.int64)
    o = 0
    for i in range(n_f):
        n = int(raw[o])
        faces[i] = raw[o + 1 : o + 4]
        o += 1 + n
    return v, faces


def read_legacy_fragment(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.frombuffer(data, "<u4", 1)[0])
    v = np.frombuffer(data, "<f4", 3 * n, 4).reshape(n, 3).astype(np.float64)
    idx = np.frombuffer(data, "<u4", offset=4 + 12 * n).astype(np.int64)
    return v, idx.reshape(-1, 3)


def to_display(v_nm: np.ndarray) -> np.ndarray:
    v = np.array(v_nm, dtype=np.float64, copy=True) / 1000.0
    v[:, 0] *= -1
    return v


def drop_degenerate(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = f[(f >= 0).all(1) & (f < len(v)).all(1)]
    used = np.zeros(len(v), dtype=bool)
    used[f.ravel()] = True
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[used] = np.arange(used.sum())
    return v[used], remap[f]


def cluster(v: np.ndarray, f: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    keys = np.round(v / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    acc = np.zeros((inv.max() + 1, 3))
    cnt = np.zeros(inv.max() + 1)
    np.add.at(acc, inv, v)
    np.add.at(cnt, inv, 1)
    nv = acc / cnt[:, None]
    nf = inv[f]
    return drop_degenerate(nv, nf)


def loop_once(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One Loop subdivision. Used only on the coarse region meshes."""
    edge: dict[tuple[int, int], int] = {}
    nv = [v]
    def mid(a: int, b: int) -> int:
        k = (a, b) if a < b else (b, a)
        if k not in edge:
            edge[k] = len(v) + len(edge)
            nv.append(((v[a] + v[b]) / 2)[None])
        return edge[k]
    faces = []
    for a, b, c in f:
        ab, bc, ca = mid(int(a), int(b)), mid(int(b), int(c)), mid(int(c), int(a))
        faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.concatenate([v, np.concatenate(nv[1:])], axis=0) if edge else v, np.array(faces, dtype=np.int64)


def face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return n


def load_regions() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache = paths.RAW / "flywire" / "JFRC2NP.surf.fw.zip"
    if not cache.exists():
        cache.write_bytes(_get(ZIP_URL))
    out = {}
    with zipfile.ZipFile(cache) as z:
        for name in z.namelist():
            if not name.endswith(".ply") or Path(name).name.startswith("."):
                continue
            v, f = read_ply(z.read(name))
            v = to_display(v)
            f = f[:, ::-1]  # mirror flipped the winding
            if len(f) < 4000:
                v, f = loop_once(v, f)
            out[Path(name).stem] = drop_degenerate(v, f)
    return out


def load_outline() -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(_get(OUTLINE_MESH + "1:0"))
    v, f = read_legacy_fragment(_get(OUTLINE_MESH + manifest["fragments"][0]))
    v = to_display(v)  # fragment is nanometres
    f = f[:, ::-1]
    voxel = 2.0
    for _ in range(8):
        cv, cf = cluster(v, f, voxel)
        print(f"outline voxel {voxel:.1f} um -> {len(cf)} triangles")
        v, f = cv, cf
        if len(f) <= 30000:
            break
        voxel *= 1.25
    return v, f


def region_stats() -> tuple[dict, dict[str, list[dict]]]:
    cols = ["pre_pt_root_id", "post_pt_root_id", "neuropil", "syn_count", *NTS]
    df = pd.read_feather(paths.FLYWIRE_CONNECTIONS, columns=cols)
    df = df[df["neuropil"].notna()]
    syn = df.groupby("neuropil")["syn_count"].sum()
    weighted = pd.DataFrame(
        df["syn_count"].to_numpy()[:, None] * df[list(NTS)].fillna(0).to_numpy(),
        columns=list(NTS),
    )
    weighted["neuropil"] = df["neuropil"].to_numpy()
    nt = weighted.groupby("neuropil")[list(NTS)].sum()
    by_neuron = (
        pd.concat(
            [
                df.groupby(["neuropil", "pre_pt_root_id"])["syn_count"].sum().rename("syn"),
                df.groupby(["neuropil", "post_pt_root_id"])["syn_count"].sum().rename("syn"),
            ]
        )
        .groupby(level=[0, 1])
        .sum()
    )
    ann = pd.read_csv(
        paths.FLYWIRE_ANNOTATIONS,
        sep="\t",
        usecols=["root_id", "cell_type", "super_class", "side", "soma_x", "soma_y", "soma_z"],
        dtype=str,
    )
    ann["root_id"] = ann["root_id"].astype(np.int64)
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    stats = {}
    picks: dict[str, list[dict]] = {}
    names = sorted(set(syn.index) & set(by_neuron.index.get_level_values(0)))
    for name in names:
        block = by_neuron.xs(name)
        # groupby of one column comes back as a Series; sort_values then takes no column name
        if isinstance(block, pd.Series):
            block = block.rename("syn").sort_values(ascending=False).to_frame()
        else:
            block = block.sort_values("syn", ascending=False)
        ids = block.index.to_numpy()
        weights = {c: float(nt.loc[name, c]) if name in nt.index else 0.0 for c in NTS}
        winner = max(weights, key=weights.get)
        types = []
        seen = set()
        chosen = []
        for bid in ids:
            if bid not in ann.index:
                continue
            row = ann.loc[bid]
            ctype = row["cell_type"] if isinstance(row["cell_type"], str) and row["cell_type"] not in ("", "nan") else row["super_class"]
            if not isinstance(ctype, str) or ctype in seen:
                continue
            seen.add(ctype)
            types.append({"type": ctype, "synapses": int(block.loc[bid, "syn"])})
            soma = None
            if row["soma_x"] not in (None, "nan") and pd.notna(row["soma_x"]):
                soma = flywire_um(np.array([float(row["soma_x"]), float(row["soma_y"]), float(row["soma_z"])])).round(2).tolist()
            chosen.append({"id": int(bid), "type": ctype, "super_class": str(row["super_class"]), "regions": [name], "soma": soma})
            if len(chosen) >= 4:
                break
        stats[name] = {
            "synapses": int(syn.loc[name]),
            "neurons": int(len(block)),
            "transmitter": NT_NAME[winner],
            "top_types": types[:5],
        }
        picks[name] = chosen
    return stats, picks


def main() -> None:
    regions = load_regions()
    print(f"{len(regions)} region meshes")
    outline_v, outline_f = load_outline()
    print(f"outline triangles {len(outline_f)}")
    cache_path = paths.RAW / "flywire" / "anatomy_stats.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        stats, picks = cached["stats"], cached["picks"]
        print(f"cached region stats {len(stats)}")
    else:
        stats, picks = region_stats()
        cache_path.write_text(json.dumps({"stats": stats, "picks": picks}))
        print(f"region stats {len(stats)}")
    # one neuron list, a neuron that tops several regions is fetched once
    chosen: dict[int, dict] = {}
    for name, rows in picks.items():
        if name not in regions:
            continue
        for row in rows:
            slot = chosen.setdefault(row["id"], {**row, "regions": []})
            if name not in slot["regions"]:
                slot["regions"].append(name)
    print(f"featured neurons {len(chosen)}")
    raw = skeletons.fetch_all(FLYWIRE_URL, list(chosen), paths.RAW / "flywire" / "skeletons", ".precomputed", workers=12)
    ann = pd.read_csv(paths.FLYWIRE_ANNOTATIONS, sep="\t", usecols=["root_id", "soma_x", "soma_y", "soma_z"])
    ann["root_id"] = ann["root_id"].astype(np.int64)
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    neurons = []
    for bid, spec in chosen.items():
        if bid not in raw:
            continue
        ids, xyz, parents = skeletons.read_precomputed(raw[bid])
        xyz[:, 0] *= -1
        ids, xyz, parents = skeletons.prune_twigs(ids, xyz, parents, min_length=8.0)
        near = None
        if bid in ann.index and pd.notna(ann.loc[bid, "soma_x"]):
            near = flywire_um(ann.loc[bid, ["soma_x", "soma_y", "soma_z"]].to_numpy(dtype=float))
        parents = skeletons.reroot(ids, xyz, parents, near)
        xyz_ds, par_ds = skeletons.downsample(ids, xyz, parents, 4.0)
        neurons.append(
            {
                "id": bid,
                "role": spec["super_class"],
                "type": spec["type"],
                "super_class": spec["super_class"],
                "regions": spec["regions"],
                "xyz": xyz_ds,
                "parents": par_ds,
                "dist": skeletons.path_distance(xyz_ds, par_ds),
            }
        )
    paths.WEB_DATA.mkdir(parents=True, exist_ok=True)
    skeletons.pack(neurons, paths.WEB_DATA / "anatomy_neurons.bin", paths.WEB_DATA / "anatomy_neurons.json")

    # pack surfaces
    pieces = [("outline", outline_v, outline_f)] + [(n, *regions[n]) for n in sorted(regions)]
    all_v = np.concatenate([p[1] for p in pieces])
    origin = np.floor(all_v.min(0))
    span = (all_v.max(0) - origin).max()
    scale = 32.0 if span * 64 > 65000 else 64.0
    pos = []
    nrm = []
    idx = []
    manifest = []
    v_off = i_off = 0
    for name, v, f in pieces:
        q = np.clip(np.round((v - origin) * scale), 0, 65535).astype(np.uint16)
        normals = np.clip(np.round(face_normals(v, f) * 127), -127, 127).astype(np.int8)
        pos.append(q)
        nrm.append(normals)
        idx.append((f + v_off).astype(np.uint32))
        entry = {
            "name": name,
            "vertexOffset": v_off,
            "vertexCount": int(len(v)),
            "indexOffset": i_off,
            "indexCount": int(len(f) * 3),
            "centroid": v.mean(0).round(1).tolist(),
            "box": [v.min(0).round(1).tolist(), v.max(0).round(1).tolist()],
        }
        if name != "outline":
            entry |= stats.get(name, {"synapses": 0, "neurons": 0, "transmitter": "", "top_types": []})
        manifest.append(entry)
        v_off += len(v)
        i_off += len(f) * 3
    blob = paths.WEB_DATA / "anatomy.bin"
    with open(blob, "wb") as fh:
        for chunk in pos:
            fh.write(chunk.astype("<u2").tobytes())
        for chunk in nrm:
            fh.write(chunk.tobytes())
        for chunk in idx:
            fh.write(chunk.astype("<u4").tobytes())
    meta = {
        "origin": origin.round(2).tolist(),
        "scale": scale,
        "n_vertices": v_off,
        "n_indices": i_off,
        "midline": sorted(MIDLINE),
        "regions": manifest,
    }
    (paths.WEB_DATA / "anatomy.json").write_text(json.dumps(meta))
    tris = sum(r["indexCount"] // 3 for r in manifest if r["name"] != "outline")
    print(f"anatomy.bin {blob.stat().st_size / 1e6:.1f} MB, region triangles {tris}, neurons {len(neurons)}")


if __name__ == "__main__":
    main()
