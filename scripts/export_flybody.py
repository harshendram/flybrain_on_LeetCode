"""The fly for the fly page: flybody (Vaxenburg et al., Nature 2025; Google DeepMind + HHMI Janelia; Apache-2.0),
decimated for the web.

MuJoCo compiles fruitfly.xml, which re-centres every mesh on its inertial frame, so geom poses are only correct
after compilation. We read the compiled body tree, visual geoms (mesh vertices/faces, pose, material colour) and
hinge joints, simplify every mesh to a triangle budget, and write:
  web/public/fly/flybody.glb   one node per body, geoms as child meshes (animate by rotating body nodes)
  web/public/fly/rig.json      bodies, hinge joints (axis, pivot, range, springref) and units

Bristles (the black meshes: hundreds of thin hairs each) turn into flat shards under quadric decimation, so each
elongated hair is rebuilt as a 5-sided cone along its principal axis instead. Joint springrefs are flybody's rest
pose: folded wings, retracted proboscis and, for the legs, the tucked posture used in flight.
Then compress with:  npx @gltf-transform/cli meshopt flybody.glb flybody.glb
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import fast_simplification
import mujoco
import numpy as np
import trimesh

from leetfly import paths

BASE = "https://raw.githubusercontent.com/TuragaLab/flybody/main/flybody/fruitfly/assets/"
API = "https://api.github.com/repos/TuragaLab/flybody/contents/flybody/fruitfly/assets"
SRC = paths.RAW / "flybody"
OUT = paths.ROOT / "web" / "public" / "fly"
FACE_BUDGET = 80_000
SKIP = ("collision", "fluid", "inertial", "adhesion")
BRISTLES = ("_black", "bristle")  # geoms made of many separate hairs
GENTLE = ("wing_",)  # thin curved veins: decimate half as hard
CONE_SIDES = 5


def download() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as r:
        files = [f["name"] for f in json.load(r) if f["type"] == "file"]

    def get(name: str) -> None:
        dest = SRC / name
        if not dest.exists():
            urllib.request.urlretrieve(BASE + name, dest)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(get, files))


def quat_to_mat(q) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = trimesh.transformations.quaternion_matrix(q)[:3, :3]  # MuJoCo quats are (w, x, y, z), same as trimesh
    return m


def hair_cones(verts: np.ndarray, faces: np.ndarray, body_centre: np.ndarray, ratio: float):
    """Replace each elongated connected component (a hair) with a cone; decimate the rest (patches) normally."""
    mesh = trimesh.Trimesh(verts, faces, process=True)
    out_v, out_f = [], []
    n = 0
    for comp in trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces))):
        f = mesh.faces[comp]
        idx = np.unique(f)
        p = mesh.vertices[idx]
        c = p.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((p - c).T)) if len(p) > 3 else (np.ones(3), np.eye(3))
        if len(p) >= 8 and evals[2] > 9 * max(evals[1], 1e-18):
            axis = evecs[:, 2]
            t = (p - c) @ axis
            if np.linalg.norm(c + axis * t.min() - body_centre) > np.linalg.norm(c + axis * t.max() - body_centre):
                axis, t = -axis, -t  # orient root (nearer the body) -> tip
            a, b = c + axis * t.min(), c + axis * t.max()
            near = t <= t.min() + 0.25 * (t.max() - t.min())
            radial = (p - c) - np.outer(t, axis)
            r = float(np.percentile(np.linalg.norm(radial[near] if near.any() else radial, axis=1), 90))
            u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
            u /= np.linalg.norm(u)
            w = np.cross(axis, u)
            ring = [a + r * (np.cos(k * 2 * np.pi / CONE_SIDES) * u + np.sin(k * 2 * np.pi / CONE_SIDES) * w) for k in range(CONE_SIDES)]
            out_v += ring + [b]
            out_f += [[n + k, n + (k + 1) % CONE_SIDES, n + CONE_SIDES] for k in range(CONE_SIDES)]
            n += CONE_SIDES + 1
        else:
            remap = {v: i for i, v in enumerate(idx)}
            pv, pf = p, np.vectorize(remap.get)(f)
            target = max(12, int(len(pf) * ratio))
            if len(pf) > target:
                pv, pf = fast_simplification.simplify(pv, pf, target_reduction=1 - target / len(pf))
            out_v += list(pv)
            out_f += (np.asarray(pf) + n).tolist()
            n += len(pv)
    return np.array(out_v), np.array(out_f, dtype=np.int64)


def main() -> None:
    download()
    model = mujoco.MjModel.from_xml_path(str(SRC / "fruitfly.xml"))
    name = lambda kind, i: mujoco.mj_id2name(model, kind, i) or f"{kind.name.lower()}_{i}"

    geoms = [
        g for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and not any(s in name(mujoco.mjtObj.mjOBJ_GEOM, g) for s in SKIP)
    ]
    geom_name = lambda g: name(mujoco.mjtObj.mjOBJ_GEOM, g)
    faces_total = sum(int(model.mesh_facenum[model.geom_dataid[g]]) for g in geoms)
    solid = sum(int(model.mesh_facenum[model.geom_dataid[g]]) for g in geoms if not any(s in geom_name(g) for s in BRISTLES))
    ratio = min(1.0, FACE_BUDGET / solid)

    scene = trimesh.Scene()
    bodies = []
    for b in range(1, model.nbody):  # 0 is the world
        bname, parent = name(mujoco.mjtObj.mjOBJ_BODY, b), int(model.body_parentid[b])
        pname = "world" if parent == 0 else name(mujoco.mjtObj.mjOBJ_BODY, parent)
        tf = quat_to_mat(model.body_quat[b])
        tf[:3, 3] = model.body_pos[b]
        scene.graph.update(frame_from=pname, frame_to=bname, matrix=tf)
        bodies.append({"name": bname, "parent": pname})

    kept = 0
    for g in geoms:
        mid = int(model.geom_dataid[g])
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        verts = np.array(model.mesh_vert[va : va + vn], dtype=np.float64)
        faces = np.array(model.mesh_face[fa : fa + fn], dtype=np.int64)
        if any(s in geom_name(g) for s in BRISTLES):
            verts, faces = hair_cones(verts, faces, np.zeros(3), ratio)
        else:
            # MuJoCo stores unwelded vertices (three per triangle); the simplifier needs shared edges or it just
            # deletes isolated triangles, so weld first
            welded = trimesh.Trimesh(verts, faces, process=True)
            verts, faces = welded.vertices, welded.faces
            target = max(60, int(fn * min(1.0, ratio * (2 if any(s in geom_name(g) for s in GENTLE) else 1))))
            if len(faces) > target:
                verts, faces = fast_simplification.simplify(verts, faces, target_reduction=1 - target / len(faces))
        mesh = trimesh.Trimesh(verts, faces, process=True)
        rgba = model.mat_rgba[model.geom_matid[g]] if model.geom_matid[g] >= 0 else model.geom_rgba[g]
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[float(c) for c in rgba],
            metallicFactor=0.0,
            roughnessFactor=0.45 if rgba[3] < 1 else 0.6,
            alphaMode="BLEND" if rgba[3] < 1 else "OPAQUE",
            doubleSided=bool(rgba[3] < 1),
        ))
        tf = quat_to_mat(model.geom_quat[g])
        tf[:3, 3] = model.geom_pos[g]
        body = name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
        gname = name(mujoco.mjtObj.mjOBJ_GEOM, g)
        # geoms often share their body's name ("thorax"); a node with the body's name would re-parent the body itself
        scene.add_geometry(mesh, node_name=f"{gname}_geom", geom_name=gname, parent_node_name=body, transform=tf)
        kept += len(mesh.faces)

    joints = [
        {
            "name": name(mujoco.mjtObj.mjOBJ_JOINT, j),
            "body": name(mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])),
            "axis": model.jnt_axis[j].round(6).tolist(),
            "pivot": model.jnt_pos[j].round(7).tolist(),
            "range": model.jnt_range[j].round(5).tolist(),
            "spring": round(float(model.qpos_spring[model.jnt_qposadr[j]]), 5),
        }
        for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    scene.export(OUT / "flybody.glb", include_normals=True)  # trimesh only writes normals it has cached otherwise
    (OUT / "rig.json").write_text(json.dumps({
        "source": "flybody (Vaxenburg et al., Nature 2025), Google DeepMind + HHMI Janelia, Apache-2.0",
        "units": "cm (MuJoCo model units)",
        "bodies": bodies,
        "joints": joints,
    }))
    print(f"{len(geoms)} visual geoms, faces {faces_total} -> {kept}, {len(bodies)} bodies, {len(joints)} hinge joints; "
          f"glb {(OUT / 'flybody.glb').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
