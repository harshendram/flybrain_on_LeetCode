"""The shipped fly model (web/public/fly): every part hangs off the body tree, has normals, and is rigged.

Guards two silent export bugs: geom nodes named like their body re-parenting the body to itself (the whole fly
detached from the scene), and trimesh dropping vertex normals (every triangle lit with garbage).
"""

import json
import struct

import pytest

from leetfly import paths

FLY = paths.ROOT / "web" / "public" / "fly"


def glb_json(path):
    data = path.read_bytes()
    magic, _, _ = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67  # "glTF"
    length, kind = struct.unpack("<II", data[12:20])
    assert kind == 0x4E4F534A  # JSON chunk
    return json.loads(data[20 : 20 + length])


@pytest.fixture(scope="module")
def gltf():
    path = FLY / "flybody.glb"
    if not path.exists():
        pytest.skip("run scripts/export_flybody.py first")
    return glb_json(path)


def test_every_node_reachable_from_the_scene(gltf):
    nodes = gltf["nodes"]
    seen, stack = set(), list(gltf["scenes"][gltf.get("scene", 0)]["nodes"])
    while stack:
        i = stack.pop()
        seen.add(i)
        stack += nodes[i].get("children", [])
    assert len(seen) == len(nodes)
    assert sum("mesh" in nodes[i] for i in seen) >= 80  # flybody has 85 visual geoms


def test_every_mesh_has_normals(gltf):
    for mesh in gltf["meshes"]:
        for prim in mesh["primitives"]:
            assert "NORMAL" in prim["attributes"], mesh.get("name")


def test_rig_bodies_exist_and_springrefs_fold_the_wings(gltf):
    rig = json.loads((FLY / "rig.json").read_text())
    names = {n.get("name") for n in gltf["nodes"]}
    assert {j["body"] for j in rig["joints"]} <= names
    spring = {j["name"]: j["spring"] for j in rig["joints"]}
    # mirrored wing frames: both wings fold with the same angles
    for kind in ("yaw", "roll", "pitch"):
        assert spring[f"wing_{kind}_left"] == spring[f"wing_{kind}_right"] != 0
