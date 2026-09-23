"""The anatomy page is the female FlyWire brain: 78 JFRC2 neuropil meshes, real stats, real skeletons.

Extra strings in the connection table (neuropils the template has no mesh for) are reported and never
turned into a surface.
"""

from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leetfly import paths

ROOT_PARENT = 65535
WEB = paths.WEB_DATA
ZIP = paths.RAW / "flywire" / "JFRC2NP.surf.fw.zip"


def mesh_names() -> set[str]:
    if not ZIP.exists():
        pytest.skip("download JFRC2NP.surf.fw.zip first (scripts/export_anatomy.py)")
    with zipfile.ZipFile(ZIP) as z:
        return {Path(n).stem for n in z.namelist() if n.endswith(".ply") and not Path(n).name.startswith(".")}


def test_zip_has_the_78_template_neuropils():
    names = mesh_names()
    assert len(names) == 78
    assert "AL_L" in names and "EB" in names and "GNG" in names


def test_exported_regions_match_the_zip_and_the_connectome():
    meta_path = WEB / "anatomy.json"
    if not meta_path.exists():
        pytest.skip("run scripts/export_anatomy.py first")
    meta = json.loads(meta_path.read_text())
    names = mesh_names()
    got = {r["name"] for r in meta["regions"]}
    assert got - {"outline"} == names
    conn = pd.read_feather(paths.FLYWIRE_CONNECTIONS, columns=["neuropil"])
    present = set(conn["neuropil"].dropna().unique())
    missing = names - present
    extra = present - names
    assert not missing, f"meshes with no synapses in the proofread table: {sorted(missing)}"
    if extra:
        warnings.warn(f"neuropil strings with no JFRC2 mesh (not invented): {sorted(extra)}", stacklevel=1)
    assert not (got & extra)
    outline = next(r for r in meta["regions"] if r["name"] == "outline")
    o0, o1 = np.array(outline["box"][0]), np.array(outline["box"][1])
    # brain_mesh_v3 stops short of the lamina and the ocelli; a wrong mirror or scale would miss by hundreds of µm
    for region in meta["regions"]:
        if region["name"] == "outline":
            continue
        b0, b1 = np.array(region["box"][0]), np.array(region["box"][1])
        assert np.all(b0 >= o0 - 120) and np.all(b1 <= o1 + 120), region["name"]
        assert region["synapses"] > 0
        assert region["neurons"] > 0
        assert region["transmitter"]


def test_featured_skeletons_exist_and_cable_distance_grows_from_the_root():
    meta_path = WEB / "anatomy_neurons.json"
    bin_path = WEB / "anatomy_neurons.bin"
    if not meta_path.exists():
        pytest.skip("run scripts/export_anatomy.py first")
    meta = json.loads(meta_path.read_text())
    raw = bin_path.read_bytes()
    n = meta["n_nodes"]
    scale = meta["scale"]
    origin = np.array(meta["origin"], dtype=np.float64)
    q = np.frombuffer(raw, "<u2", 3 * n).reshape(n, 3)
    parents = np.frombuffer(raw, "<u2", n, offset=6 * n)
    dist = np.frombuffer(raw, "<u2", n, offset=8 * n).astype(np.float64) / meta["dist_scale"]
    xyz = q.astype(np.float64) / scale + origin
    assert meta["neurons"], "no featured neurons were packed"
    regions = {r["name"] for r in json.loads((WEB / "anatomy.json").read_text())["regions"]}
    for neuron in meta["neurons"]:
        assert neuron["count"] > 1
        assert set(neuron["regions"]) <= regions
        sl = slice(neuron["offset"], neuron["offset"] + neuron["count"])
        d = dist[sl]
        p = parents[sl]
        for i, parent in enumerate(p):
            if parent == ROOT_PARENT:
                assert d[i] == 0
            else:
                assert d[i] + 0.2 >= d[parent]
                step = float(np.linalg.norm(xyz[sl][i] - xyz[sl][parent]))
                assert abs((d[i] - d[parent]) - step) < 1.0
