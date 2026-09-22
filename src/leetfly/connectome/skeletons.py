"""Neuron skeletons for the 3D site: download SWCs, downsample, and pack them into a compact binary.

Binary layout (little endian), described by a JSON manifest:
  positions: uint16[3 * n_nodes]  (x, y, z) quantized as round((p - origin) * scale), p in microns (JRC2018U)
  parents:   uint16[n_nodes]      parent index local to its neuron, 65535 for the root
Neurons are stored back to back in manifest order.
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT_PARENT = 65535


def read_swc(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (ids, xyz, parent_ids)."""
    rows = [line.split() for line in text.splitlines() if line and not line.startswith("#")]
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0].astype(np.int64), arr[:, 2:5], arr[:, 6].astype(np.int64)


def read_precomputed(data: bytes, nm_to_um: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neuroglancer precomputed skeleton (FlyWire): uint32 n_vertices, n_edges, float32 xyz, uint32 edge pairs.
    Edges are undirected, so the tree is re-rooted at vertex 0. Returns (ids, xyz, parent_ids) like read_swc."""
    nv, ne = np.frombuffer(data, "<u4", 2)
    xyz = np.frombuffer(data, "<f4", 3 * nv, 8).reshape(nv, 3).astype(np.float64)
    edges = np.frombuffer(data, "<u4", 2 * ne, 8 + 12 * nv).reshape(ne, 2)
    if nm_to_um:
        xyz = xyz / 1000.0
    adj: list[list[int]] = [[] for _ in range(nv)]
    for a, b in edges:
        adj[a].append(int(b))
        adj[b].append(int(a))
    parent = np.full(nv, -1, dtype=np.int64)
    seen = np.zeros(nv, dtype=bool)
    for root in range(nv):  # a skeleton can have several connected pieces
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    parent[v] = u
                    stack.append(v)
    ids = np.arange(nv) + 1
    return ids, xyz, np.where(parent >= 0, parent + 1, -1)


def downsample(ids: np.ndarray, xyz: np.ndarray, parents: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep roots, branch points, leaves, and nodes every `step` microns of cable. Returns (xyz, local parents)."""
    index = {int(n): i for i, n in enumerate(ids)}
    par = np.array([index.get(int(p), -1) for p in parents])
    n_children = np.bincount(par[par >= 0], minlength=len(ids))
    children: list[list[int]] = [[] for _ in ids]
    for i, p in enumerate(par):
        if p >= 0:
            children[p].append(i)

    kept_parent: dict[int, int] = {}  # original index -> original index of the kept ancestor
    order: list[int] = []
    stack = [(i, -1, 0.0) for i in np.where(par < 0)[0]]
    while stack:
        node, last_kept, dist = stack.pop()
        keep = last_kept < 0 or n_children[node] != 1 or dist >= step
        if keep:
            kept_parent[node] = last_kept
            order.append(node)
            last_kept, dist = node, 0.0
        for c in children[node]:
            stack.append((c, last_kept, dist + float(np.linalg.norm(xyz[c] - xyz[node]))))

    local = {node: i for i, node in enumerate(order)}
    out_par = np.array([local[kept_parent[n]] if kept_parent[n] >= 0 else ROOT_PARENT for n in order], dtype=np.int64)
    return xyz[order], out_par


def fetch_all(url_template: str, body_ids: list[int], cache_dir: Path, suffix: str, workers: int = 16) -> dict[int, bytes]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get(bid: int) -> tuple[int, bytes]:
        path = cache_dir / f"{bid}{suffix}"
        if not path.exists():
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url_template.format(body_id=bid), timeout=60) as r:
                        path.write_bytes(r.read())
                    break
                except OSError:
                    if attempt == 3:
                        raise
        return bid, path.read_bytes()

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(get, body_ids))


def pack(neurons: list[dict], out_bin: Path, out_json: Path, scale: float = 64.0, extra: dict | None = None) -> None:
    """neurons: dicts with keys id, role, type, xyz (n,3), parents (n,), plus anything JSON-serialisable."""
    all_xyz = np.concatenate([n["xyz"] for n in neurons])
    origin = np.floor(all_xyz.min(axis=0))
    q = np.round((all_xyz - origin) * scale)
    assert q.max() < 65535, "scene too large for uint16 quantization; lower the scale"
    parents = np.concatenate([n["parents"] for n in neurons])
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(q.astype("<u2").tobytes())
        f.write(parents.astype("<u2").tobytes())
    offset = 0
    manifest = []
    for n in neurons:
        count = len(n["xyz"])
        manifest.append({k: v for k, v in n.items() if k not in ("xyz", "parents")} | {"offset": offset, "count": count})
        offset += count
    meta = {"origin": origin.tolist(), "scale": scale, "n_nodes": int(offset), "neurons": manifest} | (extra or {})
    out_json.write_text(json.dumps(meta, separators=(",", ":")))
