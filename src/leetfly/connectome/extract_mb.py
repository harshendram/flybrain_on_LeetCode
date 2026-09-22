"""Extract one mushroom body (one hemisphere) from a connectome into a small cached circuit file.

A circuit holds uniglomerular olfactory PNs, the Kenyon cells of one side, and the PN->KC synapse-count matrix
(thresholded). It also keeps the side's MBONs, DANs and APL for the 3D view and the compartment structure.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as feather
from scipy import sparse

from leetfly import paths
from leetfly.connectome.glomeruli import glomerulus_of


@dataclass
class MBCircuit:
    name: str
    glomeruli: np.ndarray  # (G,) str, sorted
    pn_ids: np.ndarray  # (n_pn,) int64
    pn_glom: np.ndarray  # (n_pn,) int index into glomeruli
    kc_ids: np.ndarray  # (n_kc,) int64
    kc_types: np.ndarray  # (n_kc,) str
    w_pn_kc: sparse.csr_matrix  # (n_kc, n_pn) synapse counts >= min_syn
    min_syn: int
    other_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # MBON/DAN/APL on this side
    other_types: np.ndarray = field(default_factory=lambda: np.zeros(0, "<U16"))

    @property
    def n_kc(self) -> int:
        return len(self.kc_ids)

    @property
    def n_glom(self) -> int:
        return len(self.glomeruli)

    def w_glom(self) -> sparse.csr_matrix:
        """KC x glomerulus synapse counts. Sister PNs of a glomerulus carry the same signal, so they are summed."""
        onehot = sparse.csr_matrix(
            (np.ones(len(self.pn_glom)), (np.arange(len(self.pn_glom)), self.pn_glom)),
            shape=(len(self.pn_glom), self.n_glom),
        )
        return (self.w_pn_kc @ onehot).tocsr()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        w = self.w_pn_kc.tocoo()
        np.savez_compressed(
            path,
            name=self.name,
            glomeruli=self.glomeruli,
            pn_ids=self.pn_ids,
            pn_glom=self.pn_glom,
            kc_ids=self.kc_ids,
            kc_types=self.kc_types,
            w_row=w.row,
            w_col=w.col,
            w_val=w.data,
            min_syn=self.min_syn,
            other_ids=self.other_ids,
            other_types=self.other_types,
        )

    @classmethod
    def load(cls, path: Path) -> "MBCircuit":
        z = np.load(path)
        n_kc, n_pn = len(z["kc_ids"]), len(z["pn_ids"])
        w = sparse.csr_matrix((z["w_val"], (z["w_row"], z["w_col"])), shape=(n_kc, n_pn))
        return cls(
            name=str(z["name"]),
            glomeruli=z["glomeruli"],
            pn_ids=z["pn_ids"],
            pn_glom=z["pn_glom"],
            kc_ids=z["kc_ids"],
            kc_types=z["kc_types"],
            w_pn_kc=w,
            min_syn=int(z["min_syn"]),
            other_ids=z["other_ids"],
            other_types=z["other_types"],
        )


def circuit_path(name: str, min_syn: int) -> Path:
    return paths.PROCESSED / f"mb_{name}_syn{min_syn}.npz"


def _edges_among(weights_path: Path, pre_ids: np.ndarray, post_ids: np.ndarray) -> pd.DataFrame:
    table = feather.read_table(weights_path, memory_map=True)
    mask = pc.and_(
        pc.is_in(table["body_pre"], value_set=pc.cast(pre_ids, "int64")),
        pc.is_in(table["body_post"], value_set=pc.cast(post_ids, "int64")),
    )
    return table.filter(mask).to_pandas()


def build_circuit(
    name: str,
    ann: pd.DataFrame,
    edges: pd.DataFrame,
    side: str,
    min_syn: int,
) -> MBCircuit:
    """`ann` needs columns bodyId, type, somaSide; `edges` needs body_pre, body_post, weight."""
    ann = ann.assign(glom=ann["type"].map(glomerulus_of))
    pns = ann[ann["glom"].notna()]
    kcs = ann[ann["type"].fillna("").str.startswith("KC") & (ann["somaSide"] == side)]

    pe = edges[edges["body_pre"].isin(pns["bodyId"]) & edges["body_post"].isin(kcs["bodyId"])]
    pe = pe[pe["weight"] >= min_syn]

    # keep PNs that actually reach this MB (almost all ipsilateral) and KCs with >=1 olfactory input
    pn_ids = np.sort(pe["body_pre"].unique())
    kc_keep = kcs[kcs["bodyId"].isin(pe["body_post"])].sort_values("bodyId")
    kc_ids = kc_keep["bodyId"].to_numpy()

    pn_glom_names = pns.set_index("bodyId").loc[pn_ids, "glom"].to_numpy()
    glomeruli = np.array(sorted(set(pn_glom_names)))
    g_index = {g: i for i, g in enumerate(glomeruli)}
    pn_glom = np.array([g_index[g] for g in pn_glom_names])

    kc_pos = pd.Series(np.arange(len(kc_ids)), index=kc_ids)
    pn_pos = pd.Series(np.arange(len(pn_ids)), index=pn_ids)
    w = sparse.csr_matrix(
        (
            pe["weight"].to_numpy(np.float32),
            (kc_pos.loc[pe["body_post"]].to_numpy(), pn_pos.loc[pe["body_pre"]].to_numpy()),
        ),
        shape=(len(kc_ids), len(pn_ids)),
    )

    others = ann[
        (ann["somaSide"] == side)
        & ann["type"].fillna("").str.match(r"^(MBON\d+|PAM\d+|PPL1\d+|APL)")
    ].sort_values("bodyId")

    return MBCircuit(
        name=name,
        glomeruli=glomeruli,
        pn_ids=pn_ids.astype(np.int64),
        pn_glom=pn_glom,
        kc_ids=kc_ids.astype(np.int64),
        kc_types=kc_keep["type"].to_numpy().astype("<U16"),
        w_pn_kc=w,
        min_syn=min_syn,
        other_ids=others["bodyId"].to_numpy(np.int64),
        other_types=others["type"].to_numpy().astype("<U16"),
    )


def extract_malecns(sides=("R", "L"), min_syn: int = 5) -> list[MBCircuit]:
    ann = pd.read_feather(paths.MALECNS_ANNOTATIONS, columns=["bodyId", "type", "somaSide"])
    ann["type"] = ann["type"].astype(object)
    pn_ids = ann.loc[ann["type"].map(glomerulus_of).notna(), "bodyId"].to_numpy()
    kc_ids = ann.loc[ann["type"].fillna("").str.startswith("KC"), "bodyId"].to_numpy()
    edges = _edges_among(paths.MALECNS_WEIGHTS, pn_ids, kc_ids)
    out = []
    for side in sides:
        c = build_circuit(f"malecns_{side}", ann, edges, side, min_syn)
        c.save(circuit_path(c.name, min_syn))
        out.append(c)
    return out


if __name__ == "__main__":
    import argparse

    from leetfly.connectome.stats import describe

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-syn", type=int, nargs="+", default=[5])
    args = ap.parse_args()
    for m in args.min_syn:
        for circuit in extract_malecns(min_syn=m):
            print(describe(circuit))
