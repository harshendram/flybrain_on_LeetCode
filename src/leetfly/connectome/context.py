"""Neurons the learning claim is about, beyond the male's right mushroom body already on the page.

The set is a query, not a quota: the left mushroom body (same filters as the circuit), every annotated
central-brain neuron one synapse from that learner, and lateral-horn cells whose type name starts with LH.
Optic-lobe, descending, and nerve-cord neurons are not added to fill empty space.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from leetfly import paths
from leetfly.connectome.extract_mb import MBCircuit, circuit_path

# Faint background arbors: coarse sampling keeps 20k neurons near 20 MB (at 12 µm the pack was 72 MB).
PARTNER_STEP = 36.0
ROLE_STEP = {"KC": 6.0, "PN": 6.0, "MBON": 5.0, "PAM": 6.0, "PPL1": 6.0, "APL": 5.0, "LH": 10.0, "partner": PARTNER_STEP}
TWIG_UM = {"LH": 10.0, "partner": 10.0}  # terminal twigs shorter than this are dropped before sampling


def _padding(superclass: str) -> bool:
    s = str(superclass)
    if s.startswith(("ol_", "visual", "optic")):
        return True
    if s.startswith(("descending", "ascending", "sensory_ascending", "sensory_descending", "efferent")):
        return True
    return s.startswith("vnc")


def _role_of(cell_type: str) -> str:
    if cell_type.startswith("MBON"):
        return "MBON"
    if cell_type.startswith("PAM"):
        return "PAM"
    if cell_type.startswith("PPL1"):
        return "PPL1"
    if cell_type == "APL" or cell_type.startswith("APL"):
        return "APL"
    return cell_type


def context_specs() -> list[dict]:
    """One dict per neuron to draw: id, role, group. Excludes ids already in the right-hemisphere pack."""
    drawn = {
        int(n["id"])
        for n in json.loads((paths.WEB_DATA / "malecns_R_skeletons.json").read_text())["neurons"]
    }
    left = MBCircuit.load(circuit_path("malecns_L", 5))
    left_rows = (
        [{"id": int(b), "role": "PN", "group": "left"} for b in left.pn_ids]
        + [{"id": int(b), "role": "KC", "group": "left"} for b in left.kc_ids]
        + [{"id": int(b), "role": _role_of(str(t)), "group": "left"} for b, t in zip(left.other_ids, left.other_types)]
    )
    left_ids = {row["id"] for row in left_rows}
    seed = np.array(sorted(drawn | left_ids), np.int64)

    ann = pd.read_feather(paths.MALECNS_ANNOTATIONS, columns=["bodyId", "type", "superclass"])
    ann["type"] = ann["type"].fillna("").astype(str)
    ann["superclass"] = ann["superclass"].fillna("").astype(str)
    ann = ann.drop_duplicates("bodyId").set_index("bodyId")

    table = feather.read_table(paths.MALECNS_WEIGHTS)
    seed_arr = pa.array(seed)
    touch = pc.or_(pc.is_in(table["body_pre"], value_set=seed_arr), pc.is_in(table["body_post"], value_set=seed_arr))
    sub = table.filter(touch).to_pandas()
    touched = np.unique(np.concatenate([sub["body_pre"].to_numpy(), sub["body_post"].to_numpy()])).astype(np.int64)
    del sub, table
    seed_set = set(map(int, seed))

    lh_ids = {int(i) for i in ann.index[ann["type"].str.match(r"^LH")]}
    known = ann.loc[ann.index.intersection(touched)].copy()
    known = known.loc[~known.index.isin(seed_set | lh_ids)]
    known = known.loc[~known["superclass"].map(_padding)]
    partner_ids = [int(i) for i in known.index]

    specs = [row for row in left_rows if row["id"] not in drawn]
    specs += [{"id": i, "role": "LH", "group": "lh"} for i in sorted(lh_ids - drawn - left_ids)]
    specs += [{"id": i, "role": "partner", "group": "partner"} for i in sorted(partner_ids)]
    return specs
