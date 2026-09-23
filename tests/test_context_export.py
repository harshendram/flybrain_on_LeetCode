"""The context pack is the query: left mushroom body, lateral horn, one-synapse partners. Nothing else."""

import json

import pandas as pd
import pytest

from leetfly import paths
from leetfly.connectome.context import context_specs


def _padding(superclass: str) -> bool:
    s = str(superclass)
    if s.startswith(("ol_", "visual", "optic")):
        return True
    if s.startswith(("descending", "ascending", "sensory_ascending", "sensory_descending", "efferent")):
        return True
    return s.startswith("vnc")


@pytest.fixture(scope="module")
def packed():
    path = paths.WEB_DATA / "malecns_context.json"
    if not path.exists():
        pytest.skip("run scripts/export_context.py first")
    return json.loads(path.read_text())


def test_packed_ids_are_the_query(packed):
    want = {row["id"] for row in context_specs()}
    got = {int(n["id"]) for n in packed["neurons"]}
    missing = set(packed["missing"])
    assert not (got & missing)
    assert got | missing == want
    drawn = {int(n["id"]) for n in json.loads((paths.WEB_DATA / "malecns_R_skeletons.json").read_text())["neurons"]}
    assert not (got & drawn)


def test_optic_lobe_and_nerve_cord_are_absent(packed):
    ann = pd.read_feather(paths.MALECNS_ANNOTATIONS, columns=["bodyId", "superclass"])
    ann = ann.drop_duplicates("bodyId").set_index("bodyId")
    ids = [int(n["id"]) for n in packed["neurons"]]
    superclasses = ann["superclass"].reindex(ids).fillna("")
    assert not superclasses.map(_padding).any()
