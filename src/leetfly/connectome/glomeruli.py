"""Map projection-neuron cell types to antennal-lobe glomeruli.

Only uniglomerular olfactory PNs are kept. Thermo/hygrosensory VP glomeruli, multiglomerular (M_*) PNs, and mixed
types such as "VP1d+VP4_l2PN1" are excluded. Type names follow the FlyWire / hemibrain / MaleCNS convention
"<glomerulus>_<lineage>PN", e.g. "DA1_lPN", "VM5d_adPN", "V_ilPN".
"""

import re

_UPN = re.compile(r"^(?P<glom>(?:DA|DC|DL|DM|DP|VA|VC|VL|VM)\d+[a-z]*|D|V)_[a-z0-9]+PN\d*$")

# Hemibrain v1.2 names a few PN types differently from FlyWire / MaleCNS (which agree on all 51 glomeruli).
# From FlyWire's own `hemibrain_type` cross-matching: hemibrain VC3l = VC3, VC3m = VC5, VC5_adPN = VM6_adPN,
# and hemibrain VC5_lvPN corresponds to FlyWire CB3383, which is not a uniglomerular olfactory PN.
HEMIBRAIN_TYPE_ALIASES: dict[str, str | None] = {
    "VC3l_adPN": "VC3_adPN",
    "VC3m_lvPN": "VC5_lvPN",
    "VC5_adPN": "VM6_adPN",
    "VC5_lvPN": None,
}


def glomerulus_of(cell_type: str | None) -> str | None:
    if not isinstance(cell_type, str):
        return None
    m = _UPN.match(cell_type.strip())
    return None if m is None else m.group("glom")
