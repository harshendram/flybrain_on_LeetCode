"""Map projection-neuron cell types to antennal-lobe glomeruli.

Only uniglomerular olfactory PNs are kept. Thermo/hygrosensory VP glomeruli, multiglomerular (M_*) PNs, and mixed
types such as "VP1d+VP4_l2PN1" are excluded. Type names follow the FlyWire / hemibrain / MaleCNS convention
"<glomerulus>_<lineage>PN", e.g. "DA1_lPN", "VM5d_adPN", "V_ilPN".
"""

import re

_UPN = re.compile(r"^(?P<glom>(?:DA|DC|DL|DM|DP|VA|VC|VL|VM)\d+[a-z]*|D|V)_[a-z0-9]+PN\d*$")

# Dataset-specific spellings mapped onto the MaleCNS / FlyWire names (extended in Phase 2 for hemibrain).
ALIASES: dict[str, str] = {}


def glomerulus_of(cell_type: str | None) -> str | None:
    if not isinstance(cell_type, str):
        return None
    m = _UPN.match(cell_type.strip())
    if m is None:
        return None
    g = m.group("glom")
    return ALIASES.get(g, g)
