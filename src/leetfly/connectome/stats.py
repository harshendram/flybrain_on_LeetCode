"""Descriptive statistics of a mushroom-body circuit, for sanity checks against the literature."""

import numpy as np

from leetfly.connectome.extract_mb import MBCircuit


def kc_class(kc_type: str) -> str:
    if kc_type.startswith("KCa'b'"):
        return "a'b'"
    if kc_type.startswith("KCab"):
        return "ab"
    if kc_type.startswith("KCg"):
        return "g"
    return "other"


def coverage(circuit: MBCircuit) -> np.ndarray:
    """Number of KCs receiving input from each glomerulus."""
    return np.asarray((circuit.w_glom() > 0).sum(axis=0)).ravel()


def claws(circuit: MBCircuit) -> np.ndarray:
    """Number of distinct glomeruli converging on each KC."""
    return np.asarray((circuit.w_glom() > 0).sum(axis=1)).ravel()


def describe(circuit: MBCircuit) -> str:
    cov = coverage(circuit)
    cl = claws(circuit)
    classes, counts = np.unique([kc_class(t) for t in circuit.kc_types], return_counts=True)
    pns_per_glom = np.bincount(circuit.pn_glom, minlength=circuit.n_glom)
    order = np.argsort(-cov)
    lines = [
        f"== {circuit.name} (PN->KC >= {circuit.min_syn} synapses) ==",
        f"glomeruli {circuit.n_glom} | uPNs {len(circuit.pn_ids)} | KCs {circuit.n_kc} "
        + " ".join(f"{c}:{n}" for c, n in zip(classes, counts)),
        f"PN->KC edges {circuit.w_pn_kc.nnz} | synapses/edge median {np.median(circuit.w_pn_kc.data):.0f}",
        f"glomeruli per KC: mean {cl.mean():.2f} sd {cl.std():.2f} (min {cl.min()}, max {cl.max()})",
        f"KCs per glomerulus: max {cov.max()} min {cov.min()} ratio {cov.max() / max(cov.min(), 1):.1f}x",
        "  most sampled: " + ", ".join(f"{circuit.glomeruli[i]}={cov[i]}" for i in order[:6]),
        "  least sampled: " + ", ".join(f"{circuit.glomeruli[i]}={cov[i]}" for i in order[-6:]),
        f"uPNs per glomerulus: max {pns_per_glom.max()} ({circuit.glomeruli[pns_per_glom.argmax()]})",
    ]
    return "\n".join(lines)
