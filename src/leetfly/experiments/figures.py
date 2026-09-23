"""Figures for the README / write-up, from results/phase2.json, results/describe.json and results/evolve/."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from leetfly import paths

OUT = paths.RESULTS / "figures"
MBS = ["malecns_R", "malecns_L", "flywire_R", "flywire_L", "hemibrain_R"]
LABEL = {"malecns_R": "male R", "malecns_L": "male L", "flywire_R": "female-1 R", "flywire_L": "female-1 L",
         "hemibrain_R": "female-2 R"}
WIRING = {"real": ("real wiring", "#2b8cbe"), "dp": ("coverage-preserving\nscramble", "#7bccc4"),
          "uniform": ("uniform\nscramble", "#bdbdbd")}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def evolvability(r: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    rng = np.random.default_rng(0)
    for i, (k, (label, color)) in enumerate(WIRING.items()):
        e = np.array([x["E"] for x in r["evolution_runs"] if x["wiring"] == k])
        ax.bar(i, e.mean(), color=color, width=0.6)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(e)), e, s=9, color="#333", alpha=0.6, zorder=3)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(range(3), [v[0] for v in WIRING.values()])
    ax.set_ylabel("test AUROC gain over random noses")
    ax.set_title("Evolving the nose: gain on future problems", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "evolvability.png")


def mechanism(r: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    rho = r["H3_rho_by_wiring"]
    for i, k in enumerate(WIRING):
        m, (lo, hi) = rho[k]["mean"], rho[k]["ci95"]
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=WIRING[k][1], ecolor="#333", capsize=4, ms=8)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(range(3), [v[0] for v in WIRING.values()])
    ax.set_ylabel("Spearman ρ (informativeness,\ncoverage of assigned glomerulus)")
    ax.set_title("Evolution puts informative receptors\non the most-sampled glomeruli", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "mechanism.png")


def transplant(r: dict) -> None:
    rows = {(x["source"], x["target"]): x for x in r["transplants"]}
    m = np.array([[rows[(s, t)]["gain"] for t in MBS] for s in MBS])
    ex = r["exploratory_dev_transplants"]["by_kind"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
    v = np.abs(m).max()
    im = ax.imshow(m, cmap="RdBu", vmin=-v, vmax=v)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{m[i, j] * 1000:+.1f}", ha="center", va="center", fontsize=7.5)
    ax.set_xticks(range(5), [LABEL[x] for x in MBS], rotation=35, ha="right")
    ax.set_yticks(range(5), [LABEL[x] for x in MBS])
    ax.set_xlabel("target fly (learns with the transplanted nose)")
    ax.set_ylabel("nose evolved on")
    ax.set_title("Future-problem gain (AUROC × 1000)", fontsize=9.5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    kinds = ["self", "within-fly", "same-sex", "cross-sex"]
    x = np.arange(len(kinds))
    for dx, (key, label, color) in zip(
        (-0.27, 0, 0.27),
        (("mean_gain_real_nose", "nose evolved on real wiring", "#2b8cbe"),
         ("mean_gain_dp_nose", "… on coverage-preserving scramble", "#7bccc4"),
         ("mean_gain_uniform_nose", "… on uniform scramble", "#bdbdbd")),
    ):
        bx.bar(x + dx, [ex[k][key] for k in kinds], width=0.26, color=color, label=label)
    bx.set_xticks(x, ["self\n(in-sample)", "other side,\nsame fly", "another\nfemale", "other\nsex"])
    bx.set_ylabel("dev AUROC gain over random noses")
    bx.set_title("Exploratory: transplants on the problems\nthe noses evolved for (no drift)", fontsize=9.5)
    bx.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "transplant.png")


def coverage(d: dict) -> None:
    glom = d["glomeruli"]
    cov = np.array([d["coverage_fraction"][m] for m in MBS if m in d["coverage_fraction"]])
    names = [m for m in MBS if m in d["coverage_fraction"]]
    order = np.argsort(-cov.mean(axis=0))
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    for row, m in zip(cov, names):
        ax.plot(range(len(glom)), row[order], marker="o", ms=2.5, lw=1, label=LABEL[m])
    ax.set_xticks(range(len(glom)), [glom[i] for i in order], rotation=90, fontsize=6.5)
    ax.set_ylabel("fraction of KCs reached")
    ax.set_title("Glomerulus coverage is shared across hemispheres, individuals and sexes", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=5)
    fig.tight_layout()
    fig.savefig(OUT / "coverage.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads((paths.RESULTS / "phase2.json").read_text())
    evolvability(r)
    mechanism(r)
    transplant(r)
    if (paths.RESULTS / "describe.json").exists():
        coverage(json.loads((paths.RESULTS / "describe.json").read_text()))
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
