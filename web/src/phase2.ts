// Phase 2 on the site: watch evolution rewire the male fly's nose, then transplant it into a female fly.
import * as THREE from "three";
import type { Phase2Meta, Skeletons } from "./data";
import { decodeArrays, problemText, type Fly, type ModelMeta, type Smell } from "./fly";
import { Brain, type FlyScene } from "./scene";

import { TECH_COLORS } from "./colors";

export { TECH_COLORS };

export type FemaleNose = "born" | "evolved" | "transplant";

export class Phase2 {
  readonly female: Brain;
  private tints: THREE.Color[];

  constructor(
    private fly: Fly,
    private scene: FlyScene,
    private male: Brain,
    readonly meta: Phase2Meta,
    bin: ArrayBuffer,
    femaleSkeletons: Skeletons,
  ) {
    const a = decodeArrays({ arrays: meta.arrays } as unknown as ModelMeta, bin) as Record<string, any>;
    const maleW = fly.wiringOf("real");
    const femaleW = { indptr: a["female.w_indptr"], indices: a["female.w_indices"], data: a["female.w_data"] };
    for (const key of Object.keys(meta.variants)) {
      const w = meta.variants[key].fly === "male" ? maleW : femaleW;
      fly.addVariant(key, {
        ...w,
        kcScale: a[`${key}.kc_scale`],
        wPlus: a[`${key}.w_plus`],
        wMinus: a[`${key}.w_minus`],
        perm: a[`${key}.perm`],
        gain: a[`${key}.gain`],
      });
    }
    this.tints = TECH_COLORS.map((c) => new THREE.Color(c));
    this.female = new Brain(femaleSkeletons, fly.C, scene.now);
    scene.add(this.female, (male.radius + this.female.radius) * 1.15);
    this.female.group.visible = false; // only on stage for the transplant
  }

  /** Colour each glomerulus by the technique its assigned receptor smells most; size it by its neuron count. */
  styleNose(brain: Brain, perm: ArrayLike<number>, gain: ArrayLike<number>) {
    const receptorOf = new Int32Array(perm.length);
    for (let r = 0; r < perm.length; r++) receptorOf[perm[r]] = r;
    const tints = Array.from(receptorOf, (r) => this.tints[this.meta.receptor_technique[r]]);
    brain.setGlomerulusStyle(tints, gain);
  }

  frames(): number {
    return this.meta.replay.perm.length;
  }

  showGeneration(frame: number) {
    const r = this.meta.replay;
    this.styleNose(this.male, r.perm[frame], r.gain[frame]);
  }

  enterEvolve() {
    // She stays on stage: the nose being evolved is something that can be put into another fly.
    this.female.group.visible = true;
    this.scene.focus([this.male, this.female]);
    this.showGeneration(this.frames() - 1);
    const born = this.fly.noseOf("female:born");
    this.styleNose(this.female, born.perm, born.gain);
  }

  enterTransplant(nose: FemaleNose) {
    this.female.group.visible = true;
    this.scene.focus([this.male, this.female]);
    const evolved = this.fly.noseOf("male:evolved");
    this.styleNose(this.male, evolved.perm, evolved.gain);
    this.setFemaleNose(nose);
  }

  setFemaleNose(nose: FemaleNose) {
    const n = this.fly.noseOf(`female:${nose}`);
    this.styleNose(this.female, n.perm, n.gain);
  }

  exit() {
    this.male.setGlomerulusStyle(null, null);
    this.female.setGlomerulusStyle(null, null);
    this.female.group.visible = false;
    this.scene.focus([this.male]);
  }

  /** The same problem smelled by both flies: the male with his evolved nose, the female with the chosen one. */
  smellBoth(title: string, desc: string, nose: FemaleNose): { male: Smell; female: Smell } {
    const r = this.fly.receptors(problemText(title, desc));
    const male = this.fly.smellReceptors(r, "male:evolved");
    const female = this.fly.smellReceptors(r, `female:${nose}`);
    this.male.smell(male.pn, male.active, male.scores);
    this.female.smell(female.pn, female.active, female.scores);
    return { male, female };
  }
}

/** Dev-fitness curve over generations (1-based), with a marker at the replayed generation. */
export function renderCurve(svg: SVGSVGElement, best: number[], generation: number) {
  const W = 340;
  const H = 130;
  const pad = { l: 42, r: 10, t: 10, b: 28 };
  const lo = Math.floor(Math.min(...best) * 200) / 200;
  const hi = Math.ceil(Math.max(...best) * 200) / 200;
  const x = (g: number) => pad.l + ((W - pad.l - pad.r) * g) / (best.length - 1);
  const y = (v: number) => H - pad.b - ((H - pad.t - pad.b) * (v - lo)) / (hi - lo || 1);
  const path = best.map((v, g) => `${g ? "L" : "M"}${x(g).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const area = `${path}L${x(best.length - 1).toFixed(1)},${H - pad.b}L${x(0)},${H - pad.b}Z`;
  const gen = Math.min(generation - 1, best.length - 1);
  const ticksY = [lo, (lo + hi) / 2, hi];
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = `
    ${ticksY.map((v) => `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(v)}" y2="${y(v)}" class="grid"/>
      <text x="${pad.l - 6}" y="${y(v) + 4}" text-anchor="end" class="tick">${v.toFixed(3)}</text>`).join("")}
    ${[1, 50, 100, 150].filter((g) => g <= best.length).map((g) => `<text x="${x(g - 1)}" y="${H - 10}" text-anchor="middle" class="tick">${g}</text>`).join("")}
    <text x="${(pad.l + W - pad.r) / 2}" y="${H - 0.5}" text-anchor="middle" class="axis">generation</text>
    <path d="${area}" class="area"/><path d="${path}" class="line"/>
    <line x1="${x(gen)}" x2="${x(gen)}" y1="${pad.t}" y2="${H - pad.b}" class="now"/>
    <circle cx="${x(gen)}" cy="${y(best[gen])}" r="4" class="dot"/>`;
}
