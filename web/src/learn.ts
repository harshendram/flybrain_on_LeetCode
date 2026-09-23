// "Watch it learn": a naive fly learns LeetCode one problem at a time, with the exact trial-by-trial dopamine rule
// (web/tests/learn.test.ts checks that streaming every problem ends at the published Python weights).
import type { Learning, LearningSet } from "./data";
import { Fly, type Problem } from "./fly";
import type { Brain } from "./scene";

const VARIANT = "learn";

export class LearnMode {
  playing = false;
  speed = 10; // training problems per second
  private i = 0;
  private carry = 0;
  private curve: [number, number][] = [];
  private rates: { dc: Float64Array; dnot: Float64Array };
  private lastVisual = -1;

  constructor(
    private fly: Fly,
    private brain: Brain,
    private data: Learning,
    private problems: Problem[],
    private chance: number, // top-1 of "always guess the most common technique"
    private el: { progress: HTMLElement; now: HTMLElement; curve: SVGSVGElement; note: HTMLElement; play: HTMLElement },
  ) {
    fly.cloneVariant("real", VARIANT);
    const s = data.meta.sets.dev;
    this.rates = Fly.dopamineRates(s.class_counts, s.n, data.meta.eta);
    this.reset();
  }

  reset() {
    this.fly.resetNaive(VARIANT);
    this.i = 0;
    this.carry = 0;
    this.curve = [[0, this.accuracy()]];
    this.render(null);
  }

  toggle() {
    if (this.i >= this.data.dev.n) this.reset();
    this.playing = !this.playing;
    this.el.play.textContent = this.playing ? "Pause" : "Play";
  }

  private active(set: LearningSet, i: number): Uint16Array {
    return set.active.subarray(i * set.kMax, (i + 1) * set.kMax);
  }

  /** Top-1 on future problems. Ties share the credit, so a naive fly (all outputs equal) sits at chance (~11%). */
  accuracy(): number {
    const f = this.data.future;
    const C = this.fly.C;
    let hits = 0;
    for (let i = 0; i < f.n; i++) {
      const s = this.fly.score(Int32Array.from(this.active(f, i)).filter((j) => j !== 65535), VARIANT);
      let best = -Infinity;
      for (let c = 0; c < C; c++) best = Math.max(best, s[c]);
      let tied = 0;
      let right = 0;
      for (let c = 0; c < C; c++) {
        if (s[c] >= best - 1e-12) {
          tied++;
          if (f.techniques[i] & (1 << c)) right++;
        }
      }
      hits += right / tied;
    }
    return hits / f.n;
  }

  /** Called every frame while the mode is open. */
  tick(t: number, dt: number) {
    if (!this.playing) return;
    this.carry += dt * this.speed;
    let steps = Math.floor(this.carry);
    this.carry -= steps;
    const dev = this.data.dev;
    let shown: { i: number; guess: number } | null = null;
    while (steps-- > 0 && this.i < dev.n) {
      const act = this.active(dev, this.i);
      const guess = this.fly.topGuess(VARIANT, act); // what it believed before this dopamine
      this.fly.learnStep(VARIANT, act, dev.techniques[this.i], this.rates);
      shown = { i: this.i, guess };
      this.i++;
      const every = this.speed >= 50 ? 40 : 10;
      if (this.i % every === 0 || this.i === dev.n) this.curve.push([this.i, this.accuracy()]);
    }
    if (shown && t - this.lastVisual > Math.max(0.12, 0.9 / this.speed)) {
      this.lastVisual = t;
      this.visualise(shown.i, shown.guess);
    }
    if (shown) this.render(shown);
    if (this.i >= dev.n) {
      this.playing = false;
      this.el.play.textContent = "Replay";
    }
  }

  private visualise(i: number, guess: number) {
    const dev = this.data.dev;
    const act = this.active(dev, i);
    const firing = Int32Array.from(Array.from(act).filter((j) => j !== 65535));
    const pn = this.fly.pnRates(dev.receptors.subarray(i * 51, (i + 1) * 51), VARIANT);
    const scores = this.fly.score(firing, VARIANT);
    const slow = this.speed <= 2;
    this.brain.smell(pn, firing, scores, slow ? 1.4 : 0.7);
    const right = (dev.techniques[i] & (1 << guess)) !== 0;
    // teacher signal: reward DANs for the true techniques' compartments, punishment for the rest
    this.brain.dopamine(right ? "PAM" : "PPL1", slow ? 1 : 0.7);
  }

  private render(shown: { i: number; guess: number } | null) {
    const dev = this.data.dev;
    const acc = this.curve[this.curve.length - 1][1];
    this.el.progress.textContent = `problem ${this.i.toLocaleString()} / ${dev.n.toLocaleString()}  ·  future-problem top-1 ${(100 * acc).toFixed(1)}%`;
    if (shown) {
      const p = this.problems[dev.problem[shown.i]];
      const names = this.data.meta.techniques;
      const truth = names.filter((_, c) => dev.techniques[shown.i] & (1 << c));
      const right = truth.includes(names[shown.guess]);
      this.el.now.innerHTML =
        `<a href="https://leetcode.com/problems/${encodeURIComponent(p.slug)}/" target="_blank" rel="noopener">${esc(titleCase(p.title))}</a>` +
        `<span class="guess ${right ? "ok" : "bad"}">guessed ${esc(names[shown.guess])} ${right ? "→ sugar" : "→ shock"}</span>` +
        `<span class="truth">answer: ${esc(truth.join(", "))}</span>`;
    }
    renderAccuracy(this.el.curve, this.curve, dev.n, this.chance);
    const first = this.curve[0][1];
    this.el.note.innerHTML =
      this.i === 0
        ? `A naive fly has no preferences: <b>${(100 * first).toFixed(0)}%</b> on problems from its future, pure chance. Press play.`
        : `From <b>${(100 * first).toFixed(0)}%</b> to <b>${(100 * acc).toFixed(1)}%</b> after ${this.i.toLocaleString()} problems. ` +
          `Always guessing the most common technique gets ${(100 * this.chance).toFixed(0)}%.`;
  }
}

function renderAccuracy(svg: SVGSVGElement, pts: [number, number][], n: number, chance: number) {
  const W = 340;
  const H = 140;
  const pad = { l: 36, r: 10, t: 10, b: 26 };
  const hi = 0.5;
  const x = (v: number) => pad.l + ((W - pad.l - pad.r) * v) / n;
  const y = (v: number) => H - pad.b - ((H - pad.t - pad.b) * v) / hi;
  const path = pts.map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join("");
  const last = pts[pts.length - 1];
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = `
    ${[0, 0.1, 0.2, 0.3, 0.4, 0.5].map((v) => `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(v)}" y2="${y(v)}" class="grid"/>
      <text x="${pad.l - 5}" y="${y(v) + 3.5}" text-anchor="end" class="tick">${Math.round(v * 100)}%</text>`).join("")}
    <line x1="${pad.l}" x2="${W - pad.r}" y1="${y(chance)}" y2="${y(chance)}" class="chance"/>
    <text x="${W - pad.r}" y="${y(chance) - 4}" text-anchor="end" class="tick">always guess the most common</text>
    ${[0, 500, 1000, 1500].filter((v) => v <= n).map((v) => `<text x="${x(v)}" y="${H - 10}" text-anchor="middle" class="tick">${v}</text>`).join("")}
    <text x="${(pad.l + W - pad.r) / 2}" y="${H - 0.5}" text-anchor="middle" class="axis">problems learned</text>
    <path d="${path}" class="line"/>
    <circle cx="${x(last[0])}" cy="${y(last[1])}" r="4" class="dot"/>`;
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const titleCase = (s: string) => s.replace(/\b[a-z]/g, (c) => c.toUpperCase());
