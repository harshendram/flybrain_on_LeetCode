// The whole fly, in the browser. A line-by-line mirror of the Python pipeline (src/leetfly):
// text cleaning -> tokens -> TF-IDF -> NMF receptors -> nose -> antennal lobe -> KC drive (homeostasis)
// -> APL k-winners-take-all -> dopamine-trained MBON readout. web/tests/parity.test.ts checks it against Python.

type Dtype = "<f4" | "<f8" | "<u2" | "<u4";
type Typed = Float32Array | Float64Array | Uint16Array | Uint32Array;

export interface ArraySpec {
  offset: number;
  length: number;
  dtype: Dtype;
  shape: number[];
}

export interface Problem {
  slug: string;
  title: string;
  difficulty: string;
  tags: string[];
}

export interface ModelMeta {
  circuit: string;
  glomeruli: string[];
  techniques: string[];
  config: { sigma: number; m: number; sparsity: number; eta: number; k: number; n_exponent: number; nmf_iters: number };
  n_kc: number;
  vocab: string[];
  stop_words: string[];
  receptor_terms: string[][];
  problems: Problem[];
  scores_on_future_problems: Record<string, { "hit@1": number; "hit@3": number; macro_auroc: number }>;
  arrays: Record<string, ArraySpec>;
}

export type VariantName = "real" | "scrambled";

interface Variant {
  indptr: Uint32Array;
  indices: Uint16Array;
  data: Float32Array;
  kcScale: Float64Array;
  wPlus: Float64Array; // (n_kc, C) row-major, mutable: "train your fly" edits these
  wMinus: Float64Array;
  wPlus0: Float64Array; // trained-on-LeetCode weights, for reset
  wMinus0: Float64Array;
}

export interface Smell {
  receptors: Float64Array; // K receptor types
  glomeruli: Float64Array; // G glomerular inputs (after the nose)
  pn: Float64Array; // G projection-neuron rates
  drive: Float64Array; // n_kc
  active: Int32Array; // indices of firing KCs, strongest first
  scores: Float64Array; // C technique scores
  ranking: number[]; // technique indices, best first
}

const EXAMPLES = /Example\s*\d*:[\s\S]*?(?=Example\s*\d*:|Constraints:|$)/g;
const BIGO = /O\(([a-z])([2-9])\)/g;
const POW = /(?<![\w.^])(-?)10([1-9])(?![\d.])/g;
const TOKEN = /[a-z0-9^]+/g;
const EPS = 1e-9;

export function cleanText(text: string): string {
  let s = text.normalize("NFKD").replace(/[^\x00-\x7f]/g, "");
  s = s.replace(EXAMPLES, " ");
  s = s.replace(BIGO, "O($1^$2)");
  return s.replace(POW, "$1pow$2");
}

export function problemText(title: string, description: string): string {
  return `${title}. ${title}. ${cleanText(description)}`.trim();
}

export function analyze(text: string, stop: Set<string>): string[] {
  const words = (text.toLowerCase().match(TOKEN) ?? []).filter(
    (t) => t.length >= 2 && !stop.has(t) && !/^\d+$/.test(t),
  );
  const out = words.slice();
  for (let i = 0; i + 1 < words.length; i++) out.push(`${words[i]} ${words[i + 1]}`);
  return out;
}

export function decodeArrays(meta: ModelMeta, buf: ArrayBuffer): Record<string, Typed> {
  const out: Record<string, Typed> = {};
  for (const [name, s] of Object.entries(meta.arrays)) {
    const ctor = { "<f4": Float32Array, "<f8": Float64Array, "<u2": Uint16Array, "<u4": Uint32Array }[s.dtype];
    out[name] = new ctor(buf, s.offset, s.length);
  }
  return out;
}

export class Fly {
  readonly K: number;
  readonly V: number;
  readonly C: number;
  readonly nKc: number;
  private vocabIndex: Map<string, number>;
  private stop: Set<string>;
  private idf: Float64Array;
  private comps: Float32Array;
  private gram: Float64Array;
  private scale: Float64Array;
  readonly perm: Uint16Array;
  readonly gain: Float64Array;
  private variants: Record<VariantName, Variant>;
  private poolStart: Uint32Array;
  private poolActive: Uint16Array;

  constructor(readonly meta: ModelMeta, arrays: Record<string, Typed>) {
    this.K = meta.glomeruli.length;
    this.V = meta.vocab.length;
    this.C = meta.techniques.length;
    this.nKc = meta.n_kc;
    this.vocabIndex = new Map(meta.vocab.map((t, i) => [t, i]));
    this.stop = new Set(meta.stop_words);
    this.idf = arrays["idf"] as Float64Array;
    this.comps = arrays["components"] as Float32Array;
    this.gram = arrays["components_gram"] as Float64Array;
    this.scale = arrays["receptor_scale"] as Float64Array;
    this.perm = arrays["nose_perm"] as Uint16Array;
    this.gain = arrays["nose_gain"] as Float64Array;
    const variant = (p: string): Variant => {
      const wPlus0 = Float64Array.from(arrays[`${p}.w_plus`]);
      const wMinus0 = Float64Array.from(arrays[`${p}.w_minus`]);
      return {
        indptr: arrays[`${p}.w_indptr`] as Uint32Array,
        indices: arrays[`${p}.w_indices`] as Uint16Array,
        data: arrays[`${p}.w_data`] as Float32Array,
        kcScale: arrays[`${p}.kc_scale`] as Float64Array,
        wPlus: wPlus0.slice(),
        wMinus: wMinus0.slice(),
        wPlus0,
        wMinus0,
      };
    };
    this.variants = { real: variant("real"), scrambled: variant("scrambled") };
    const counts = arrays["pool_counts"] as Uint16Array;
    this.poolStart = new Uint32Array(counts.length + 1);
    for (let i = 0; i < counts.length; i++) this.poolStart[i + 1] = this.poolStart[i] + counts[i];
    this.poolActive = arrays["pool_active"] as Uint16Array;
  }

  /** l2-normalized sublinear TF-IDF, as a sparse vector. */
  tfidf(text: string): { idx: number[]; val: number[] } {
    const counts = new Map<number, number>();
    for (const tok of analyze(text, this.stop)) {
      const i = this.vocabIndex.get(tok);
      if (i !== undefined) counts.set(i, (counts.get(i) ?? 0) + 1);
    }
    const idx = [...counts.keys()].sort((a, b) => a - b);
    const val = idx.map((i) => (1 + Math.log(counts.get(i)!)) * this.idf[i]);
    const norm = Math.sqrt(val.reduce((s, v) => s + v * v, 0));
    return { idx, val: norm > 0 ? val.map((v) => v / norm) : val };
  }

  /** NMF codes by the same fixed number of multiplicative updates as Python, then per-receptor scaling. */
  receptors(text: string): Float64Array {
    const { K, V } = this;
    const x = this.tfidf(text);
    const xht = new Float64Array(K);
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let j = 0; j < x.idx.length; j++) s += x.val[j] * this.comps[k * V + x.idx[j]];
      xht[k] = s;
    }
    const w = new Float64Array(K);
    for (let k = 0; k < K; k++) w[k] = Math.max(xht[k] / this.gram[k * K + k], 1e-6);
    const denom = new Float64Array(K);
    for (let it = 0; it < this.meta.config.nmf_iters; it++) {
      for (let k = 0; k < K; k++) {
        let s = 0;
        for (let l = 0; l < K; l++) s += w[l] * this.gram[l * K + k];
        denom[k] = s;
      }
      for (let k = 0; k < K; k++) w[k] *= xht[k] / (denom[k] + EPS);
    }
    for (let k = 0; k < K; k++) w[k] /= this.scale[k];
    return w;
  }

  smell(title: string, description: string, which: VariantName = "real"): Smell {
    const { K, C, nKc } = this;
    const { sigma, m, k: nActive, n_exponent: n } = this.meta.config;
    const receptors = this.receptors(problemText(title, description));

    const glomeruli = new Float64Array(K);
    for (let r = 0; r < K; r++) glomeruli[this.perm[r]] = receptors[r];
    let total = 0;
    for (let g = 0; g < K; g++) {
      glomeruli[g] *= this.gain[g];
      total += glomeruli[g];
    }
    const pn = new Float64Array(K);
    const lateral = Math.pow(m * total, n);
    const sn = Math.pow(sigma, n);
    for (let g = 0; g < K; g++) {
      const xn = Math.pow(glomeruli[g], n);
      pn[g] = xn / (sn + xn + lateral);
    }

    const v = this.variants[which];
    const drive = new Float64Array(nKc);
    for (let j = 0; j < nKc; j++) {
      let s = 0;
      for (let p = v.indptr[j]; p < v.indptr[j + 1]; p++) s += v.data[p] * pn[v.indices[p]];
      drive[j] = s / v.kcScale[j];
    }
    const order = Array.from({ length: nKc }, (_, j) => j).sort((a, b) => drive[b] - drive[a] || a - b);
    const active = Int32Array.from(order.slice(0, nActive).filter((j) => drive[j] > 0));

    const scores = this.score(active, which);
    const ranking = Array.from({ length: C }, (_, c) => c).sort((a, b) => scores[b] - scores[a] || a - b);
    return { receptors, glomeruli, pn, drive, active, scores, ranking };
  }

  score(active: Int32Array, which: VariantName): Float64Array {
    const { C } = this;
    const v = this.variants[which];
    const scores = new Float64Array(C);
    for (const j of active) for (let c = 0; c < C; c++) scores[c] += v.wPlus[j * C + c] - v.wMinus[j * C + c];
    const denom = Math.max(active.length, 1);
    for (let c = 0; c < C; c++) scores[c] /= denom;
    return scores;
  }

  /** Train your fly: sugar (PAM) or a shock (PPL1) on the compartment of the technique it guessed. */
  reward(active: Int32Array, which: VariantName, guess: number, correct: boolean, delta = 0.5): void {
    const v = this.variants[which];
    const w = correct ? v.wMinus : v.wPlus;
    for (const j of active) w[j * this.C + guess] *= 1 - delta;
  }

  resetTraining(which: VariantName): void {
    const v = this.variants[which];
    v.wPlus.set(v.wPlus0);
    v.wMinus.set(v.wMinus0);
  }

  /** FlyHash: the problems whose KC codes overlap most with this one (Jaccard). */
  similar(active: Int32Array, k = 5, exclude?: string): { problem: Problem; jaccard: number }[] {
    const mask = new Uint8Array(this.nKc);
    for (const j of active) mask[j] = 1;
    const n = this.poolStart.length - 1;
    const sims = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let inter = 0;
      for (let p = this.poolStart[i]; p < this.poolStart[i + 1]; p++) inter += mask[this.poolActive[p]];
      const union = active.length + (this.poolStart[i + 1] - this.poolStart[i]) - inter;
      sims[i] = union > 0 ? inter / union : 0;
    }
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => sims[b] - sims[a] || a - b);
    return order
      .filter((i) => this.meta.problems[i].slug !== exclude)
      .slice(0, k)
      .map((i) => ({ problem: this.meta.problems[i], jaccard: sims[i] }));
  }
}
