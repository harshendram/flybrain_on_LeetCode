import { Fly, decodeArrays, type ModelMeta } from "./fly";

export interface NeuronInfo {
  id: number;
  role: "PN" | "KC" | "MBON" | "DAN" | "APL";
  type: string;
  glom?: number;
  kc_index?: number;
  offset: number;
  count: number;
}

export interface SkeletonMeta {
  origin: [number, number, number];
  scale: number;
  n_nodes: number;
  dist_scale?: number;
  neurons: NeuronInfo[];
  circuit: string;
  glomeruli: string[];
  glomerulus_xyz: [number, number, number][];
}

export interface Skeletons {
  meta: SkeletonMeta;
  positions: Float32Array; // microns, display axes
  parents: Uint16Array;
  dist: Float32Array | null; // cable distance from the soma (microns), for spike pulses
}

export interface CloudMeta {
  n: number;
  origin: [number, number, number];
  scale: number;
  regions: string[];
  region_counts: number[];
  circuit: string;
  circuit_dot: number[]; // skeleton-manifest neuron -> cloud dot (-1 if its soma is unknown)
}

export interface Cloud {
  meta: CloudMeta;
  positions: Float32Array;
  region: Uint8Array;
}

export interface LearningMeta {
  techniques: string[];
  eta: number;
  balanced: boolean;
  k: number;
  sets: Record<"dev" | "future", { n: number; k_max: number; class_counts: number[] }>;
  arrays: Record<string, { offset: number; length: number; dtype: string; shape: number[] }>;
}

export interface LearningSet {
  n: number;
  kMax: number;
  problem: Uint16Array; // index into model.json's problems (titles/slugs)
  techniques: Uint16Array; // bitmask over techniques
  receptors: Float32Array; // (n, 51)
  active: Uint16Array; // (n, kMax) firing KCs, padded with 65535
}

export interface Learning {
  meta: LearningMeta;
  dev: LearningSet;
  future: LearningSet;
}

/** Shared download-progress bookkeeping across several files. */
export class Progress {
  loaded = 0;
  total = 0;
  constructor(private onChange: (loaded: number, total: number) => void = () => {}) {}
  addTotal = (n: number) => {
    this.total += n;
  };
  tick = (n: number) => {
    this.loaded += n;
    this.onChange(this.loaded, this.total);
  };
}

async function fetchWithProgress(url: string, p: Progress): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`failed to load ${url}: ${res.status}`);
  p.addTotal(Number(res.headers.get("content-length") ?? 0));
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
    p.tick(value.length);
  }
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out.buffer;
}

/** Some hosts (claude.ai artifacts) don't serve raw .bin files; there the build ships base64 text instead. */
const BASE64_DATA = (globalThis as { __LEETFLY_B64__?: boolean }).__LEETFLY_B64__ === true;
const BASE = "./data/";

export async function fetchBinary(name: string, p: Progress, base = BASE): Promise<ArrayBuffer> {
  if (!BASE64_DATA) return fetchWithProgress(base + name, p);
  const text = new TextDecoder().decode(await fetchWithProgress(`${base}${name}.b64.txt`, p));
  const raw = atob(text.trim());
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes.buffer;
}

const fetchJson = <T>(name: string): Promise<T> =>
  fetch(BASE + name).then((r) => {
    if (!r.ok) throw new Error(`failed to load ${name}: ${r.status}`);
    return r.json() as Promise<T>;
  });

export async function loadModel(p: Progress): Promise<Fly> {
  const [meta, bin] = await Promise.all([fetchJson<ModelMeta>("model.json"), fetchBinary("model.bin", p)]);
  return new Fly(meta, decodeArrays(meta, bin));
}

export async function loadSkeletons(circuit: string, p: Progress): Promise<Skeletons> {
  const [meta, bin] = await Promise.all([
    fetchJson<SkeletonMeta>(`${circuit}_skeletons.json`),
    fetchBinary(`${circuit}_skeletons.bin`, p),
  ]);
  const n = meta.n_nodes;
  const q = new Uint16Array(bin, 0, 3 * n);
  const parents = new Uint16Array(bin, 6 * n, n);
  const positions = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) positions[3 * i + a] = q[3 * i + a] / meta.scale + meta.origin[a];
  }
  let dist: Float32Array | null = null;
  if (meta.dist_scale) {
    const d = new Uint16Array(bin, 8 * n, n);
    dist = Float32Array.from(d, (v) => v / meta.dist_scale!);
  }
  return { meta, positions, parents, dist };
}

export interface ContextPack {
  positions: Float32Array;
  /** Absolute node index of the parent, or 65535 at a root. Wider than the packed local parents. */
  parents: Uint32Array;
  /** 0 partner, 1 lateral horn, 2 the other mushroom body. One value per node. */
  kind: Float32Array;
  role: Float32Array;
  /** Cable distance from the soma (µm) and a per-neuron random seed, for the resting shimmer. */
  dist: Float32Array;
  seed: Float32Array;
}

const CONTEXT_ROLE: Record<string, number> = { KC: 0, PN: 1, MBON: 2, PAM: 3, PPL1: 4, APL: 5 };

/** Left mushroom body, lateral horn, and one-synapse partners. Not part of the spike texture. */
export async function loadContext(p: Progress): Promise<ContextPack> {
  const [meta, bin] = await Promise.all([
    fetchJson<{ origin: number[]; scale: number; n_nodes: number; dist_scale?: number; neurons: { role: string; group: string; offset: number; count: number }[] }>(
      "malecns_context.json",
    ),
    fetchBinary("malecns_context.bin", p),
  ]);
  const n = meta.n_nodes;
  const q = new Uint16Array(bin, 0, 3 * n);
  const parents = new Uint16Array(bin, 6 * n, n);
  const positions = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) positions[3 * i + a] = q[3 * i + a] / meta.scale + meta.origin[a];
  const kind = new Float32Array(n);
  const role = new Float32Array(n);
  const seed = new Float32Array(n);
  const dist = new Float32Array(n);
  if (meta.dist_scale) {
    const qd = new Uint16Array(bin, 8 * n, n);
    for (let i = 0; i < n; i++) dist[i] = qd[i] / meta.dist_scale;
  }
  meta.neurons.forEach((neuron, j) => {
    const k = neuron.group === "left" ? 2 : neuron.group === "lh" ? 1 : 0;
    const r = CONTEXT_ROLE[neuron.role] ?? 0;
    const sd = (Math.sin(j * 12.9898) * 43758.5453) % 1;
    for (let i = neuron.offset; i < neuron.offset + neuron.count; i++) {
      kind[i] = k;
      role[i] = r;
      seed[i] = Math.abs(sd);
    }
  });
  // parents in the pack are local to each neuron; the line index needs absolute node ids
  const absolute = new Uint32Array(n);
  absolute.fill(0xffffffff);
  for (const neuron of meta.neurons) {
    for (let i = 0; i < neuron.count; i++) {
      const p = parents[neuron.offset + i];
      absolute[neuron.offset + i] = p === 65535 ? 0xffffffff : neuron.offset + p;
    }
  }
  return { positions, parents: absolute, kind, role, dist, seed };
}

export async function loadCloud(dataset: string, p: Progress): Promise<Cloud> {
  const [meta, bin] = await Promise.all([fetchJson<CloudMeta>(`${dataset}_cloud.json`), fetchBinary(`${dataset}_cloud.bin`, p)]);
  const q = new Uint16Array(bin, 0, 3 * meta.n);
  const positions = new Float32Array(3 * meta.n);
  for (let i = 0; i < meta.n; i++) {
    for (let a = 0; a < 3; a++) positions[3 * i + a] = q[3 * i + a] / meta.scale + meta.origin[a];
  }
  return { meta, positions, region: new Uint8Array(bin, 6 * meta.n, meta.n) };
}

export async function loadLearning(p: Progress): Promise<Learning> {
  const [meta, bin] = await Promise.all([fetchJson<LearningMeta>("learning.json"), fetchBinary("learning.bin", p)]);
  const a = decodeArrays({ arrays: meta.arrays } as unknown as ModelMeta, bin) as Record<string, any>;
  const set = (name: "dev" | "future"): LearningSet => ({
    n: meta.sets[name].n,
    kMax: meta.sets[name].k_max,
    problem: a[`${name}.problem`],
    techniques: a[`${name}.techniques`],
    receptors: a[`${name}.receptors`],
    active: a[`${name}.active`],
  });
  return { meta, dev: set("dev"), future: set("future") };
}

export interface Phase2Meta {
  flies: Record<"male" | "female", { circuit: string; label: string }>;
  variants: Record<string, { fly: "male" | "female"; nose: string }>;
  scores: Record<string, { dev_auroc: number; test_auroc: number; test_hit1: number; test_hit3: number }>;
  receptor_informativeness: number[];
  receptor_technique: number[];
  replay: { generations: number[]; perm: number[][]; gain: number[][]; best_fitness: number[]; seed: number };
  summary: Record<string, unknown>;
  exploratory: Record<string, { median_TR: number; mean_gain_real_nose: number; mean_gain_dp_nose: number; mean_gain_uniform_nose: number }> | null;
  arrays: Record<string, { offset: number; length: number; dtype: string; shape: number[] }>;
}

/** Phase 2 (evolve + transplant): the female fly's skeletons + cloud and every fly/nose variant. Optional. */
export async function loadPhase2(p: Progress): Promise<{ meta: Phase2Meta; bin: ArrayBuffer; female: Skeletons; femaleCloud: Cloud } | null> {
  const res = await fetch(BASE + "phase2.json");
  if (!res.ok) return null;
  const meta = (await res.json()) as Phase2Meta;
  const [bin, female, femaleCloud] = await Promise.all([
    fetchBinary("phase2.bin", p),
    loadSkeletons(meta.flies.female.circuit, p),
    loadCloud(meta.flies.female.circuit.split("_")[0], p),
  ]);
  return { meta, bin, female, femaleCloud };
}

/** Everything the brain page needs to start: the fly model, the male circuit and his whole-brain cloud. */
export async function loadAll(onProgress: (loaded: number, total: number) => void) {
  const p = new Progress(onProgress);
  const [fly, skeletons, cloud] = await Promise.all([loadModel(p), loadSkeletons("malecns_R", p), loadCloud("malecns", p)]);
  return { fly, skeletons, cloud };
}
