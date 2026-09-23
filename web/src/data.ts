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
  neurons: NeuronInfo[];
  circuit: string;
  glomeruli: string[];
  glomerulus_xyz: [number, number, number][];
}

export interface Skeletons {
  meta: SkeletonMeta;
  positions: Float32Array; // microns, JRC2018U
  parents: Uint16Array;
}

async function fetchWithProgress(
  url: string,
  onTotal: (n: number) => void,
  onBytes: (n: number) => void,
): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`failed to load ${url}: ${res.status}`);
  onTotal(Number(res.headers.get("content-length") ?? 0));
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
    onBytes(value.length);
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

async function fetchBinary(url: string, onTotal: (n: number) => void, onBytes: (n: number) => void) {
  if (!BASE64_DATA) return fetchWithProgress(url, onTotal, onBytes);
  const text = new TextDecoder().decode(await fetchWithProgress(`${url}.b64.txt`, onTotal, onBytes));
  const raw = atob(text.trim());
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes.buffer;
}

function decodeSkeletons(meta: SkeletonMeta, bin: ArrayBuffer): Skeletons {
  const n = meta.n_nodes;
  const q = new Uint16Array(bin, 0, 3 * n);
  const parents = new Uint16Array(bin, 6 * n, n);
  const positions = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) positions[3 * i + a] = q[3 * i + a] / meta.scale + meta.origin[a];
  }
  return { meta, positions, parents };
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

/** Phase 2 (evolve + transplant): the female fly's skeletons and every fly/nose variant. Optional. */
export async function loadPhase2(
  onProgress: (loadedBytes: number, totalBytes: number) => void,
): Promise<{ meta: Phase2Meta; bin: ArrayBuffer; female: Skeletons } | null> {
  const base = "./data/";
  const metaRes = await fetch(base + "phase2.json");
  if (!metaRes.ok) return null;
  const meta = (await metaRes.json()) as Phase2Meta;
  const skelMeta = (await fetch(base + `${meta.flies.female.circuit}_skeletons.json`).then((r) => r.json())) as SkeletonMeta;
  let loaded = 0;
  let total = 0;
  const [bin, skelBin] = await Promise.all([
    fetchBinary(base + "phase2.bin", (n) => (total += n), (n) => onProgress((loaded += n), total)),
    fetchBinary(base + `${meta.flies.female.circuit}_skeletons.bin`, (n) => (total += n), (n) => onProgress((loaded += n), total)),
  ]);
  return { meta, bin, female: decodeSkeletons(skelMeta, skelBin) };
}

export async function loadAll(
  onProgress: (loadedBytes: number, totalBytes: number) => void,
): Promise<{ fly: Fly; skeletons: Skeletons }> {
  let loaded = 0;
  let total = 0;
  const addTotal = (n: number) => (total += n);
  const tick = (n: number) => onProgress((loaded += n), total);
  const base = "./data/";
  const [modelMeta, skelMeta] = await Promise.all([
    fetch(base + "model.json").then((r) => r.json() as Promise<ModelMeta>),
    fetch(base + "malecns_R_skeletons.json").then((r) => r.json() as Promise<SkeletonMeta>),
  ]);
  const [modelBin, skelBin] = await Promise.all([
    fetchBinary(base + "model.bin", addTotal, tick),
    fetchBinary(base + "malecns_R_skeletons.bin", addTotal, tick),
  ]);
  const fly = new Fly(modelMeta, decodeArrays(modelMeta, modelBin));
  return { fly, skeletons: decodeSkeletons(skelMeta, skelBin) };
}
