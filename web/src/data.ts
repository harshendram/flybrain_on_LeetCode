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

  const n = skelMeta.n_nodes;
  const q = new Uint16Array(skelBin, 0, 3 * n);
  const parents = new Uint16Array(skelBin, 6 * n, n);
  const positions = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) positions[3 * i + a] = q[3 * i + a] / skelMeta.scale + skelMeta.origin[a];
  }
  return { fly, skeletons: { meta: skelMeta, positions, parents } };
}
