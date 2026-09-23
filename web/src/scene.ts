import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import type { Cloud, Skeletons } from "./data";

const ROLES = { KC: 0, PN: 1, MBON: 2, PAM: 3, PPL1: 4, APL: 5 } as const;
export type Band = "PN" | "KC" | "OUT" | "DA";
// [base rgb, glow rgb, resting opacity, peak opacity]. Normal alpha blending: a bundle of ~2,000 overlapping KC axons
// saturates to its own colour instead of blowing out to white; the bloom pass supplies the glow of firing cells.
const PALETTE: [number, number, number, number, number, number, number, number][] = [
  [0.16, 0.28, 0.7, 0.45, 0.95, 1.0, 0.035, 0.6], // KC
  [0.75, 0.38, 0.12, 1.0, 0.72, 0.35, 0.07, 0.75], // PN
  [0.2, 0.55, 0.3, 0.55, 1.0, 0.65, 0.06, 1.0], // MBON
  [0.6, 0.45, 0.12, 1.0, 0.85, 0.4, 0.025, 0.28], // PAM (reward): ~160 cells with dense arbours
  [0.65, 0.16, 0.22, 1.0, 0.45, 0.5, 0.06, 0.8], // PPL1 (punishment): 8 cells
  [0.4, 0.28, 0.65, 0.8, 0.65, 1.0, 0.04, 0.45], // APL
];
export const ROLE_COLORS = PALETTE.map(([r, g, b]) => `rgb(${(r * 255) | 0}, ${(g * 255) | 0}, ${(b * 255) | 0})`);
export const BAND_COLORS: Record<Band, string> = { PN: "#ffb35c", KC: "#66f0ff", OUT: "#8cf5a8", DA: "#ffd166" };

const TEX = 64; // activity texture: TEX x TEX texels, one per neuron (<= 4,096 neurons per brain)
const GLOM_RADIUS = 2.6;
const PULSE_SPEED = 320; // microns per second along the arbor
const PULSE_WIDTH = 0.05; // seconds (rise)
const PULSE_TAIL = 0.16; // seconds (decay behind the front)
export const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const VERT = /* glsl */ `
  attribute float aNid;
  attribute float aRole;
  attribute float aDist;
  uniform sampler2D uAct;
  uniform float uTime;
  uniform vec3 uBase[6];
  uniform vec3 uGlow[6];
  uniform float uRest[6];
  uniform float uPeak[6];
  uniform float uShowResting;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    int nid = int(aNid + 0.5);
    vec4 act = texelFetch(uAct, ivec2(nid % ${TEX}, nid / ${TEX}), 0);
    float a = clamp(act.r, 0.0, 1.0);
    // a spike: a bright front travelling out from the soma along the real arbor
    float dt = uTime - act.g - aDist / ${PULSE_SPEED.toFixed(1)};
    float pulse = act.b * (dt < 0.0 ? exp(-dt * dt / ${(PULSE_WIDTH * PULSE_WIDTH).toFixed(5)}) : exp(-dt / ${PULSE_TAIL.toFixed(3)}));
    int role = int(aRole + 0.5);
    vColor = mix(uBase[role], uGlow[role], max(a, pulse)) + vec3(0.55) * pulse * pulse;
    vAlpha = max(max(uRest[role] * uShowResting, a * uPeak[role]), min(1.0, pulse * 1.1));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;
const FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() { gl_FragColor = vec4(vColor, vAlpha); }`;

// ---------- whole-brain point cloud ----------
const REGION_TINTS = [
  [0.22, 0.62, 0.72], // optic lobe
  [0.52, 0.45, 0.95], // central brain
  [0.8, 0.8, 1.0], // descending / ascending
  [0.3, 0.45, 0.95], // ventral nerve cord
  [0.55, 0.55, 0.6], // other
];
const MAX_FLASH = 64;
const CLOUD_VERT = /* glsl */ `
  attribute float aRegion;
  attribute float aIndex;
  uniform float uTime;
  uniform float uSize;
  uniform float uPixelRatio;
  uniform vec3 uTint[5];
  uniform float uFlashIdx[${MAX_FLASH}];
  uniform float uFlashT[${MAX_FLASH}];
  uniform vec3 uFlashColor[${MAX_FLASH}];
  uniform float uSparkle;
  varying vec3 vColor;
  varying float vAlpha;
  float hash(float n) { return fract(sin(n) * 43758.5453123); }
  void main() {
    vec3 tint = uTint[int(aRegion + 0.5)];
    float slow = 0.75 + 0.25 * sin(uTime * 0.7 + hash(aIndex) * 6.283);
    // sparse spontaneous "spikes": each dot lights up now and then, a different random set every 0.2 s
    float tick = floor(uTime * 5.0 + hash(aIndex * 0.37) * 5.0);
    float spark = step(1.0 - uSparkle, hash(aIndex * 1.13 + tick * 7.1)) * (1.0 - fract(uTime * 5.0 + hash(aIndex * 0.37) * 5.0));
    vec3 col = tint * (0.7 * slow) + vec3(1.0, 1.0, 1.0) * spark * 1.3;
    float alpha = 0.34 * slow + 0.85 * spark;
    float size = uSize + 2.6 * spark;
    for (int i = 0; i < ${MAX_FLASH}; i++) {
      if (abs(uFlashIdx[i] - aIndex) < 0.5) {
        float f = exp(-(uTime - uFlashT[i]) * 2.2);
        col = mix(col, uFlashColor[i] * 1.6, f);
        alpha = max(alpha, f);
        size += 5.0 * f;
      }
    }
    vColor = col;
    vAlpha = alpha;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * uPixelRatio * (320.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const CLOUD_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float d = dot(p, p);
    if (d > 1.0) discard;
    gl_FragColor = vec4(vColor, vAlpha * (1.0 - d));
  }`;

export class PointCloud {
  readonly points: THREE.Points;
  readonly box = new THREE.Box3();
  private material: THREE.ShaderMaterial;
  private flashSlot = 0;

  constructor(cloud: Cloud, toScene: (x: number, y: number, z: number) => [number, number, number], lod = 1) {
    const n = Math.ceil(cloud.meta.n / lod);
    const pos = new Float32Array(3 * n);
    const region = new Float32Array(n);
    const index = new Float32Array(n);
    const v = new THREE.Vector3();
    for (let i = 0, j = 0; i < cloud.meta.n; i += lod, j++) {
      const p = toScene(cloud.positions[3 * i], cloud.positions[3 * i + 1], cloud.positions[3 * i + 2]);
      pos.set(p, 3 * j);
      region[j] = cloud.region[i];
      index[j] = i;
      this.box.expandByPoint(v.fromArray(p));
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("aRegion", new THREE.BufferAttribute(region, 1));
    geom.setAttribute("aIndex", new THREE.BufferAttribute(index, 1));
    this.material = new THREE.ShaderMaterial({
      vertexShader: CLOUD_VERT,
      fragmentShader: CLOUD_FRAG,
      uniforms: {
        uTime: { value: 0 },
        uSize: { value: 2.3 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
        uTint: { value: REGION_TINTS.map((c) => new THREE.Vector3(...c)) },
        uFlashIdx: { value: new Array(MAX_FLASH).fill(-1) },
        uFlashT: { value: new Array(MAX_FLASH).fill(-99) },
        uFlashColor: { value: Array.from({ length: MAX_FLASH }, () => new THREE.Vector3(1, 1, 1)) },
        uSparkle: { value: REDUCED_MOTION ? 0 : 0.004 },
      },
      transparent: true,
      depthWrite: false,
    });
    this.points = new THREE.Points(geom, this.material);
  }

  /** Light one real neuron's soma dot (e.g. a Kenyon cell that just fired). */
  flash(dot: number, t: number, color: THREE.Color) {
    const u = this.material.uniforms;
    const s = this.flashSlot++ % MAX_FLASH;
    u.uFlashIdx.value[s] = dot;
    u.uFlashT.value[s] = t;
    u.uFlashColor.value[s].set(color.r, color.g, color.b);
  }

  update(t: number) {
    this.material.uniforms.uTime.value = t;
  }
}

// ---------- one fly's circuit ----------
interface Train {
  rate: number; // spikes per second
  next: number; // time of the next spike
  end: number;
  strength: number;
}

export interface SpikeEvent {
  band: Band;
  row: number; // 0..1 position within the band
  t: number;
  neuron: number;
  color?: string; // overrides the band colour (punishment dopamine is red)
}

/** One fly's mushroom-body circuit: real neuron skeletons, glomeruli, its whole-brain cloud, and spiking activity. */
export class Brain {
  readonly group = new THREE.Group();
  readonly radius: number;
  readonly kcNeuron: Int32Array; // model KC index -> neuron index
  readonly pnByGlom: number[][];
  readonly displayMbons: number[]; // one real MBON drawn per technique
  readonly glom: THREE.InstancedMesh;
  readonly counts: Record<"PN" | "KC" | "MBON" | "DAN" | "all", number>;
  readonly toScene: (x: number, y: number, z: number) => [number, number, number];
  cloud: PointCloud | null = null;
  firingKcs = 0;
  onSpike: (e: SpikeEvent) => void = () => {};
  private cloudDot: Int32Array | null = null;
  private material: THREE.ShaderMaterial;
  private actData: Float32Array;
  private actTex: THREE.DataTexture;
  private target: Float32Array;
  private current: Float32Array;
  private start: Float32Array;
  private flicker = new Set<number>();
  private trains = new Map<number, Train>();
  private band: (Band | null)[];
  private bandRow: Float32Array;
  private glomTarget: Float32Array;
  private glomCurrent: Float32Array;
  private glomTint: THREE.Color[] | null = null; // evolution view: colour by assigned receptor
  private glomSize: Float32Array;
  private glomPos: THREE.Vector3[];
  private pam: number[] = [];
  private ppl1: number[] = [];
  private apl = -1;
  private nextBackground = 0;

  constructor(sk: Skeletons, nTechniques: number, private now: () => number) {
    const { meta, positions, parents, dist } = sk;
    const nNeurons = meta.neurons.length;
    // centre on the mushroom body + antennal lobe; dorsal up, anterior towards the camera
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (const n of meta.neurons) {
      if (n.role !== "KC") continue;
      for (let i = n.offset; i < n.offset + n.count; i += 7) box.expandByPoint(v.fromArray(positions, 3 * i));
    }
    for (const g of meta.glomerulus_xyz) box.expandByPoint(v.fromArray(g));
    const c = box.getCenter(new THREE.Vector3());
    this.radius = box.getSize(new THREE.Vector3()).length() / 2;
    this.toScene = (x, y, z) => [x - c.x, -(y - c.y), -(z - c.z)];

    const nNodes = meta.n_nodes;
    const pos = new Float32Array(3 * nNodes);
    const nid = new Float32Array(nNodes);
    const role = new Float32Array(nNodes);
    const index: number[] = [];
    this.kcNeuron = new Int32Array(meta.neurons.filter((n) => n.role === "KC").length);
    this.pnByGlom = meta.glomeruli.map(() => []);
    this.band = new Array(nNeurons).fill(null);
    this.bandRow = new Float32Array(nNeurons);
    const mbonByType = new Map<string, number>();
    const roleCount = { PN: 0, KC: 0, MBON: 0, DAN: 0 };
    meta.neurons.forEach((n, k) => {
      const r =
        n.role === "DAN" ? (n.type.startsWith("PAM") ? ROLES.PAM : ROLES.PPL1) : ROLES[n.role as keyof typeof ROLES];
      if (n.role === "KC") this.kcNeuron[n.kc_index!] = k;
      if (n.role === "PN") this.pnByGlom[n.glom!].push(k);
      if (n.role === "MBON" && !mbonByType.has(n.type)) mbonByType.set(n.type, k);
      if (r === ROLES.PAM) this.pam.push(k);
      if (r === ROLES.PPL1) this.ppl1.push(k);
      if (n.role === "APL") this.apl = k;
      if (n.role in roleCount) roleCount[n.role as keyof typeof roleCount]++;
      for (let i = n.offset; i < n.offset + n.count; i++) {
        pos.set(this.toScene(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]), 3 * i);
        nid[i] = k;
        role[i] = r;
        const p = parents[i];
        if (p !== 65535) index.push(i, n.offset + p);
      }
    });
    this.counts = { ...roleCount, all: nNeurons };
    // Each technique's approach/avoid MBON pair is abstract in the model; we draw it on a distinct real MBON type.
    this.displayMbons = [...mbonByType.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .slice(0, nTechniques)
      .map(([, k]) => k);
    // raster rows: where each neuron sits inside its band
    const pns = meta.neurons.map((n, k) => (n.role === "PN" ? k : -1)).filter((k) => k >= 0);
    pns.forEach((k, i) => ((this.band[k] = "PN"), (this.bandRow[k] = (i + 0.5) / pns.length)));
    this.kcNeuron.forEach((k, i) => ((this.band[k] = "KC"), (this.bandRow[k] = (i + 0.5) / this.kcNeuron.length)));
    this.displayMbons.forEach((k, i) => ((this.band[k] = "OUT"), (this.bandRow[k] = (i + 0.5) / this.displayMbons.length)));
    [...this.pam, ...this.ppl1].forEach((k, i, all) => ((this.band[k] = "DA"), (this.bandRow[k] = (i + 0.5) / all.length)));

    this.actData = new Float32Array(TEX * TEX * 4);
    this.actTex = new THREE.DataTexture(this.actData, TEX, TEX, THREE.RGBAFormat, THREE.FloatType);
    for (let k = 0; k < TEX * TEX; k++) this.actData[4 * k + 1] = -99; // no spike yet
    this.actTex.needsUpdate = true;
    this.target = new Float32Array(nNeurons);
    this.current = new Float32Array(nNeurons);
    this.start = new Float32Array(nNeurons);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("aNid", new THREE.BufferAttribute(nid, 1));
    geom.setAttribute("aRole", new THREE.BufferAttribute(role, 1));
    geom.setAttribute("aDist", new THREE.BufferAttribute(dist ?? new Float32Array(nNodes), 1));
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(index), 1));
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uAct: { value: this.actTex },
        uTime: { value: 0 },
        uBase: { value: PALETTE.map((p) => new THREE.Vector3(p[0], p[1], p[2])) },
        uGlow: { value: PALETTE.map((p) => new THREE.Vector3(p[3], p[4], p[5])) },
        uRest: { value: PALETTE.map((p) => p[6]) },
        uPeak: { value: PALETTE.map((p) => p[7]) },
        uShowResting: { value: 1 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });
    this.group.add(new THREE.LineSegments(geom, this.material));

    const G = meta.glomeruli.length;
    this.glom = new THREE.InstancedMesh(
      new THREE.SphereGeometry(1, 18, 12),
      new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }),
      G,
    );
    this.glomPos = meta.glomerulus_xyz.map((g) => new THREE.Vector3(...this.toScene(g[0], g[1], g[2])));
    this.glomSize = new Float32Array(G).fill(1);
    this.glomTarget = new Float32Array(G);
    this.glomCurrent = new Float32Array(G);
    for (let g = 0; g < G; g++) this.glom.setColorAt(g, new THREE.Color(0.25, 0.14, 0.06));
    this.placeGlomeruli();
    this.group.add(this.glom);
  }

  /** Put this fly's whole brain around its circuit: one dot per real neuron at its measured soma. */
  addCloud(cloud: Cloud, lod = 1): PointCloud {
    this.cloud = new PointCloud(cloud, this.toScene, lod);
    this.cloudDot = Int32Array.from(cloud.meta.circuit_dot, (d) => (d >= 0 && d % lod === 0 ? d : -1));
    this.group.add(this.cloud.points);
    return this.cloud;
  }

  private placeGlomeruli() {
    const m = new THREE.Matrix4();
    const s = new THREE.Vector3();
    const q = new THREE.Quaternion();
    this.glomPos.forEach((p, g) => {
      const r = GLOM_RADIUS * this.glomSize[g];
      this.glom.setMatrixAt(g, m.compose(p, q, s.set(r, r, r)));
    });
    this.glom.instanceMatrix.needsUpdate = true;
  }

  private set(neuron: number, value: number, delay: number) {
    this.target[neuron] = value;
    this.start[neuron] = this.now() + delay;
  }

  private train(neuron: number, rate: number, delay: number, duration: number, strength = 1) {
    const t = this.now() + delay;
    if (rate <= 0) return this.trains.delete(neuron);
    this.trains.set(neuron, { rate, next: t + Math.random() / rate, end: t + duration, strength });
  }

  private spike(neuron: number, t: number, strength: number) {
    this.actData[4 * neuron + 1] = t;
    this.actData[4 * neuron + 2] = strength;
    const band = this.band[neuron];
    if (band) this.onSpike({ band, row: this.bandRow[neuron], t, neuron, color: band === "DA" && this.ppl1.includes(neuron) ? "#ff5c75" : undefined });
    const dot = this.cloudDot?.[neuron] ?? -1;
    if (dot >= 0 && this.cloud && band === "KC") this.cloud.flash(dot, t, new THREE.Color(0.55, 0.95, 1.0));
  }

  /** Play one "smell": glomeruli -> PN spike trains -> the sparse KCs (APL) -> output neurons, over ~3 s. */
  smell(pn: Float64Array, activeKcs: Int32Array, scores: Float64Array, duration = 3.2) {
    const maxPn = Math.max(...pn, 1e-9);
    this.flicker.clear();
    this.trains.clear();
    for (let k = 0; k < this.target.length; k++) this.set(k, 0, 0);
    pn.forEach((rate, g) => {
      const r = rate / maxPn;
      this.glomTarget[g] = r;
      for (const k of this.pnByGlom[g]) {
        this.set(k, 0.7 * r, 0.15);
        this.train(k, 1 + 11 * r * r, 0.1, duration, 0.4 + 0.6 * r);
      }
    });
    for (const j of activeKcs) {
      const k = this.kcNeuron[j];
      this.set(k, 0.8, 0.75 + Math.random() * 0.35);
      this.train(k, 3 + 3 * Math.random(), 0.6 + Math.random() * 0.3, duration - 0.6);
      this.flicker.add(k);
    }
    this.firingKcs = activeKcs.length;
    if (this.apl >= 0) this.set(this.apl, 0.55, 0.7);
    const lo = Math.min(...scores);
    const hi = Math.max(...scores);
    scores.forEach((s, c) => {
      const k = this.displayMbons[c];
      if (k === undefined) return;
      const x = Math.pow((s - lo) / (hi - lo || 1), 3);
      this.set(k, x, 1.45);
      this.train(k, 1 + 14 * x, 1.2, duration - 1.2, 0.5 + 0.5 * x);
    });
  }

  /** Sugar lights the reward (PAM) cluster, a shock the punishment (PPL1) cluster: a burst of dopamine spikes. */
  dopamine(kind: "PAM" | "PPL1", strength = 1) {
    const cluster = kind === "PAM" ? this.pam : this.ppl1;
    for (const k of cluster) {
      this.set(k, 0.6 * strength, Math.random() * 0.1);
      this.train(k, 12 + 8 * Math.random(), Math.random() * 0.05, 0.35, strength);
    }
    setTimeout(() => cluster.forEach((k) => this.set(k, 0, 0)), 900);
  }

  /** Evolution view: tint each glomerulus by the receptor it expresses and size it by its sensory-neuron count. */
  setGlomerulusStyle(tints: THREE.Color[] | null, sizes: ArrayLike<number> | null) {
    this.glomTint = tints;
    for (let g = 0; g < this.glomSize.length; g++) this.glomSize[g] = sizes ? Math.sqrt(sizes[g]) : 1;
    this.placeGlomeruli();
  }

  setShowResting(show: boolean) {
    this.material.uniforms.uShowResting.value = show ? 1 : 0;
  }

  update(t: number, rate: number) {
    // spike trains
    for (const [k, tr] of this.trains) {
      while (tr.next <= t) {
        this.spike(k, tr.next, tr.strength);
        tr.next += (0.6 + 0.8 * Math.random()) / tr.rate;
      }
      if (tr.next > tr.end) this.trains.delete(k);
    }
    // sparse spontaneous activity keeps the circuit alive between smells
    if (!REDUCED_MOTION && t > this.nextBackground && this.group.visible) {
      const k = this.kcNeuron[(Math.random() * this.kcNeuron.length) | 0];
      this.spike(k, t, 0.35);
      this.nextBackground = t + 0.08 + Math.random() * 0.12;
    }
    for (let k = 0; k < this.target.length; k++) {
      if (t < this.start[k]) continue;
      let goal = this.target[k];
      if (goal > 0 && !REDUCED_MOTION && this.flicker.has(k)) goal *= 0.72 + 0.28 * Math.sin(t * 9 + k * 1.7);
      this.current[k] += (goal - this.current[k]) * rate;
      this.actData[4 * k] = this.current[k];
    }
    this.actTex.needsUpdate = true;
    this.material.uniforms.uTime.value = t;
    this.cloud?.update(t);
    const col = new THREE.Color();
    for (let g = 0; g < this.glomTarget.length; g++) {
      this.glomCurrent[g] += (this.glomTarget[g] - this.glomCurrent[g]) * rate;
      const a = this.glomCurrent[g];
      if (this.glomTint) col.copy(this.glomTint[g]).multiplyScalar(0.45 + 0.55 * a);
      else col.setRGB(0.25 + 0.75 * a, 0.14 + 0.6 * a, 0.06 + 0.25 * a);
      this.glom.setColorAt(g, col);
    }
    this.glom.instanceColor!.needsUpdate = true;
  }
}

/** Renderer, camera, bloom, and any number of brains laid out side by side. */
export class FlyScene {
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;
  readonly brains: Brain[] = [];
  readonly camera = new THREE.PerspectiveCamera(38, 1, 1, 12000);
  onHoverGlomerulus: (brain: Brain | null, g: number | null, x: number, y: number) => void = () => {};
  onFrame: (t: number) => void = () => {};
  private scene = new THREE.Scene();
  private composer: EffectComposer;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2(2, 2);
  private t0 = performance.now();
  private lastMs = performance.now();
  private tNow = 0;
  private goalTarget = new THREE.Vector3();
  private goalDistance = 1;
  private easeUntil = 0; // camera glides to a new framing only briefly, so it never fights the user's zoom
  private easeRate = 3;
  private narrow = false;
  private portrait = false;

  constructor(private container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x03040a, 1);
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotate = !REDUCED_MOTION;
    this.controls.autoRotateSpeed = 0.35;
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(256, 256), 0.8, 0.35, 0.45));
    this.composer.addPass(new OutputPass());

    this.resize();
    new ResizeObserver(() => this.resize()).observe(container);
    this.renderer.domElement.addEventListener("pointermove", (e) => {
      const r = this.renderer.domElement.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.hover(e.clientX, e.clientY);
    });
    this.renderer.domElement.addEventListener("pointerleave", () => this.onHoverGlomerulus(null, null, 0, 0));
    this.renderer.setAnimationLoop(() => this.frame());
  }

  readonly now = () => this.tNow;

  add(brain: Brain, x: number) {
    brain.group.position.x = x;
    this.brains.push(brain);
    this.scene.add(brain.group);
  }

  private distanceFor(span: number): number {
    const aspect = this.container.clientWidth / Math.max(this.container.clientHeight, 1);
    return span * 3.9 * (this.portrait ? Math.min(2.2, 1.1 / aspect) : 1);
  }

  /** Ease the camera to frame these brains (all brains stay in the scene). */
  focus(brains: Brain[], instant = false, rate = 3) {
    const box = new THREE.Box3();
    for (const b of brains) {
      box.expandByPoint(b.group.position.clone().addScalar(-b.radius));
      box.expandByPoint(b.group.position.clone().addScalar(b.radius));
    }
    this.goalTarget.copy(box.getCenter(new THREE.Vector3())).setY(0).setZ(0);
    const span = box.getSize(new THREE.Vector3()).x / 2;
    this.goalDistance = this.distanceFor(span);
    if (instant) {
      this.controls.target.copy(this.goalTarget);
      const dir = new THREE.Vector3(-0.15, 0.1, 1).normalize();
      this.camera.position.copy(this.goalTarget).addScaledVector(dir, this.goalDistance);
    }
    this.controls.minDistance = span * 0.3;
    this.controls.maxDistance = span * 14;
    this.easeRate = rate;
    this.easeUntil = this.tNow + 6 / rate;
  }

  /** Start wide on the whole brain (its point cloud), then glide in onto the learning circuit. */
  intro(brain: Brain) {
    const cloudBox = brain.cloud?.box;
    if (!cloudBox || REDUCED_MOTION) return this.focus([brain], true);
    const span = cloudBox.getSize(new THREE.Vector3()).length() / 2;
    this.controls.target.copy(cloudBox.getCenter(new THREE.Vector3()));
    this.camera.position.copy(this.controls.target).addScaledVector(new THREE.Vector3(-0.35, 0.3, 1).normalize(), span * 3.4);
    setTimeout(() => this.focus([brain], false, 0.75), 1400);
  }

  /** Pixels per micron at the orbit target (for the scale bar). */
  pixelsPerMicron(): number {
    const t = this.controls.target.clone();
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0).normalize();
    const a = t.clone().project(this.camera);
    const b = t.clone().addScaledVector(right, 100).project(this.camera);
    return (Math.abs(b.x - a.x) * this.container.clientWidth) / 2 / 100;
  }

  private resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.narrow = w <= 820;
    this.portrait = w / h < 1;
    // the control panel covers part of the stage: bottom half on phones, a left column on desktop
    if (this.narrow) this.camera.setViewOffset(w, h, 0, h * 0.2, w, h);
    else this.camera.setViewOffset(w, h, -200, 0, w, h);
    this.camera.updateProjectionMatrix();
  }

  private hover(x: number, y: number) {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    for (const b of this.brains) {
      if (!b.group.visible) continue;
      const hit = this.raycaster.intersectObject(b.glom, false)[0];
      if (hit?.instanceId !== undefined) return this.onHoverGlomerulus(b, hit.instanceId, x, y);
    }
    this.onHoverGlomerulus(null, null, x, y);
  }

  private frame() {
    const ms = performance.now();
    const dt = Math.min((ms - this.lastMs) / 1000, 0.05);
    this.lastMs = ms;
    const t = (this.tNow = (ms - this.t0) / 1000);
    const rate = 1 - Math.exp(-dt * 7);
    for (const b of this.brains) if (b.group.visible) b.update(t, rate);
    if (t < this.easeUntil) {
      const ease = 1 - Math.exp(-dt * this.easeRate);
      const offset = this.camera.position.clone().sub(this.controls.target);
      this.controls.target.lerp(this.goalTarget, ease);
      offset.setLength(offset.length() + (this.goalDistance - offset.length()) * ease);
      this.camera.position.copy(this.controls.target).add(offset);
    }
    this.controls.update();
    this.onFrame(t);
    this.composer.render();
  }
}
