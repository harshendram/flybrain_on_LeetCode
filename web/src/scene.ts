import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import type { Skeletons } from "./data";

// role index -> [base colour, glow colour, resting brightness]
const ROLES = {
  KC: 0,
  PN: 1,
  MBON: 2,
  PAM: 3,
  PPL1: 4,
  APL: 5,
} as const;
// [base rgb, glow rgb, resting opacity, peak opacity]. Normal alpha blending: a bundle of 1,880 overlapping KC axons
// saturates to its own colour instead of blowing out to white; the bloom pass supplies the glow of firing cells.
const PALETTE: [number, number, number, number, number, number, number, number][] = [
  [0.16, 0.28, 0.7, 0.45, 0.95, 1.0, 0.035, 0.6], // KC
  [0.75, 0.38, 0.12, 1.0, 0.72, 0.35, 0.07, 0.75], // PN
  [0.2, 0.55, 0.3, 0.55, 1.0, 0.65, 0.06, 1.0], // MBON
  [0.6, 0.45, 0.12, 1.0, 0.85, 0.4, 0.025, 0.28], // PAM (reward): ~160 cells with dense arbours
  [0.65, 0.16, 0.22, 1.0, 0.45, 0.5, 0.06, 0.8], // PPL1 (punishment): 8 cells
  [0.4, 0.28, 0.65, 0.8, 0.65, 1.0, 0.04, 0.45], // APL
];
export const ROLE_COLORS = PALETTE.map(([r, g, b]) => `rgb(${r * 255 | 0}, ${g * 255 | 0}, ${b * 255 | 0})`);

const TEX = 64; // activity texture is TEX x TEX texels, one per neuron

const VERT = /* glsl */ `
  attribute float aNid;
  attribute float aRole;
  uniform sampler2D uAct;
  uniform vec3 uBase[6];
  uniform vec3 uGlow[6];
  uniform float uRest[6];
  uniform float uPeak[6];
  uniform float uShowResting;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    int nid = int(aNid + 0.5);
    float a = clamp(texelFetch(uAct, ivec2(nid % ${TEX}, nid / ${TEX}), 0).r, 0.0, 1.0);
    int role = int(aRole + 0.5);
    vColor = mix(uBase[role], uGlow[role], a);
    vAlpha = max(uRest[role] * uShowResting, a * uPeak[role]);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;
const FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() { gl_FragColor = vec4(vColor, vAlpha); }`;

export class FlyScene {
  readonly renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private composer: EffectComposer;
  private material: THREE.ShaderMaterial;
  private actData: Float32Array;
  private actTex: THREE.DataTexture;
  private target: Float32Array;
  private start: Float32Array; // time (s) at which each neuron starts moving to its target
  private current: Float32Array;
  private flicker = new Set<number>();
  private glom: THREE.InstancedMesh;
  private glomTarget: Float32Array;
  private glomCurrent: Float32Array;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2(2, 2);
  private t0 = performance.now();
  private lastMs = performance.now();
  private tNow = 0;
  private reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  readonly kcNeuron: Int32Array; // model KC index -> neuron index
  readonly pnByGlom: number[][];
  readonly displayMbons: number[]; // one real MBON drawn per technique
  private pam: number[] = [];
  private ppl1: number[] = [];
  private apl = -1;
  onHoverGlomerulus: (g: number | null, x: number, y: number) => void = () => {};

  constructor(container: HTMLElement, sk: Skeletons, nTechniques: number) {
    const { meta, positions, parents } = sk;
    const nNeurons = meta.neurons.length;

    // --- centre on the mushroom body + antennal lobe, dorsal up, anterior facing the camera ---
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (const n of meta.neurons) {
      if (n.role !== "KC") continue;
      for (let i = n.offset; i < n.offset + n.count; i += 7) box.expandByPoint(v.fromArray(positions, 3 * i));
    }
    for (const g of meta.glomerulus_xyz) box.expandByPoint(v.fromArray(g));
    const c = box.getCenter(new THREE.Vector3());
    const toScene = (x: number, y: number, z: number): [number, number, number] => [x - c.x, -(y - c.y), -(z - c.z)];

    const nNodes = meta.n_nodes;
    const pos = new Float32Array(3 * nNodes);
    const nid = new Float32Array(nNodes);
    const role = new Float32Array(nNodes);
    const index: number[] = [];
    this.kcNeuron = new Int32Array(meta.neurons.filter((n) => n.role === "KC").length);
    this.pnByGlom = meta.glomeruli.map(() => []);
    const mbonByType = new Map<string, number>();
    meta.neurons.forEach((n, k) => {
      const r =
        n.role === "DAN" ? (n.type.startsWith("PAM") ? ROLES.PAM : ROLES.PPL1) : ROLES[n.role as keyof typeof ROLES];
      if (n.role === "KC") this.kcNeuron[n.kc_index!] = k;
      if (n.role === "PN") this.pnByGlom[n.glom!].push(k);
      if (n.role === "MBON" && !mbonByType.has(n.type)) mbonByType.set(n.type, k);
      if (r === ROLES.PAM) this.pam.push(k);
      if (r === ROLES.PPL1) this.ppl1.push(k);
      if (n.role === "APL") this.apl = k;
      for (let i = n.offset; i < n.offset + n.count; i++) {
        pos.set(toScene(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]), 3 * i);
        nid[i] = k;
        role[i] = r;
        const p = parents[i];
        if (p !== 65535) index.push(i, n.offset + p);
      }
    });
    // Each technique's approach/avoid MBON pair is abstract in the model; we draw it on a distinct real MBON type.
    this.displayMbons = [...mbonByType.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .slice(0, nTechniques)
      .map(([, k]) => k);

    // --- renderer / camera / controls / bloom ---
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x03040a, 1);
    container.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(38, 1, 1, 5000);
    const radius = box.getSize(new THREE.Vector3()).length() / 2;
    this.camera.position.set(-radius * 0.5, radius * 0.35, radius * 3.3);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotate = !this.reducedMotion;
    this.controls.autoRotateSpeed = 0.35;
    this.controls.minDistance = radius * 0.4;
    this.controls.maxDistance = radius * 5;
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(256, 256), 0.7, 0.3, 0.5));
    this.composer.addPass(new OutputPass());

    // --- neurons: one LineSegments, activity looked up per neuron from a float texture ---
    this.actData = new Float32Array(TEX * TEX);
    this.actTex = new THREE.DataTexture(this.actData, TEX, TEX, THREE.RedFormat, THREE.FloatType);
    this.actTex.needsUpdate = true;
    this.target = new Float32Array(nNeurons);
    this.current = new Float32Array(nNeurons);
    this.start = new Float32Array(nNeurons);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("aNid", new THREE.BufferAttribute(nid, 1));
    geom.setAttribute("aRole", new THREE.BufferAttribute(role, 1));
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(index), 1));
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uAct: { value: this.actTex },
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
    this.scene.add(new THREE.LineSegments(geom, this.material));

    // --- glomeruli: glowing beads in the antennal lobe ---
    const G = meta.glomeruli.length;
    this.glom = new THREE.InstancedMesh(
      new THREE.SphereGeometry(2.6, 18, 12),
      new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }),
      G,
    );
    const m = new THREE.Matrix4();
    meta.glomerulus_xyz.forEach((g, i) => {
      m.makeTranslation(...toScene(g[0], g[1], g[2]));
      this.glom.setMatrixAt(i, m);
      this.glom.setColorAt(i, new THREE.Color(0.25, 0.14, 0.06));
    });
    this.glomTarget = new Float32Array(G);
    this.glomCurrent = new Float32Array(G);
    this.scene.add(this.glom);

    // --- events ---
    const baseDistance = this.camera.position.length();
    const onResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      this.renderer.setSize(w, h);
      this.composer.setSize(w, h);
      this.camera.aspect = w / h;
      // the brain is wide: back off in portrait, and on phones draw it in the top half (the panel covers the bottom)
      const portrait = w / h < 1;
      this.camera.position.setLength(baseDistance * (portrait ? Math.min(2.2, 1.1 / (w / h)) : 1));
      if (w <= 820) this.camera.setViewOffset(w, h, 0, h * 0.2, w, h);
      else this.camera.clearViewOffset();
      this.camera.updateProjectionMatrix();
    };
    new ResizeObserver(onResize).observe(container);
    onResize();
    this.renderer.domElement.addEventListener("pointermove", (e) => {
      const r = this.renderer.domElement.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.hover(e.clientX, e.clientY);
    });
    this.renderer.domElement.addEventListener("pointerleave", () => this.onHoverGlomerulus(null, 0, 0));
    this.renderer.setAnimationLoop(() => this.frame());
  }

  private now(): number {
    return this.tNow;
  }

  private set(neuron: number, value: number, delay: number) {
    this.target[neuron] = value;
    this.start[neuron] = this.now() + delay;
  }

  /** Play one "smell": glomeruli -> PNs -> sparse KCs (APL) -> output neurons. */
  smell(pn: Float64Array, activeKcs: Int32Array, scores: Float64Array) {
    const maxPn = Math.max(...pn, 1e-9);
    this.flicker.clear();
    for (let k = 0; k < this.target.length; k++) this.set(k, 0, 0);
    pn.forEach((rate, g) => {
      this.glomTarget[g] = rate / maxPn;
      for (const k of this.pnByGlom[g]) this.set(k, 0.95 * (rate / maxPn), 0.15);
    });
    for (const j of activeKcs) {
      const k = this.kcNeuron[j];
      this.set(k, 1, 0.75 + Math.random() * 0.35);
      this.flicker.add(k);
    }
    if (this.apl >= 0) this.set(this.apl, 0.55, 0.7);
    const lo = Math.min(...scores);
    const hi = Math.max(...scores);
    scores.forEach((s, cIdx) => {
      const k = this.displayMbons[cIdx];
      if (k !== undefined) this.set(k, Math.pow((s - lo) / (hi - lo || 1), 3), 1.45);
    });
  }

  /** Sugar lights the reward (PAM) cluster, a shock the punishment (PPL1) cluster. */
  dopamine(kind: "PAM" | "PPL1") {
    const cluster = kind === "PAM" ? this.pam : this.ppl1;
    for (const k of cluster) this.set(k, 1, Math.random() * 0.15);
    setTimeout(() => cluster.forEach((k) => this.set(k, 0, 0)), 1100);
  }

  setShowResting(show: boolean) {
    this.material.uniforms.uShowResting.value = show ? 1 : 0;
  }

  private hover(x: number, y: number) {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObject(this.glom, false)[0];
    this.onHoverGlomerulus(hit?.instanceId ?? null, x, y);
  }

  private frame() {
    const ms = performance.now();
    const dt = Math.min((ms - this.lastMs) / 1000, 0.05);
    this.lastMs = ms;
    const t = (this.tNow = (ms - this.t0) / 1000);
    const rate = 1 - Math.exp(-dt * 7);
    for (let k = 0; k < this.target.length; k++) {
      if (t < this.start[k]) continue;
      let goal = this.target[k];
      if (goal > 0 && !this.reducedMotion && this.flicker.has(k)) goal *= 0.72 + 0.28 * Math.sin(t * 9 + k * 1.7);
      this.current[k] += (goal - this.current[k]) * rate;
      this.actData[k] = this.current[k];
    }
    this.actTex.needsUpdate = true;
    const col = new THREE.Color();
    for (let g = 0; g < this.glomTarget.length; g++) {
      this.glomCurrent[g] += (this.glomTarget[g] - this.glomCurrent[g]) * rate;
      const a = this.glomCurrent[g];
      this.glom.setColorAt(g, col.setRGB(0.25 + 0.75 * a, 0.14 + 0.6 * a, 0.06 + 0.25 * a));
    }
    this.glom.instanceColor!.needsUpdate = true;
    this.controls.update();
    this.composer.render();
  }
}
