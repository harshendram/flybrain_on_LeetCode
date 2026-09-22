import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import type { Skeletons } from "./data";

const ROLES = { KC: 0, PN: 1, MBON: 2, PAM: 3, PPL1: 4, APL: 5 } as const;
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

const TEX = 64; // activity texture: TEX x TEX texels, one per neuron (<= 4,096 neurons per brain)
const GLOM_RADIUS = 2.6;
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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

/** One fly's mushroom-body circuit: real neuron skeletons, glomeruli, and their animated activity. */
export class Brain {
  readonly group = new THREE.Group();
  readonly radius: number;
  readonly kcNeuron: Int32Array; // model KC index -> neuron index
  readonly pnByGlom: number[][];
  readonly displayMbons: number[]; // one real MBON drawn per technique
  readonly glom: THREE.InstancedMesh;
  private material: THREE.ShaderMaterial;
  private actData: Float32Array;
  private actTex: THREE.DataTexture;
  private target: Float32Array;
  private current: Float32Array;
  private start: Float32Array;
  private flicker = new Set<number>();
  private glomTarget: Float32Array;
  private glomCurrent: Float32Array;
  private glomTint: THREE.Color[] | null = null; // evolution view: colour by assigned receptor
  private glomSize: Float32Array;
  private glomPos: THREE.Vector3[];
  private pam: number[] = [];
  private ppl1: number[] = [];
  private apl = -1;

  constructor(sk: Skeletons, nTechniques: number, private now: () => number) {
    const { meta, positions, parents } = sk;
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
    this.group.add(new THREE.LineSegments(geom, this.material));

    const G = meta.glomeruli.length;
    this.glom = new THREE.InstancedMesh(
      new THREE.SphereGeometry(1, 18, 12),
      new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }),
      G,
    );
    this.glomPos = meta.glomerulus_xyz.map((g) => new THREE.Vector3(...toScene(g[0], g[1], g[2])));
    this.glomSize = new Float32Array(G).fill(1);
    this.glomTarget = new Float32Array(G);
    this.glomCurrent = new Float32Array(G);
    for (let g = 0; g < G; g++) this.glom.setColorAt(g, new THREE.Color(0.25, 0.14, 0.06));
    this.placeGlomeruli();
    this.group.add(this.glom);
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
    scores.forEach((s, c) => {
      const k = this.displayMbons[c];
      if (k !== undefined) this.set(k, Math.pow((s - lo) / (hi - lo || 1), 3), 1.45);
    });
  }

  /** Sugar lights the reward (PAM) cluster, a shock the punishment (PPL1) cluster. */
  dopamine(kind: "PAM" | "PPL1") {
    const cluster = kind === "PAM" ? this.pam : this.ppl1;
    for (const k of cluster) this.set(k, 1, Math.random() * 0.15);
    setTimeout(() => cluster.forEach((k) => this.set(k, 0, 0)), 1100);
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
    for (let k = 0; k < this.target.length; k++) {
      if (t < this.start[k]) continue;
      let goal = this.target[k];
      if (goal > 0 && !REDUCED_MOTION && this.flicker.has(k)) goal *= 0.72 + 0.28 * Math.sin(t * 9 + k * 1.7);
      this.current[k] += (goal - this.current[k]) * rate;
      this.actData[k] = this.current[k];
    }
    this.actTex.needsUpdate = true;
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
  onHoverGlomerulus: (brain: Brain | null, g: number | null, x: number, y: number) => void = () => {};
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(38, 1, 1, 8000);
  private composer: EffectComposer;
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2(2, 2);
  private t0 = performance.now();
  private lastMs = performance.now();
  private tNow = 0;
  private goalTarget = new THREE.Vector3();
  private goalDistance = 1;
  private easeUntil = 0; // camera glides to a new framing only briefly, so it never fights the user's zoom
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
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(256, 256), 0.7, 0.3, 0.5));
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

  /** Ease the camera to frame these brains (all brains stay in the scene). */
  focus(brains: Brain[], instant = false) {
    const box = new THREE.Box3();
    for (const b of brains) {
      box.expandByPoint(b.group.position.clone().addScalar(-b.radius));
      box.expandByPoint(b.group.position.clone().addScalar(b.radius));
    }
    this.goalTarget.copy(box.getCenter(new THREE.Vector3())).setY(0).setZ(0);
    const span = box.getSize(new THREE.Vector3()).x / 2;
    const aspect = this.container.clientWidth / Math.max(this.container.clientHeight, 1);
    this.goalDistance = span * 3.2 * (this.portrait ? Math.min(2.2, 1.1 / aspect) : 1);
    if (instant || this.camera.position.lengthSq() === 0) {
      this.controls.target.copy(this.goalTarget);
      const dir = new THREE.Vector3(-0.15, 0.1, 1).normalize();
      this.camera.position.copy(this.goalTarget).addScaledVector(dir, this.goalDistance);
    }
    this.easeUntil = this.tNow + 1.8;
    this.controls.minDistance = span * 0.4;
    this.controls.maxDistance = span * 6;
  }

  private resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.narrow = w <= 820;
    this.portrait = w / h < 1;
    // on phones the panel covers the bottom half, so draw the brains in the top half
    if (this.narrow) this.camera.setViewOffset(w, h, 0, h * 0.2, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
  }

  private hover(x: number, y: number) {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    for (const b of this.brains) {
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
    for (const b of this.brains) b.update(t, rate);
    if (t < this.easeUntil) {
      const ease = 1 - Math.exp(-dt * 3);
      const offset = this.camera.position.clone().sub(this.controls.target);
      this.controls.target.lerp(this.goalTarget, ease);
      offset.setLength(offset.length() + (this.goalDistance - offset.length()) * ease);
      this.camera.position.copy(this.controls.target).add(offset);
    }
    this.controls.update();
    this.composer.render();
  }
}
