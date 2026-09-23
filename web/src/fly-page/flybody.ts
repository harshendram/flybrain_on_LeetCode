// The fly: flybody (Vaxenburg et al., Nature 2025; Google DeepMind + HHMI Janelia; Apache-2.0), decimated for the
// web by scripts/export_flybody.py. MuJoCo convention: the fly faces +x, its left is +y, up is +z; every hinge joint
// turns its body about the body's own origin, so a pose is just a rotation per body.
//
// Poses come from the model itself: joint angle 0 is flybody's standing posture, and each joint's springref is its
// rest angle: wings folded over the back, proboscis retracted, and legs tucked up (flybody's flight tasks start the
// legs there). The wing frames are mirrored left/right, so both wings take the same angles.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import { fetchBinary, type Progress } from "../data";

interface RigJoint {
  name: string;
  body: string;
  axis: [number, number, number];
  range: [number, number];
  spring: number;
}

interface BodyRig {
  node: THREE.Object3D;
  rest: THREE.Quaternion;
  joints: { name: string; axis: THREE.Vector3 }[];
}

const SIDES = ["left", "right"] as const;
const isLeg = (j: string) => /_T[123]_/.test(j);
const isWing = (j: string) => j.startsWith("wing_");
/** Proboscis out: rostrum swung down, haustellum unfolded; the head dips to the food. */
const DRINK: Record<string, number> = { rostrum: -0.9, haustellum: -0.7, head: 0.2 };
/** Standing: flybody's zero pose is on tiptoe; bend every leg a little toward its tucked (springref) angle. */
const CROUCH = 0.3;
/** Wild-type Drosophila: tan-brown cuticle, brick-red eyes (flybody's MuJoCo colours are display values). */
const PALETTE = { cuticle: "#a8743f", eye: "#a3170f", dark: "#0c0806", vein: "#4a2c16", ocelli: "#2a140a" };

export class FlyBody {
  readonly root = new THREE.Group(); // placement in the arena: position, heading, bank
  /** height of the thorax above the surface the fly stands on, in arena units */
  readonly footDrop: number;
  private model: THREE.Object3D;
  private bodies = new Map<string, BodyRig>();
  private joints = new Map<string, RigJoint>();
  private angles = new Map<string, number>();
  private ghosts: { node: THREE.Object3D; body: BodyRig; lag: number }[] = [];
  /** 0 = standing, 1 = flying (legs tucked, wings beating) */
  flight = 0;
  /** 0 = proboscis in, 1 = extended (drinking) */
  feed = 0;
  private t = 0;

  private constructor(gltfScene: THREE.Object3D, rig: { joints: RigJoint[] }, scale: number) {
    this.model = gltfScene;
    this.model.rotation.x = -Math.PI / 2; // MuJoCo z-up -> three.js y-up (fly faces +x, its left is -z)
    this.model.scale.setScalar(scale);
    this.root.add(this.model);
    for (const j of rig.joints) {
      const node = this.model.getObjectByName(j.body);
      if (!node) continue;
      if (!this.bodies.has(j.body)) this.bodies.set(j.body, { node, rest: node.quaternion.clone(), joints: [] });
      this.bodies.get(j.body)!.joints.push({ name: j.name, axis: new THREE.Vector3(...j.axis).normalize() });
      this.joints.set(j.name, j);
    }
    this.polishMaterials();
    this.makeWingGhosts();
    this.update(0);
    this.root.updateMatrixWorld(true);
    this.footDrop = -new THREE.Box3().setFromObject(this.root, true).min.y;
  }

  static async load(p: Progress, scale: number, base = "./fly/"): Promise<FlyBody> {
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    const [glb, rig] = await Promise.all([fetchBinary("flybody.glb", p, base), fetch(`${base}rig.json`).then((r) => r.json())]);
    const gltf = await loader.parseAsync(glb, base);
    return new FlyBody(gltf.scene, rig, scale);
  }

  /** Waxy tan cuticle, matte black bristles, glossy deep-red compound eyes, iridescent wing membranes. */
  private polishMaterials() {
    this.model.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = true;
      const name = mesh.name.toLowerCase();
      if (name.includes("membrane")) {
        mesh.castShadow = false;
        mesh.material = new THREE.MeshPhysicalMaterial({
          color: new THREE.Color(0.72, 0.8, 0.92),
          transparent: true,
          opacity: 0.16,
          roughness: 0.15,
          iridescence: 1,
          iridescenceIOR: 1.35,
          iridescenceThicknessRange: [220, 560],
          envMapIntensity: 0.6,
          side: THREE.DoubleSide,
          depthWrite: false,
        });
      } else if (name.includes("red")) {
        const eye = new THREE.Color(PALETTE.eye);
        mesh.material = new THREE.MeshPhysicalMaterial({
          color: eye,
          roughness: 0.3,
          clearcoat: 1,
          clearcoatRoughness: 0.1,
          emissive: eye.clone().multiplyScalar(0.25),
        });
      } else if (name.includes("black") || name.includes("bristle")) {
        mesh.material = new THREE.MeshStandardMaterial({ color: PALETTE.dark, roughness: 0.6 });
      } else if (name.includes("wing") || name.includes("claw")) {
        mesh.material = new THREE.MeshStandardMaterial({ color: PALETTE.vein, roughness: 0.5, side: THREE.DoubleSide });
      } else if (name.includes("ocelli")) {
        mesh.material = new THREE.MeshStandardMaterial({ color: PALETTE.ocelli, roughness: 0.35 });
      } else {
        mesh.material = new THREE.MeshPhysicalMaterial({ color: PALETTE.cuticle, roughness: 0.55, clearcoat: 0.3, clearcoatRoughness: 0.4 });
      }
    });
  }

  /** Faint copies of each wing at trailing stroke phases: the blur you see on a real fly's 218 Hz wingbeat. */
  private makeWingGhosts() {
    for (const side of SIDES) {
      const body = this.bodies.get(`wing_${side}`);
      if (!body || !body.node.parent) continue;
      for (let k = 1; k <= 4; k++) {
        const g = body.node.clone(true);
        g.traverse((o) => {
          const m = o as THREE.Mesh;
          if (!m.isMesh) return;
          const mat = (m.material as THREE.Material).clone() as THREE.MeshStandardMaterial;
          mat.transparent = true;
          mat.opacity = (mat.opacity ?? 1) * (0.3 - k * 0.06);
          mat.depthWrite = false;
          m.material = mat;
          m.castShadow = false;
        });
        g.visible = false;
        body.node.parent.add(g);
        this.ghosts.push({ node: g, body, lag: k * 0.07 });
      }
    }
  }

  private set(joint: string, angle: number) {
    const j = this.joints.get(joint);
    if (j) this.angles.set(joint, Math.min(j.range[1], Math.max(j.range[0], angle)));
  }

  private rest(joint: string): number {
    return this.joints.get(joint)?.spring ?? 0;
  }

  private orient(target: THREE.Quaternion, b: BodyRig, angle: (j: string) => number) {
    const q = new THREE.Quaternion();
    target.copy(b.rest);
    for (const j of b.joints) {
      const a = angle(j.name);
      if (a !== 0) target.multiply(q.setFromAxisAngle(j.axis, a));
    }
  }

  /** One wingbeat, in the model's stroke frame: yaw sweeps forward/back, roll lifts, pitch flips at reversal. */
  private stroke(phase: number): Record<"yaw" | "roll" | "pitch", number> {
    const w = 2 * Math.PI * phase;
    return { yaw: 0.05 + 1.15 * Math.sin(w), roll: 0.12 * Math.sin(2 * w), pitch: 0.8 * Math.tanh(2.5 * Math.cos(w)) };
  }

  private wingAngle(joint: string, phase: number): number {
    const kind = joint.split("_")[1] as "yaw" | "roll" | "pitch";
    return this.rest(joint) * (1 - this.flight) + this.stroke(phase)[kind] * this.flight;
  }

  update(dt: number) {
    this.t += dt;
    const f = this.flight;
    const beat = this.t * 17;
    for (const [name] of this.joints) {
      if (isLeg(name)) this.set(name, this.rest(name) * (CROUCH + (1 - CROUCH) * f)); // standing <-> tucked (springref)
      else if (isWing(name)) this.set(name, this.wingAngle(name, beat));
      else this.set(name, this.rest(name) * (1 - this.feed) + (DRINK[name] ?? this.rest(name)) * this.feed);
    }
    // alive: antennae twitch, the head scans a little while it stands
    for (const s of SIDES) this.set(`antenna_${s}`, 0.12 * Math.sin(this.t * 5.3 + (s === "left" ? 0 : 1.7)));
    this.set("head_twist", 0.1 * Math.sin(this.t * 1.3) * (1 - f) * (1 - this.feed));
    for (const b of this.bodies.values()) this.orient(b.node.quaternion, b, (j) => this.angles.get(j) ?? 0);
    // motion blur: ghosts ride the same stroke a little behind in phase
    for (const g of this.ghosts) {
      g.node.visible = f > 0.6;
      if (!g.node.visible) continue;
      g.node.position.copy(g.body.node.position);
      this.orient(g.node.quaternion, g.body, (j) => this.wingAngle(j, beat - g.lag));
    }
  }
}
