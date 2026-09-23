// Translucent neuropil glass. Every region is one draw: per-vertex colour and alpha, so a click
// can fade the rest without 78 transparent passes. Hidden sides are dropped from the index.
import * as THREE from "three";
import { regionInfo } from "./regions";

export interface MeshPart {
  name: string;
  vertexOffset: number;
  vertexCount: number;
  indexOffset: number;
  indexCount: number;
  centroid: number[];
  box: number[][];
  synapses?: number;
  neurons?: number;
  transmitter?: string;
  top_types?: { type: string; synapses: number }[];
}

export interface AnatomyMeta {
  origin: number[];
  scale: number;
  n_vertices: number;
  n_indices: number;
  midline: string[];
  regions: MeshPart[];
}

const VERT = /* glsl */ `
  attribute vec3 aColor;
  attribute float aAlpha;
  varying vec3 vNormal;
  varying vec3 vView;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = normalize(-mv.xyz);
    vColor = aColor;
    vAlpha = aAlpha;
    gl_Position = projectionMatrix * mv;
  }`;

const FRAG = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vView;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    if (vAlpha < 0.01) discard;
    float fres = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), 2.2);
    float a = vAlpha * (0.35 + 0.65 * fres);
    vec3 col = vColor * (0.55 + 0.9 * fres);
    gl_FragColor = vec4(col, clamp(a, 0.0, 0.85));
  }`;

export class Atlas {
  readonly group = new THREE.Group();
  readonly mesh: THREE.Mesh;
  private readonly source: Uint32Array;
  private readonly meta: AnatomyMeta;
  private readonly alpha: Float32Array;
  private readonly alphaAttr: THREE.BufferAttribute;
  private triName: string[] = [];
  private side: "both" | "left" | "right" = "both";

  constructor(meta: AnatomyMeta, bin: ArrayBuffer) {
    this.meta = meta;
    const n = meta.n_vertices;
    const q = new Uint16Array(bin, 0, n * 3);
    const normals = new Int8Array(bin, n * 6, n * 3);
    this.source = new Uint32Array(meta.n_indices);
    new Uint8Array(this.source.buffer).set(new Uint8Array(bin, n * 9, meta.n_indices * 4));
    const pos = new Float32Array(n * 3);
    const nrm = new Float32Array(n * 3);
    const color = new Float32Array(n * 3);
    this.alpha = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < 3; a++) {
        pos[3 * i + a] = q[3 * i + a] / meta.scale + meta.origin[a];
        nrm[3 * i + a] = normals[3 * i + a] / 127;
      }
    }
    const c = new THREE.Color();
    for (const part of meta.regions) {
      const outline = part.name === "outline";
      c.set(outline ? "#9fb4d8" : regionInfo(part.name).color);
      const a = outline ? 0.05 : 0.22;
      for (let i = part.vertexOffset; i < part.vertexOffset + part.vertexCount; i++) {
        color[3 * i] = c.r;
        color[3 * i + 1] = c.g;
        color[3 * i + 2] = c.b;
        this.alpha[i] = a;
      }
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
    geom.setAttribute("aColor", new THREE.BufferAttribute(color, 3));
    this.alphaAttr = new THREE.BufferAttribute(this.alpha, 1);
    geom.setAttribute("aAlpha", this.alphaAttr);
    this.mesh = new THREE.Mesh(
      geom,
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        depthWrite: false,
        side: THREE.FrontSide,
      }),
    );
    this.mesh.frustumCulled = false;
    this.group.add(this.mesh);
    this.rebuild();
  }

  private shown(name: string) {
    if (name === "outline") return true;
    const info = regionInfo(name);
    return this.side === "both" || info.side === "" || info.side === this.side;
  }

  private rebuild() {
    let n = 0;
    for (const part of this.meta.regions) if (this.shown(part.name)) n += part.indexCount;
    const kept = new Uint32Array(n);
    const names: string[] = [];
    let o = 0;
    for (const part of this.meta.regions) {
      if (!this.shown(part.name)) continue;
      kept.set(this.source.subarray(part.indexOffset, part.indexOffset + part.indexCount), o);
      o += part.indexCount;
      const tris = part.indexCount / 3;
      for (let t = 0; t < tris; t++) names.push(part.name);
    }
    this.triName = names;
    this.mesh.geometry.setIndex(new THREE.BufferAttribute(kept, 1));
  }

  setSide(side: "both" | "left" | "right") {
    this.side = side;
    this.rebuild();
  }

  /** Fade every region except `name`. Pass null to restore the overview. */
  focus(name: string | null) {
    for (const part of this.meta.regions) {
      const outline = part.name === "outline";
      const a = outline ? (name ? 0.02 : 0.05) : !name || part.name === name ? (name ? 0.55 : 0.22) : 0.035;
      this.alpha.fill(a, part.vertexOffset, part.vertexOffset + part.vertexCount);
    }
    this.alphaAttr.needsUpdate = true;
  }

  raycast(raycaster: THREE.Raycaster): string | null {
    const hits = raycaster.intersectObject(this.mesh, false);
    for (const hit of hits) {
      const name = this.triName[hit.faceIndex ?? -1];
      if (name && name !== "outline") return name;
    }
    return null;
  }
}
