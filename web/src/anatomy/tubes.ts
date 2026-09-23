// Lit tubes for the selected region's real neurons. Each skeleton edge is a camera-facing quad
// with cylinder shading. Spikes travel by cable distance, the same idea as the learning-brain page.
// Precomputed skeletons have no per-node radius, so width is a micrometre-scale stand-in, thicker near the soma.
import * as THREE from "three";
import type { Skeletons } from "../data";

const VERT = /* glsl */ `
  attribute vec3 aStart;
  attribute vec3 aEnd;
  attribute float aSide;
  attribute float aAlong;
  attribute float aDist;
  uniform float uTime;
  varying float vSide;
  varying float vPulse;
  void main() {
    vec3 axis = aEnd - aStart;
    vec3 mid = mix(aStart, aEnd, aAlong);
    vec3 toCam = normalize(cameraPosition - mid);
    vec3 side = normalize(cross(normalize(axis + vec3(1e-4)), toCam));
    float width = mix(1.15, 0.45, clamp(aDist / 400.0, 0.0, 1.0));
    vec3 p = mid + side * aSide * width;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vSide = aSide;
    float dt = uTime - aDist / 280.0;
    vPulse = exp(-dt * dt / 0.004) ;
    gl_Position = projectionMatrix * mv;
  }`;

const FRAG = /* glsl */ `
  uniform vec3 uColor;
  varying float vSide;
  varying float vPulse;
  void main() {
    float ndot = sqrt(max(0.0, 1.0 - vSide * vSide));
    float spec = pow(ndot, 12.0);
    float rim = pow(1.0 - ndot, 2.0);
    vec3 col = uColor * (0.25 + 0.75 * ndot) + vec3(1.0) * spec * 0.45 + uColor * rim * 0.35;
    col += vec3(0.7, 0.95, 1.0) * vPulse;
    gl_FragColor = vec4(col, 0.92);
  }`;

export class Tubes {
  readonly mesh: THREE.Mesh;
  private material: THREE.ShaderMaterial;

  constructor() {
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: { uTime: { value: 0 }, uColor: { value: new THREE.Color("#d7f7ff") } },
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), this.material);
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
  }

  setTime(t: number) {
    this.material.uniforms.uTime.value = t;
  }

  setColor(hex: string) {
    this.material.uniforms.uColor.value.set(hex);
  }

  /** Show the neurons whose manifest index is in `which`. Empty hides the tubes. */
  show(skel: Skeletons, which: number[]) {
    const segs: number[] = [];
    for (const k of which) {
      const n = skel.meta.neurons[k];
      // long optic-lobe cells are tens of thousands of nodes; draw every other edge past a few thousand
      const step = n.count > 6000 ? 2 : 1;
      for (let i = n.offset; i < n.offset + n.count; i += step) {
        const p = skel.parents[i];
        if (p === 65535) continue;
        segs.push(i, n.offset + p);
      }
    }
    this.mesh.visible = segs.length > 0;
    if (!segs.length) return;
    const n = segs.length / 2;
    const start = new Float32Array(n * 4 * 3);
    const end = new Float32Array(n * 4 * 3);
    const side = new Float32Array(n * 4);
    const along = new Float32Array(n * 4);
    const dist = new Float32Array(n * 4);
    const index: number[] = [];
    const corner = [
      [-1, 0],
      [1, 0],
      [1, 1],
      [-1, 1],
    ];
    for (let s = 0; s < n; s++) {
      const a = segs[s * 2];
      const b = segs[s * 2 + 1];
      const da = skel.dist ? skel.dist[a] : 0;
      const db = skel.dist ? skel.dist[b] : 0;
      for (let c = 0; c < 4; c++) {
        const o = (s * 4 + c) * 3;
        start.set(skel.positions.subarray(a * 3, a * 3 + 3), o);
        end.set(skel.positions.subarray(b * 3, b * 3 + 3), o);
        side[s * 4 + c] = corner[c][0];
        along[s * 4 + c] = corner[c][1];
        dist[s * 4 + c] = da + (db - da) * corner[c][1];
      }
      const b0 = s * 4;
      index.push(b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("aStart", new THREE.BufferAttribute(start, 3));
    g.setAttribute("aEnd", new THREE.BufferAttribute(end, 3));
    g.setAttribute("aSide", new THREE.BufferAttribute(side, 1));
    g.setAttribute("aAlong", new THREE.BufferAttribute(along, 1));
    g.setAttribute("aDist", new THREE.BufferAttribute(dist, 1));
    g.setIndex(index);
    this.mesh.geometry.dispose();
    this.mesh.geometry = g;
  }
}
