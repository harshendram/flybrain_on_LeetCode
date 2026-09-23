// The fly's world: a dark arena ringed by 14 glowing feeders, one per algorithm technique, an odour plume that
// carries a problem's smell, and the little effects (sugar sparkles, shock zaps) that teach the fly.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

export const ARENA_R = 230; // feeder ring radius
export const PERCH_Y = 26; // top of the centre pedestal

/** A name tag that stays the same size on screen however far away its feeder is. */
function labelSprite(text: string, color: string): THREE.Sprite {
  const c = document.createElement("canvas");
  const s = 2;
  const ctx = c.getContext("2d")!;
  const font = `600 ${20 * s}px 'IBM Plex Sans', system-ui, sans-serif`;
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 26 * s;
  c.width = w;
  c.height = 36 * s;
  ctx.font = font;
  ctx.fillStyle = "rgba(6,9,20,0.78)";
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * s;
  ctx.beginPath();
  ctx.roundRect(s, s, w - 2 * s, 34 * s, 10 * s);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#e8ecf8";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 13 * s, 18.5 * s);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, transparent: true, sizeAttenuation: false }));
  sprite.userData.aspect = w / c.height;
  sprite.center.set(0.5, 0);
  return sprite;
}

function floorTexture(): THREE.CanvasTexture {
  const size = 1024;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "#0b1022");
  g.addColorStop(1, "#03040a");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = "rgba(124,196,255,0.10)";
  for (let r = 40; r < size / 2; r += 40) {
    ctx.lineWidth = r % 200 === 0 ? 2 : 1;
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  for (let a = 0; a < 72; a++) {
    const t = (a / 72) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(size / 2 + Math.cos(t) * 60, size / 2 + Math.sin(t) * 60);
    ctx.lineTo(size / 2 + Math.cos(t) * (size / 2), size / 2 + Math.sin(t) * (size / 2));
    ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

const STAND_H = 22;
const DISH_R = 24; // room for the fly to stand beside the drop
const DROP_R = 7;

/** A feeding station: a pedestal with a flat dish and a glowing sugar drop in the middle. */
export class Feeder {
  readonly group = new THREE.Group();
  readonly surface = STAND_H + 0.1; // the dish floor the fly stands on
  readonly label: THREE.Sprite;
  private drop: THREE.Mesh;
  private rim: THREE.Mesh;
  private halo: THREE.Mesh;
  private glow = 0.6;
  private target = 0.6;

  constructor(readonly index: number, readonly name: string, readonly color: THREE.Color, angle: number) {
    this.group.position.set(Math.cos(angle) * ARENA_R, 0, Math.sin(angle) * ARENA_R);
    const metal = new THREE.MeshStandardMaterial({ color: 0x141a2e, roughness: 0.38, metalness: 0.65 });
    const stand = new THREE.Mesh(new THREE.CylinderGeometry(DISH_R, DISH_R + 7, STAND_H, 56), metal);
    stand.position.y = STAND_H / 2;
    stand.receiveShadow = true;
    const plate = new THREE.Mesh(
      new THREE.CircleGeometry(DISH_R - 1.2, 56),
      new THREE.MeshStandardMaterial({ color: 0x0d1224, roughness: 0.25, metalness: 0.3, emissive: color, emissiveIntensity: 0.05 }),
    );
    plate.rotation.x = -Math.PI / 2;
    plate.position.y = this.surface;
    plate.receiveShadow = true;
    this.rim = new THREE.Mesh(
      new THREE.TorusGeometry(DISH_R - 0.6, 0.8, 10, 96),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.8 }),
    );
    this.rim.rotation.x = -Math.PI / 2;
    this.rim.position.y = STAND_H + 0.3;
    this.drop = new THREE.Mesh(
      new THREE.SphereGeometry(DROP_R, 40, 20, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshPhysicalMaterial({ color, emissive: color, emissiveIntensity: 0.6, roughness: 0.05, clearcoat: 1 }),
    );
    this.drop.position.y = this.surface;
    this.drop.scale.y = 0.6;
    this.halo = new THREE.Mesh(
      new THREE.RingGeometry(DISH_R + 9, DISH_R + 17, 64),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthWrite: false }),
    );
    this.halo.rotation.x = -Math.PI / 2;
    this.halo.position.y = 0.3;
    this.label = labelSprite(name, `#${color.getHexString()}`);
    this.label.position.y = STAND_H + 22;
    this.group.add(stand, plate, this.rim, this.drop, this.halo, this.label);
  }

  /** Where a fly arriving from `from` stands to drink: on the dish, beside the drop, facing it. */
  landing(from: THREE.Vector3, footDrop: number, reach = DROP_R + 7.5): THREE.Vector3 {
    const d = from.clone().sub(this.group.position).setY(0).normalize();
    return this.group.position.clone().addScaledVector(d, reach).setY(this.surface + footDrop);
  }

  /** The drop, for sparkles. */
  get top(): THREE.Vector3 {
    return this.group.position.clone().setY(this.surface + DROP_R * 0.6);
  }

  highlight(on: boolean) {
    this.target = on ? 1.6 : 0.6;
  }

  /** Label height in pixels -> sprite scale (sizeAttenuation off: scale is a fraction of the view height). */
  setLabelSize(px: number, viewHeight: number, fov: number) {
    const s = (px / (viewHeight / 2)) * Math.tan(THREE.MathUtils.degToRad(fov) / 2);
    this.label.scale.set(s * this.label.userData.aspect, s, 1);
  }

  update(t: number, dt: number) {
    this.glow += (this.target - this.glow) * (1 - Math.exp(-dt * 5));
    const pulse = 0.92 + 0.08 * Math.sin(t * 3 + this.index);
    (this.drop.material as THREE.MeshPhysicalMaterial).emissiveIntensity = this.glow * pulse;
    (this.rim.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.5 + 0.9 * this.glow * pulse;
    (this.halo.material as THREE.MeshBasicMaterial).opacity = 0.14 + 0.18 * this.glow;
  }
}

/** A cloud of odour particles carrying the problem's smell, coloured by the receptors it drives. */
export class Plume {
  readonly points: THREE.Points;
  private pos: Float32Array;
  private vel: Float32Array;
  private age: Float32Array;
  private col: Float32Array;
  private alpha: Float32Array;
  private n: number;
  private next = 0;
  private emitting = 0;
  private palette: THREE.Color[] = [new THREE.Color(1, 1, 1)];
  private weights: number[] = [1];

  constructor(n = 2600) {
    this.n = n;
    this.pos = new Float32Array(3 * n);
    this.vel = new Float32Array(3 * n);
    this.age = new Float32Array(n).fill(99);
    this.col = new Float32Array(3 * n);
    this.alpha = new Float32Array(n);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(this.col, 3));
    geom.setAttribute("alpha", new THREE.BufferAttribute(this.alpha, 1));
    const mat = new THREE.ShaderMaterial({
      uniforms: { uScale: { value: window.innerHeight / 2 } },
      vertexShader: `attribute float alpha; attribute vec3 color; varying float vA; varying vec3 vC; uniform float uScale;
        void main() { vA = alpha; vC = color; vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = min((3.0 + 5.0 * alpha) * uScale / -mv.z, 14.0); gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `varying float vA; varying vec3 vC;
        void main() { vec2 p = gl_PointCoord * 2.0 - 1.0; float d = dot(p, p); if (d > 1.0) discard;
          gl_FragColor = vec4(vC, vA * (1.0 - d)); }`,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(geom, mat);
  }

  /** Start releasing a smell: colours sampled in proportion to receptor activity. */
  release(colors: THREE.Color[], weights: number[], seconds = 2.2) {
    this.palette = colors;
    this.weights = weights;
    this.emitting = seconds;
  }

  private sampleColor(): THREE.Color {
    const total = this.weights.reduce((a, b) => a + b, 0) || 1;
    let r = Math.random() * total;
    for (let i = 0; i < this.weights.length; i++) if ((r -= this.weights[i]) <= 0) return this.palette[i];
    return this.palette[this.palette.length - 1];
  }

  update(t: number, dt: number) {
    if (this.emitting > 0) {
      this.emitting -= dt;
      let k = Math.floor(dt * 650);
      while (k-- > 0) {
        const i = this.next++ % this.n;
        const a = Math.random() * Math.PI * 2;
        const r = Math.random() * 10;
        this.pos.set([Math.cos(a) * r, PERCH_Y + 18 + Math.random() * 10, Math.sin(a) * r], 3 * i);
        const sp = 18 + Math.random() * 40;
        this.vel.set([Math.cos(a) * sp, 6 + Math.random() * 14, Math.sin(a) * sp], 3 * i);
        this.age[i] = 0;
        this.col.set(this.sampleColor().toArray(), 3 * i);
      }
    }
    for (let i = 0; i < this.n; i++) {
      if (this.age[i] > 6) {
        this.alpha[i] = 0;
        continue;
      }
      this.age[i] += dt;
      const x = this.pos[3 * i];
      const z = this.pos[3 * i + 2];
      // gentle swirl + turbulence
      this.vel[3 * i] += (Math.sin(z * 0.03 + t) * 6 - x * 0.02) * dt;
      this.vel[3 * i + 2] += (Math.cos(x * 0.03 + t * 1.3) * 6 - z * 0.02) * dt;
      this.vel[3 * i + 1] -= 1.5 * dt;
      for (let a = 0; a < 3; a++) this.pos[3 * i + a] += this.vel[3 * i + a] * dt * 0.8;
      const age = this.age[i];
      this.alpha[i] = Math.min(1, age * 3) * Math.max(0, 1 - age / 6) * 0.7;
    }
    const g = this.points.geometry;
    g.attributes.position.needsUpdate = true;
    g.attributes.color.needsUpdate = true;
    g.attributes.alpha.needsUpdate = true;
  }
}

/** Short-lived sparkles: gold sugar rising from a feeder, or a blue shock crackle. */
export class Sparks {
  readonly points: THREE.Points;
  private pos = new Float32Array(3 * 400);
  private vel = new Float32Array(3 * 400);
  private age = new Float32Array(400).fill(9);
  private alpha = new Float32Array(400);
  private next = 0;
  private color = new THREE.Color();

  constructor() {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    geom.setAttribute("alpha", new THREE.BufferAttribute(this.alpha, 1));
    const mat = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: this.color }, uScale: { value: window.innerHeight / 2 } },
      vertexShader: `attribute float alpha; varying float vA; uniform float uScale;
        void main() { vA = alpha; vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = min(5.0 * uScale / -mv.z, 16.0); gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `varying float vA; uniform vec3 uColor;
        void main() { vec2 p = gl_PointCoord * 2.0 - 1.0; float d = dot(p, p); if (d > 1.0) discard;
          gl_FragColor = vec4(uColor * 1.6, vA * (1.0 - d)); }`,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(geom, mat);
  }

  burst(at: THREE.Vector3, color: THREE.Color, n: number, spread: number, rise: number) {
    this.color.copy(color);
    for (let k = 0; k < n; k++) {
      const i = this.next++ % 400;
      this.pos.set([at.x + (Math.random() - 0.5) * spread, at.y + Math.random() * 4, at.z + (Math.random() - 0.5) * spread], 3 * i);
      this.vel.set([(Math.random() - 0.5) * 20, rise * (0.4 + Math.random()), (Math.random() - 0.5) * 20], 3 * i);
      this.age[i] = 0;
    }
  }

  update(dt: number) {
    for (let i = 0; i < 400; i++) {
      this.age[i] += dt;
      if (this.age[i] > 1.6) {
        this.alpha[i] = 0;
        continue;
      }
      for (let a = 0; a < 3; a++) this.pos[3 * i + a] += this.vel[3 * i + a] * dt;
      this.alpha[i] = Math.max(0, 1 - this.age[i] / 1.6);
    }
    this.points.geometry.attributes.position.needsUpdate = true;
    this.points.geometry.attributes.alpha.needsUpdate = true;
  }
}

export class Arena {
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(42, 1, 1, 6000);
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;
  readonly feeders: Feeder[];
  readonly plume = new Plume();
  readonly sparks = new Sparks();
  private composer: EffectComposer;

  constructor(container: HTMLElement, names: string[], colors: string[]) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    container.appendChild(this.renderer.domElement);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.45; // a white studio room: reflections only, not the key light
    this.scene.background = new THREE.Color(0x03040a);
    this.scene.fog = new THREE.Fog(0x03040a, 520, 1300);

    const key = new THREE.DirectionalLight(0xfff4e6, 1.7);
    key.position.set(160, 320, 120);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = key.shadow.camera.bottom = -320;
    key.shadow.camera.right = key.shadow.camera.top = 320;
    const rim = new THREE.DirectionalLight(0x7cc4ff, 1.4);
    rim.position.set(-200, 120, -260);
    this.scene.add(key, rim, new THREE.HemisphereLight(0x8fa6ff, 0x05060c, 0.5));

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(420, 96),
      new THREE.MeshStandardMaterial({ map: floorTexture(), roughness: 0.85, metalness: 0.1 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    const perch = new THREE.Mesh(
      new THREE.CylinderGeometry(16, 22, PERCH_Y, 40),
      new THREE.MeshStandardMaterial({ color: 0x1a2140, roughness: 0.35, metalness: 0.7, emissive: 0x0b1a33 }),
    );
    perch.position.y = PERCH_Y / 2;
    perch.receiveShadow = true;
    this.scene.add(floor, perch, this.plume.points, this.sparks.points);

    this.feeders = names.map((n, i) => {
      const f = new Feeder(i, n, new THREE.Color(colors[i]), (i / names.length) * Math.PI * 2 - Math.PI / 2);
      this.scene.add(f.group);
      return f;
    });

    this.camera.position.set(0, 330, 470);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 30, 0);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.47;
    this.controls.minDistance = 40;
    this.controls.maxDistance = 1200;

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(256, 256), 0.6, 0.5, 0.9));
    this.composer.addPass(new OutputPass());
    const resize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      this.renderer.setSize(w, h);
      this.composer.setSize(w, h);
      this.camera.aspect = w / h;
      if (w > 820) this.camera.setViewOffset(w, h, -190, 0, w, h);
      else this.camera.setViewOffset(w, h, 0, h * 0.24, w, h); // keep the subject above the bottom sheet
      this.camera.updateProjectionMatrix();
      for (const f of this.feeders) f.setLabelSize(w > 820 ? 24 : 20, h, this.camera.fov);
    };
    new ResizeObserver(resize).observe(container);
    resize();
  }

  render(t: number, dt: number) {
    this.controls.update();
    this.camera.updateMatrixWorld();
    const v = new THREE.Vector3();
    for (const f of this.feeders) {
      f.update(t, dt);
      // labels fade out before they slide under the page header
      v.setFromMatrixPosition(f.label.matrixWorld).project(this.camera);
      const fromTop = (1 - v.y) / 2;
      (f.label.material as THREE.SpriteMaterial).opacity = v.z < 1 ? THREE.MathUtils.smoothstep(fromTop, 0.1, 0.18) : 0;
    }
    this.plume.update(t, dt);
    this.sparks.update(dt);
    this.composer.render();
  }
}
