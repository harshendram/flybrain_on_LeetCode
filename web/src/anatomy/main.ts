import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "../style.css";
import "./anatomy.css";
import { Progress, fetchBinary, loadCloud, type Skeletons } from "../data";
import { PointCloud } from "../scene";
import { scaleBar } from "../hud";
import { Atlas, type AnatomyMeta, type MeshPart } from "./atlas";
import { SMELL_TOUR, SYSTEM_ORDER, regionInfo } from "./regions";
import { Tubes } from "./tubes";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

async function loadNeurons(p: Progress): Promise<Skeletons> {
  const [meta, bin] = await Promise.all([
    fetch("./data/anatomy_neurons.json").then((r) => r.json()),
    fetchBinary("anatomy_neurons.bin", p),
  ]);
  const n = meta.n_nodes as number;
  const q = new Uint16Array(bin, 0, 3 * n);
  const parents = new Uint16Array(bin, 6 * n, n);
  const positions = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) positions[3 * i + a] = q[3 * i + a] / meta.scale + meta.origin[a];
  const dist = meta.dist_scale ? Float32Array.from(new Uint16Array(bin, 8 * n, n), (v) => v / meta.dist_scale) : null;
  return { meta, positions, parents, dist };
}

async function main() {
  const p = new Progress((l, t) => ($("bar-fill").style.width = `${Math.min(100, (100 * l) / Math.max(t, 1))}%`));
  const [meta, bin, skel, cloud] = await Promise.all([
    fetch("./data/anatomy.json").then((r) => r.json() as Promise<AnatomyMeta>),
    fetchBinary("anatomy.bin", p),
    loadNeurons(p),
    loadCloud("flywire", p),
  ]);
  const stage = $("stage");
  const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setClearColor(0x03040a);
  stage.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 1, 8000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const atlas = new Atlas(meta, bin);
  scene.add(atlas.group);
  const tubes = new Tubes();
  scene.add(tubes.mesh);
  const somas = new PointCloud(cloud, (x, y, z) => [x, y, z], 2);
  scene.add(somas.points);

  const box = new THREE.Box3().setFromObject(atlas.group);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(-size * 0.15, size * 0.2, size * 0.85));

  const byName = new Map(meta.regions.map((r) => [r.name, r]));
  const neuronsOf = (name: string) =>
    skel.meta.neurons.map((n, i) => ((n as { regions?: string[] }).regions ?? []).includes(name) ? i : -1).filter((i) => i >= 0);

  let selected: string | null = null;
  let side: "both" | "left" | "right" = "both";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function applySide() {
    atlas.setSide(side);
    // AL_L sits on negative x after the display mirror, so anatomical left is x < 0.
    const x = side === "left" ? [-1e9, 0] : side === "right" ? [0, 1e9] : [-1e9, 1e9];
    somas.setXRange(x[0], x[1]);
  }

  function select(name: string | null, fly = true) {
    selected = name;
    atlas.focus(name);
    const showNeurons = $<HTMLInputElement>("neurons").checked && !!name;
    tubes.show(skel, showNeurons && name ? neuronsOf(name) : []);
    if (name) tubes.setColor(regionInfo(name).color);
    const card = $("card");
    card.hidden = !name;
    document.querySelectorAll(".region-btn").forEach((b) => b.classList.toggle("on", (b as HTMLElement).dataset.name === name));
    if (!name) return;
    const info = regionInfo(name);
    const part = byName.get(name)!;
    $("card-title").textContent = info.title;
    $("card-about").textContent = info.about;
    const syn = part.synapses ? part.synapses.toLocaleString() : "—";
    const neurons = part.neurons ? part.neurons.toLocaleString() : "—";
    $("card-stats").textContent = `${syn} synapses · ${neurons} neurons · ${part.transmitter || "transmitter unknown"}`;
    $("card-types").innerHTML = (part.top_types ?? []).map((t) => `<span class="pill" style="--c:${info.color}">${esc(t.type)}</span>`).join(" ");
    if (fly) {
      const c = new THREE.Vector3(...(part.centroid as [number, number, number]));
      const from = camera.position.clone().sub(controls.target);
      const dest = c.clone().add(from.normalize().multiplyScalar(Math.max(80, from.length() * 0.45)));
      glide(c, dest);
    }
  }

  let glideT = 1;
  const glideFrom = { target: new THREE.Vector3(), cam: new THREE.Vector3() };
  const glideTo = { target: new THREE.Vector3(), cam: new THREE.Vector3() };
  function glide(target: THREE.Vector3, cam: THREE.Vector3) {
    glideFrom.target.copy(controls.target);
    glideFrom.cam.copy(camera.position);
    glideTo.target.copy(target);
    glideTo.cam.copy(cam);
    glideT = 0;
  }

  // list
  const list = $("list");
  const grouped = new Map<string, MeshPart[]>();
  for (const part of meta.regions) {
    if (part.name === "outline") continue;
    const info = regionInfo(part.name);
    const arr = grouped.get(info.system) ?? [];
    arr.push(part);
    grouped.set(info.system, arr);
  }
  for (const system of [...SYSTEM_ORDER, ...grouped.keys()].filter((s, i, a) => grouped.has(s) && a.indexOf(s) === i)) {
    const block = document.createElement("div");
    block.className = "sys";
    block.dataset.system = system;
    block.innerHTML = `<h3>${esc(system)}</h3>` + (grouped.get(system) ?? [])
      .map((part) => {
        const info = regionInfo(part.name);
        return `<button type="button" class="region-btn" data-name="${esc(part.name)}"><span class="swatch" style="background:${info.color}"></span>${esc(info.title)}</button>`;
      })
      .join("");
    list.appendChild(block);
  }
  list.addEventListener("click", (e) => {
    const name = (e.target as HTMLElement).closest<HTMLElement>(".region-btn")?.dataset.name;
    if (name) select(name);
  });
  $("search").addEventListener("input", () => {
    const q = $<HTMLInputElement>("search").value.trim().toLowerCase();
    document.querySelectorAll<HTMLElement>(".region-btn").forEach((b) => {
      b.hidden = q.length > 0 && !b.textContent!.toLowerCase().includes(q) && !b.dataset.name!.toLowerCase().includes(q);
    });
  });
  $("side").addEventListener("click", (e) => {
    const s = (e.target as HTMLElement).dataset.side as typeof side | undefined;
    if (!s) return;
    side = s;
    for (const b of document.querySelectorAll<HTMLButtonElement>("#side button")) b.classList.toggle("on", b.dataset.side === s);
    applySide();
  });
  $("somas").addEventListener("change", () => (somas.points.visible = $<HTMLInputElement>("somas").checked));
  $("neurons").addEventListener("change", () => select(selected, false));

  let tourI = -1;
  const tourLines = [
    "The smell arrives in the antennal lobe. LeetFly's 51 channels enter here.",
    "Projection neurons meet Kenyon cells in the calyx. This is the wiring LeetFly keeps.",
    "The vertical lobe is where dopamine changes an approach into an avoidance.",
    "The lateral horn handles innate smell. LeetFly leaves it alone.",
  ];
  $("tour").addEventListener("click", () => {
    tourI = (tourI + 1) % SMELL_TOUR.length;
    if (side !== "both") {
      side = "both";
      for (const b of document.querySelectorAll<HTMLButtonElement>("#side button")) b.classList.toggle("on", b.dataset.side === "both");
      applySide();
    }
    select(SMELL_TOUR[tourI]);
    $("card-about").textContent = tourLines[tourI];
  });

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2(2, 2);
  const hover = $("hover");
  renderer.domElement.addEventListener("pointermove", (e) => {
    const r = renderer.domElement.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    hover.style.left = `${e.clientX}px`;
    hover.style.top = `${e.clientY}px`;
    raycaster.setFromCamera(pointer, camera);
    const hit = atlas.raycast(raycaster);
    hover.hidden = !hit;
    if (hit) hover.textContent = regionInfo(hit).title;
  });
  renderer.domElement.addEventListener("click", () => {
    raycaster.setFromCamera(pointer, camera);
    const hit = atlas.raycast(raycaster);
    select(hit);
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") select(null);
  });
  const about = $<HTMLDialogElement>("about");
  $("about-open").addEventListener("click", () => about.showModal());
  $("about-close").addEventListener("click", () => about.close());

  // axes: compare the antennal lobe (anterior) with the calyx (posterior) once the meshes exist
  const al = byName.get("AL_R")?.centroid;
  const ca = byName.get("MB_CA_R")?.centroid;
  const anterior = new THREE.Vector3(0, -1, 0);
  const dorsal = new THREE.Vector3(0, 0, 1);
  if (al && ca) {
    const d = new THREE.Vector3(al[0] - ca[0], al[1] - ca[1], al[2] - ca[2]);
    const ax = ["x", "y", "z"] as const;
    const dom = ax.reduce((a, b) => (Math.abs(d[b]) > Math.abs(d[a]) ? b : a));
    anterior.set(0, 0, 0);
    anterior[dom] = Math.sign(d[dom]) || 1;
    dorsal.set(0, 0, 1);
    if (dom === "z") dorsal.set(0, 1, 0);
  }
  const left = new THREE.Vector3(-1, 0, 0);
  const gizmo = $<HTMLCanvasElement>("orient");
  const gctx = gizmo.getContext("2d")!;

  const onResize = () => {
    const w = stage.clientWidth;
    const h = stage.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    if (w <= 820) camera.setViewOffset(w, h, 0, h * 0.18, w, h);
    else camera.clearViewOffset();
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(onResize).observe(stage);
  onResize();
  $("loading").classList.add("done");

  let last = performance.now();
  let elapsed = 0;
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    elapsed += dt;
    const t = elapsed;
    if (glideT < 1) {
      glideT = Math.min(1, glideT + dt * 0.8);
      const e = glideT * glideT * (3 - 2 * glideT);
      controls.target.lerpVectors(glideFrom.target, glideTo.target, e);
      camera.position.lerpVectors(glideFrom.cam, glideTo.cam, e);
    }
    controls.update();
    somas.update(t);
    if (!reduced) tubes.setTime(t);
    renderer.render(scene, camera);
    const dist = camera.position.distanceTo(controls.target);
    const px = (renderer.domElement.clientHeight / 2) / (dist * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
    scaleBar($("scalebar"), px);
    drawGizmo(gctx, camera);
    (window as unknown as { __anatomy: () => unknown }).__anatomy = () => ({ selected, side });
  });

  function drawGizmo(c: CanvasRenderingContext2D, cam: THREE.Camera) {
    const w = gizmo.clientWidth;
    const h = gizmo.clientHeight;
    const dpr = Math.min(window.devicePixelRatio, 2);
    if (gizmo.width !== Math.round(w * dpr)) {
      gizmo.width = Math.round(w * dpr);
      gizmo.height = Math.round(h * dpr);
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    const project = (v: THREE.Vector3) => v.clone().applyQuaternion(cam.quaternion.clone().invert());
    const axes: [THREE.Vector3, string, string][] = [
      [anterior, "A", "#ffd166"],
      [dorsal, "D", "#66f0ff"],
      [left, "L", "#ffb35c"],
    ];
    const cx = w / 2;
    const cy = h / 2;
    for (const [dir, label, color] of axes) {
      const p = project(dir);
      c.strokeStyle = color;
      c.fillStyle = color;
      c.beginPath();
      c.moveTo(cx, cy);
      c.lineTo(cx + p.x * 28, cy - p.y * 28);
      c.stroke();
      c.font = "600 11px 'IBM Plex Sans', sans-serif";
      c.fillText(label, cx + p.x * 34 - 4, cy - p.y * 34 + 4);
    }
  }
}

main().catch((err) => {
  console.error(err);
  document.querySelector(".loading-inner p")!.textContent = `Could not start: ${err.message}`;
});
