// The fly page: a real fruit fly (flybody) flies to the algorithm its mushroom body votes for.
// The decision is the same model as the brain page (40% top-1 on future problems). The flight only visualises it:
// a confident fly surges straight to its choice; an unsure one casts (zig-zags) between its top candidates.
import * as THREE from "three";
import "../style.css";
import "./fly.css";
import { TECH_COLORS } from "../colors";
import { Progress, loadLearning, loadModel, type Learning } from "../data";
import examples from "../examples.json";
import type { Smell } from "../fly";
import { Arena, PERCH_Y } from "./arena";
import { BrainCam } from "./braincam";
import { FlyBody } from "./flybody";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const titleCase = (s: string) => s.replace(/\b[a-z]/g, (c) => c.toUpperCase());
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const HOVER = new THREE.Vector3(0, PERCH_Y + 46, 0);
const FLY_SCALE = 95; // flybody is in cm (~0.25 cm long); the arena is in "units"
const STUDIO = location.hash === "#studio"; // freezes the choreography so poses can be inspected (see __leetfly)

interface Problem {
  title: string;
  link?: string;
  truth: number[] | null; // technique indices, null when unknown (a pasted problem)
  smell: Smell;
}

type Phase = "idle" | "release" | "sniff" | "fly" | "land" | "verdict";

async function main() {
  const p = new Progress((l, t) => ($("bar-fill").style.width = `${Math.min(100, (100 * l) / Math.max(t, 1))}%`));
  const [fly, learning, phase2, body] = await Promise.all([
    loadModel(p),
    loadLearning(p),
    fetch("./data/phase2.json").then((r) => r.json() as Promise<{ receptor_technique: number[] }>),
    FlyBody.load(p, FLY_SCALE),
  ]);
  const names = fly.meta.techniques;
  const arena = new Arena($("stage"), names, TECH_COLORS);
  arena.scene.add(body.root);
  const cam = new BrainCam($<HTMLCanvasElement>("braincam"), fly.nKc, names, TECH_COLORS);
  $("loading").classList.add("done");

  // ---------- state ----------
  const pos = HOVER.clone().setY(PERCH_Y + body.footDrop);
  (window as unknown as { __leetfly: unknown }).__leetfly = { body, arena, pos, THREE, aim: (h: number) => (heading = h) };
  let heading = 0;
  let bank = 0;
  let pitch = 0;
  const landAt = new THREE.Vector3();
  let phase: Phase = "idle";
  let phaseT = 0;
  let problem: Problem | null = null;
  let target = -1;
  const tried = new Set<number>();
  let path: { from: THREE.Vector3; to: THREE.Vector3; dur: number; cast: number } | null = null;
  let follow = true;
  let score = { first: 0, total: 0 };
  let jolt = 0;
  arena.controls.addEventListener("start", () => (follow = false));

  const caption = (html: string) => ($("caption").innerHTML = html);
  const setPhase = (p: Phase) => {
    phase = p;
    phaseT = 0;
  };

  // ---------- problems ----------
  const receptorColors = (r: Float64Array) => {
    const cols = Array.from(r, (_, i) => new THREE.Color(TECH_COLORS[phase2.receptor_technique[i]]));
    return { cols, weights: Array.from(r, (v) => Math.max(0, v)) };
  };

  function start(pr: Problem) {
    problem = pr;
    tried.clear();
    for (const f of arena.feeders) f.highlight(false);
    const { cols, weights } = receptorColors(pr.smell.receptors);
    arena.plume.release(cols, weights);
    follow = true;
    $("teach").hidden = true;
    renderStatus();
    caption(`Releasing the smell of <b>${esc(pr.title)}</b>`);
    setPhase("release");
  }

  function fromExample(i: number): Problem {
    const e = examples[i];
    return {
      title: e.title,
      truth: e.techniques.map((t) => names.indexOf(t)).filter((c) => c >= 0),
      smell: fly.smell(e.title, e.description, "real"),
    };
  }

  function randomReal(data: Learning): Problem {
    const f = data.future;
    const i = (Math.random() * f.n) | 0;
    const meta = fly.meta.problems[f.problem[i]];
    const receptors = Float64Array.from(f.receptors.subarray(i * 51, (i + 1) * 51));
    return {
      title: titleCase(meta.title),
      link: `https://leetcode.com/problems/${meta.slug}/`,
      truth: names.map((_, c) => c).filter((c) => f.techniques[i] & (1 << c)),
      smell: fly.smellReceptors(receptors, "real"),
    };
  }

  function choose(): number {
    const s = fly.score(problem!.smell.active, "real");
    problem!.smell = { ...problem!.smell, scores: s, ranking: [...s.keys()].sort((a, b) => s[b] - s[a] || a - b) };
    return problem!.smell.ranking.find((c) => !tried.has(c)) ?? problem!.smell.ranking[0];
  }

  function confidence(): number {
    const r = problem!.smell.ranking.filter((c) => !tried.has(c));
    const s = problem!.smell.scores;
    const lo = Math.min(...s);
    return r.length > 1 ? (s[r[0]] - s[r[1]]) / Math.max(s[r[0]] - lo, 1e-9) : 1;
  }

  function flyTo(c: number) {
    target = c;
    tried.add(c);
    const f = arena.feeders[c];
    landAt.copy(f.landing(pos, body.footDrop));
    // arrive a little short of and above the spot, then settle onto it facing the drop
    const back = landAt.clone().sub(f.group.position).setY(0).normalize();
    const to = landAt.clone().addScaledVector(back, 12).add(new THREE.Vector3(0, 12, 0));
    const dist = pos.distanceTo(to);
    path = { from: pos.clone(), to, dur: Math.max(2.2, dist / 95), cast: 70 * (1 - Math.min(1, confidence() * 2.2)) };
    const conf = confidence();
    caption(
      conf > 0.35
        ? `Surging toward <b style="color:${TECH_COLORS[c]}">${esc(names[c])}</b>`
        : `Unsure… casting between <b style="color:${TECH_COLORS[c]}">${esc(names[c])}</b> and <b>${esc(names[problem!.smell.ranking.find((x) => !tried.has(x)) ?? c])}</b>`,
    );
    setPhase("fly");
  }

  function verdict(correct: boolean) {
    const f = arena.feeders[target];
    fly.reward(problem!.smell.active, "real", target, correct);
    cam.dopamine(correct ? "PAM" : "PPL1", clock);
    if (correct) {
      arena.sparks.burst(f.top.clone(), new THREE.Color(1, 0.82, 0.35), 90, 20, 45);
      f.highlight(true);
      if (tried.size === 1) score.first++;
      score.total++;
      caption(`<b style="color:#ffd166">Correct!</b> It drinks the sugar: reward dopamine (PAM) strengthens this choice.`);
      renderStatus(true);
      setPhase("verdict");
      phaseT = 0;
      finished = true;
    } else {
      arena.sparks.burst(f.top.clone(), new THREE.Color(0.45, 0.7, 1), 120, 34, 25);
      jolt = 1;
      const answer = problem!.truth ? problem!.truth.map((c) => names[c]).join(", ") : "";
      if (tried.size >= 3 || !problem!.truth) {
        if (problem!.truth) score.total++;
        for (const c of problem!.truth ?? []) arena.feeders[c].highlight(true);
        caption(`<b style="color:#ff5c75">Shock!</b> Punishment dopamine (PPL1).${answer ? ` The answer was <b>${esc(answer)}</b>.` : ""}`);
        renderStatus(false);
        setPhase("verdict");
        finished = true;
      } else {
        caption(`<b style="color:#ff5c75">Shock!</b> Punishment dopamine (PPL1) weakens that choice. It tries again…`);
        renderStatus(false);
        setPhase("verdict");
        finished = false;
      }
    }
  }
  let finished = true;

  function renderStatus(correct?: boolean) {
    if (!problem) return;
    const title = problem.link
      ? `<a href="${problem.link}" target="_blank" rel="noopener">${esc(problem.title)}</a>`
      : esc(problem.title);
    const guesses = [...tried].map((c) => `<span class="pill" style="--c:${TECH_COLORS[c]}">${esc(names[c])}</span>`).join(" ");
    $("status").innerHTML =
      `<div class="st-title">${title}</div>` +
      (guesses ? `<div class="st-row">flew to ${guesses}${correct === true ? " ✓" : correct === false ? " ✗" : ""}</div>` : "") +
      (problem.truth && correct !== undefined ? `<div class="st-row muted">answer: ${esc(problem.truth.map((c) => names[c]).join(", "))}</div>` : "");
    $("score").innerHTML = score.total
      ? `First try: <b>${score.first} / ${score.total}</b> · the brain behind it scores 40% on problems it never saw`
      : "";
  }

  // ---------- controls ----------
  $("examples").innerHTML = examples.map((e, i) => `<button data-i="${i}">${esc(e.short)}</button>`).join("");
  $("examples").addEventListener("click", (e) => {
    const i = (e.target as HTMLElement).dataset.i;
    if (i !== undefined && phaseIdle()) start(fromExample(Number(i)));
  });
  $("real").addEventListener("click", () => phaseIdle() && start(randomReal(learning)));
  $("release").addEventListener("click", () => {
    const title = $<HTMLInputElement>("p-title").value;
    const desc = $<HTMLTextAreaElement>("p-desc").value;
    if (!phaseIdle() || (!title.trim() && !desc.trim())) return;
    start({ title: title || "Your problem", truth: null, smell: fly.smell(title, desc, "real") });
  });
  $("sugar").addEventListener("click", () => phase === "land" && problem && !problem.truth && ((problem.truth = [target]), verdict(true)));
  $("shock").addEventListener("click", () => phase === "land" && problem && !problem.truth && verdict(false));
  const about = $<HTMLDialogElement>("about");
  $("about-open").addEventListener("click", () => about.showModal());
  $("about-close").addEventListener("click", () => about.close());
  const phaseIdle = () => phase === "idle" || (phase === "verdict" && finished && phaseT > 1.2);

  // ---------- the loop ----------
  let clock = 0;
  let last = performance.now();
  const camGoal = new THREE.Vector3();
  const lookGoal = new THREE.Vector3();
  arena.renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    clock += dt;
    phaseT += dt;
    const prev = pos.clone();

    if (!STUDIO) switch (phase) {
      case "idle":
        body.flight += (0 - body.flight) * (1 - Math.exp(-dt * 3));
        break;
      case "release":
        if (phaseT > 0.8) {
          setPhase("sniff");
          cam.show(problem!.smell, clock + 0.6);
          caption(`Sniffing… <span class="muted">its mushroom body is reading the smell</span>`);
        }
        break;
      case "sniff": {
        body.flight += (1 - body.flight) * (1 - Math.exp(-dt * 4));
        const bob = new THREE.Vector3(Math.sin(clock * 1.3) * 7, Math.sin(clock * 2.1) * 3, Math.cos(clock * 1.1) * 7);
        pos.lerp(HOVER.clone().add(bob), 1 - Math.exp(-dt * 2.2));
        if (phaseT > 2.6) flyTo(choose());
        break;
      }
      case "fly": {
        const u = Math.min(1, phaseT / path!.dur);
        const e = u * u * (3 - 2 * u);
        const base = path!.from.clone().lerp(path!.to, e);
        base.y += 34 * Math.sin(Math.PI * u);
        const dir = path!.to.clone().sub(path!.from).setY(0).normalize();
        const side = new THREE.Vector3(-dir.z, 0, dir.x);
        base.addScaledVector(side, path!.cast * Math.sin(2 * Math.PI * 2.1 * u) * Math.pow(1 - u, 1.2));
        pos.copy(base);
        body.flight = 1;
        if (u >= 1) {
          setPhase("land");
          caption(`Landing on <b style="color:${TECH_COLORS[target]}">${esc(names[target])}</b>`);
        }
        break;
      }
      case "land": {
        pos.lerp(landAt, 1 - Math.exp(-dt * 4));
        body.flight += (0 - body.flight) * (1 - Math.exp(-dt * 5));
        if (phaseT > 0.9 && problem!.truth) verdict(problem!.truth.includes(target));
        else if (phaseT > 0.9 && !problem!.truth && $("teach").hidden) {
          $("teach").hidden = false;
          caption(`Is <b>${esc(names[target])}</b> right? Teach it with sugar or a shock.`);
        }
        break;
      }
      case "verdict": {
        const correct = problem?.truth?.includes(target) ?? false;
        body.feed += ((correct ? 1 : 0) - body.feed) * (1 - Math.exp(-dt * 4));
        if (!finished && phaseT > 1.4) {
          body.feed = 0;
          flyTo(choose());
        }
        if (finished && phaseT > 6 && correct) body.feed += (0 - body.feed) * (1 - Math.exp(-dt * 2));
        break;
      }
    }

    // orientation: face the direction of travel, bank into turns
    const v = pos.clone().sub(prev);
    if (v.lengthSq() > 1e-4) {
      const want = Math.atan2(-v.z, v.x);
      let d = want - heading;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      heading += d * (1 - Math.exp(-dt * 6));
      bank += (Math.max(-0.6, Math.min(0.6, -d * 3)) - bank) * (1 - Math.exp(-dt * 4));
    } else bank *= 0.95;
    jolt *= Math.exp(-dt * 4);
    body.root.position.copy(pos);
    body.root.rotation.set(0, 0, 0);
    body.root.rotateY(heading + (REDUCED ? 0 : jolt * Math.sin(clock * 60) * 0.3));
    body.root.rotateX(bank);
    // nose up in the air (flybody hovers at 47.5 degrees), level on the ground
    const pitchGoal = body.flight > 0.5 ? (phase === "fly" ? 0.3 : 0.55) : 0;
    pitch += (pitchGoal - pitch) * (1 - Math.exp(-dt * 3));
    body.root.rotateZ(pitch);
    body.update(dt);

    // camera: follow the fly while it works, orbit when idle
    if (STUDIO) {
      // camera and pose are driven from the console
    } else if (follow && phase !== "idle") {
      const back = new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
      const zoom = Math.min(2, Math.max(1, 1.05 / arena.camera.aspect)); // portrait screens see less sideways
      if (phase === "fly") camGoal.copy(pos).addScaledVector(back, -105 * zoom).add(new THREE.Vector3(0, 34 * zoom, 0));
      else if (phase === "release" || phase === "sniff") camGoal.copy(pos).add(new THREE.Vector3(70, 24, 92).multiplyScalar(zoom));
      else camGoal.copy(pos).add(new THREE.Vector3(Math.cos(clock * 0.3) * 95, 48, Math.sin(clock * 0.3) * 95).multiplyScalar(zoom));
      lookGoal.copy(pos).add(new THREE.Vector3(0, 8, 0));
      arena.camera.position.lerp(camGoal, 1 - Math.exp(-dt * 2.2));
      arena.controls.target.lerp(lookGoal, 1 - Math.exp(-dt * 3));
    } else if (phase === "idle") {
      arena.controls.autoRotate = !REDUCED;
      arena.controls.autoRotateSpeed = 0.4;
    }
    arena.render(clock, dt);
    cam.draw(clock);
  });

  // open with a problem already in the air
  if (!STUDIO) setTimeout(() => start(fromExample(0)), 900);
}

main().catch((err) => {
  console.error(err);
  document.querySelector(".loading-inner p")!.textContent = `Could not start: ${err.message}`;
});
