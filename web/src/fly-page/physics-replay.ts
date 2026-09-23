// "Physics" mode: replays real MuJoCo flights from Phase 3 (branch research/embodied). The mushroom body chose a
// feeder and a flight style; DeepMind/Janelia's pretrained flybody flight controller flew that path in physics; the
// body's actual centre of mass and orientation were recorded every 4 ms. Here they are played back 8x slower, in the
// arena's units. Touchdown and drinking are scripted (the controller's training data has no landings).
import * as THREE from "three";
import { ARENA_R } from "./arena";

export interface PhysicsFlight {
  label: string;
  stage: "early" | "late" | "test";
  n_seen: number; // training problems the brain had learned from before this flight
  title: string;
  slug: string;
  goal: number;
  reached: number; // -1: reached no feeder
  correct: boolean;
  margin: number;
  t_arrive: number | null; // seconds of real (simulated) time
  err_mm: number | null;
  dt: number; // seconds between samples
  path: [number, number, number][]; // cm, MuJoCo frame (mirrored so feeder i sits where the site draws it)
  quat: [number, number, number, number][]; // (w, x, y, z), same frame
}

export interface PhysicsData {
  ring_cm: number;
  arrive_cm: number;
  slow: number;
  summary: Record<string, unknown>;
  flights: PhysicsFlight[];
}

export async function loadPhysics(): Promise<PhysicsData | null> {
  try {
    const r = await fetch("./data/embodied_flights.json");
    return r.ok ? ((await r.json()) as PhysicsData) : null;
  } catch {
    return null;
  }
}

/** MuJoCo (x, y, z; z up) -> three.js (x, z, -y): positions, and the matching conjugation for orientations. */
export function toThree(p: [number, number, number], scale: number, lift: number): THREE.Vector3 {
  return new THREE.Vector3(p[0] * scale, p[2] * scale + lift, -p[1] * scale);
}

export function quatToThree(q: [number, number, number, number]): THREE.Quaternion {
  const [w, x, y, z] = q;
  return new THREE.Quaternion(x, z, -y, w);
}

export class PhysicsReplay {
  private flight: PhysicsFlight | null = null;
  private t = 0;
  readonly scale: number;
  done = true;

  constructor(readonly data: PhysicsData, private lift: number) {
    this.scale = ARENA_R / data.ring_cm;
  }

  play(f: PhysicsFlight) {
    this.flight = f;
    this.t = 0;
    this.done = false;
  }

  get current(): PhysicsFlight | null {
    return this.flight;
  }

  /** Advance by dt of wall time; returns the body pose to show, or null when there is no flight. */
  step(dt: number): { pos: THREE.Vector3; quat: THREE.Quaternion; progress: number } | null {
    const f = this.flight;
    if (!f || !f.path.length) return null;
    this.t += dt / this.data.slow;
    const x = Math.min(this.t / f.dt, f.path.length - 1);
    const i = Math.floor(x);
    const j = Math.min(i + 1, f.path.length - 1);
    const u = x - i;
    const pos = toThree(f.path[i], this.scale, this.lift).lerp(toThree(f.path[j], this.scale, this.lift), u);
    const quat = quatToThree(f.quat[i]).slerp(quatToThree(f.quat[j]), u);
    if (x >= f.path.length - 1) this.done = true;
    return { pos, quat, progress: x / Math.max(1, f.path.length - 1) };
  }
}
