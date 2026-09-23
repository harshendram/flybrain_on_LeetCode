// Manual flight. Pure so vitest can step it without a scene: heading, damped speed, climb,
// cylinder collisions (perch and feeder stands), the arena wall and the floor.
// y is the thorax height, the same frame as FlyBody's root (feet sit footDrop below it).

export interface Surface {
  kind: "perch" | "feeder";
  index: number;
  x: number;
  z: number;
  top: number;
  r: number;
  side: number;
}

export interface FlightInput {
  /** 1 = full thrust (W), -1 = brake (S). */
  thrust: number;
  /** -1 = yaw left (A), +1 = yaw right (D). */
  yaw: number;
  /** 1 = climb (Space), -1 = descend (Shift / C). */
  climb: number;
}

export interface FlightState {
  x: number;
  y: number;
  z: number;
  heading: number;
  speed: number;
  vy: number;
  grounded: boolean;
  pad: "floor" | "perch" | "feeder" | null;
  padIndex: number;
}

export interface World {
  surfaces: Surface[];
  footDrop: number;
  wallR?: number;
}

export interface StepResult {
  bank: number;
  pitch: number;
  /** 0..1, how hard the wings should beat. */
  wing: number;
  event: "none" | "takeoff" | "land";
}

export const WALL_R = 400;
const MAX_SPEED = 95;
const ACCEL = 80;
const BRAKE = 160;
const YAW_RATE = 2.1;
const CLIMB = 46;
const SINK = 22;

export function flightInputFromKeys(down: Set<string>): FlightInput {
  const thrust = (down.has("KeyW") || down.has("ArrowUp") ? 1 : 0) - (down.has("KeyS") || down.has("ArrowDown") ? 1 : 0);
  const yaw = (down.has("KeyD") || down.has("ArrowRight") ? 1 : 0) - (down.has("KeyA") || down.has("ArrowLeft") ? 1 : 0);
  const climb = (down.has("Space") ? 1 : 0) - (down.has("ShiftLeft") || down.has("ShiftRight") || down.has("KeyC") ? 1 : 0);
  return { thrust, yaw, climb };
}

export function isFlightKey(code: string): boolean {
  return ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space", "ShiftLeft", "ShiftRight", "KeyC"].includes(code);
}

export function resting(footDrop: number, pad: FlightState["pad"] = "perch", padIndex = -1): FlightState {
  return { x: 0, y: footDrop + (pad === "perch" ? 26 : 0), z: 0, heading: 0, speed: 0, vy: 0, grounded: true, pad, padIndex };
}

function padHeight(world: World, state: FlightState): number {
  if (state.pad === "floor" || state.pad === null) return world.footDrop;
  const s = world.surfaces.find((p) => p.kind === state.pad && p.index === state.padIndex);
  return world.footDrop + (s?.top ?? 0);
}

/** The raised surface directly under (x, z), if the fly has descended onto its top. */
function under(world: World, x: number, z: number, y: number): Surface | null {
  let best: Surface | null = null;
  for (const s of world.surfaces) {
    const dx = x - s.x;
    const dz = z - s.z;
    if (dx * dx + dz * dz > s.r * s.r) continue;
    const stand = s.top + world.footDrop;
    if (y > stand + 8) continue;
    if (!best || s.top > best.top) best = s;
  }
  return best;
}

function pushCylinders(state: FlightState, world: World) {
  for (const s of world.surfaces) {
    const dx = state.x - s.x;
    const dz = state.z - s.z;
    const d = Math.hypot(dx, dz) || 1e-6;
    // The stand occupies y from the floor up to its top. Above the dish the fly flies over it.
    const feet = state.y - world.footDrop;
    if (feet > s.top - 0.5) continue;
    if (d >= s.side) continue;
    const nx = dx / d;
    const nz = dz / d;
    state.x = s.x + nx * s.side;
    state.z = s.z + nz * s.side;
    const inward = state.speed * (Math.cos(state.heading) * nx + -Math.sin(state.heading) * nz);
    if (inward < 0) state.speed = Math.max(0, state.speed + inward);
  }
  const wall = world.wallR ?? WALL_R;
  const r = Math.hypot(state.x, state.z);
  if (r > wall) {
    state.x *= wall / r;
    state.z *= wall / r;
    state.speed *= 0.4;
  }
}

export function stepFlight(state: FlightState, input: FlightInput, dt: number, world: World): StepResult {
  const thrust = Math.max(-1, Math.min(1, input.thrust));
  const yaw = Math.max(-1, Math.min(1, input.yaw));
  const climb = Math.max(-1, Math.min(1, input.climb));
  let event: StepResult["event"] = "none";

  if (state.grounded && (thrust > 0.2 || climb > 0.2)) {
    state.grounded = false;
    state.pad = null;
    state.vy = climb > 0 ? 28 : 12;
    state.speed = Math.max(state.speed, thrust > 0 ? 18 : 0);
    event = "takeoff";
  }

  state.heading += yaw * YAW_RATE * dt;
  if (!state.grounded) {
    if (thrust > 0) state.speed += thrust * ACCEL * dt;
    else if (thrust < 0) state.speed += thrust * BRAKE * dt;
    else state.speed *= Math.exp(-dt * 0.7);
    state.speed = Math.max(0, Math.min(MAX_SPEED, state.speed));
    state.x += Math.cos(state.heading) * state.speed * dt;
    state.z += -Math.sin(state.heading) * state.speed * dt;
    // W holds altitude so thrust actually flies; Space and Shift climb and descend; letting go glides down
    const wantVy = climb !== 0 ? climb * CLIMB : thrust > 0.2 ? 0 : -SINK;
    state.vy += (wantVy - state.vy) * (1 - Math.exp(-dt * 4));
    state.y += state.vy * dt;
    pushCylinders(state, world);

    const hit = state.vy <= 2 ? under(world, state.x, state.z, state.y) : null;
    if (hit && state.y <= hit.top + world.footDrop + 1) {
      state.y = hit.top + world.footDrop;
      state.vy = 0;
      state.speed *= 0.2;
      state.grounded = true;
      state.pad = hit.kind;
      state.padIndex = hit.index;
      event = "land";
    } else if (state.y <= world.footDrop) {
      state.y = world.footDrop;
      state.vy = 0;
      state.speed *= 0.2;
      state.grounded = true;
      state.pad = "floor";
      state.padIndex = -1;
      event = "land";
    }
  } else {
    state.speed *= Math.exp(-dt * 6);
    state.vy = 0;
    state.y = padHeight(world, state);
  }

  const wing = state.grounded ? 0 : Math.min(1, 0.35 + 0.65 * Math.max(0, thrust) + Math.min(0.3, state.speed / MAX_SPEED));
  return {
    bank: Math.max(-0.7, Math.min(0.7, -yaw * 0.55)),
    pitch: state.grounded ? 0 : 0.35 + 0.2 * Math.max(0, climb),
    wing,
    event,
  };
}
