import { describe, expect, it } from "vitest";
import { WALL_R, resting, stepFlight, type Surface, type World } from "../src/fly-page/pilot";

const footDrop = 20;
const perch: Surface = { kind: "perch", index: -1, x: 0, z: 0, top: 26, r: 16, side: 19 };
const dish: Surface = { kind: "feeder", index: 3, x: 80, z: 0, top: 22, r: 24, side: 27.5 };
const world: World = { surfaces: [perch, dish], footDrop, wallR: WALL_R };

describe("stepFlight", () => {
  it("takes off from the perch on thrust and stays airborne", () => {
    const s = resting(footDrop);
    s.y = 26 + footDrop;
    const a = stepFlight(s, { thrust: 1, yaw: 0, climb: 0 }, 0.05, world);
    expect(a.event).toBe("takeoff");
    expect(s.grounded).toBe(false);
    for (let i = 0; i < 40; i++) stepFlight(s, { thrust: 1, yaw: 0, climb: 0 }, 0.05, world);
    expect(s.grounded).toBe(false);
    expect(s.x).toBeGreaterThan(20);
    for (let i = 0; i < 10; i++) stepFlight(s, { thrust: 1, yaw: 0, climb: 1 }, 0.05, world);
    expect(s.grounded).toBe(false);
    expect(s.y).toBeGreaterThan(26 + footDrop);
    expect(s.x).toBeGreaterThan(0);
  });

  it("lands on a feeder dish when it descends onto the top", () => {
    const s = resting(footDrop);
    s.grounded = false;
    s.pad = null;
    s.x = 80;
    s.z = 0;
    s.y = 22 + footDrop + 30;
    s.vy = -10;
    s.speed = 0;
    let landed = false;
    for (let i = 0; i < 80 && !landed; i++) {
      const r = stepFlight(s, { thrust: 0, yaw: 0, climb: -1 }, 0.05, world);
      landed = r.event === "land";
    }
    expect(landed).toBe(true);
    expect(s.pad).toBe("feeder");
    expect(s.padIndex).toBe(3);
    expect(s.y).toBeCloseTo(22 + footDrop, 0);
  });

  it("lands on the floor when nothing is underneath", () => {
    const s = resting(footDrop);
    s.grounded = false;
    s.pad = null;
    s.x = 200;
    s.z = 0;
    s.y = footDrop + 40;
    s.vy = -5;
    let pad: string | null = null;
    for (let i = 0; i < 80; i++) {
      const r = stepFlight(s, { thrust: 0, yaw: 0, climb: -1 }, 0.05, world);
      if (r.event === "land") pad = s.pad;
    }
    expect(pad).toBe("floor");
    expect(s.y).toBeCloseTo(footDrop, 0);
  });

  it("slides off a feeder stand instead of flying through it", () => {
    const s = resting(footDrop);
    s.grounded = false;
    s.pad = null;
    s.x = 80 - 10;
    s.z = 0;
    s.y = footDrop + 8;
    s.speed = 40;
    s.heading = 0;
    stepFlight(s, { thrust: 1, yaw: 0, climb: 0 }, 0.05, world);
    expect(Math.hypot(s.x - 80, s.z)).toBeGreaterThanOrEqual(27.5 - 0.2);
  });

  it("stays inside the arena wall", () => {
    const s = resting(footDrop);
    s.grounded = false;
    s.pad = null;
    s.x = WALL_R - 5;
    s.y = footDrop + 40;
    s.speed = MAX_SPEED_SAFE();
    s.heading = 0;
    for (let i = 0; i < 30; i++) stepFlight(s, { thrust: 1, yaw: 0, climb: 1 }, 0.05, world);
    expect(Math.hypot(s.x, s.z)).toBeLessThanOrEqual(WALL_R + 0.1);
  });
});

function MAX_SPEED_SAFE() {
  return 95;
}
