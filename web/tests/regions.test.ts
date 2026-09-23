import { describe, expect, it } from "vitest";
import { everyStem, regionInfo, systemColor } from "../src/anatomy/regions";

describe("neuropil copy", () => {
  it("gives every stem a system, a colour, and a description", () => {
    const stems = everyStem();
    expect(stems.length).toBe(43);
    for (const info of stems) {
      expect(info.system).not.toBe("other");
      expect(info.about.length).toBeGreaterThan(12);
      expect(systemColor(info.system)).toMatch(/^#[0-9a-f]{6}$/);
      expect(info.color).toBe(systemColor(info.system));
    }
  });

  it("keeps the smell-path regions pointed at LeetFly", () => {
    expect(regionInfo("AL_L").side).toBe("left");
    expect(regionInfo("AL_R").about).toMatch(/LeetFly/);
    expect(regionInfo("MB_CA_R").about).toMatch(/Kenyon/);
    expect(regionInfo("LH_L").about).toMatch(/LeetFly/);
    expect(regionInfo("EB").side).toBe("");
  });
});
