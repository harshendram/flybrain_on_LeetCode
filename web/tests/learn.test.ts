// "Watch it learn" replays the dopamine rule trial by trial. Streaming every training problem into a naive fly must
// end exactly where the Python fly ended (model.bin's trained weights), and score its published 39.8% top-1.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { Fly, decodeArrays, type ModelMeta } from "../src/fly";

const dir = resolve(__dirname, "..", "public", "data");
const buf = (name: string) => {
  const b = readFileSync(resolve(dir, name));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const meta: ModelMeta = JSON.parse(readFileSync(resolve(dir, "model.json"), "utf8"));
const arrays = decodeArrays(meta, buf("model.bin"));
const fly = new Fly(meta, arrays);
const lmeta = JSON.parse(readFileSync(resolve(dir, "learning.json"), "utf8"));
const la = decodeArrays({ arrays: lmeta.arrays } as unknown as ModelMeta, buf("learning.bin")) as Record<string, any>;

describe("watch it learn", () => {
  it("streaming the dev set from naive weights reproduces the trained fly", () => {
    fly.cloneVariant("real", "learn");
    fly.resetNaive("learn");
    const { n, k_max: k, class_counts } = lmeta.sets.dev;
    const rates = Fly.dopamineRates(class_counts, n, lmeta.eta);
    for (let i = 0; i < n; i++) fly.learnStep("learn", la["dev.active"].subarray(i * k, (i + 1) * k), la["dev.techniques"][i], rates);
    const trained = { plus: arrays["real.w_plus"] as Float32Array, minus: arrays["real.w_minus"] as Float32Array };
    let worst = 0;
    // compare through a public path: scores of every KC in isolation
    for (let j = 0; j < meta.n_kc; j++) {
      const s = fly.score(Int32Array.of(j), "learn");
      for (let c = 0; c < fly.C; c++) {
        const ref = trained.plus[j * fly.C + c] - trained.minus[j * fly.C + c];
        worst = Math.max(worst, Math.abs(s[c] - ref));
      }
    }
    expect(worst).toBeLessThan(2e-6);
  });

  it("the trained fly scores its published top-1 on future problems", () => {
    const { n, k_max: k } = lmeta.sets.future;
    let hits = 0;
    for (let i = 0; i < n; i++) {
      const guess = fly.topGuess("learn", la["future.active"].subarray(i * k, (i + 1) * k));
      if (la["future.techniques"][i] & (1 << guess)) hits++;
    }
    expect(hits / n).toBeCloseTo(meta.scores_on_future_problems["fly (real wiring)"]["hit@1"], 3);
  });
});
