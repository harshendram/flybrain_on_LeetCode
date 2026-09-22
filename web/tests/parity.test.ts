// The browser fly must behave exactly like the Python fly that produced the reported numbers.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { Fly, decodeArrays, type ModelMeta } from "../src/fly";

const dir = resolve(__dirname, "..");
const meta: ModelMeta = JSON.parse(readFileSync(resolve(dir, "public/data/model.json"), "utf8"));
const bin = readFileSync(resolve(dir, "public/data/model.bin"));
const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const fly = new Fly(meta, decodeArrays(meta, buf));
const golden: any[] = JSON.parse(readFileSync(resolve(dir, "tests/golden.json"), "utf8"));
const examples: any[] = JSON.parse(readFileSync(resolve(dir, "src/examples.json"), "utf8"));

function closeRel(a: ArrayLike<number>, b: number[], tol: number) {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < b.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i])));
}

describe("JS fly == Python fly", () => {
  golden.forEach((g, i) => {
    for (const which of ["real", "scrambled"] as const) {
      it(`${g.title} (${which})`, () => {
        const s = fly.smell(examples[i].title, examples[i].description, which);
        closeRel(s.receptors, g.receptors, 1e-6);
        closeRel(s.pn, g[which].pn, 1e-6);
        expect([...s.active].sort((a, b) => a - b)).toEqual(g[which].active);
        closeRel(s.scores, g[which].scores, 1e-5);
        const pyTop3 = g[which].scores
          .map((v: number, c: number) => [v, c])
          .sort((a: number[], b: number[]) => b[0] - a[0])
          .slice(0, 3)
          .map((p: number[]) => p[1]);
        expect(s.ranking.slice(0, 3)).toEqual(pyTop3);
      });
    }
  });
});
