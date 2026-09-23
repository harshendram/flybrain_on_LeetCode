// Headless smoke test + screenshots of the LeetFly site.
//   node scripts/shot.mjs <url> <outPrefix> <width> <height> <actions>
// actions, comma-separated: click:sel  chip:0  wait:400  shot:name  key:KeyW/800  state:__fly  fps
import puppeteer from "puppeteer-core";

const url = process.argv[2] ?? "http://localhost:5173/";
const out = process.argv[3] ?? "shot";
const width = Number(process.argv[4] ?? 1440);
const height = Number(process.argv[5] ?? 900);
const actions = process.argv[6] ?? "";

const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: "new",
  args: ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"],
  defaultViewport: { width, height, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
const t0 = Date.now();
await page.goto(url, { waitUntil: "load" });
await page.waitForSelector(".loading.done", { timeout: 90000 });
console.log(`loaded in ${Date.now() - t0} ms`);
const gl = await page.evaluate(() => {
  const c = document.createElement("canvas").getContext("webgl2");
  const ext = c && c.getExtension("WEBGL_debug_renderer_info");
  return c ? (ext ? c.getParameter(ext.UNMASKED_RENDERER_WEBGL) : "webgl2 ok") : "no webgl2";
});
console.log("renderer:", gl);
await new Promise((r) => setTimeout(r, Number(process.env.SHOT_WAIT ?? 2500)));
await page.screenshot({ path: `${out}-1.png` });
for (const step of actions.split(",").filter(Boolean)) {
  const [kind, arg] = step.split(":");
  try {
    if (kind === "click") await page.click(arg);
    if (kind === "chip") await page.click(`#examples button[data-i="${arg}"]`);
    if (kind === "wait") await new Promise((r) => setTimeout(r, Number(arg)));
    if (kind === "shot") await page.screenshot({ path: `${out}-${arg}.png` });
    if (kind === "key") {
      const [key, ms] = arg.split("/");
      await page.keyboard.down(key);
      await new Promise((r) => setTimeout(r, Number(ms ?? 500)));
      await page.keyboard.up(key);
    }
    if (kind === "state") console.log(arg, JSON.stringify(await page.evaluate((name) => window[name]?.(), arg)));
    if (kind === "text") console.log(arg, JSON.stringify(await page.$eval(arg, (el) => el.textContent)));
    if (kind === "move") {
      const [fx, fy] = arg.split("/").map(Number);
      const box = await page.$("#stage canvas").then((el) => el.boundingBox());
      await page.mouse.move(box.x + box.width * fx, box.y + box.height * fy);
      await new Promise((r) => setTimeout(r, 200));
    }
    if (kind === "drag") {
      const [dx, dy] = arg.split("/").map(Number);
      const box = await page.$("#stage canvas").then((el) => el.boundingBox());
      const x = box.x + box.width * 0.62;
      const y = box.y + box.height * 0.45;
      await page.mouse.move(x, y);
      await page.mouse.down();
      await page.mouse.move(x + dx, y + dy, { steps: 10 });
      await page.mouse.up();
    }
    if (kind === "fps") {
      const fps = await page.evaluate(
        () =>
          new Promise((res) => {
            let n = 0;
            const s = performance.now();
            const f = () => (++n, performance.now() - s < 2000 ? requestAnimationFrame(f) : res((1000 * n) / (performance.now() - s)));
            requestAnimationFrame(f);
          }),
      );
      console.log("fps:", fps.toFixed(1));
    }
  } catch (e) {
    console.log(`step "${step}" failed: ${e.message.split("\n")[0]}`);
  }
}
console.log(logs.filter((l) => !l.includes("[vite]")).join("\n") || "(no console output)");
await browser.close();
