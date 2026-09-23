// Link-preview cards (1200x630): the live 3D scene on the right, a title block on the left.
//   node scripts/og.mjs <url> <out.png> <waitMs> <title> <line> <facts>
import puppeteer from "puppeteer-core";

const [url, out, waitMs, title, line, facts] = process.argv.slice(2);
const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: "new",
  args: ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"],
  defaultViewport: { width: 1200, height: 630, deviceScaleFactor: 1 },
});
const page = await browser.newPage();
await page.goto(url, { waitUntil: "load" });
await page.waitForSelector(".loading.done", { timeout: 90000 });
await new Promise((r) => setTimeout(r, Number(waitMs)));
await page.evaluate(
  (title, line, facts) => {
    const css = document.createElement("style");
    css.textContent = `.topbar,.panel,.scorecard,.raster,.legend,#legend,.counters,#counters,.scalebar,#scalebar,.braincam,.caption,.loading,.joystick,.site-nav,.orient,.hover-label,.card{display:none!important}
      #og-card{position:fixed;left:60px;top:0;bottom:0;width:440px;display:flex;flex-direction:column;justify-content:center;gap:16px;z-index:99;color:#e8ecf8;font-family:var(--sans)}
      #og-card .brand{font:700 64px/1 var(--sans);letter-spacing:-0.02em;background:linear-gradient(90deg,#ffb35c,#66f0ff);-webkit-background-clip:text;background-clip:text;color:transparent}
      #og-card .line{font:600 30px/1.2 var(--sans);text-wrap:balance}
      #og-card .facts{font:400 17px/1.5 var(--mono);color:#8e98b8}`;
    document.head.appendChild(css);
    const card = document.createElement("div");
    card.id = "og-card";
    card.innerHTML = `<div class="brand">${title}</div><div class="line">${line}</div><div class="facts">${facts}</div>`;
    document.body.appendChild(card);
  },
  title,
  line,
  facts,
);
await new Promise((r) => setTimeout(r, 400));
await page.screenshot({ path: out });
await browser.close();
console.log("wrote", out);
