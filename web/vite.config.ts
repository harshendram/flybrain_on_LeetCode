import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

// Link previews (og:image / og:url) need absolute URLs. On Vercel the production domain is known at build time.
const site = process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}/` : "./";
// LEETFLY_PAGES=fly.html builds a single page with no shared chunk (scripts/build_artifact.py inlines it)
const wanted = process.env.LEETFLY_PAGES?.split(",") ?? ["index.html", "fly.html"];
const pages = wanted.filter((p) => existsSync(resolve(__dirname, p)));

// relative base so the static build works from any sub-path (Vercel, GitHub Pages, a claude.ai artifact)
export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500,
    rollupOptions: { input: Object.fromEntries(pages.map((p) => [p.replace(".html", ""), resolve(__dirname, p)])) },
  },
  plugins: [{ name: "site-url", transformIndexHtml: (html) => html.replaceAll("%SITE_URL%", site) }],
});
