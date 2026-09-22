import { defineConfig } from "vite";

// relative base so the static build works from any GitHub Pages sub-path
export default defineConfig({
  base: "./",
  build: { target: "es2022", chunkSizeWarningLimit: 1500 },
});
