// The instrument layer: a scrolling spike raster (like an electrophysiology recording), live counters, a scale bar.
import { BAND_COLORS, type Band, type SpikeEvent } from "./scene";

const WINDOW = 4; // seconds of history shown
const BANDS: { band: Band; label: string; weight: number }[] = [
  { band: "PN", label: "PN", weight: 1 },
  { band: "KC", label: "KC", weight: 1.6 },
  { band: "OUT", label: "OUT", weight: 0.9 },
  { band: "DA", label: "DA", weight: 0.8 },
];

export class Raster {
  private ctx: CanvasRenderingContext2D;
  private events: SpikeEvent[] = [];
  readouts: Partial<Record<Band, string>> = {};

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
  }

  push(e: SpikeEvent) {
    this.events.push(e);
    if (this.events.length > 6000) this.events.splice(0, this.events.length - 5000);
  }

  clear() {
    this.events = [];
  }

  draw(now: number) {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    const c = this.ctx;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    const labelW = 34;
    const readW = w > 560 ? 196 : 0;
    const plotW = w - labelW - readW;
    const total = BANDS.reduce((s, b) => s + b.weight, 0);
    const gap = 3;
    const usable = h - gap * (BANDS.length - 1);
    let y = 0;
    const tops: Record<string, [number, number]> = {};
    for (const b of BANDS) {
      const bh = (usable * b.weight) / total;
      tops[b.band] = [y, bh];
      c.fillStyle = "rgba(255,255,255,0.025)";
      c.fillRect(labelW, y, plotW, bh);
      c.fillStyle = BAND_COLORS[b.band];
      c.font = "600 10px 'IBM Plex Mono', ui-monospace, monospace";
      c.textBaseline = "middle";
      c.fillText(b.label, 4, y + bh / 2);
      if (readW && this.readouts[b.band]) {
        c.fillStyle = "rgba(200,210,235,0.85)";
        c.font = "10px 'IBM Plex Mono', ui-monospace, monospace";
        c.fillText(this.readouts[b.band]!, labelW + plotW + 8, y + bh / 2);
      }
      y += bh + gap;
    }
    // time grid: one tick per second
    c.strokeStyle = "rgba(140,170,255,0.08)";
    for (let s = 0; s <= WINDOW; s++) {
      const x = labelW + plotW - (s / WINDOW) * plotW;
      c.beginPath();
      c.moveTo(x, 0);
      c.lineTo(x, h);
      c.stroke();
    }
    // spikes
    const keep: SpikeEvent[] = [];
    for (const e of this.events) {
      const age = now - e.t;
      if (age > WINDOW) continue;
      keep.push(e);
      if (age < 0) continue;
      const [top, bh] = tops[e.band];
      const x = labelW + plotW - (age / WINDOW) * plotW;
      const yy = top + e.row * bh;
      c.globalAlpha = age < 0.15 ? 1 : Math.max(0.25, 1 - age / WINDOW);
      c.fillStyle = e.color ?? BAND_COLORS[e.band];
      c.fillRect(x, yy - (e.band === "KC" ? 1 : 1.5), age < 0.15 ? 2 : 1.2, e.band === "KC" ? 2 : 3);
    }
    c.globalAlpha = 1;
    this.events = keep;
  }
}

/** A scale bar with a "nice" length that stays between ~40 and ~140 px as you zoom. */
export function scaleBar(el: HTMLElement, pxPerMicron: number) {
  if (!Number.isFinite(pxPerMicron) || pxPerMicron <= 0) return;
  const nice = [10, 20, 50, 100, 200, 500, 1000];
  const um = nice.find((v) => v * pxPerMicron >= 40) ?? 1000;
  const bar = el.querySelector<HTMLElement>(".bar-line")!;
  bar.style.width = `${Math.min(160, um * pxPerMicron).toFixed(1)}px`;
  el.querySelector<HTMLElement>(".bar-label")!.textContent = `${um} µm`;
}
