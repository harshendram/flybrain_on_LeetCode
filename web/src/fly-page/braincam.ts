// "Brain cam": the fly's mushroom body as a live instrument, in sync with its flight. Glomeruli ring -> the 1,880
// Kenyon cells (sunflower disc; the firing 2.5% light up) -> its 14 technique outputs.
import type { Smell } from "../fly";

const GOLDEN = Math.PI * (3 - Math.sqrt(5));

export class BrainCam {
  private ctx: CanvasRenderingContext2D;
  private smell: Smell | null = null;
  private t0 = -99;
  private da: { kind: "PAM" | "PPL1"; t: number } | null = null;
  private firing = new Set<number>();

  constructor(private canvas: HTMLCanvasElement, private nKc: number, private techniques: string[], private colors: string[]) {
    this.ctx = canvas.getContext("2d")!;
  }

  show(smell: Smell, t: number) {
    this.smell = smell;
    this.t0 = t;
    this.firing = new Set(smell.active);
  }

  dopamine(kind: "PAM" | "PPL1", t: number) {
    this.da = { kind, t };
  }

  draw(t: number) {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    if (this.canvas.width !== Math.round(w * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    const c = this.ctx;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    const dt = t - this.t0;
    const ramp = (a: number, b: number) => Math.min(1, Math.max(0, (dt - a) / (b - a)));
    const s = this.smell;
    const narrow = w < 300; // phone inset: fewer, larger rows
    const barsW = narrow ? 92 : 118;
    const cy = h / 2 + 6;
    const R = Math.min((w - barsW - 26) / 2, h / 2 - 16);
    const cx = 12 + R;

    // Kenyon cells: sunflower disc, firing cells glow
    const kcOn = ramp(0.7, 1.1);
    for (let i = 0; i < this.nKc; i++) {
      const r = Math.sqrt((i + 0.5) / this.nKc) * (R - 14);
      const a = i * GOLDEN;
      const x = cx + r * Math.cos(a);
      const y = cy + r * Math.sin(a);
      const on = this.firing.has(i) ? kcOn * (0.75 + 0.25 * Math.sin(t * 9 + i)) : 0;
      c.fillStyle = on > 0 ? `rgba(120,245,255,${0.35 + 0.65 * on})` : "rgba(70,100,190,0.22)";
      const sz = on > 0 ? 2.6 : 1.2;
      c.fillRect(x - sz / 2, y - sz / 2, sz, sz);
    }
    // glomeruli ring
    if (s) {
      const glOn = ramp(0.15, 0.5);
      const max = Math.max(...s.pn, 1e-9);
      s.pn.forEach((v, g) => {
        const a = (g / s.pn.length) * Math.PI * 2 - Math.PI / 2;
        const x = cx + (R - 4) * Math.cos(a);
        const y = cy + (R - 4) * Math.sin(a);
        const k = (v / max) * glOn;
        c.fillStyle = `rgba(255,${150 + 80 * k | 0},${60 + 60 * k | 0},${0.25 + 0.75 * k})`;
        c.beginPath();
        c.arc(x, y, 2 + 3.5 * k, 0, Math.PI * 2);
        c.fill();
      });
    }
    // dopamine flash over the disc
    if (this.da && t - this.da.t < 1.2) {
      const f = 1 - (t - this.da.t) / 1.2;
      c.fillStyle = this.da.kind === "PAM" ? `rgba(255,209,102,${0.28 * f})` : `rgba(255,92,117,${0.3 * f})`;
      c.beginPath();
      c.arc(cx, cy, R, 0, Math.PI * 2);
      c.fill();
    }
    // outputs
    const x0 = w - barsW;
    c.font = "600 9.5px 'IBM Plex Mono', ui-monospace, monospace";
    c.fillStyle = "rgba(142,152,184,0.9)";
    c.fillText(narrow ? "KCs" : "KENYON CELLS", 12, 13);
    c.fillText("OUTPUTS", x0, 13);
    if (s) {
      const outOn = ramp(1.3, 1.9);
      const lo = Math.min(...s.scores);
      const hi = Math.max(...s.scores);
      const rows = Math.min(this.techniques.length, Math.max(3, Math.floor((h - 26) / 11)));
      const rowH = (h - 26) / rows;
      s.ranking.slice(0, rows).forEach((tech, rank) => {
        const y = 22 + rank * rowH;
        const v = ((s.scores[tech] - lo) / (hi - lo || 1)) * outOn;
        c.fillStyle = "rgba(255,255,255,0.05)";
        c.fillRect(x0, y, barsW - 8, rowH - 2);
        c.fillStyle = this.colors[tech];
        c.globalAlpha = rank === 0 ? 1 : 0.55;
        c.fillRect(x0, y, (barsW - 8) * v, rowH - 2);
        c.globalAlpha = 1;
        c.fillStyle = rank === 0 && outOn > 0.9 ? "#fff" : "rgba(220,226,245,0.85)";
        c.font = `${rank === 0 ? 600 : 400} 9px 'IBM Plex Sans', system-ui, sans-serif`;
        c.fillText(short(this.techniques[tech]), x0 + 3, y + rowH / 2 + 3);
      });
    }
  }
}

function short(t: string): string {
  return t.replace("Dynamic Programming", "Dynamic Prog.").replace("Bit Manipulation", "Bit Manip.");
}
