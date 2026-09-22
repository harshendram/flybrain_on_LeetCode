import "./style.css";
import { loadAll } from "./data";
import examples from "./examples.json";
import type { Fly, Smell, VariantName } from "./fly";
import { FlyScene, ROLE_COLORS } from "./scene";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fmtPct = (x: number) => `${(100 * x).toFixed(0)}%`;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

interface Current {
  title: string;
  desc: string;
  smell: Smell;
  slug?: string;
}

async function main() {
  const barFill = $("bar-fill");
  const { fly, skeletons } = await loadAll((loaded, total) => {
    barFill.style.width = `${Math.min(100, (100 * loaded) / Math.max(total, 1))}%`;
  });
  const scene = new FlyScene($("stage"), skeletons, fly.C);
  $("loading").classList.add("done");

  const nPn = skeletons.meta.neurons.filter((n) => n.role === "PN").length;
  const receptorOf = new Int32Array(fly.K);
  fly.perm.forEach((g, r) => (receptorOf[g] = r));
  let variant: VariantName = "real";
  let current: Current | null = null;

  // ---------- smelling ----------
  function run(title: string, desc: string) {
    if (!title.trim() && !desc.trim()) return;
    const smell = fly.smell(title, desc, variant);
    current = { title, desc, smell };
    scene.smell(smell.pn, smell.active, smell.scores);
    $("result").hidden = false;
    $("pipeline").innerHTML =
      `${fly.K} glomeruli → ${nPn} projection neurons → ${fly.nKc.toLocaleString()} Kenyon cells, ` +
      `<b>${smell.active.length} firing</b> (APL keeps ${((100 * smell.active.length) / fly.nKc).toFixed(1)}%) → ` +
      `${fly.C} technique output pairs` +
      (variant === "scrambled" ? ` · <b>scrambled wiring</b>` : "");
    renderGuesses(smell.scores);
    renderSimilar(smell);
    $("train-note").textContent = "";
    $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderGuesses(scores: Float64Array) {
    const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a] || a - b);
    const lo = scores[order[order.length - 1]];
    const hi = scores[order[0]];
    $("guesses").innerHTML = order
      .slice(0, 6)
      .map((c, i) => {
        const w = 8 + 92 * ((scores[c] - lo) / (hi - lo || 1));
        return `<li class="${i >= 3 ? "minor" : ""}"><span class="rank">${i + 1}</span>
          <span class="name"><i style="width:${w}%"></i>${esc(fly.meta.techniques[c])}</span>
          <span class="val">${scores[c].toFixed(2)}</span></li>`;
      })
      .join("");
  }

  function renderSimilar(smell: Smell) {
    $("similar").innerHTML = fly
      .similar(smell.active, 5)
      .map(
        ({ problem: p, jaccard }) =>
          `<li><a href="https://leetcode.com/problems/${encodeURIComponent(p.slug)}/" target="_blank" rel="noopener">${esc(
            titleCase(p.title),
          )}</a> <span class="muted">· ${fmtPct(jaccard)} shared KCs</span>
          <span class="tags">${esc(p.difficulty)} · ${esc(p.tags.join(", "))}</span></li>`,
      )
      .join("");
  }

  // ---------- training with dopamine ----------
  function train(correct: boolean) {
    if (!current) return;
    const s = current.smell;
    const guess = [...s.scores.keys()].sort((a, b) => s.scores[b] - s.scores[a] || a - b)[0];
    fly.reward(s.active, variant, guess, correct);
    scene.dopamine(correct ? "PAM" : "PPL1");
    const scores = fly.score(s.active, variant);
    current.smell = { ...s, scores };
    renderGuesses(scores);
    const tech = fly.meta.techniques[guess];
    $("train-note").textContent = correct
      ? `Sugar: reward neurons (PAM) released dopamine in the "${tech}" compartment. The synapses from the ${s.active.length} firing Kenyon cells onto its "avoid" output weakened, so smells like this now pull harder towards ${tech}.`
      : `Shock: punishment neurons (PPL1) released dopamine in the "${tech}" compartment. The synapses from the ${s.active.length} firing Kenyon cells onto its "approach" output weakened, so smells like this now pull away from ${tech}.`;
  }

  // ---------- controls ----------
  const titleEl = $<HTMLInputElement>("title");
  const descEl = $<HTMLTextAreaElement>("desc");
  $("examples").innerHTML = examples.map((e, i) => `<button data-i="${i}">${esc(e.short)}</button>`).join("");
  $("examples").addEventListener("click", (e) => {
    const i = (e.target as HTMLElement).dataset.i;
    if (i === undefined) return;
    const ex = examples[Number(i)];
    titleEl.value = ex.title;
    descEl.value = ex.description;
    run(ex.title, ex.description);
  });
  $("smell").addEventListener("click", () => run(titleEl.value, descEl.value));
  descEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(titleEl.value, descEl.value);
  });
  $("sugar").addEventListener("click", () => train(true));
  $("shock").addEventListener("click", () => train(false));
  $("reset").addEventListener("click", () => {
    fly.resetTraining(variant);
    if (current) run(current.title, current.desc);
    $("train-note").textContent = "Training forgotten: back to what it learned from LeetCode.";
  });

  const setVariant = (v: VariantName) => {
    variant = v;
    $("wire-real").classList.toggle("on", v === "real");
    $("wire-scrambled").classList.toggle("on", v === "scrambled");
    if (current) run(current.title, current.desc);
  };
  $("wire-real").addEventListener("click", () => setVariant("real"));
  $("wire-scrambled").addEventListener("click", () => setVariant("scrambled"));

  let resting = true;
  $("toggle-resting").addEventListener("click", () => {
    resting = !resting;
    scene.setShowResting(resting);
    $("toggle-resting").textContent = resting ? "Hide resting" : "Show resting";
  });

  const about = $<HTMLDialogElement>("about");
  $("about-open").addEventListener("click", () => about.showModal());
  $("about-close").addEventListener("click", () => about.close());

  // ---------- static panels ----------
  renderScorecard(fly);
  const narrow = window.matchMedia("(max-width: 820px)");
  const placeScorecard = () => (narrow.matches ? $("panel") : document.body).appendChild($("scorecard"));
  narrow.addEventListener("change", placeScorecard);
  placeScorecard();
  const legend: [string, number][] = [
    ["Projection neurons", 1],
    ["Kenyon cells", 0],
    ["APL (inhibition)", 5],
    ["Output neurons", 2],
    ["PAM (reward)", 3],
    ["PPL1 (punishment)", 4],
  ];
  $("legend").innerHTML = legend.map(([n, r]) => `<span style="--c:${ROLE_COLORS[r]}">${n}</span>`).join("");

  const tip = $("tooltip");
  scene.onHoverGlomerulus = (g, x, y) => {
    if (g === null) {
      tip.hidden = true;
      return;
    }
    const r = receptorOf[g];
    const rate = current ? current.smell.pn[g] : null;
    tip.innerHTML =
      `Glomerulus <b>${esc(fly.meta.glomeruli[g])}</b><br>` +
      `Receptor ${r} plugged in here smells: <i>${esc(fly.meta.receptor_terms[r].join(", "))}</i>` +
      (rate !== null ? `<br>Projection-neuron rate: ${rate.toFixed(2)}` : "");
    tip.style.left = `${x + 14}px`;
    tip.style.top = `${y + 14}px`;
    tip.hidden = false;
  };

  // start with an example so the brain is alive on arrival
  const first = examples[0];
  titleEl.value = first.title;
  descEl.value = first.description;
  setTimeout(() => run(first.title, first.description), 700);
}

function renderScorecard(fly: Fly) {
  const s = fly.meta.scores_on_future_problems;
  const rows: [string, string, boolean][] = [
    ["This fly (real wiring)", "fly (real wiring)", true],
    ["Same fly, scrambled wiring", "scrambled wiring (mean of 20)", false],
    ["Always guess the most common", "B0 frequency", false],
    ["Classifier on the same smell", "B1 LR on same 51 channels", false],
    ["Classifier reading every word", "B2 LR on full TF-IDF", false],
  ];
  $("scores").innerHTML =
    `<tr><td></td><td class="num muted">top-1</td><td class="num muted">top-3</td></tr>` +
    rows
      .filter(([, key]) => s[key])
      .map(
        ([label, key, isFly]) =>
          `<tr class="${isFly ? "fly" : ""}"><td>${label}</td><td class="num">${fmtPct(s[key]["hit@1"])}</td>` +
          `<td class="num">${fmtPct(s[key]["hit@3"])}</td></tr>`,
      )
      .join("");
}

function titleCase(s: string) {
  return s.replace(/\b[a-z]/g, (c) => c.toUpperCase());
}

main().catch((err) => {
  console.error(err);
  document.querySelector(".loading-inner p")!.textContent = `Could not start: ${err.message}`;
});
