import "./style.css";
import { loadAll, loadPhase2 } from "./data";
import examples from "./examples.json";
import type { Fly, Smell, VariantName } from "./fly";
import { Phase2, TECH_COLORS, renderCurve, type FemaleNose } from "./phase2";
import { Brain, FlyScene, ROLE_COLORS } from "./scene";

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
  const scene = new FlyScene($("stage"));
  const brain = new Brain(skeletons, fly.C, scene.now);
  scene.add(brain, 0);
  scene.focus([brain], true);
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
    brain.smell(smell.pn, smell.active, smell.scores);
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
    brain.dopamine(correct ? "PAM" : "PPL1");
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
    for (const b of scene.brains) b.setShowResting(resting);
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
  scene.onHoverGlomerulus = (_brain, g, x, y) => {
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

  // ---------- Phase 2 (evolve + transplant), loaded in the background ----------
  const p2data = await loadPhase2(() => {}).catch((e) => (console.warn("phase 2 unavailable", e), null));
  if (!p2data) return;
  const p2 = new Phase2(fly, scene, brain, p2data.meta, p2data.bin, p2data.female);
  setupPhase2(p2, fly, () => current, (m) => {
    $("mode-smell").hidden = m !== "smell";
    $("result").hidden = m !== "smell" || !current;
    $("wiring").hidden = m !== "smell";
    $("scorecard").hidden = m !== "smell";
    if (m === "smell" && current) run(current.title, current.desc);
  });
}

type Mode = "smell" | "evolve" | "transplant";

function setupPhase2(p2: Phase2, fly: Fly, _current: () => Current | null, onMode: (m: Mode) => void) {
  const meta = p2.meta;
  const sc = meta.scores;
  const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
  let femaleNose: FemaleNose = "born";
  let timer: number | undefined;
  const last = p2.frames() - 1;

  // --- mode tabs ---
  $("modes").hidden = false;
  const setMode = (m: Mode) => {
    for (const b of $("modes").querySelectorAll("button")) b.classList.toggle("on", b.dataset.mode === m);
    $("mode-evolve").hidden = m !== "evolve";
    $("mode-transplant").hidden = m !== "transplant";
    onMode(m);
    if (m === "evolve") {
      p2.enterEvolve();
      showFrame(last);
    } else if (m === "transplant") {
      stop();
      p2.enterTransplant(femaleNose);
      renderTransplant();
    } else {
      stop();
      p2.exit();
    }
  };
  $("modes").addEventListener("click", (e) => {
    const m = (e.target as HTMLElement).dataset.mode as Mode | undefined;
    if (m) setMode(m);
  });

  // --- evolve: replay ---
  const gen = $<HTMLInputElement>("gen");
  gen.max = String(last);
  gen.value = String(last);
  const showFrame = (f: number) => {
    gen.value = String(f);
    p2.showGeneration(f);
    const g = meta.replay.generations[f];
    $("gen-label").textContent = `generation ${g}`;
    renderCurve($<SVGSVGElement & HTMLElement>("curve"), meta.replay.best_fitness, g);
  };
  const stop = () => {
    if (timer) clearInterval(timer);
    timer = undefined;
    $("play").textContent = "Play";
  };
  gen.addEventListener("input", () => (stop(), showFrame(Number(gen.value))));
  $("play").addEventListener("click", () => {
    if (timer) return stop();
    let f = Number(gen.value) >= last ? 0 : Number(gen.value);
    $("play").textContent = "Pause";
    timer = window.setInterval(() => {
      showFrame(f);
      if (++f > last) stop();
    }, 140);
  });
  $("tech-legend").innerHTML = fly.meta.techniques
    .map((t, i) => `<span style="--c:${TECH_COLORS[i]}">${esc(t)}</span>`)
    .join("");
  const born = sc["male:born"];
  const evo = sc["male:evolved"];
  $("evolve-scores").innerHTML =
    `<tr><th></th><th>past problems<br>(AUROC)</th><th>future problems<br>(AUROC)</th><th>future<br>top-1</th></tr>` +
    [["Born nose (random)", born], ["Evolved nose", evo]]
      .map(([label, s]: any) => `<tr><td>${label}</td><td>${s.dev_auroc.toFixed(3)}</td><td>${s.test_auroc.toFixed(3)}</td><td>${pct(s.test_hit1)}</td></tr>`)
      .join("");
  const devGain = evo.dev_auroc - born.dev_auroc;
  const testGain = evo.test_auroc - born.test_auroc;
  $("evolve-note").innerHTML =
    `On the past problems it evolved for, the new nose lifts learning by <b>${devGain >= 0 ? "+" : ""}${devGain.toFixed(3)}</b> AUROC. ` +
    (testGain > 0.005
      ? `On newer problems it keeps <b>${testGain >= 0 ? "+" : ""}${testGain.toFixed(3)}</b>.`
      : `On newer problems that edge <b>does not survive</b> (${testGain >= 0 ? "+" : ""}${testGain.toFixed(3)}): the niche drifted.`);

  // --- transplant ---
  const renderTransplant = () => {
    const rows: [FemaleNose, string][] = [["born", "Born nose (random)"], ["evolved", "Her own evolved nose"], ["transplant", "His evolved nose"]];
    $("transplant-scores").innerHTML =
      `<tr><th>female fly with…</th><th>past (AUROC)</th><th>future (AUROC)</th><th>future top-1</th></tr>` +
      rows
        .map(([k, label]) => {
          const s = sc[`female:${k}`];
          return `<tr class="${k === femaleNose ? "on" : ""}"><td>${label}</td><td>${s.dev_auroc.toFixed(3)}</td><td>${s.test_auroc.toFixed(3)}</td><td>${pct(s.test_hit1)}</td></tr>`;
        })
        .join("");
    const maleGain = sc["male:evolved"].test_auroc - sc["male:born"].test_auroc;
    const caveat =
      maleGain <= 0
        ? `<p class="note">Heads-up: this particular male fly is the one case where evolution's gain did <b>not</b> ` +
          `carry over to newer problems (${maleGain >= 0 ? "+" : ""}${maleGain.toFixed(3)}), so his nose is a weak donor. ` +
          `The pattern across all flies is below.</p>`
        : "";
    $("transplant-summary").innerHTML = caveat + summaryHtml(meta);
    for (const b of $("female-nose").querySelectorAll("button")) b.classList.toggle("on", b.dataset.nose === femaleNose);
  };
  $("female-nose").addEventListener("click", (e) => {
    const n = (e.target as HTMLElement).dataset.nose as FemaleNose | undefined;
    if (!n) return;
    femaleNose = n;
    p2.setFemaleNose(n);
    renderTransplant();
    if (lastBoth) smellBoth(lastBoth.title, lastBoth.description);
  });
  let lastBoth: { title: string; description: string } | null = null;
  const smellBoth = (title: string, description: string) => {
    lastBoth = { title, description };
    const { male, female } = p2.smellBoth(title, description, femaleNose);
    const top = (s: Smell) => s.ranking.slice(0, 3).map((c) => `<li>${esc(fly.meta.techniques[c])}</li>`).join("");
    $("both-guesses").innerHTML =
      `<div><h4>Male (his evolved nose)</h4><ol>${top(male)}</ol></div>` +
      `<div><h4>Female (${femaleNose === "transplant" ? "his nose" : femaleNose === "evolved" ? "her evolved nose" : "born nose"})</h4><ol>${top(female)}</ol></div>`;
  };
  $("examples2").innerHTML = examples.map((e, i) => `<button data-i="${i}">${esc(e.short)}</button>`).join("");
  $("examples2").addEventListener("click", (e) => {
    const i = (e.target as HTMLElement).dataset.i;
    if (i !== undefined) smellBoth(examples[Number(i)].title, examples[Number(i)].description);
  });
}

function summaryHtml(meta: import("./data").Phase2Meta): string {
  const s = meta.summary as Record<string, any>;
  const ex = meta.exploratory;
  if (!s || s.H1 === undefined) return "";
  const pct = (x: number) => `${Math.round(100 * x)}%`;
  const verdict = (ok: boolean) => (ok ? `<span class="yes">supported</span>` : `<span class="no">not supported</span>`);
  const k = s.transplant_gain_by_kind as Record<string, number>;
  return (
    `<h3>The full experiment: 5 mushroom bodies, 3 flies, 75 evolution runs</h3><ul class="findings">` +
    `<li>Evolving only the nose improved learning on <i>future</i> problems in ${s.runs_positive} of ${s.runs_total} runs, ` +
    `but by just +${s.E.real.toFixed(3)} AUROC, and not in every fly. <small>Pre-registered H1 ${verdict(s.H1)}.</small></li>` +
    `<li>Brains are evolvable <b>because some glomeruli reach far more Kenyon cells</b>: real wiring (+${s.E.real.toFixed(4)}) ` +
    `and coverage-preserving scrambles (+${s.E.dp.toFixed(4)}) gain ${(s.E.real / s.E.uniform).toFixed(1)}× more than uniformly scrambled wiring (+${s.E.uniform.toFixed(4)}). <small>H2 ${verdict(s.H2)}.</small></li>` +
    `<li>Evolution rediscovered the trick on its own: it puts the most informative receptors on the most-sampled glomeruli ` +
    `(ρ = +${s.H3_rho.real.toFixed(2)} on real wiring, +${s.H3_rho.uniform.toFixed(2)} when coverage is flattened). <small>H3 supported.</small></li>` +
    `<li>A transplanted nose keeps <b>${pct(s.transplant_ratio_of_means)}</b> of the benefit on future problems (${s.transplants_positive} of ${s.transplants_total} transplants help), ` +
    `just as well across sexes (+${k["cross-sex"].toFixed(4)}) as between females (+${k["same-sex"].toFixed(4)}) or within one fly (+${k["within-fly"].toFixed(4)}). ` +
    `<small>The stricter pre-registered criterion (ratio ≥ 0.7 in every group) was ${s.H4 ? "met" : "not met"}.</small></li>` +
    (ex
      ? `<li>Exploratory, on the problems the noses evolved for: noses evolved on coverage-preserving scrambles transfer as well as real ones ` +
        `(+${ex["cross-sex"].mean_gain_dp_nose.toFixed(3)} vs +${ex["cross-sex"].mean_gain_real_nose.toFixed(3)} across sexes). What evolution can use is the <b>inherited coverage bias</b>, not each fly's own wiring quirks.</li>`
      : "") +
    `</ul>`
  );
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
