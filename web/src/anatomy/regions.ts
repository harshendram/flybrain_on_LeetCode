// Ito et al. 2014 neuropil names, one entry per stem. Left and right meshes share the text.
export interface RegionInfo {
  name: string;
  title: string;
  system: string;
  color: string;
  about: string;
  side: "" | "left" | "right";
}

const SYSTEMS: Record<string, string> = {
  "optic lobe": "#7eb6ff",
  "antennal lobe": "#ffb35c",
  "mushroom body": "#66f0ff",
  "lateral horn": "#f0a3ff",
  "central complex": "#ffd166",
  "lateral complex": "#ffe08a",
  "superior neuropils": "#c4b5fd",
  "inferior neuropils": "#93c5fd",
  "ventromedial neuropils": "#86efac",
  "ventrolateral neuropils": "#6ee7b7",
  "periesophageal neuropils": "#fda4af",
  "gnathal ganglia": "#fb7185",
  ocellar: "#e2e8f0",
};

interface Stem {
  system: keyof typeof SYSTEMS;
  title: string;
  about: string;
}

const STEMS: Record<string, Stem> = {
  LA: { system: "optic lobe", title: "lamina", about: "The first optic neuropil. Photoreceptors end here." },
  ME: { system: "optic lobe", title: "medulla", about: "The second optic neuropil, where motion and colour start to separate." },
  LO: { system: "optic lobe", title: "lobula", about: "A deep optic neuropil that detects visual features." },
  LOP: { system: "optic lobe", title: "lobula plate", about: "Detects wide-field motion, the signal the fly uses to steer." },
  AME: { system: "optic lobe", title: "accessory medulla", about: "A small optic neuropil tied to the circadian clock." },
  AL: {
    system: "antennal lobe",
    title: "antennal lobe",
    about: "The fly's smell centre: one glomerulus per receptor type. LeetFly's 51 channels enter the brain here.",
  },
  MB_CA: {
    system: "mushroom body",
    title: "mushroom body calyx",
    about: "Where projection neurons meet Kenyon cells. LeetFly's real PN→KC synapses live in this neuropil.",
  },
  MB_PED: {
    system: "mushroom body",
    title: "mushroom body pedunculus",
    about: "The stalk that carries Kenyon-cell axons out of the calyx toward the lobes.",
  },
  MB_VL: {
    system: "mushroom body",
    title: "mushroom body vertical lobe",
    about: "An output lobe. Dopamine here changes which smells the fly will approach.",
  },
  MB_ML: {
    system: "mushroom body",
    title: "mushroom body medial lobe",
    about: "The other output lobe. LeetFly's technique votes are a model of these approach and avoid outputs.",
  },
  LH: {
    system: "lateral horn",
    title: "lateral horn",
    about: "Innate smell, parallel to the mushroom body. LeetFly does not train this pathway.",
  },
  FB: { system: "central complex", title: "fan-shaped body", about: "A central-complex neuropil used in heading and steering." },
  EB: { system: "central complex", title: "ellipsoid body", about: "Carries a heading compass: a bump of activity that tracks which way the fly faces." },
  PB: { system: "central complex", title: "protocerebral bridge", about: "Links the compass to the left and right turning commands." },
  NO: { system: "central complex", title: "noduli", about: "A small central-complex neuropil involved in angular and translational speed." },
  BU: { system: "lateral complex", title: "bulb", about: "Relays the compass into the ellipsoid body." },
  GA: { system: "lateral complex", title: "gall", about: "A lateral-complex neuropil beside the bulb." },
  LAL: { system: "lateral complex", title: "lateral accessory lobe", about: "Sends steering commands from the central complex toward the body." },
  SMP: { system: "superior neuropils", title: "superior medial protocerebrum", about: "A large superior neuropil, mixed with mushroom-body output." },
  SIP: { system: "superior neuropils", title: "superior intermediate protocerebrum", about: "Sits between the superior medial and superior lateral protocerebrum." },
  SLP: { system: "superior neuropils", title: "superior lateral protocerebrum", about: "A superior neuropil above the lateral horn." },
  CRE: { system: "inferior neuropils", title: "crepine", about: "An inferior neuropil wrapped around the mushroom-body pedunculus." },
  SCL: { system: "inferior neuropils", title: "superior clamp", about: "Clamps the mushroom-body pedunculus from above." },
  ICL: { system: "inferior neuropils", title: "inferior clamp", about: "Clamps the mushroom-body pedunculus from below." },
  IB: { system: "inferior neuropils", title: "inferior bridge", about: "A bridge of neuropil under the central complex." },
  ATL: { system: "inferior neuropils", title: "antler", about: "A small neuropil beside the antennal lobe." },
  VES: { system: "ventromedial neuropils", title: "vest", about: "Receives wind and mechanosensory input from the antenna." },
  SPS: { system: "ventromedial neuropils", title: "superior posterior slope", about: "A posterior slope neuropil, part of the ventromedial group." },
  IPS: { system: "ventromedial neuropils", title: "inferior posterior slope", about: "The lower posterior slope." },
  EPA: { system: "ventromedial neuropils", title: "epaulette", about: "A small ventromedial neuropil." },
  AVLP: { system: "ventrolateral neuropils", title: "anterior ventrolateral protocerebrum", about: "Mixes smell and sight on the way to descending neurons." },
  PVLP: { system: "ventrolateral neuropils", title: "posterior ventrolateral protocerebrum", about: "A ventrolateral neuropil behind the anterior one." },
  PLP: { system: "ventrolateral neuropils", title: "posterior lateral protocerebrum", about: "Receives optic-glomerulus input." },
  WED: { system: "ventrolateral neuropils", title: "wedge", about: "A wedge of neuropil between the ventrolateral protocerebrum and the lateral horn." },
  AOTU: { system: "ventrolateral neuropils", title: "anterior optic tubercle", about: "An optic glomerulus: a visual relay into the central brain." },
  AMMC: { system: "periesophageal neuropils", title: "antennal mechanosensory and motor centre", about: "Hears and feels with the antenna, and moves it." },
  GOR: { system: "periesophageal neuropils", title: "gorget", about: "A periesophageal neuropil beside the oesophagus." },
  FLA: { system: "periesophageal neuropils", title: "flange", about: "A periesophageal neuropil." },
  CAN: { system: "periesophageal neuropils", title: "cantle", about: "A periesophageal neuropil." },
  PRW: { system: "periesophageal neuropils", title: "prow", about: "A midline neuropil in front of the oesophagus." },
  SAD: { system: "periesophageal neuropils", title: "saddle", about: "A midline neuropil behind the oesophagus." },
  GNG: { system: "gnathal ganglia", title: "gnathal ganglia", about: "Taste and the motor neurons of the mouthparts." },
  OCG: { system: "ocellar", title: "ocellar ganglion", about: "Where the simple eyes on top of the head, the ocelli, send their axons." },
};

export const SYSTEM_ORDER = Object.keys(SYSTEMS);

/** One info row per neuropil stem, without a left/right suffix. */
export function everyStem(): RegionInfo[] {
  return Object.keys(STEMS).map((stem) => regionInfo(stem));
}

export function regionInfo(name: string): RegionInfo {
  const side: RegionInfo["side"] = name.endsWith("_L") ? "left" : name.endsWith("_R") ? "right" : "";
  const stem = name.replace(/_(L|R)$/, "");
  const row = STEMS[stem];
  if (!row) return { name, title: name, system: "other", color: "#94a3b8", about: "", side };
  const titled = side ? `${row.title}, ${side}` : row.title;
  return { name, title: titled, system: row.system, color: SYSTEMS[row.system], about: row.about, side };
}

export function systemColor(system: string): string {
  return SYSTEMS[system] ?? "#94a3b8";
}

/** The smell tour, on the right hemisphere, in the order a real odour travels. */
export const SMELL_TOUR = ["AL_R", "MB_CA_R", "MB_VL_R", "LH_R"];
