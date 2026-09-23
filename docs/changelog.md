# What changed after the Claude Code session

Handoff for the next agent. Claude Code's last commit on `main` is `994e555` ("Artifact previews for both pages", 23 Sep 2026, 08:35 IST). That session then died mid-plan, having edited some fly-page files in the working tree and not finished the anatomy page. Everything below is Cursor work after that commit.

Do not re-run Phase 1 or Phase 2. Those results are committed and the AWS instance that produced them was deleted. Do not start a second Vite: port 5173 is already the dev server. PowerShell on this machine does not accept `&&`.

## 1. Commit `3870c82` (pushed to `origin/main`)

Message: "Let visitors fly the fly, and open the female brain region by region."

This is the approved plan "fly it yourself, a real-anatomy brain page, and a real local website." `994e555..3870c82` is what Vercel rebuilds from.

### Local site, not artifacts

`README.md` no longer opens with claude.ai artifact URLs. It leads with:

- http://localhost:5173/ learning brain
- http://localhost:5173/anatomy.html
- http://localhost:5173/fly.html

`scripts/build_artifact.py` is still in the tree and is not used. Hosting stays Vercel, root directory `web`. `web/vite.config.ts` now lists `anatomy.html` next to `index.html` and `fly.html`, so `npm run build` emits all three. Dev server serves any HTML file even without that list. Nav hrefs are `./index.html`, `./anatomy.html`, `./fly.html` because Vite `base` is `"./"`. Vercel `cleanUrls` also serves `/fly` and `/anatomy`; the local dev server does not.

Shared nav "Learning brain · Anatomy · Fly" is on all three pages. On the brain page it replaced only the "See the fly fly" link. Smell, Watch it learn, Evolve a nose, Nose transplant, and Real vs Scrambled stay. On the fly page it replaced "Inside its brain". Below 820px `.site-nav` wraps to its own row in `web/src/style.css` so the brain top bar does not overflow. The same rules are duplicated in `web/src/fly-page/fly.css` because that page loads both stylesheets.

### Fly page: the visitor flies

`web/src/fly-page/main.ts` no longer auto-starts example 0 (Ways Up the Staircase) after 900ms. The fly stands on the centre perch. Caption: pick a problem, then let its brain fly, or take the controls.

`web/src/fly-page/pilot.ts` is new. `stepFlight` is pure. Heading 0 faces +x. Motion is `x += cos(heading) * speed * dt`, `z += -sin(heading) * speed * dt`. Yaw rate 2.1 rad/s, accel 80, brake 160, max speed 95. W holds altitude (`wantVy = 0` while thrust > 0.2 and climb is 0). Space climbs, Shift or C descends, letting go of W sinks at 22. Takeoff if grounded and thrust or climb exceeds 0.2. Landing fires once, on the frame the fly settles onto the highest surface under it, or the floor. Wall is `WALL_R = 400` (the floor edge, not the feeder ring at 230). Cylinder collision only applies when the feet are below the stand top, so the fly can pass over a feeder.

Keys: W/Up thrust, S/Down brake, A/D and arrows yaw, Space up, Shift/C down. Ignored while an input or textarea is focused. Any flight key or a deflected joystick switches the segmented control to You. `web/src/fly-page/joystick.ts` is the bottom-left stick. Its pointer handlers call `stopPropagation` so the stick does not orbit the camera. Desktop CSS puts the stick at `left: 420px` so it clears the 380px panel. On a phone it sits above the bottom sheet.

Its brain still surge/casts from the current position to the top untried technique. Landing on a feeder while a problem is active is the answer: `fly.reward` sugar/PAM if right, shock/PPL1 if wrong. Score line is "You: x/y · its brain: x/y". A wrong guess that is not finished can be scored again on the next landing. A finished problem ignores later touchdowns. Pasted problems have `truth === null` and open the sugar/shock buttons instead of auto-verdict. The brain's top untried feeder ring pulses via `Feeder.instinct`. `prefers-reduced-motion` still kills auto-rotate and the shock jolt. Keys and the stick still work.

Camera: there is no `follow = false` latch. Every frame the orbit target is the fly, the camera is placed at target + `camOffset`, `controls.update()` runs inside `arena.render`, then `camOffset` is read back from the camera. A drag or a zoom therefore sticks. Desktop `setViewOffset` of 190px to the left is unchanged (`web/src/fly-page/arena.ts`) because the panel covers the left. Idle orbit starts at camera `(55, 90, 160)`, target `(0, PERCH_Y + 10, 0)`.

`web/src/fly-page/flybody.ts`: wingbeat is `this.t * 17 * (0.45 + 0.55 * this.beatScale)`. You-mode sets `beatScale` from thrust. Brain mode forces `beatScale = 1`. While `who === "you"`, pitch and bank come from `stepFlight` and are not overwritten by the brain's hover pose.

`arena.ts` exports `STAND_H`, `DISH_R`, `PERCH_TOP_R`, `PERCH_BASE_R`. `Arena.surfaces()` returns the perch plus one cylinder per feeder for the pilot. `web/tests/pilot.test.ts` covers takeoff, feeder landing, floor landing, sliding off a stand, and the wall. W alone must stay airborne and travel, which is why sink is suppressed during thrust.

`window.__fly()` is set every frame for the headless checker: position, who, phase, grounded, camera offset.

Checked in Chrome on the Iris Xe, 23 Sep 2026: idle on the perch, who=brain, phase=idle. W switches to You and leaves the perch. A scripted flight (turn with D, climb with Space, thrust with W, descend with C) landed on a feeder and scored You 1/1 with the sugar caption. Dragging the canvas changed `cam`. Phone 390×844 also took off on W.

### Anatomy page

`web/anatomy.html` plus `web/src/anatomy/`. Same `.loading` / `.loading.done` shell the screenshot script waits for.

`scripts/export_anatomy.py` packs the female FlyWire brain. It does not import fafbseg (GPL). It reads the already-downloaded zip `data/raw/flywire/JFRC2NP.surf.fw.zip` (Ito et al. 2014 surfaces in FlyWire nm). PLY is binary little-endian, xyz float32, faces `list int int` (count is int32, not uchar). Display transform is `/1000` then `x *= -1`, and the face winding is flipped. Meshes with fewer than 4000 faces get one Loop subdivision. That produced 78 region meshes.

The outline is not at `brain_mesh_v3:0` (that URL 404s). It is the directory prefix `https://storage.googleapis.com/flywire_neuropil_meshes/whole_neuropil/brain_mesh_v3/mesh/` then object `1:0` (a JSON fragment list) then fragment `1:0:0`. Coordinates are nm. The fragment is clustered from voxel 2.0 µm upward until it is under 30,000 triangles. Shipped outline is 21,901 triangles. The lamina and the ocelli stick out of that shell by up to about 105 µm. That is the mesh, not a bad transform. `tests/test_anatomy_export.py` allows 120 µm.

Stats come from `proofread_connections_783.feather`. The first export died on `Series.sort_values("syn")` because a one-column groupby is a Series. The fix sorts with `ascending=False` and no column name. Dominant transmitter is the synapse-weighted winner of gaba, ach, glut, oct, ser, da. Neuron count is the number of distinct pre or post roots in that neuropil. Top types are up to 4 distinct `cell_type` values, falling back to `super_class`. Stats are cached at `data/raw/flywire/anatomy_stats.json` (gitignored with the rest of `data/raw`).

Featured skeletons: those neurons, fetched from `https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/{body_id}`, pruned at 8 µm, rerooted at the soma, downsampled at 4 µm. They are not indexed by the learning-circuit `STEP` table. Packed as `web/public/data/anatomy_neurons.bin` and `.json` (271 neurons, 394,954 nodes, about 3.8 MB). `role` in that JSON is the FlyWire `super_class` string, not PN/KC/MBON. Each neuron has a `regions` list.

Surfaces: `web/public/data/anatomy.bin` (5.3 MB) and `anatomy.json`. uint16 xyz, int8 normals, uint32 indices. Scale 64. `n_vertices` is 166,877, so the normal block (`n * 9` bytes) is not 4-byte aligned. `web/src/anatomy/atlas.ts` copies the index bytes into a fresh `Uint32Array` instead of viewing them in place. Region triangles in the file: 312,032. Midline names with no `_L`/`_R`, kept visible when a side is hidden: EB, FB, NO, PB, GNG, OCG, PRW, SAD. After the x mirror, anatomical left is negative x (`AL_L` sits around x = −523 to −410). The soma cloud is the existing `flywire_cloud` (118,104 points, five superclass bins, no per-neuropil byte). Left/right soma toggle uses `PointCloud.setXRange` in `web/src/scene.ts` (`uXMin` / `uXMax` discard). Do not re-export that cloud.

`atlas.ts` draws every region as one mesh, one shader, per-vertex colour and alpha. 78 separate transparent draws plus a per-frame raycast was about 20 fps. One draw, raycast only on pointer move, soma LOD 2, and `antialias: false` measured about 60 fps overview and 57 fps with a region selected, on the Iris Xe at 1440×900. FrontSide. Hidden sides are dropped from the index, not discarded in the shader. Click fades the rest by rewriting per-vertex alpha. Hover label uses the same raycast. Esc or a miss clears the selection.

`web/src/anatomy/regions.ts` has 43 stems (system, title, colour, one line). Left and right share the text. `everyStem()` is what `web/tests/regions.test.ts` checks. AL, MB calyx, MB lobes, and LH mention LeetFly. Tour order is `AL_R`, `MB_CA_R`, `MB_VL_R`, `LH_R`. Starting the tour while a side filter is on resets the filter to both, otherwise the toured mesh is not in the index. The info card for `MB_VL_R` read 113,967 synapses, 2,824 neurons, acetylcholine.

`web/src/anatomy/tubes.ts` draws the selected region's featured neurons as camera-facing quads with cylinder shading. Spikes move by cable distance. Precomputed skeletons have no radius, so width is a stand-in, thicker near the soma. Neurons over 6,000 nodes draw every other edge.

HUD: canvas gizmo. Anterior is the dominant axis from calyx centroid to antennal-lobe centroid. Dorsal is +z unless that axis is z, in which case dorsal is +y. Left is −x. `scaleBar` from `web/src/hud.ts`.

The neuropil string `UNASGD` is in the connection table and has no JFRC2 mesh. The test warns and asserts it was not turned into a surface. AppleDouble entries `._AL_L.ply` inside the zip are skipped.

`web/scripts/shot.mjs` and `web/scripts/og.mjs` are the headless Chrome checkers (puppeteer-core is a devDependency; Chrome is `C:/Program Files/Google/Chrome/Application/chrome.exe` with `--use-angle=d3d11`). `shot.mjs` actions include `key:KeyW/700`, `drag:160/40`, `state:__fly`, `text:#caption`, `fps`. `web/public/og-anatomy.png` is the 1200×630 card.

`window.__anatomy()` returns `{ selected, side }`.

Pytest `tests/test_anatomy_export.py`: 78 zip meshes, every mesh name is a real neuropil column value, boxes inside the outline plus 120 µm, positive synapses and neuron counts, featured skeletons exist, cable distance is non-decreasing from the root within 1 µm of the quantized step. Full `pytest` was 19 passed. `npm test` was 37 passed. `npm run build` emits `dist/anatomy.html`.

## 2. Not committed yet: the neurons the claim is about

Started 23 Sep 2026 after the user sent screenshots of the learning brain and said it looks hollow. A first plan to add 12,000 random background neurons was rejected. Those numbers were invented, and a random optic-lobe fill would hide the actual claim (evolve only the nose, then put that nose in a different fly). The optic lobes stay a point cloud. There are 90,810 of those somas.

The hollow look has two causes, both in code that `994e555` already shipped:

- `web/src/scene.ts` drew the 2,208 right-mushroom-body neurons at resting opacity 0.025–0.07. The screenshot's own counter said 47 Kenyon cells were firing, so the loaded circuit was invisible.
- The page only loads `malecns_R`. `extract_malecns` already builds `malecns_L` (`data/processed/mb_malecns_L_syn5.npz`, gitignored). The female brain is constructed in `web/src/phase2.ts` and was `visible = false` until transplant mode.

### Query, not a quota

`src/leetfly/connectome/context.py` `context_specs()` returns the set. No target count.

- Left mushroom body: Kenyon cells, the projection neurons that reach them, and that side's MBON / PAM / PPL1 / APL, same filters as `build_circuit` (synapse weight at least 5). Ids already in `web/public/data/malecns_R_skeletons.json` are dropped. Result: 2,179.
- Lateral horn: annotation `type` matches `^LH`. There are 2,028, all `superclass` `cb_intrinsic`. They are included because the column names them. They are the pathway the model does not train.
- One synapse away: body ids in `connectome-weights-male-cns-v1.0-minconf-0.5.feather` that touch the right-circuit ids or the left-circuit ids, are in the annotation table, and are not optic-lobe, descending/ascending, or VNC (`_padding`). Result: 16,269 partners.

Total 20,476. The finished export printed `{'partner': 16269, 'left': 2179, 'lh': 2028}`.

### First pack finished, then a coarser re-pack started

`scripts/export_context.py` finished once, exit 0, at 2026-09-23T05:58:43Z (about 49 minutes). It downloaded every MaleCNS SWC from the URL already in `scripts/export_skeletons.py`, rotated with `data/processed/malecns_em_to_display.json`, rerooted at the soma, and downsampled with the circuit steps for known roles (PAM and PPL1 at 4 µm, not collapsed to one DAN colour) and at 12 µm for unnamed partners. Cache: `data/raw/malecns/skeletons_em/`. Result of that run: **20,476 neurons, 0 missing, 7,196,028 nodes, 72.0 MB** written to `web/public/data/malecns_context.bin` and `.json`.

That geometry is too heavy for the Iris Xe page (about 7.2 million line vertices, plus a float position and four float attributes per vertex). `src/leetfly/connectome/context.py` was then changed and a second `scripts/export_context.py` was started from the SWC cache (no new downloads). Do not start another copy. Current steps in that file: partners 36 µm, KC / PN / PAM / PPL1 6 µm, MBON / APL 5 µm, lateral horn 10 µm. Terminal twigs shorter than 10 µm are dropped for lateral-horn and partner neurons before sampling. The 72 MB files stay on disk until that re-pack's final `skeletons.pack` overwrites them.

`tests/test_context_export.py` checks ids, not sampling. It recomputes `context_specs()` and asserts packed ids plus `missing` equal the query, none of the packed ids are already in the right circuit, and no packed id has an optic-lobe, descending, or VNC superclass. It has not been run yet. Run it after the re-pack exits, not while two copies are writing the same JSON.

### Page code already edited, waiting on the file

Uncommitted:

- `web/src/scene.ts`. Resting opacities are now KC 0.11, PN 0.14, MBON 0.12, PAM 0.08, PPL1 0.12, APL 0.10. Peak opacities are unchanged. `Brain.radius` is no longer `readonly`. `addContext` draws one `LineSegments` in the male brain's group, through the same `toScene` as the circuit, and expands `radius` to the farthest context node so the camera can frame both lobes. Kind 2 is the left mushroom body (palette colour, alpha 0.20), kind 1 is the lateral horn (alpha 0.22), kind 0 is a partner (alpha 0.07). They do not use the 64×64 spike texture. Circuit lines have `renderOrder = 1`.
- `web/src/data.ts` `loadContext`. Packed parents are uint16 and local to each neuron. The loader rewrites them to absolute uint32 indexes. Root sentinel in that array is `0xffffffff`, not 65535, because a real node can sit at index 65535 once the pack exceeds 65,536 nodes.
- `web/src/main.ts` loads the context in parallel with the model and calls `brain.addContext` before the cloud. The page will throw on fetch if `malecns_context.json` is not there yet.
- `web/src/phase2.ts` `enterEvolve` no longer hides the female. She stays visible with her born nose, and the camera frames both flies. Transplant mode was already both flies. `exit()` still hides her and frames the male.

Not done, and blocked on the coarser re-pack: browser check that the mushroom body reads as a lobe when almost nothing is firing, that partner arbors sit around it, that the optic lobes are still dots, and that evolve and transplant show two brains. Do not refresh the learning brain against the 72 MB pack. `npx tsc --noEmit` in `web/` passed after the first context edits (an earlier run failed because `Brain.radius` was still `readonly`; that is fixed). Pytest for the context pack has not been run.

## 3. Facts the next agent should not rediscover

- MaleCNS right mushroom body on the page: 2,208 neurons, 576,801 nodes (`malecns_R_skeletons.json`). Roles: 1,880 KC, 166 DAN, 112 PN, 49 MBON, 1 APL.
- Soma cloud: 141,781. Region counts in `malecns_cloud.json`: optic lobe 90,810, central brain 32,172, descending/ascending 3,132, VNC 13,547, other 2,120.
- Activity texture is 64×64. Do not put the 20,476 context neurons into it.
- FlyWire connection table has a column `neuropil`. The only value with no JFRC2 mesh is `UNASGD`.
- Display frame for MaleCNS is the saved Procrustes rotation in `data/processed/malecns_em_to_display.json` (EM voxels are 8 nm). FlyWire display is nm to µm, or voxels `[0.004, 0.004, 0.040]`, then `x *= -1`.
- `scripts/` is not a Python package. Do not `from scripts.... import`.
- Machine: i5-12500H, Iris Xe, 16 GB, often little free RAM. Anatomy overview was ~60 fps after the single-mesh change. The fly page during flight was about 35 fps. That is the flybody model, not the anatomy meshes.
- `web/dist/` is gitignored. `data/raw/` and `data/processed/` are gitignored. The site binaries under `web/public/data/` are committed.

## 4. Claude Code, 23 Sep 2026 (after the Cursor handoff): the context pack shipped, and it moves

- `scripts/export_context.py` finished: 20,476 neurons, 0 missing, 7.2M nodes, **72 MB**. Too heavy for a page, so
  `src/leetfly/connectome/context.py` now samples coarser (partners 36 µm, LH 10 µm, left-MB roles 5–6 µm), and
  `TWIG_UM` prunes terminal twigs < 10 µm on partner/LH cells before sampling. Re-packed from the SWC cache (no
  downloads): 1,809,907 nodes, **18.1 MB**. `tests/test_context_export.py` passes.
- Resting look: 1.8M overlapping segments at Cursor's alphas (0.07–0.22) saturated to a white blob under bloom.
  Context alphas are now left MB 0.07, LH 0.05, partners 0.016; partners draw every other segment (halves fill).
  The camera frames the 95th-percentile context node instead of the farthest axon. Brain-page bloom 0.8 → 0.55.
- Alive: `addContext` passes `aDist` (cable distance from the pack's dist block) and a per-neuron `aSeed`.
  Every context neuron fires spontaneously on a seeded period (≈2–5 s at rest, ≈0.7–2 s while busy); the spike
  runs out from the soma at the circuit's PULSE_SPEED. `Brain.smell()` / `dopamine()` set `busyUntil`, and
  `uActivity` eases to 1 while the circuit fires, so partners and the left MB flicker harder, then settle.
  GPU only; the 64×64 spike texture is still circuit-only. `prefers-reduced-motion` freezes it.
- Measured on the Iris Xe at 1440×900 in headless Chrome: learning brain 52 fps, evolve mode shows both flies
  (259,885 neurons), anatomy 60 fps, fly 60 fps.

## 5. Phase 3 (branch `research/embodied`, merged 23 Sep 2026): the brain flies a physics fly

- `src/leetfly/embodied/`:
  - `brain.py`: online MB with bandit dopamine (one compartment per landing); full feedback equals the closed form.
  - `plan.py`: turn-rate-limited pursuit, 11 quantized cast levels.
  - `body.py`: flybody in MuJoCo with the pretrained flight policy, loaded without acme/reverb.
  - `physics.py`: one flight from the centre to a 6 cm ring.
  - `bank.py`: the parallel flight bank.
  - `run.py`: streams, with PerfectBody / BankBody / live.
- AWS runner: `scripts/aws_embodied.py` (Python 3.10 + TF 2.8; tfp-nightly removed; figshare zips shipped in the
  payload; m7a.2xlarge for 8 workers).
- Bank: 2,702 real flights in 68 min on 8 cores. 70 of 2,772 were lost when a partial download overwrote the final
  one before the S3 copy was deleted.
- Results (README "Phase 3 — results"): B1 −4.1 points (narrowly not supported); B2 not supported; E1 100%; E2 0.000;
  E3 not measurable (all flights 0.36 s).
- Web: fly page "Physics" mode (`web/src/fly-page/physics-replay.ts`), 19 real flights in
  `web/public/data/embodied_flights.json` (`scripts/export_embodied.py`).
