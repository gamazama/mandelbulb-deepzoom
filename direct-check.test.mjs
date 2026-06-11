/* =============================================================================
   direct-check.test.mjs — NON-CIRCULAR verification of the perturbation renderer
   Run from this folder:  node direct-check.test.mjs
   -----------------------------------------------------------------------------
   The dive gate compares GPU-HDR vs CPU-f64 of the SAME §3 math — it cannot catch
   a formulation/seam bug both sides share (the slice-5 lesson: the first-order Δθ
   passed three suites). This script compares the GPU perturbed de against referees
   that share NOTHING with the kernel:
     • shallow (≥1e-12): direct f64 triplex loop at the explicit position
       (__directCheck — f64 absolute floor ~1e-16 kills it below ~1e-12), AND
     • all depths: BigInt fixed-point direct march (__directCheckHP — no reference
       orbit, no §3, no HDR, no f64 position; P=256 ≈ 77 digits → valid to 1e-50+).
   In the overlap region the two referees cross-validate each other.
   If placement fails at a requested depth, the GPU is still certified at the
   ACHIEVED depth (read back via the BigInt referee at t=0).
   ============================================================================= */
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import path from 'path';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PAGE = 'file://' + path.join(__dirname, 'index.html').replace(/\\/g, '/');

const DEPTHS = [1e-5, 1e-6, 1e-8, 1e-9, 1e-11, 1e-12, 1e-15, 1e-20, 1e-30, 1e-40, 1e-50];

const browser = await chromium.launch({ channel: 'chrome', headless: false,
  args: ['--use-gl=angle', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--enable-gpu'] });
try {
  const page = await browser.newPage({ viewport: { width: 640, height: 400 } });
  await page.goto(PAGE, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForFunction(() => window.__deepReady !== undefined, null, { timeout: 30000 });
  // BLA=1 enables the GPU skip-table path (lazily compiles the BLA program, ~40 s once).
  // The referee (__directCheckHP, BigInt direct) shares ZERO math with the kernel either way.
  const BLA = process.env.BLA === '1';
  if (BLA) { await page.evaluate(() => window.__setBla(true));
    const e = await page.evaluate(() => window.__blaErr()); if (e) { console.log('BLA COMPILE FAILED:\n' + e); process.exit(1); } }
  console.log('BLA path:', BLA ? 'ON' : 'off');
  let pass = 0, ran = 0;
  for (const depth of DEPTHS) {
    // base 32: +8 margin over escape time so boundary points don't sit at the budget
    // edge (measured at 1e-50: iters=92 flipped the escape flag, 96+ agreed)
    const iters = Math.min(120, 32 + Math.ceil(1.35 * -Math.log10(depth)));
    // placement is ray-specific (thin folds / near-critical parking) — same fallback
    // list as harness-dive.test.mjs; certification itself is the BigInt referee below
    const RAYS = [
      { pos: [0.1, 0.15, 2.6], yaw: -Math.PI / 2, pitch: 0 },
      { pos: [0.4, 0.3, 2.5], yaw: -Math.PI / 2, pitch: 0 },
      { pos: [-0.3, 0.2, 2.6], yaw: -Math.PI / 2, pitch: 0 },
      { pos: [2.6, 0.15, 0.1], yaw: Math.PI, pitch: 0 },
    ];
    let dv = null, rayUsed = -1;
    for (let ri = 0; ri < RAYS.length; ri++) {
      const c = RAYS[ri];
      await page.evaluate(({ p, yaw, pitch, it }) => { window.__setCamHP(p); window.__cam.yaw = yaw; window.__cam.pitch = pitch; window.__cam.iters = it; }, { p: c.pos, yaw: c.yaw, pitch: c.pitch, it: iters });
      dv = await page.evaluate(d => window.__dive(d), depth);
      if (!dv.error && dv.relDE < 1e-3) { rayUsed = ri; break; }
    }
    let placeNote = rayUsed > 0 ? `  [ray ${rayUsed}]` : '';
    if (dv.error) placeNote = `  [dive: ${dv.error} — certifying at ACHIEVED depth]`;
    else if (rayUsed < 0) placeNote = '  [no ray certified clean — using last placement]';
    // achieved depth from the BigInt referee at t=0 (non-circular by construction)
    const probe0 = await page.evaluate(() => window.__directCheckHP([0]));
    const deAch = Math.abs(probe0[0].direct);
    if (!(deAch > 0) || deAch > 1e-2) { console.log(`depth ${depth.toExponential(0).padStart(6)}  SKIP — camera not deep (achieved de=${deAch.toExponential(1)})${placeNote}`); continue; }
    // probes exercise δ≠0 at increasing reach AT THE RENDERING-RELEVANT SCALE: pixels at
    // depth d are ~d apart, so probe at min(deAch, depth) — when the dive parks in a
    // crevice (de at camera ≫ requested depth), deAch-scaled probes would overshoot into
    // fold-crossings that no pixel at this depth ever samples.
    const ps = Math.min(deAch, depth);
    const tProbe = [0, ps * 1e-3, ps * 0.1, ps * 0.5];
    const ptsHP = await page.evaluate(tv => window.__directCheckHP(tv), tProbe);
    ran++;
    // dive-gate semantics: escape-flag mismatch = 1; both-unescaped = garbage-vs-garbage,
    // flag agreement IS the test (skip rel); both-escaped: min(raw rel, |diff|/de(t=0)).
    const de0 = Math.max(Math.abs(ptsHP[0].direct), 1e-300);
    let worstHP = -1;
    for (const p of ptsHP) {
      if (p.escG !== p.escD) worstHP = Math.max(worstHP, 1);
      else if (p.escG) worstHP = Math.max(worstHP, Math.min(p.rel, Math.abs(p.gpu - p.direct) / de0));
    }
    if (worstHP < 0) worstHP = 1;
    let xval = '';
    if (depth >= 1e-12) {           // overlap region: cross-validate the two referees
      const ptsF64 = await page.evaluate(tv => window.__directCheck(tv), tProbe);
      const xrel = Math.max(...ptsF64.map((p, i) => (p.escD && ptsHP[i].escD) ? Math.abs(p.direct - ptsHP[i].direct) / Math.max(Math.abs(ptsHP[i].direct), 1e-300) : 0));
      xval = `  refXval f64-vs-HP=${xrel.toExponential(1)}`;
    }
    const ok = worstHP < 1e-3;
    if (ok) pass++;
    console.log(`depth ${depth.toExponential(0).padStart(6)} it=${String(iters).padStart(3)} achieved=${deAch.toExponential(1)}  worst GPUvsHP=${worstHP.toExponential(1)}  ${ok ? 'PASS' : 'FAIL'}${xval}${placeNote}`);
    for (const p of ptsHP)
      console.log(`        t=${p.t.toExponential(2)}  gpu=${p.gpu.toExponential(6)}  direct=${p.direct.toExponential(6)}  pert=${p.pert.toExponential(6)}  esc G/D=${p.escG}/${p.escD}  rel=${p.rel.toExponential(1)}`);
  }
  console.log(`\n${pass}/${ran} depths: GPU perturbed de matches the INDEPENDENT BigInt direct referee.`);
} finally { await browser.close(); }
