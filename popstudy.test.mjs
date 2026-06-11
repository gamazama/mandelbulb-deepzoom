/* =============================================================================
   popstudy.test.mjs — near-critical PIXEL POPULATION study at depth
   Run from this folder:  node popstudy.test.mjs
   -----------------------------------------------------------------------------
   Single near-critical probes can flip escape flags (f32 chain, orbit-conditioned).
   The shipping question: at 1e-40/1e-50, what FRACTION of a real view's pixels
   flip, and is the spatial pattern scattered (reads as high-iter fizz, fine) or
   clustered (structured artifact, needs work)? Method: 64×64 grid of pertOrbit
   evaluations at transverse offsets δc = spread·(uv·frame) — spread chosen at the
   rendering-relevant scale (the achieved camera de, and 10× it) — GPU vs the CPU
   f64-perturbed referee (proven ≡ BigInt direct everywhere).
   GPU load: 64×64 × ~100 iters ≈ trivial, no TDR risk.
   ============================================================================= */
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import path from 'path';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PAGE = 'file://' + path.join(__dirname, 'index.html').replace(/\\/g, '/');

const DEPTHS = [1e-20, 1e-30, 1e-40, 1e-50];
const RAYS = [
  { pos: [0.1, 0.15, 2.6], yaw: -Math.PI / 2, pitch: 0 },
  { pos: [0.4, 0.3, 2.5], yaw: -Math.PI / 2, pitch: 0 },
  { pos: [-0.3, 0.2, 2.6], yaw: -Math.PI / 2, pitch: 0 },
  { pos: [2.6, 0.15, 0.1], yaw: Math.PI, pitch: 0 },
];

const browser = await chromium.launch({ channel: 'chrome', headless: false,
  args: ['--use-gl=angle', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--enable-gpu'] });
try {
  const page = await browser.newPage({ viewport: { width: 640, height: 400 } });
  await page.goto(PAGE, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForFunction(() => window.__deepReady !== undefined, null, { timeout: 30000 });
  for (const depth of DEPTHS) {
    const iters = Math.min(120, 32 + Math.ceil(1.35 * -Math.log10(depth)));
    let ok = false;
    for (const c of RAYS) {
      await page.evaluate(({ p, yaw, pitch, it }) => { window.__setCamHP(p); window.__cam.yaw = yaw; window.__cam.pitch = pitch; window.__cam.iters = it; }, { p: c.pos, yaw: c.yaw, pitch: c.pitch, it: iters });
      const dv = await page.evaluate(d => window.__dive(d), depth);
      if (!dv.error) { ok = true; break; }
    }
    if (!ok) { console.log(`depth ${depth.toExponential(0).padStart(6)}  SKIP (placement failed)`); continue; }
    const deAch = await page.evaluate(() => Math.abs(window.__directCheckHP([0])[0].direct));
    console.log(`\ndepth ${depth.toExponential(0)} it=${iters} achieved de=${deAch.toExponential(1)}`);
    for (const mult of [0.001, 0.01, 0.1, 1]) {
      const r = await page.evaluate(({ s, W }) => window.__popStudy(s, W), { s: deAch * mult, W: 64 });
      console.log(`  [probe2 pointwise] spread=${r.spread.toExponential(1)} (de×${mult})  flips=${r.flips}/${r.W * r.W} (${(100 * r.flipFrac).toFixed(2)}%)  ` +
        `clustered=${(100 * r.clusterFrac).toFixed(0)}%  escFrac=${(100 * r.escFrac).toFixed(0)}%  ` +
        `deRel median=${r.medianRel.toExponential(1)} worst=${r.worstRel.toExponential(1)}`);
    }
    // DECISIVE follow-up: the FULL-march population — the user-visible surface POSITION.
    const mp = await page.evaluate(W => window.__marchPop(W), 64);
    const verdict = mp.subPixelP95 ? 'SUB-PIXEL ✓ — SHIPS' : 'NOT sub-pixel — needs f32-chain work';
    console.log(`  [probe3 MARCH] pixAng=${mp.pixAng.toExponential(2)}  gHit=${(100*mp.gHitFrac).toFixed(0)}% cHit=${(100*mp.cHitFrac).toFixed(0)}%  ` +
      `hitDisagree=${mp.hitDisagree}/${mp.W*mp.W} (${(100*mp.hitDisagreeFrac).toFixed(2)}%)\n` +
      `                 |ΔtHit/tHit| median=${mp.median.toExponential(1)} p95=${mp.p95.toExponential(1)} worst=${mp.worst.toExponential(1)}  ` +
      `>pixAng:${(100*mp.badFrac).toFixed(1)}% (clustered ${(100*mp.badClusterFrac).toFixed(0)}%)  →  ${verdict}`);
  }
  console.log(`\nREADING: probe2 (pointwise de at the boundary-adjacent t=0 plane) is a WORST-CASE
slice — flips there affect step lengths, not necessarily the rendered surface. probe3 (the FULL
march) gives the user-visible quantity: |ΔtHit/tHit| = how far the rendered surface POSITION moves
GPU-vs-CPU. DECISION RULE (SLICE 6b): p95 |ΔtHit/tHit| < pixAng ⇒ surface agrees within a display
pixel ⇒ 1e-50 ships visually. If not sub-pixel, the clustered >pixAng pixels are where f32-chain
tightening (dd-mantissa δ at flagged pixels) would go.`);
} finally { await browser.close(); }
