/* =============================================================================
   gpu-eft.test.mjs  —  GPU verification loop + double-single EFT survival check
   Run from this folder:  node gpu-eft.test.mjs
   (Playwright 1.59.1 is in dev/node_modules; node resolves it up the tree.)
   -----------------------------------------------------------------------------
   Launches headless/real Chromium on several backends, loads gpu-ds.html, runs
   twoSum/twoProd on the GPU, reads back (s,e,p,e2) as RGBA32F, and diffs the ERROR
   terms against an EXACT (double) oracle. If a backend's error term collapses to 0
   (or is wildly off) while the oracle's is nonzero, the driver's shader compiler
   broke the error-free transform → double-single is unsafe there.
   This is also THE reusable template for every future GPU numeric test.
   ============================================================================= */
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import path from 'path';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PAGE = 'file://' + path.join(__dirname, 'gpu-ds.html').replace(/\\/g, '/');

// exact f32-EFT oracle: errors are exactly representable, computed in double
const f = Math.fround;
function oracle(a, b) {
  const s = f(a + b), e = (a + b) - s;     // exact twoSum error
  const p = f(a * b), e2 = (a * b) - p;    // exact twoProd error
  return { s, e, p, e2 };
}

// REAL GPU ONLY (SwiftShader is too slow + sidesteps the optimizer risk we care about).
// No swiftshader fallback flag → if the real GPU doesn't engage, it errors loudly (good).
const BACKENDS = [
  { name: 'd3d11', opts: { channel: 'chrome', headless: false,
      args: ['--use-gl=angle','--use-angle=d3d11','--ignore-gpu-blocklist','--enable-gpu'] } },
  { name: 'gl', opts: { channel: 'chrome', headless: false,
      args: ['--use-gl=angle','--use-angle=gl','--ignore-gpu-blocklist','--enable-gpu'] } },
  { name: 'vulkan', opts: { channel: 'chrome', headless: false,
      args: ['--use-gl=angle','--use-angle=vulkan','--ignore-gpu-blocklist','--enable-gpu','--enable-features=Vulkan'] } },
];

const relBad = (g, t) => {
  if (t === 0) return Math.abs(g) > 1e-12;
  return Math.abs(g - t) / Math.abs(t) > 1e-3;     // >0.1% off, or collapsed to 0
};

/* ── HDR oracle ──────────────────────────────────────────────────────────────
   f64 ground truth for the 8 HDR shader cases (case semantics mirror the
   shader switch in gpu-ds.html runHdrTest). Values like 2^-166 (~1e-50) and
   2^-314 are fine in f64. Tolerances: ~few-ulp-of-f32-mantissa for single
   ops; looser for 32-iter / repeated-squaring chains.                       */
const HDR_CASES = (() => {
  const v = (m, e) => f(m) * Math.pow(2, e);
  return [
    { name: 'mul @1e-50',        tol: 5e-7, exact: (a) => v(1.5,-83) * v(1.25,-84) },
    { name: 'add same-exp',      tol: 5e-7, exact: (a) => v(1.5,-166) + v(1.75,-166) },
    { name: 'add gap-10',        tol: 5e-7, exact: (a) => v(1.5,-160) + v(1.9,-170) },
    { name: 'add gap-50 (drop)', tol: 5e-7, exact: (a) => v(1.5,-150) + v(1.9,-200) },
    { name: 'sub cancel',        tol: 5e-7, exact: (a) => v(1.5,-166) - v(1.25,-166) },
    { name: 'chain h=hK+C ×32',  tol: 2e-5, exact: (a) => {
        let h = v(1.1,-150); const K = v(1.002,0), C = v(1.3,-170);
        for (let k = 0; k < 32; k++) h = h * K + C; return h; } },
    { name: 'square ×4 → 2^-314',tol: 1e-5, exact: (a) => {
        let h = v(1.3,-20); for (let k = 0; k < 4; k++) h = h * h; return h; } },
    { name: 'from/toFloat trip', tol: 5e-7, exact: (a) => v(1.5,-10) * v(1.25,-15) },
  ];
})();

function checkHdr(results) {
  let fails = 0, worst = 0; const rows = [];
  results.forEach((row, i) => {
    const c = HDR_CASES[i];
    const got = row.m * Math.pow(2, row.e);          // exact in f64: m is f32, 2^e exact
    const exact = c.exact();
    const rel = exact === 0 ? Math.abs(got) : Math.abs(got - exact) / Math.abs(exact);
    const bad = rel > c.tol;
    if (bad) fails++;
    worst = Math.max(worst, rel);
    rows.push(`     ${bad ? '✗' : '✓'} ${c.name.padEnd(20)} got m=${row.m.toPrecision(8)} e=${row.e}  rel=${rel.toExponential(2)}`);
  });
  return { fails, worst, rows };
}

for (const be of BACKENDS) {
  let browser;
  try {
    browser = await chromium.launch(be.opts);
    const page = await browser.newPage();
    await page.goto(PAGE, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForFunction(() => window.__ready === true, null, { timeout: 30000 });
    const r = await page.evaluate(() => window.runEftTest());
    if (r.error) { console.log(`\n[${be.name}]  ${r.renderer || ''}\n   ERROR: ${r.error}`); await browser.close(); continue; }
    console.log(`\n[${be.name}]`);
    console.log(`   renderer: ${r.renderer}`);
    let fails = 0, worstE = 0;
    for (const row of r.results) {
      const o = oracle(row.a, row.b);
      const badE = relBad(row.e, o.e), badE2 = relBad(row.e2, o.e2);
      if (badE || badE2) fails++;
      worstE = Math.max(worstE, o.e === 0 ? 0 : Math.abs(row.e - o.e) / Math.abs(o.e));
    }
    const real = /nvidia|rtx|geforce|radeon|intel/i.test(r.renderer);
    console.log(`   EFT survival: ${fails === 0 ? 'PASS' : 'FAIL ('+fails+'/8 broken)'}  | worst twoSum-err rel = ${worstE.toExponential(2)}  | real-GPU: ${real ? 'YES' : 'no (software)'}`);
    const h = await page.evaluate(() => window.runHdrTest());
    if (h.error) { console.log(`   HDR test ERROR: ${h.error}`); }
    else {
      const hc = checkHdr(h.results);
      console.log(`   HDR survival: ${hc.fails === 0 ? 'PASS' : 'FAIL ('+hc.fails+'/8 broken)'}  | worst rel = ${hc.worst.toExponential(2)}`);
      console.log(hc.rows.join('\n'));
    }
  } catch (e) {
    console.log(`\n[${be.name}]  launch/run failed: ${String(e).split('\n')[0]}`);
  } finally { if (browser) await browser.close(); }
}
console.log(`\nINTERPRET:
 • A backend whose renderer names NVIDIA/RTX = your real 2080 engaged. That's the one
   that matters for the optimizer risk and for perf.
 • EFT survival PASS on the real-GPU backend → double-single is safe on GPU; green-light
   the precision ladder. FAIL → the ANGLE/D3D11 compiler fused the error-free transforms;
   mitigations to try next: --use-angle=gl backend, or wrap EFT subtractions to block
   contraction. Knowing which BEFORE building the kernel is the whole point.
 • If NO backend shows a real GPU here, headless GPU isn't engaging — use the
   channel:'chrome', headless:false config (or run on the desktop) for real-GPU tests.
 • HDR survival PASS = mantissa+exponent arithmetic (GMT fractalKernel.ts helpers) is
   accurate on this GPU at exponents far past f32 underflow (verified to 2^-314 ≈ 3e-95).
   HDR needs no error-free transforms, so the NVIDIA EFT collapse can't touch it.
   RESULT 2026-06-10 (RTX 2070): EFT FAIL 8/8 on all 3 backends, HDR PASS 8/8 on all 3
   (worst rel 1.3e-6 over a 32-iter δ-shaped chain) → the precision ladder uses HDR-δ
   (f32 mantissa ≈ 7 digits) + aggressive rebasing, NOT double-single.`);
