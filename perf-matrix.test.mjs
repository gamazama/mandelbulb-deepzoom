/* =============================================================================
   perf-matrix.test.mjs — fps vs depth × iteration budget × resolution divisor
   Run from this folder:  node perf-matrix.test.mjs
   -----------------------------------------------------------------------------
   Places the camera at each depth (same dive + ray fallback as the other suites),
   then measures rendered frames/sec at each deep-mode resolution divisor by
   counting requestAnimationFrame callbacks over a 2.5 s window (vsync-capped at
   the display rate — a reading at the cap means "fully interactive").
   The deep march cost ≈ steps × orbit-iters × ~50 HDR flops per pixel, so fps is
   expected to fall with depth (iters) and rise ~quadratically with the divisor.
   ============================================================================= */
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import path from 'path';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PAGE = 'file://' + path.join(__dirname, 'index.html').replace(/\\/g, '/');

const DEPTHS = [1e-9, 1e-20, 1e-40];
/* ÷1 (full res) at depth is TDR territory: single frames exceed the ~2 s Windows GPU
   watchdog, the device resets, and the WHOLE DESKTOP freezes while it happens (same
   GPU). Measured before the resets: ~1.1 fps @1e-9, ~1.8 fps @1e-20. Only include ÷1
   with an explicit --full flag, and know what you're signing up for. */
const DIVS = process.argv.includes('--full') ? [8, 4, 2, 1] : [8, 4, 2];
const RAYS = [
  { pos: [0.1, 0.15, 2.6], yaw: -Math.PI / 2, pitch: 0 },
  { pos: [0.4, 0.3, 2.5], yaw: -Math.PI / 2, pitch: 0 },
  { pos: [-0.3, 0.2, 2.6], yaw: -Math.PI / 2, pitch: 0 },
];

async function newSession() {
  const browser = await chromium.launch({ channel: 'chrome', headless: false,
    args: ['--use-gl=angle', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--enable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto(PAGE, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForFunction(() => window.__deepReady !== undefined, null, { timeout: 30000 });
  if (process.env.BLA === '1') { await page.evaluate(() => window.__setBla(true));
    const e = await page.evaluate(() => window.__blaErr()); if (e) { console.log('BLA COMPILE FAILED:\n' + e); } }
  return { browser, page };
}
let ses = await newSession();
try {
  console.log(`viewport 1280×800 (deep render res = viewport / divisor)   BLA path: ${process.env.BLA === '1' ? 'ON' : 'off'}\n`);
  console.log('depth     iters   ' + DIVS.map(d => ('÷' + d).padStart(8)).join(''));
  for (const depth of DEPTHS) {
    const iters = Math.min(120, 32 + Math.ceil(1.35 * -Math.log10(depth)));
    let placed = false;
    try {
      for (const c of RAYS) {
        // dive renders live frames — make sure the loop is at the cheap divisor first
        await ses.page.evaluate(({ p, yaw, pitch, it }) => { window.__setDiv(8); window.__setCamHP(p); window.__cam.yaw = yaw; window.__cam.pitch = pitch; window.__cam.iters = it; }, { p: c.pos, yaw: c.yaw, pitch: c.pitch, it: iters });
        const r = await ses.page.evaluate(d => window.__dive(d), depth);
        if (!r.error && r.relDE < 1e-3) { placed = true; break; }
      }
    } catch {
      console.log(`${depth.toExponential(0).padStart(6)}  (TDR during dive — relaunching)`);
      try { await ses.browser.close(); } catch {}
      ses = await newSession();
      continue;
    }
    if (!placed) { console.log(`${depth.toExponential(0).padStart(6)}  (placement failed)`); continue; }
    const cells = [];
    let lastFps = Infinity;
    for (const div of DIVS) {
      // a single frame slower than ~2 s trips the D3D11 watchdog (device reset kills the
      // page) — skip divisors that the previous reading predicts are TDR-doomed
      if (lastFps / (4 * 2) < 0.55) { cells.push('TDR-skip'.padStart(8)); continue; }
      try {
        await ses.page.evaluate(d => window.__setDiv(d), div);
        await ses.page.evaluate(() => new Promise(res => { let n = 0; const f = () => (++n > 5 ? res() : requestAnimationFrame(f)); requestAnimationFrame(f); }));
        const fps = await ses.page.evaluate(() => new Promise(res => {
          let n = 0; const t0 = performance.now();
          const f = () => { n++; const el = performance.now() - t0; el < 2500 ? requestAnimationFrame(f) : res(n / (el / 1000)); };
          requestAnimationFrame(f);
        }));
        cells.push(fps.toFixed(1).padStart(8));
        lastFps = fps;
      } catch {
        cells.push('TDR'.padStart(8));            // device reset — relaunch for the next depth
        try { await ses.browser.close(); } catch {}
        ses = await newSession();
        break;
      }
    }
    console.log(`${depth.toExponential(0).padStart(6)}    ${String(iters).padStart(3)}   ` + cells.join(''));
  }
  console.log('\n(fps at the display refresh cap = fully interactive at that setting;');
  console.log(' TDR = a single frame exceeded the ~2 s Windows GPU watchdog at that setting)');
} finally { try { await ses.browser.close(); } catch {} }
