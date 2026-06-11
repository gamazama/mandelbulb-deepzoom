/* Non-circular certification at power n ≠ 8 and at phase ψ ≠ 0. The BigInt direct referee
   (__directCheckHP → hpDirectDE + hp.P8) is now power/phase-general and shares ZERO math with the
   GPU kernel, so GPU-vs-referee agreement certifies the generalization. Shallow placements (1e-6)
   keep dive placement easy; the point is the de-field match at the new power/phase, not depth. */
import { chromium } from 'playwright';
import { fileURLToPath } from 'url'; import path from 'path';
const PAGE = 'file://' + path.join(path.dirname(fileURLToPath(import.meta.url)), 'index.html').replace(/\\/g, '/');
const RAYS = [[0.1,0.15,2.6],[0.4,0.3,2.5],[-0.3,0.2,2.6],[2.6,0.15,0.1]];
const YAW  = [-Math.PI/2,-Math.PI/2,-Math.PI/2,Math.PI];
const CASES = [  // {power, phase:[θ,φ], depth}
  { power: 5, phase:[0,0],     depth: 1e-6 },
  { power: 6, phase:[0,0],     depth: 1e-6 },
  { power: 7, phase:[0,0],     depth: 1e-6 },
  { power: 8, phase:[0.3,0.5], depth: 1e-6 },
  { power: 6, phase:[0.4,0.0], depth: 1e-9 },
];
const browser = await chromium.launch({ channel: 'chrome', headless: false,
  args: ['--use-gl=angle','--use-angle=d3d11','--ignore-gpu-blocklist','--enable-gpu'] });
const page = await browser.newPage({ viewport: { width: 640, height: 400 } });
page.on('pageerror', e => console.log('PAGEERR', e.message));
await page.goto(PAGE, { waitUntil: 'commit', timeout: 15000 });
await page.waitForFunction(() => window.__deepReady !== undefined, null, { timeout: 60000 });
let pass = 0, ran = 0;
for (const c of CASES) {
  const iters = Math.min(120, 32 + Math.ceil(1.35 * -Math.log10(c.depth)));
  let placed = false;
  for (let ri = 0; ri < RAYS.length; ri++) {
    await page.evaluate(({ p, yaw, it, pw, ph }) => {
      window.__setPower(pw); window.__setPhase(ph[0], ph[1]);
      window.__setCamHP(p); window.__cam.yaw = yaw; window.__cam.pitch = 0; window.__cam.iters = it;
    }, { p: RAYS[ri], yaw: YAW[ri], it: iters, pw: c.power, ph: c.phase });
    const dv = await page.evaluate(d => window.__dive(d), c.depth);
    if (!dv.error) { placed = true; break; }
  }
  ran++;
  if (!placed) { console.log(`n=${c.power} ψ=${c.phase}  PLACEMENT FAILED`); continue; }
  const deAch = await page.evaluate(() => Math.abs(window.__directCheckHP([0])[0].direct));
  const ps = Math.min(deAch, c.depth);
  const tp = [0, ps*1e-3, ps*0.1, ps*0.5];
  const pts = await page.evaluate(tv => window.__directCheckHP(tv), tp);
  const de0 = Math.max(Math.abs(pts[0].direct), 1e-300);
  let worst = -1, flags = true;
  for (const p of pts) { if (p.escG !== p.escD) { worst = Math.max(worst,1); flags=false; }
    else if (p.escG) worst = Math.max(worst, Math.min(p.rel, Math.abs(p.gpu-p.direct)/de0)); }
  if (worst < 0) worst = 1;
  const ok = worst < 1e-3;
  if (ok) pass++;
  console.log(`n=${c.power} ψ=[${c.phase}]  achieved=${deAch.toExponential(1)}  worst GPUvsHP=${worst.toExponential(1)}  flags=${flags?'✓':'✗'}  ${ok?'PASS':'FAIL'}`);
}
console.log(`\n${pass}/${ran} power/phase cases: GPU matches the independent BigInt direct referee.`);
await browser.close();
