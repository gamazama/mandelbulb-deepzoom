/* Controlled perf for the whole pass. Place ONCE per depth, then measure fps for each config
   at the SAME camera/orbit/divisor. Two scenarios:
   - FLIGHT (rebuild the BigInt orbit every frame, as while navigating): constant CPU baseline,
     so GPU-side deltas (normals/bisection/BLA) show cleanly.
   - IDLE  (orbit rebuild skipped, as when paused): shows the idle-skip benefit + GPU-only cost. */
import { chromium } from 'playwright';
import { fileURLToPath } from 'url'; import path from 'path';
const PAGE = 'file://' + path.join(path.dirname(fileURLToPath(import.meta.url)), 'index.html').replace(/\\/g, '/');
const DEPTHS = [1e-9, 1e-20, 1e-40];
const DIV = Number(process.env.DIV || 4);
const VW = Number(process.env.VW || 640), VH = Number(process.env.VH || 400);  // small ⇒ stable, no TDR
const RAYS = [[0.1,0.15,2.6],[0.4,0.3,2.5],[-0.3,0.2,2.6],[2.6,0.15,0.1]];
const YAW = [-Math.PI/2,-Math.PI/2,-Math.PI/2,Math.PI];
const CFG = [   // label, normalMode, bisect, bla
  ['4t/22(before)', 2, 22, false],
  ['3t/12',         1, 12, false],
  ['3t/12+BLA',     1, 12, true ],
  ['screen/12',     0, 12, false],
];
const browser = await chromium.launch({ channel: 'chrome', headless: false,
  args: ['--use-gl=angle','--use-angle=d3d11','--ignore-gpu-blocklist','--enable-gpu'] });
const page = await browser.newPage({ viewport: { width: VW, height: VH } });
page.on('pageerror', e => console.log('PAGEERR', e.message));
await page.goto(PAGE, { waitUntil: 'commit', timeout: 15000 });
await page.waitForFunction(() => window.__deepReady !== undefined, null, { timeout: 60000 });
await page.evaluate(() => window.__setBla(true));   // pre-compile the BLA program once
await page.evaluate(() => window.__setAccum(false));  // measure raw render, not progressive stills
const measure = () => page.evaluate(() => new Promise(res => {
  let n=0; const t0=performance.now();
  const f=()=>{ n++; const el=performance.now()-t0; el<3000? requestAnimationFrame(f): res(n/(el/1000)); };
  requestAnimationFrame(f); }));
const warm = () => page.evaluate(() => new Promise(res => { let n=0; const f=()=> (++n>15? res(): requestAnimationFrame(f)); requestAnimationFrame(f); }));
for (const scenario of ['FLIGHT','IDLE']) {
  console.log(`\n=== ${scenario} (orbit rebuild ${scenario==='FLIGHT'?'every frame':'skipped when idle'}) @ ÷${DIV}, ${VW}×${VH} ===`);
  console.log('depth   iters  ' + CFG.map(c=>c[0].padStart(15)).join(''));
  for (const depth of DEPTHS) {
    const iters = Math.min(120, 32 + Math.ceil(1.35 * -Math.log10(depth)));
    let placed=false;
    for (let ri=0; ri<RAYS.length; ri++) {
      await page.evaluate(({p,yaw,it})=>{ window.__setBla(false); window.__forceRebuild(false); window.__setDiv(8); window.__setCamHP(p); window.__cam.yaw=yaw; window.__cam.pitch=0; window.__cam.iters=it; }, {p:RAYS[ri],yaw:YAW[ri],it:iters});
      const dv = await page.evaluate(d => window.__dive(d), depth);
      if (!dv.error) { placed=true; break; }
    }
    if (!placed) { console.log(`${depth.toExponential(0).padStart(6)}  (placement failed)`); continue; }
    await page.evaluate(({d,fr}) => { window.__setDiv(d); window.__forceRebuild(fr); }, {d:DIV, fr: scenario==='FLIGHT'});
    const cells = [];
    for (const [,nm,bis,bla] of CFG) {
      await page.evaluate(({nm,bis,bla})=>{ window.__setNormalMode(nm); window.__setBisect(bis); window.__setBla(bla); }, {nm,bis,bla});
      await warm();
      cells.push((await measure()).toFixed(1).padStart(15));
    }
    console.log(`${depth.toExponential(0).padStart(6)}   ${String(iters).padStart(3)}  ` + cells.join(''));
  }
}
await browser.close();
