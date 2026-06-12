/* Regression: once the camera is static and the temporal accumulation has converged, the canvas
   must hold a STABLE image. A prior bug let the ground-truth gate's periodic readPixels (a mid-frame
   GPU sync) glitch the blit-to-canvas of the converged still, showing as an intermittent flash.
   We screenshot the canvas over ~120 frames while static (gate ON, the default) and assert there is
   no transient outlier frame (a hash that differs from both neighbours). Real GPU via Playwright. */
import { chromium } from 'playwright';
import { fileURLToPath } from 'url'; import path from 'path';
import crypto from 'crypto';
const PAGE = 'file://' + path.join(path.dirname(fileURLToPath(import.meta.url)), 'index.html').replace(/\\/g, '/');
const b = await chromium.launch({ channel: 'chrome', headless: false,
  args: ['--use-gl=angle', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--enable-gpu'] });
const p = await b.newPage({ viewport: { width: 900, height: 600 } });
p.on('pageerror', e => console.log('PAGEERR', e.message));
await p.goto(PAGE, { waitUntil: 'commit', timeout: 15000 });
await p.waitForFunction(() => window.__deepReady !== undefined, null, { timeout: 60000 });
// place a mid-depth dive, deep mode, gate ON (default), then go fully static and let it converge
await p.evaluate(() => { window.__setCamHP([0.1, 0.15, 2.6]); window.__cam.yaw = -Math.PI / 2; window.__cam.pitch = 0; window.__cam.iters = 60; });
await p.evaluate(d => window.__dive(d), 1e-12);
const cv = p.locator('canvas');
const hashes = [];
for (let i = 0; i < 120; i++) hashes.push(crypto.createHash('md5').update(await cv.screenshot()).digest('hex').slice(0, 8));
let flashes = 0;
for (let i = 1; i < hashes.length - 1; i++)
  if (hashes[i] !== hashes[i - 1] && hashes[i] !== hashes[i + 1] && hashes[i - 1] === hashes[i + 1]) {
    flashes++; console.log(`FLASH @ sample ${i}: ${hashes[i - 1]} -> ${hashes[i]} -> ${hashes[i + 1]}`);
  }
console.log(flashes === 0 ? '\nPASS — converged canvas is stable (no flash)' : `\nFAIL — ${flashes} transient flash frame(s)`);
await b.close();
process.exit(flashes === 0 ? 0 : 1);
