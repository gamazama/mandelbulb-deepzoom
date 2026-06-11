import { chromium } from 'playwright';
import { fileURLToPath } from 'url'; import path from 'path';
const PAGE = 'file://' + path.join(path.dirname(fileURLToPath(import.meta.url)), 'index.html').replace(/\\/g, '/');
const browser = await chromium.launch({ channel: 'chrome', headless: false,
  args: ['--use-gl=angle', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--enable-gpu'] });
const page = await browser.newPage();
page.on('pageerror', e => console.log('PAGEERR', e.message));
try {
  await page.goto(PAGE, { waitUntil: 'commit', timeout: 15000 });
  await page.waitForFunction(() => window.__deepReady !== undefined, null, { timeout: 60000 });
  console.log('deep shader:', JSON.stringify(await page.evaluate(()=>({ok:window.__deepReady,err:window.__deepErr}))));
  if (process.env.BLA === '1') {
    const t = Date.now();
    await page.evaluate(() => window.__setBla(true));
    console.log('BLA compile ms', Date.now()-t, 'err:', await page.evaluate(()=>window.__blaErr()));
  }
} catch(e){ console.log('FAIL', e.message.slice(0,200)); }
await browser.close();
