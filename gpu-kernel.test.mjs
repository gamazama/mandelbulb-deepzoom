/* =============================================================================
   gpu-kernel.test.mjs  —  GPU verification of the GLSL perturbation kernel
   Run from this folder:  node gpu-kernel.test.mjs [--backend=d3d11]
   (default: all three ANGLE backends; real GPU, headed — same loop as gpu-eft.test.mjs)
   -----------------------------------------------------------------------------
   Drives gpu-kernel.html and checks each kernel building block against oracles:
     1. runPrimsTest  — new HDR helpers (mulF/divF/recip, exact-binomial (1+u)^n−1,
                        tiny atan, sin/cos residuals) vs f64.
     2. runStepTest   — one pertStep (§3 stableDelta + Δr + Δdr) vs an f64 oracle
                        that mirrors the texture's f32 rounding exactly, at δ from
                        1e-3 down to 1e-40 (far below f32 underflow).
     3. runOrbitTest  — 14-iteration perturbed orbit (texture fetch + rebase logic +
                        Δdr accumulation) vs the BigInt hp orbit (§8.3) ground truth:
                        final δ, Δr, Δdr, de — relative errors at offsets to 1e-40.
     4. runMarchTest  — march-loop stub: compiles, runs, finite outputs.
   ============================================================================= */
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import path from 'path';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PAGE = 'file://' + path.join(__dirname, 'gpu-kernel.html').replace(/\\/g, '/');

const N = 8;
const f = Math.fround;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len3 = a => Math.hypot(a[0], a[1], a[2]);
const hdrVal = (x) => { if (x === 0 || !isFinite(x)) return 0; const e = Math.floor(Math.log2(Math.abs(x))); return f(x * Math.pow(2, -e)) * Math.pow(2, e); };
const hdrOut = (pair) => pair[0] * Math.pow(2, pair[1]);   // GPU (m,e) readback → f64 value
const rel = (g, t) => t === 0 ? (g === 0 ? 0 : Math.abs(g)) : Math.abs(g - t) / Math.abs(t);

/* ────────────────────────── oracle: prims (f64) ────────────────────────── */
function primsOracle(c) {
  const a = hdrVal(c.a), b = c.b;
  switch (c.name) {
    case 'mulF tiny':        return a * b;
    case 'divF tiny':        return a / b;
    case 'recip':            return 1 / a;
    case 'powm1_8 u=1e-20':
    case 'powm1_8 u=1e-3':
    case 'powm1_8 u=-0.3':   return Math.expm1(8 * Math.log1p(a));
    case 'powm1_7 u=1e-20':  return Math.expm1(7 * Math.log1p(a));
    case 'atanT tiny':
    case 'atanT series.01':
    case 'atanT series.09':
    case 'atanT builtin.3':  return Math.atan2(a, b);
    case 'sin x=1e-25':
    case 'sin x=0.05':
    case 'sin x=0.5':        return Math.sin(a);
    case 'cosm1 x=1e-25':
    case 'cosm1 x=0.05':
    case 'cosm1 x=0.5': {    // residual form so f64 doesn't underflow at tiny x
      const s = Math.sin(a); return -s * s / (1 + Math.cos(a));
    }
    case 'mulF negative':    return a * b;
  }
  throw new Error('unknown prim case ' + c.name);
}

/* ───────────── oracle: one pertStep, mirroring the texture's f32 rounding ─────────────
   Reference quantities exactly as the GPU sees them (f32 / HDR-f32-mantissa); the step
   math itself in f64 (so the measured GPU deviation is the GLSL f32 arithmetic). */
function stepOracle(Vd, dRaw, ddrInRaw, DRtest) {
  const V = Vd.map(f);
  const Rd = len3(Vd);
  const R = f(Rd);
  const th = Math.acos(Math.max(-1, Math.min(1, Vd[2] / Rd))), ph = Math.atan2(Vd[1], Vd[0]);
  const a = N * th, b = N * ph;
  const sa = f(Math.sin(a)), ca = f(Math.cos(a)), sb = f(Math.sin(b)), cb = f(Math.cos(b));
  const Rn = hdrVal(Math.pow(Rd, N)), Rp = hdrVal(Math.pow(Rd, N - 1)), DR = hdrVal(DRtest);
  const d = dRaw.map(hdrVal);                          // GPU received the HDR-split value
  const ddrIn = hdrVal(ddrInRaw);
  // §3a / §3b / §8.1
  const q = 2 * dot(V, d) + dot(d, d);
  const r2 = Math.sqrt(R * R + q);
  const Dr = q / (r2 + R);
  const u = Dr / R;
  const dRn = Rn * Math.expm1(N * Math.log1p(u));
  const Dp = Rp * Math.expm1((N - 1) * Math.log1p(u));
  const Ddr = N * (Dp * DR + (Rp + Dp) * ddrIn);
  // §3c-§3f (residual forms)
  const Dphi = Math.atan2(V[0] * d[1] - V[1] * d[0], V[0] * (V[0] + d[0]) + V[1] * (V[1] + d[1]));
  const rho = Math.hypot(V[0], V[1]);
  // §3c EXACT Δθ (2026-06-10 fix — matches the kernel; the old first-order form
  // seeded O(δ²) error, visible here as ~5e-4 disagreement at |δ|=1e-3)
  const qr = 2 * (V[0] * d[0] + V[1] * d[1]) + d[0] * d[0] + d[1] * d[1];
  const rhoP = Math.sqrt(Math.max(rho * rho + qr, 0));
  const drho = qr / (rhoP + rho);
  const Dth = Math.atan2(drho * V[2] - d[2] * rho, rho * rhoP + V[2] * (V[2] + d[2]));
  const Da = N * Dth, Db = N * Dphi;
  const sA = Math.sin(Da), cA = Math.cos(Da), sB = Math.sin(Db), cB = Math.cos(Db);
  const cAm = -sA * sA / (1 + cA), cBm = -sB * sB / (1 + cB);
  const dsA = sa * cAm + ca * sA, dcA = ca * cAm - sa * sA;
  const dsB = sb * cBm + cb * sB, dcB = cb * cBm - sb * sB;
  const DD = [sa * dcB + dsA * cb + dsA * dcB, sa * dsB + dsA * sb + dsA * dsB, dcA];
  const Dref = [sa * cb, sa * sb, ca];
  const delta = [0, 1, 2].map(j => Rn * DD[j] + dRn * (Dref[j] + DD[j]));
  return { delta, Dr, Ddr };
}

/* ───────────── hp BigInt fixed-point factory (from hp-orbit.test.mjs / de-depth.test.mjs) ───────────── */
function makeHP(P) {
  const Pb = BigInt(P), SCALE = 1n << Pb, ONE = SCALE;
  const mul = (a, b) => (a * b) >> Pb, fdiv = (a, b) => (a << Pb) / b;
  const isqrt = (n) => { if (n < 2n) return n < 0n ? 0n : n; let x = n, y = (x + 1n) >> 1n; while (y < x) { x = y; y = (x + n / x) >> 1n; } return x; };
  const fsqrt = (a) => isqrt(a << Pb), SMALL = SCALE >> 90n;
  const fromNum = (x) => { if (x === 0) return 0n; const neg = x < 0; x = Math.abs(x); const dv = new DataView(new ArrayBuffer(8)); dv.setFloat64(0, x); const bits = dv.getBigUint64(0); const exp = Number((bits >> 52n) & 0x7ffn), mant = bits & 0xfffffffffffffn; let M, E; if (exp === 0) { M = mant; E = -1074; } else { M = mant | 0x10000000000000n; E = exp - 1075; } const sh = Pb + BigInt(E); let v = sh >= 0n ? (M << sh) : (M >> (-sh)); return neg ? -v : v; };
  const toNum = (v) => Number(v) / Number(SCALE);
  const chebT = (x, n) => { let a = ONE, b = x; if (n === 0) return a; if (n === 1) return b; for (let k = 2; k <= n; k++) { const t = 2n * mul(x, b) - a; a = b; b = t; } return b; };
  const chebU = (x, n) => { let a = ONE, b = 2n * x; if (n === 0) return a; if (n === 1) return b; for (let k = 2; k <= n; k++) { const u = 2n * mul(x, b) - a; a = b; b = u; } return b; };
  const P8 = (v) => { const [x, y, z] = v; const x2 = mul(x, x), y2 = mul(y, y), z2 = mul(z, z); const r2 = x2 + y2 + z2, rho2 = x2 + y2; const r = fsqrt(r2), rho = fsqrt(rho2); const r4 = mul(r2, r2), r8 = mul(r4, r4); const cth = fdiv(z, r), sth = fdiv(rho, r); let cph, sph; if (rho > SMALL) { cph = fdiv(x, rho); sph = fdiv(y, rho); } else { cph = ONE; sph = 0n; } const c8th = chebT(cth, N), s8th = mul(sth, chebU(cth, N - 1)); const c8ph = chebT(cph, N), s8ph = mul(sph, chebU(cph, N - 1)); return [mul(mul(r8, s8th), c8ph), mul(mul(r8, s8th), s8ph), mul(r8, c8th)]; };
  const addV = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const refOrbit = (Cfp, iters) => { const V = [Cfp.map(c => c)]; for (let i = 0; i < iters; i++) V.push(addV(P8(V[i]), Cfp)); return V; };
  const orbitDr = (Cfp, iters) => {
    let v = Cfp.map(c => c), dr = ONE;
    for (let k = 0; k < iters; k++) {
      const r2 = mul(v[0], v[0]) + mul(v[1], v[1]) + mul(v[2], v[2]);
      if (r2 > 4n * SCALE) break;
      const r6 = mul(mul(r2, r2), r2), r = fsqrt(r2), r7 = mul(r6, r);
      dr = 8n * mul(r7, dr) + ONE;
      v = addV(P8(v), Cfp);
    }
    const rF = fsqrt(mul(v[0], v[0]) + mul(v[1], v[1]) + mul(v[2], v[2]));
    return { rFinal: rF, drFinal: dr };
  };
  const radius = (v) => fsqrt(mul(v[0], v[0]) + mul(v[1], v[1]) + mul(v[2], v[2]));
  return { fromNum, toNum, refOrbit, orbitDr, radius, SCALE };
}
const HP = makeHP(384);

/* ────────────────────────── checks ────────────────────────── */
function checkPrims(r) {
  let fails = 0, worst = 0; const rows = [];
  // tolerances: single HDR ops ≈ f32 ulp; poly/atan/sincos slightly looser
  // 'atanT builtin.3' documents the GLSL builtin atan() quality on this driver (only used
  // for |ratio| > 0.1, the near-rebase regime) — ANGLE d3d11 measured ~1e-4 at small angles.
  const TOL = { default: 5e-7, 'powm1_8 u=-0.3': 2e-6, 'atanT series.09': 2e-6, 'atanT builtin.3': 5e-4, 'sin x=0.5': 2e-6, 'cosm1 x=0.5': 2e-6 };
  for (const row of r.results) {
    const got = hdrOut([row.m, row.e]);
    const want = primsOracle(row);
    const e = rel(got, want);
    const tol = TOL[row.name] || TOL.default;
    const bad = e > tol;
    if (bad) fails++;
    worst = Math.max(worst, e);
    rows.push(`     ${bad ? '✗' : '✓'} ${row.name.padEnd(18)} rel=${e.toExponential(2)}`);
  }
  return { fails, worst, rows, n: r.results.length };
}

function checkStep(r) {
  let fails = 0, worst = 0; const rows = [];
  r.results.forEach((row, i) => {
    const o = stepOracle(row.V, row.d, row.ddrIn, row.drTest);
    const dG = [hdrOut(row.dx), hdrOut(row.dy), hdrOut(row.dz)];
    const eDelta = len3([dG[0] - o.delta[0], dG[1] - o.delta[1], dG[2] - o.delta[2]]) / Math.max(len3(o.delta), 1e-300);
    const eDr = rel(hdrOut(row.Dr), o.Dr);
    const eDdr = rel(hdrOut(row.Ddr), o.Ddr);
    const e = Math.max(eDelta, eDr, eDdr);
    const tol = 2e-5;
    const bad = e > tol;
    if (bad) fails++;
    worst = Math.max(worst, e);
    const mag = len3(row.d).toExponential(0);
    rows.push(`     ${bad ? '✗' : '✓'} case${i} |δ|≈${mag.padStart(6)}  relΔ=${eDelta.toExponential(2)}  relΔr=${eDr.toExponential(2)}  relΔdr=${eDdr.toExponential(2)}`);
  });
  return { fails, worst, rows, n: r.results.length };
}

function checkOrbit(r) {
  const C = r.center, ITERS = r.iters;
  const Cfp = C.map(HP.fromNum);
  const ref = HP.refOrbit(Cfp, ITERS);
  const refDr = HP.orbitDr(Cfp, ITERS);
  let fails = 0, worst = 0; const rows = [];
  for (const row of r.results) {
    const Cd = Cfp.map((c, j) => c + HP.fromNum(row.dpos[j]));
    const pix = HP.refOrbit(Cd, ITERS);
    const pixDr = HP.orbitDr(Cd, ITERS);
    // exact BigInt subtraction → true final δ, Δr, Δdr
    const dTrue = [0, 1, 2].map(j => HP.toNum(pix[ITERS][j] - ref[ITERS][j]));
    const DrTrue = HP.toNum(HP.radius(pix[ITERS]) - HP.radius(ref[ITERS]));
    const DdrTrue = HP.toNum(pixDr.drFinal - refDr.drFinal);
    const dG = [hdrOut(row.dx), hdrOut(row.dy), hdrOut(row.dz)];
    const eDelta = len3([dG[0] - dTrue[0], dG[1] - dTrue[1], dG[2] - dTrue[2]]) / Math.max(len3(dTrue), 1e-300);
    const eDr = rel(hdrOut(row.Dr), DrTrue);
    const eDdr = rel(hdrOut(row.Ddr), DdrTrue);
    const flags = `it=${row.iters} rb=${row.rebases} esc=${row.escaped}`;
    const e = Math.max(eDelta, eDr, eDdr);
    // shallow offsets (≥1e-3) have a FORMULATION floor ~1e-3 that the CPU f64 mirror also
    // measures (de-depth.test.mjs: 9.99e-4 / 1.05e-3 for these exact cases) — not GPU error.
    // Deep offsets are pure f32-mantissa noise × 14-iter gain → expect ~1e-7..1e-6.
    const tol = row.mag >= 1e-3 ? 3e-3 : 1e-3;
    const bad = e > tol || row.iters !== ITERS;
    if (bad) fails++;
    worst = Math.max(worst, e);
    rows.push(`     ${bad ? '✗' : '✓'} δc=${row.mag.toExponential(0).padStart(6)}  relδ=${eDelta.toExponential(2)}  relΔr=${eDr.toExponential(2)}  relΔdr=${eDdr.toExponential(2)}  (${flags})`);
  }
  return { fails, worst, rows, n: r.results.length };
}

/* ────────────────────────── run ────────────────────────── */
const ALL_BACKENDS = [
  { name: 'd3d11', opts: { channel: 'chrome', headless: false, args: ['--use-gl=angle', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--enable-gpu'] } },
  { name: 'gl', opts: { channel: 'chrome', headless: false, args: ['--use-gl=angle', '--use-angle=gl', '--ignore-gpu-blocklist', '--enable-gpu'] } },
  { name: 'vulkan', opts: { channel: 'chrome', headless: false, args: ['--use-gl=angle', '--use-angle=vulkan', '--ignore-gpu-blocklist', '--enable-gpu', '--enable-features=Vulkan'] } },
];
const beArg = process.argv.find(a => a.startsWith('--backend='));
const BACKENDS = beArg ? ALL_BACKENDS.filter(b => b.name === beArg.split('=')[1]) : ALL_BACKENDS;

let anyFail = false;
for (const be of BACKENDS) {
  let browser;
  try {
    browser = await chromium.launch(be.opts);
    const page = await browser.newPage();
    page.on('pageerror', e => console.log('   [pageerror]', String(e).split('\n')[0]));
    await page.goto(PAGE, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForFunction(() => window.__ready === true, null, { timeout: 30000 });
    console.log(`\n[${be.name}]`);

    const prims = await page.evaluate(() => window.runPrimsTest());
    if (prims.error) { console.log(`   renderer: ${prims.renderer || '?'}\n   PRIMS ERROR: ${prims.error}`); anyFail = true; }
    else {
      console.log(`   renderer: ${prims.renderer}`);
      const c = checkPrims(prims);
      console.log(`   1) HDR primitives:   ${c.fails === 0 ? 'PASS' : 'FAIL (' + c.fails + '/' + c.n + ')'}  worst rel = ${c.worst.toExponential(2)}`);
      if (c.fails) { console.log(c.rows.join('\n')); anyFail = true; }
    }

    const step = await page.evaluate(() => window.runStepTest());
    if (step.error) { console.log(`   STEP ERROR: ${step.error}`); anyFail = true; }
    else {
      const c = checkStep(step);
      console.log(`   2) pertStep (§3+Δdr): ${c.fails === 0 ? 'PASS' : 'FAIL (' + c.fails + '/' + c.n + ')'}  worst rel = ${c.worst.toExponential(2)}`);
      console.log(c.rows.join('\n'));
      if (c.fails) anyFail = true;
    }

    for (const ci of [0, 1]) {
      const orbit = await page.evaluate((i) => window.runOrbitTest(i), ci);
      if (orbit.error) { console.log(`   ORBIT[${ci}] ERROR: ${orbit.error}`); anyFail = true; continue; }
      const c = checkOrbit(orbit);
      console.log(`   3) pertOrbit center[${orbit.center}]: ${c.fails === 0 ? 'PASS' : 'FAIL (' + c.fails + '/' + c.n + ')'}  worst rel = ${c.worst.toExponential(2)}  (vs hp BigInt truth)`);
      console.log(c.rows.join('\n'));
      if (c.fails) anyFail = true;
    }

    const march = await page.evaluate(() => window.runMarchTest());
    if (march.error || !march.compiled) { console.log(`   4) march stub: FAIL — ${march.error}`); anyFail = true; }
    else {
      const finite = march.results.every(r => isFinite(r.steps) && isFinite(r.tM) && isFinite(r.tE) && isFinite(r.deF));
      console.log(`   4) march stub: ${finite ? 'PASS (compiles, runs, finite)' : 'FAIL (non-finite output)'}`);
      march.results.forEach((r, i) => console.log(`     ray${i}: steps=${r.steps}  t=${(r.tM * Math.pow(2, r.tE)).toExponential(3)}  de=${r.deF.toExponential(3)}`));
      if (!finite) anyFail = true;
    }
  } catch (e) {
    console.log(`\n[${be.name}]  launch/run failed: ${String(e).split('\n')[0]}`);
    anyFail = true;
  } finally { if (browser) await browser.close(); }
}
console.log(`\n${anyFail ? 'RESULT: FAILURES PRESENT — kernel arithmetic not yet GPU-clean.' : 'RESULT: ALL PASS — kernel arithmetic is GPU-verified; march loop compiles.'}
INTERPRET:
 • Test 3 is the headline: the FULL per-pixel recurrence (orbit texture → pertStep →
   Zhuoran rebase → Δr/Δdr) run on the real GPU, measured against the arbitrary-precision
   BigInt orbit, at pixel offsets down to 1e-40 (far below f32 underflow ~1e-38).
   relδ/relΔdr ≈ 1e-5..1e-6 = f32-mantissa noise × 14-iteration gain — the expected HDR floor.
 • Rebases should read rb=0 here (bounded centers, no dip below 0.05, orbit long enough);
   the rebase PATH is compiled+branch-evaluated but not exercised — see §8.6 CPU work.
 • Oracles mirror the texture's f32 rounding, so failures indicate GLSL arithmetic bugs,
   not reference-precision differences.`);
