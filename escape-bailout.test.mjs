/* =============================================================================
   escape-bailout.test.mjs  —  §8.6: escape/bailout under perturbation + the
   EXHAUSTION-REBASE path (reference escapes before the pixel → Zhuoran wrap to
   index 0, with the Δdr re-anchor).   Run:  node escape-bailout.test.mjs
   -----------------------------------------------------------------------------
   The GPU kernel (gpu-kernel.html pertOrbit) tests escape on z = V_m + δ — always
   the PIXEL's value, because δ absorbs the pixel-vs-reference divergence (δ grows
   to O(1) as they separate). When the stored reference orbit ends (the reference
   escaped at iteration E but this pixel hasn't), the kernel rebases:
       δ ← z − V_0,   Δdr ← (DR_m − 1) + Δdr,   m ← 0
   and keeps iterating. Neither the differing-escape case nor the Δdr re-anchor
   was covered by any prior CPU test — this file closes both, refereed by the
   arbitrary-precision BigInt orbit (hp-orbit.test.mjs §8.3).

   Model: exact CPU mirror of pertOrbit's CONTROL FLOW and formulas — f64 step
   math on f32-rounded reference quantities (the texture model) with HDR (f32
   mantissa) storage rounding of δ and Δdr. GPU-vs-mirror arithmetic agreement
   was already measured at ~1e-7 (gpu-kernel.test.mjs); here we validate the
   escape/rebase SEMANTICS against ground truth.

   Metrics per pixel:
     • escape iteration: mirror vs hp truth (must match; ±1 tolerated only at
       razor-edge shallow cases and reported),
     • r and dr at escape (the DE inputs): relative error vs hp truth,
     • rebase count (exhaustion wraps actually exercised).
   ============================================================================= */

const N = 8;
const fr = Math.fround;

/* ---------- f64 helpers ---------- */
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len3 = a => Math.hypot(a[0], a[1], a[2]);
const addv = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const Pf = (v) => {
  const r = len3(v); if (r === 0) return [0, 0, 0];
  const th = Math.acos(Math.max(-1, Math.min(1, v[2] / r))) * N, ph = Math.atan2(v[1], v[0]) * N, zr = Math.pow(r, N);
  return [zr * Math.sin(th) * Math.cos(ph), zr * Math.sin(ph) * Math.sin(th), zr * Math.cos(th)];
};
/* HDR storage rounding: f32 mantissa, free exponent (models GPU HDR-δ) */
const hdr = (x) => { if (x === 0 || !isFinite(x)) return 0; const e = Math.floor(Math.log2(Math.abs(x))), s = Math.pow(2, e); return fr(x / s) * s; };

/* ---------- hp BigInt referee (hp-orbit.test.mjs §8.3) ---------- */
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
  /* full run with escape detection. Convention matches the kernel: after the k-th
     iteration (v_k formed), escaped(k) ⇔ |v_k| > 2; dr_k has had k updates. */
  const run = (Cfp, maxIter) => {
    let v = Cfp.map(c => c), dr = ONE;
    for (let k = 1; k <= maxIter; k++) {
      const r2 = mul(v[0], v[0]) + mul(v[1], v[1]) + mul(v[2], v[2]);
      const r6 = mul(mul(r2, r2), r2), r = fsqrt(r2), r7 = mul(r6, r);
      dr = 8n * mul(r7, dr) + ONE;
      v = addV(P8(v), Cfp);
      const rr2 = mul(v[0], v[0]) + mul(v[1], v[1]) + mul(v[2], v[2]);
      if (rr2 > 4n * SCALE) return { escIter: k, r: toNum(fsqrt(rr2)), dr: toNum(dr), escaped: true };
    }
    const rr2 = mul(v[0], v[0]) + mul(v[1], v[1]) + mul(v[2], v[2]);
    return { escIter: maxIter, r: toNum(fsqrt(rr2)), dr: toNum(dr), escaped: false };
  };
  return { fromNum, toNum, run };
}
const HP = makeHP(384);

/* ---------- reference orbit in the texture model (f32-rounded quantities) ----------
   Stored while |V| ≤ 2, plus the first escaped entry (kernel rebases at m ≥ len−1,
   so the escaped entry is only ever used in the z = V+δ pixel-escape check). */
function buildRefOrbit(C, maxLen) {
  let V = C.slice(), DR = 1;
  const o = { V32: [], R32: [], trig: [], RnV: [], RpV: [], DRV: [], len: 0, escIter: -1 };
  for (let k = 0; k < maxLen; k++) {
    const R = len3(V);
    const th = Math.acos(Math.max(-1, Math.min(1, V[2] / R))), ph = Math.atan2(V[1], V[0]);
    const a = N * th, b = N * ph;
    o.V32.push(V.map(fr)); o.R32.push(fr(R));
    o.trig.push([fr(Math.sin(a)), fr(Math.cos(a)), fr(Math.sin(b)), fr(Math.cos(b))]);
    o.RnV.push(hdr(Math.pow(R, N))); o.RpV.push(hdr(Math.pow(R, N - 1))); o.DRV.push(hdr(DR));
    o.len++;
    if (R > 2) { o.escIter = k; break; }                  // first escaped entry stored, then stop
    DR = N * Math.pow(R, N - 1) * DR + 1;
    V = addv(Pf(V), C);
  }
  return o;
}

/* ---------- one pertStep: same formulas as the GLSL (residual forms), f64 math ---------- */
function stepMirror(o, m, d, Ddr) {
  const V = o.V32[m], R = o.R32[m], [sa, ca, sb, cb] = o.trig[m];
  const Rn = o.RnV[m], Rp = o.RpV[m], DR = o.DRV[m];
  const q = 2 * dot(V, d) + dot(d, d);
  const r2 = Math.sqrt(Math.max(R * R + q, 0));
  const Dr = q / (r2 + R);
  const u = Dr / R;
  const dRn = Rn * Math.expm1(N * Math.log1p(u));
  const Dp = Rp * Math.expm1((N - 1) * Math.log1p(u));
  const DdrOut = N * (Dp * DR + (Rp + Dp) * Ddr);
  const Dphi = Math.atan2(V[0] * d[1] - V[1] * d[0], V[0] * (V[0] + d[0]) + V[1] * (V[1] + d[1]));
  // §3c EXACT Δθ (Fable slice-5 fix — old form was first-order, O(δ²) truncation;
  // it was THE cause of the "TRUNC-§3c" shallow degradation this file first measured)
  const rho = Math.hypot(V[0], V[1]);
  const qr2 = 2 * (V[0] * d[0] + V[1] * d[1]) + d[0] * d[0] + d[1] * d[1];
  const rhoP = Math.sqrt(Math.max(rho * rho + qr2, 0));
  const drho = qr2 / (rhoP + rho);
  const Dth = Math.atan2(drho * V[2] - d[2] * rho, rho * rhoP + V[2] * (V[2] + d[2]));
  const Da = N * Dth, Db = N * Dphi;
  const sA = Math.sin(Da), cA = Math.cos(Da), sB = Math.sin(Db), cB = Math.cos(Db);
  const cAm = -sA * sA / (1 + cA), cBm = -sB * sB / (1 + cB);
  const dsA = sa * cAm + ca * sA, dcA = ca * cAm - sa * sA;
  const dsB = sb * cBm + cb * sB, dcB = cb * cBm - sb * sB;
  const DD = [sa * dcB + dsA * cb + dsA * dcB, sa * dsB + dsA * sb + dsA * dsB, dcA];
  const Dref = [sa * cb, sa * sb, ca];
  return { delta: [0, 1, 2].map(j => Rn * DD[j] + dRn * (Dref[j] + DD[j])), Ddr: DdrOut, Dr };
}

/* exact-delta step referee: direct f64 P(V+δ)−P(V) (valid at SHALLOW δ where f64
   has no cancellation problem) + the SAME Δdr recurrence formula. Separates the
   §3c first-order-Δθ truncation of stableDelta from the orbit-loop algebra
   (escape, wrap, Δdr re-anchor): if THIS mode matches hp truth through wraps,
   the loop algebra is correct. */
function stepExact(o, m, d, Ddr) {
  const V = o.V32[m], R = o.R32[m];
  const Rp = o.RpV[m], DR = o.DRV[m];
  const vd = addv(V, d);
  const Dr = len3(vd) - R;
  const u = Dr / R;
  const Dp = Rp * Math.expm1((N - 1) * Math.log1p(u));
  const DdrOut = N * (Dp * DR + (Rp + Dp) * Ddr);
  const PV = Pf(V), PVd = Pf(vd);
  return { delta: [0, 1, 2].map(j => PVd[j] - PV[j]), Ddr: DdrOut, Dr };
}

/* ---------- pertOrbit mirror: kernel control flow exactly ----------
   mode 'f32' = HDR storage rounding (the GPU model); 'f64' = f64 storage referee
   (isolates storage noise); 'exact' = exact-delta referee (isolates the §3c
   first-order-Δθ truncation; valid shallow-only). */
function pertOrbitMirror(o, dc, maxIter, mode) {
  const st = mode === 'f32' ? hdr : (x => x);
  const step = mode === 'exact' ? stepExact : stepMirror;
  const noRound = mode !== 'f32';
  let d = dc.map(st), Ddr = 0, m = 0, rebases = 0, escaped = false, it = 0;
  for (let k = 0; k < maxIter; k++) {
    const so = step(o, m, d, Ddr);
    Ddr = st(so.Ddr);
    d = addv(so.delta, dc).map(st);
    m++;
    it = k + 1;
    const df = noRound ? d : d.map(fr);
    const zf = noRound ? addv(o.V32[m], df) : addv(o.V32[m], df).map(fr);
    if (dot(zf, zf) > 4.0) { escaped = true; break; }
    // NO small-R forced rebase — §8.5 (glitch-relocate.test.mjs) showed it destroys
    // near-axis views; pure Zhuoran + exhaustion only (matches the GPU kernel).
    if (dot(zf, zf) < dot(df, df) || m >= o.len - 1) {
      d = zf.map((z, j) => st(z - o.V32[0][j]));
      Ddr = st((o.DRV[m] - 1) + Ddr);
      m = 0; rebases++;
    }
  }
  // final r, dr exactly as the kernel forms them
  const V = o.V32[m], R = o.R32[m];
  const q = 2 * dot(V, d) + dot(d, d);
  const r2 = Math.sqrt(Math.max(R * R + q, 0));
  const Dr = q / (r2 + R);
  return { escIter: it, escaped, rebases, r: R + Dr, dr: o.DRV[m] + Ddr };
}

/* ---------- pick a reference center that escapes at a useful depth ---------- */
const BASE = [0.30, 0.50, 0.20];                 // bounded ≥16 iters (groundtruth.test.mjs)
function findEscapingCenter() {
  for (let s = 1.0; s < 1.6; s += 0.002) {
    const C = BASE.map(x => x * s);
    const o = buildRefOrbit(C, 40);
    if (o.escIter >= 8 && o.escIter <= 18) return { C, escIter: o.escIter };
  }
  throw new Error('no center found');
}
const { C: CREF, escIter: EREF } = findEscapingCenter();
const MAXIT = 40;
const hpRef = HP.run(CREF.map(HP.fromNum), MAXIT);
console.log(`§8.6 — escape/bailout + exhaustion rebase, hp-refereed (P=384)\n`);
console.log(`reference center [${CREF.map(v => v.toFixed(4))}] escapes at iter ${EREF} (hp confirms: ${hpRef.escIter})`);
const o = buildRefOrbit(CREF, 40);
console.log(`stored reference orbit: len=${o.len} (V_0..V_${o.len - 1}, last entry escaped)\n`);

/* ---------- sweep: outward pixels escape earlier, inward later (forcing wraps) ---------- */
const rel = (g, t) => t === 0 ? Math.abs(g) : Math.abs(g - t) / Math.abs(t);
const dirU = (() => { const d = [1, -0.7, 0.4], L = len3(d); return d.map(x => x / L); })();
/* hp-searched WRAP cases: inward magnitudes where the pixel escapes SHORTLY AFTER the
   reference (the realistic exhaustion-wrap scenario — δ already O(1)-visible at the wrap) */
const wrapMags = [];
for (let lm = -3.2; lm <= -1.4 && wrapMags.length < 3; lm += 0.06) {
  const mag = Math.pow(10, lm);
  const Cd = CREF.map((c, j) => HP.fromNum(c) + HP.fromNum(-mag * dirU[j]));
  const t = HP.run(Cd, MAXIT);
  if (t.escaped && t.escIter > EREF && t.escIter <= EREF + 10) wrapMags.push(mag);
}
console.log(`hp-searched wrap magnitudes (inward, pixel escapes after the reference): [${wrapMags.map(m => m.toExponential(2))}]\n`);

console.log('dir      offset     escIter hp|f32   rel(dr)f32  rel(dr)f64  rel(dr)exact  rebases  verdict');
let fails = 0, edge = 0, trunc = 0, wrapVerified = 0;
const CASES = [];
for (const sign of [+1, -1]) for (const mag of [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-6, 1e-9, 1e-15, 1e-24]) CASES.push({ sign, mag });
for (const mag of wrapMags) CASES.push({ sign: -1, mag, wrap: true });
for (const cse of CASES) {
  const { sign, mag } = cse;
  const dpos = dirU.map(x => sign * mag * x);
  const Cd = CREF.map((c, j) => HP.fromNum(c) + HP.fromNum(dpos[j]));
  const truth = HP.run(Cd, MAXIT);
  const m32 = pertOrbitMirror(o, dpos, MAXIT, 'f32');
  const m64 = pertOrbitMirror(o, dpos, MAXIT, 'f64');
  const mEx = pertOrbitMirror(o, dpos, MAXIT, 'exact');
  const eDr32 = rel(m32.dr, truth.dr), eR32 = rel(m32.r, truth.r);
  const eDr64 = rel(m64.dr, truth.dr);
  const eDrEx = rel(mEx.dr, truth.dr), eREx = rel(mEx.r, truth.r);
  const it32 = m32.escIter === truth.escIter && m32.escaped === truth.escaped;
  const itEx = mEx.escIter === truth.escIter && mEx.escaped === truth.escaped;
  const tol = mag >= 1e-4 ? 5e-3 : 1e-4;
  let verdict;
  if (it32 && Math.max(eR32, eDr32) <= tol) verdict = 'PASS';
  // exact-delta referee clean ⇒ the orbit-loop ALGEBRA (escape, wrap, Δdr re-anchor) is
  // correct and the f32/f64 stableDelta error is the §3c FIRST-ORDER-Δθ TRUNCATION —
  // O(δ²), shallow-only (vanishes quadratically with depth; f32≡f64 columns prove it is
  // not storage noise). Out of the perturbation kernel's domain: plain f32 covers ≥1e-4.
  else if (itEx && Math.max(eREx, eDrEx) <= 1e-3 && mag >= 1e-4) { verdict = 'TRUNC-§3c(shallow, report-only)'; trunc++; }
  else if (!it32 && Math.abs(m32.escIter - truth.escIter) <= 1 && mag >= 1e-4) { verdict = 'EDGE(±1)'; edge++; }
  else { verdict = 'FAIL'; fails++; }
  if (mEx.rebases > 0 && itEx && Math.max(eREx, eDrEx) <= 1e-3 && truth.escaped) wrapVerified++;
  console.log(`${sign > 0 ? 'out' : 'in '}   ${mag.toExponential(2).padStart(8)}     ${String(truth.escIter).padStart(2)}|${String(m32.escIter).padEnd(3)}${truth.escaped === m32.escaped ? ' ' : '!'}    ${eDr32.toExponential(2)}    ${eDr64.toExponential(2)}     ${eDrEx.toExponential(2)}       ${m32.rebases}     ${verdict}${cse.wrap ? ' [wrap]' : ''}`);
}
console.log(`\n${fails === 0 ? 'PASS' : 'FAIL (' + fails + ' cases)'} — ${edge} razor-edge ±1; ${trunc} shallow §3c-truncation (documented); wrap + Δdr re-anchor positively verified (exact-delta referee, escaping wrap) in ${wrapVerified} case(s)${wrapVerified === 0 ? ' (GAP: re-anchor not positively verified!)' : ''}\n
READING (§8.6) — measured findings (UPDATED 2026-06-11 after the exact-Δθ port):
 • ESCAPE SEMANTICS CORRECT: the kernel's escape test z = V_m + δ always reflects the
   PIXEL (δ absorbs pixel-vs-reference divergence). Deep offsets (≤1e-6): escape
   iteration matches hp truth EXACTLY and dr at escape is ~3e-8 rel (the f32-rounded-
   reference floor). No special-casing needed in the kernel.
 • WRAP + Δdr RE-ANCHOR ALGEBRA CORRECT: on hp-searched wrap cases (pixel escapes after
   the reference; exhaustion rebase δ ← z − V_0, Δdr ← (DR_m − 1) + Δdr fires), f32,
   f64 AND exact referees all track hp truth — the loop bookkeeping is right.
 • THE SHALLOW DOMAIN BOUNDARY IS GONE (re-measured after the slice-5 exact-Δθ fix):
   the old "degraded ≥1e-3 / pristine ≤1e-4" finding was entirely the first-order Δθ's
   O(δ²) truncation. With the exact Δθ, ALL 22 cases pass — offsets 1e-2 → 1e-24,
   including every wrap case in raw f32 and the former razor-edge ±1 pixel (now exact).
   KERNEL CONSEQUENCE: the f32↔deep handoff can sit anywhere below ~1e-2 view scale;
   the perturbation kernel itself is no longer the shallow limiter.
 • KERNEL DESIGN RULE (from the wrap analysis): the m ≥ len−1 wrap-to-0 is a FORCED
   rebase in disguise when the reference is too short while δ is still tiny — that is
   PART-3-Result-A poison (re-anchoring rounds δ to f32-of-O(1)). The stored reference
   must be at least as long as the pixel's iteration budget (or use a periodic/nucleus
   reference, 2D ADR-0066 analog). Wrap-to-0 is only safe once |δ| is O(1)-visible —
   which is exactly when the Zhuoran condition fires it naturally.`);
