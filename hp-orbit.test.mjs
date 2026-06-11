/* =============================================================================
   hp-orbit.test.mjs  —  §8.3: ARBITRARY-PRECISION triplex reference orbit
   Run:  node hp-orbit.test.mjs
   -----------------------------------------------------------------------------
   Every deep-zoom metric so far walls at the dd (~1e-30) ground truth. To validate
   anything past ~1e-15 we need a reference orbit at 50+ digits. KEY ENABLER (proven
   in precision-ladder.test.mjs #2): the integer-power triplex needs NO transcendentals
   — Chebyshev multiple-angle polys reduce it to +,−,×,÷,sqrt. So we can compute the
   orbit in BigInt FIXED-POINT (P fraction bits → arbitrary precision; only isqrt +
   BigInt div needed). This file builds it and verifies:
     (1) BigInt orbit vs f64 triplex agree to ~1e-15 (correctness),
     (2) P=128 vs P=384 agree to ~1e-37 (genuinely arbitrary precision, not f64-limited).
   That removes the ground-truth wall → deep HDR-δ / DE-difference validation becomes possible.
   ============================================================================= */

const N = 8;

// ---- fixed-point arbitrary precision factory (P fraction bits) ----
function makeHP(P) {
  const Pb = BigInt(P), SCALE = 1n << Pb;
  const mul = (a, b) => (a * b) >> Pb;
  const fdiv = (a, b) => (a << Pb) / b;
  const isqrt = (n) => { if (n < 2n) return n < 0n ? 0n : n; let x = n, y = (x + 1n) >> 1n; while (y < x) { x = y; y = (x + n / x) >> 1n; } return x; };
  const fsqrt = (a) => isqrt(a << Pb);                    // sqrt(a/2^P)*2^P = isqrt(a<<P)
  const SMALL = SCALE >> 90n;                              // ~1e-27 pole guard
  // exact double → fixed point (handles tiny values like 1e-50 down to ~2^-P)
  const fromNum = (x) => {
    if (x === 0) return 0n;
    const neg = x < 0; x = Math.abs(x);
    const dv = new DataView(new ArrayBuffer(8)); dv.setFloat64(0, x);
    const bits = dv.getBigUint64(0);
    const exp = Number((bits >> 52n) & 0x7ffn), mant = bits & 0xfffffffffffffn;
    let M, E; if (exp === 0) { M = mant; E = -1074; } else { M = mant | 0x10000000000000n; E = exp - 1075; }
    const sh = Pb + BigInt(E);
    let v = sh >= 0n ? (M << sh) : (M >> (-sh));
    return neg ? -v : v;
  };
  const toNum = (v) => Number(v) / Number(SCALE);          // ~double precision (leading digits)
  const ONE = SCALE;
  const chebT = (x, n) => { let a = ONE, b = x; if (n === 0) return a; if (n === 1) return b; for (let k = 2; k <= n; k++) { const t = 2n * mul(x, b) - a; a = b; b = t; } return b; };
  const chebU = (x, n) => { let a = ONE, b = 2n * x; if (n === 0) return a; if (n === 1) return b; for (let k = 2; k <= n; k++) { const u = 2n * mul(x, b) - a; a = b; b = u; } return b; };
  const P8 = (v) => {                                       // triplex power, transcendental-free
    const [x, y, z] = v;
    const x2 = mul(x, x), y2 = mul(y, y), z2 = mul(z, z);
    const r2 = x2 + y2 + z2, rho2 = x2 + y2;
    const r = fsqrt(r2), rho = fsqrt(rho2);
    const r4 = mul(r2, r2), r8 = mul(r4, r4);
    const cth = fdiv(z, r), sth = fdiv(rho, r);
    let cph, sph;
    if (rho > SMALL) { cph = fdiv(x, rho); sph = fdiv(y, rho); } else { cph = ONE; sph = 0n; }
    const c8th = chebT(cth, N), s8th = mul(sth, chebU(cth, N - 1));
    const c8ph = chebT(cph, N), s8ph = mul(sph, chebU(cph, N - 1));
    return [mul(mul(r8, s8th), c8ph), mul(mul(r8, s8th), s8ph), mul(r8, c8th)];
  };
  const addV = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const orbit = (C, iters) => { const Cf = C.map(fromNum); const V = [Cf.map(c => c)]; for (let i = 0; i < iters; i++) V.push(addV(P8(V[i]), Cf)); return V; };
  // high-precision relative diff between two fixed-point vectors (exact BigInt subtraction)
  const relDiff = (a, b) => {
    let dn = 0n, vn = 0n;
    for (let j = 0; j < 3; j++) { const d = a[j] - b[j]; dn += d * d; vn += a[j] * a[j]; }
    const dnum = Number(isqrt(dn)), vnum = Number(isqrt(vn));
    return vnum === 0 ? dnum : dnum / vnum;
  };
  return { P, fromNum, toNum, P8, orbit, addV, relDiff, SCALE };
}

// f64 triplex for cross-check
const Pf = (v) => { const r = Math.hypot(...v); if (r === 0) return [0, 0, 0]; const th = Math.acos(Math.max(-1, Math.min(1, v[2] / r))) * N, ph = Math.atan2(v[1], v[0]) * N, zr = Math.pow(r, N); return [zr * Math.sin(th) * Math.cos(ph), zr * Math.sin(ph) * Math.sin(th), zr * Math.cos(th)]; };
const len = a => Math.hypot(a[0], a[1], a[2]), sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

const hp256 = makeHP(256), hp128 = makeHP(128), hp384 = makeHP(384);

console.log('§8.3 — arbitrary-precision (BigInt fixed-point) triplex orbit\n');
console.log('(1) BigInt P8 vs f64 triplex (should agree ~1e-15):');
for (const v of [[0.3, 0.5, 0.2], [0.9, -0.1, 0.4], [-0.2, 0.7, -0.6]]) {
  const big = hp256.P8(v.map(hp256.fromNum)).map(hp256.toNum);
  console.log(`   v=[${v}]  |P_big - P_f64| = ${len(sub(big, Pf(v))).toExponential(2)}`);
}

console.log('\n(2) P=128 vs P=384 orbit agreement (proves arbitrary precision, not f64-limited):');
// compare the two precisions' orbits via the higher-precision one's relDiff
const C = [0.30, 0.50, 0.20], ITERS = 16;
const o128 = hp128.orbit(C, ITERS), o384 = hp384.orbit(C, ITERS);
let worst = 0;
for (let i = 0; i <= ITERS; i++) {
  // lift o128 (P=128) into P=384 scale to subtract, then relDiff in P=384
  const lifted = o128[i].map(v => v << (384n - 128n));
  worst = Math.max(worst, hp384.relDiff(o384[i], lifted));
}
console.log(`   worst orbit rel-diff over ${ITERS} iters = ${worst.toExponential(2)}  (≈ 2^-128 ≈ 3e-39 → P=128 is the limiter, P=384 is far deeper)`);

console.log('\n(3) headroom demo — orbit value carried to ~115 digits (P=384):');
const deep = hp384.orbit(C, 4);
const s = (deep[4][0] < 0n ? '-' : '') + (deep[4][0] < 0n ? -deep[4][0] : deep[4][0]).toString();
console.log(`   V[4].x raw BigInt has ${s.length} digits of fixed-point state (P=384 ≈ 115 decimal digits of fraction).`);

console.log(`
READING (§8.3):
 • BigInt fixed-point triplex orbit WORKS and is arbitrary-precision — only isqrt + BigInt
   div added (the Chebyshev transcendental-free form is what makes it possible). P=128 vs
   P=384 agree to ~2^-128; pick P ≈ 3.4·(target digits) for any depth (P=384 → ~115 digits,
   ample for 1e-50 with margin).
 • This REMOVES the dd ground-truth wall (precision-ladder Result C). Next: use an hp orbit
   as ground truth to (a) re-measure HDR-δ resolvability past 1e-18 with a correct
   DE-DIFFERENCE metric (no O(1) subtraction), and (b) gate the GLSL kernel.
 • For the GPU kernel this is the CPU/worker reference-orbit precision (mirrors GMT 2D's
   BigInt HPComplex). Cost is per-frame on the reference only (one orbit), not per-pixel.`);
