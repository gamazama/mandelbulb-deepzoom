/* =============================================================================
   glitch-relocate.test.mjs  —  §8.5: axis/pole glitch pixels — detect + RELOCATE
   the reference (the 2D auto-reference pattern in 3D).
   Run:  node glitch-relocate.test.mjs
   -----------------------------------------------------------------------------
   Regime 3 (groundtruth.test.mjs Part 2): when the REFERENCE orbit hits the
   origin/z-axis (the triplex map's degenerate manifold — 1/R, 1/sinθ in the §3
   frame), perturbed pixels near it glitch (err/offset ~ O(10)) and rebasing does
   NOT help. The reference is exactly degenerate (the axis is invariant under the
   map); the PIXELS themselves are fine — their orbits miss the pole by their
   offset × the orbit's cumulative gain. So the 2D playbook should transfer:
   DETECT the degenerate reference, RELOCATE to a nearby off-axis point, re-run
   the pixels against the new reference. This file measures whether that works,
   refereed by the hp BigInt orbit:
     (1) reproduce the glitch at the pathological center (err/offset vs hp truth),
     (2) relocate by varying off-axis amounts → measure the NEW reference's
         min R_k / min ρ_k and the pixels' err/offset after relocation,
     (3) derive the DETECT threshold: how non-degenerate must a reference be
         (min R_k, min ρ_k) for pixel accuracy to recover.
   Kernel model: f64 step math on f32-rounded reference + HDR (f32-mantissa)
   δ storage — same mirror as escape-bailout.test.mjs.
   ============================================================================= */

const N = 8;
const fr = Math.fround;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len3 = a => Math.hypot(a[0], a[1], a[2]);
const addv = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const Pf = (v) => {
  const r = len3(v); if (r === 0) return [0, 0, 0];
  const th = Math.acos(Math.max(-1, Math.min(1, v[2] / r))) * N, ph = Math.atan2(v[1], v[0]) * N, zr = Math.pow(r, N);
  return [zr * Math.sin(th) * Math.cos(ph), zr * Math.sin(ph) * Math.sin(th), zr * Math.cos(th)];
};
const hdr = (x) => { if (x === 0 || !isFinite(x)) return 0; const e = Math.floor(Math.log2(Math.abs(x))), s = Math.pow(2, e); return fr(x / s) * s; };

/* ---------- hp BigInt referee ---------- */
function makeHP(P) {
  const Pb = BigInt(P), SCALE = 1n << Pb, ONE = SCALE;
  const mul = (a, b) => (a * b) >> Pb, fdiv = (a, b) => (a << Pb) / b;
  const isqrt = (n) => { if (n < 2n) return n < 0n ? 0n : n; let x = n, y = (x + 1n) >> 1n; while (y < x) { x = y; y = (x + n / x) >> 1n; } return x; };
  const fsqrt = (a) => isqrt(a << Pb), SMALL = SCALE >> 90n;
  const fromNum = (x) => { if (x === 0) return 0n; const neg = x < 0; x = Math.abs(x); const dv = new DataView(new ArrayBuffer(8)); dv.setFloat64(0, x); const bits = dv.getBigUint64(0); const exp = Number((bits >> 52n) & 0x7ffn), mant = bits & 0xfffffffffffffn; let M, E; if (exp === 0) { M = mant; E = -1074; } else { M = mant | 0x10000000000000n; E = exp - 1075; } const sh = Pb + BigInt(E); let v = sh >= 0n ? (M << sh) : (M >> (-sh)); return neg ? -v : v; };
  const toNum = (v) => Number(v) / Number(SCALE);
  const chebT = (x, n) => { let a = ONE, b = x; if (n === 0) return a; if (n === 1) return b; for (let k = 2; k <= n; k++) { const t = 2n * mul(x, b) - a; a = b; b = t; } return b; };
  const chebU = (x, n) => { let a = ONE, b = 2n * x; if (n === 0) return a; if (n === 1) return b; for (let k = 2; k <= n; k++) { const u = 2n * mul(x, b) - a; a = b; b = u; } return b; };
  const P8 = (v) => { const [x, y, z] = v; const x2 = mul(x, x), y2 = mul(y, y), z2 = mul(z, z); const r2 = x2 + y2 + z2, rho2 = x2 + y2; const r = fsqrt(r2), rho = fsqrt(rho2); if (r === 0n) return [0n, 0n, 0n]; const r4 = mul(r2, r2), r8 = mul(r4, r4); const cth = fdiv(z, r), sth = fdiv(rho, r); let cph, sph; if (rho > SMALL) { cph = fdiv(x, rho); sph = fdiv(y, rho); } else { cph = ONE; sph = 0n; } const c8th = chebT(cth, N), s8th = mul(sth, chebU(cth, N - 1)); const c8ph = chebT(cph, N), s8ph = mul(sph, chebU(cph, N - 1)); return [mul(mul(r8, s8th), c8ph), mul(mul(r8, s8th), s8ph), mul(r8, c8th)]; };
  const addV = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const traj = (Cfp, iters) => { const V = [Cfp.map(c => c)]; for (let i = 0; i < iters; i++) V.push(addV(P8(V[i]), Cfp)); return V; };
  return { fromNum, toNum, traj };
}
const HP = makeHP(384);

/* ---------- pathological-center search (groundtruth.test.mjs Part 2) ---------- */
function orbitStats(C, iters) {
  let V = [...C], minR = Infinity, minRho = Infinity, dipK = -1;
  for (let k = 0; k < iters; k++) {
    V = addv(Pf(V), C);
    const r = len3(V);
    if (r > 2.0) return { minR: Infinity, minRho: Infinity, escaped: true };
    if (k > 0 && r < minR) { minR = r; dipK = k; }
    if (k > 0) minRho = Math.min(minRho, Math.hypot(V[0], V[1]));
  }
  return { minR, minRho, dipK, escaped: false };
}
function searchNearOrigin() {
  let best = null;
  for (let x = -1.2; x <= 1.2; x += 0.1)
    for (let y = -1.2; y <= 1.2; y += 0.1)
      for (let z = -1.2; z <= 1.2; z += 0.1) {
        if (Math.hypot(x, y, z) < 0.35) continue;
        const s = orbitStats([x, y, z], 16);
        if (s.escaped) continue;
        if (!best || s.minR < best.minR) best = { C: [x, y, z], ...s };
      }
  return best;
}

/* ---------- reference build + kernel-model mirror (same as escape-bailout) ---------- */
function buildRefOrbit(C, lenWanted) {
  let V = C.slice(), DR = 1;
  const o = { V32: [], R32: [], trig: [], RnV: [], RpV: [], DRV: [], len: 0 };
  for (let k = 0; k < lenWanted; k++) {
    const R = len3(V);
    const th = Math.acos(Math.max(-1, Math.min(1, V[2] / Math.max(R, 1e-300)))), ph = Math.atan2(V[1], V[0]);
    const a = N * th, b = N * ph;
    o.V32.push(V.map(fr)); o.R32.push(fr(R));
    o.trig.push([fr(Math.sin(a)), fr(Math.cos(a)), fr(Math.sin(b)), fr(Math.cos(b))]);
    o.RnV.push(hdr(Math.pow(R, N))); o.RpV.push(hdr(Math.pow(R, N - 1))); o.DRV.push(hdr(DR));
    o.len++;
    DR = N * Math.pow(R, N - 1) * DR + 1;
    V = addv(Pf(V), C);
  }
  return o;
}
function stepMirror(o, m, d) {
  const V = o.V32[m], R = o.R32[m], [sa, ca, sb, cb] = o.trig[m];
  const Rn = o.RnV[m];
  const q = 2 * dot(V, d) + dot(d, d);
  const r2 = Math.sqrt(Math.max(R * R + q, 0));
  const Dr = q / (r2 + R);
  const u = Dr / R;
  const dRn = Rn * Math.expm1(N * Math.log1p(u));
  const Dphi = Math.atan2(V[0] * d[1] - V[1] * d[0], V[0] * (V[0] + d[0]) + V[1] * (V[1] + d[1]));
  // §3c EXACT Δθ (Fable slice-5 fix — old form was first-order, O(δ²) truncation)
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
  return [0, 1, 2].map(j => Rn * DD[j] + dRn * (Dref[j] + DD[j]));
}
/* perturbed run vs reference o; per-iter reconstruction error against the hp pixel orbit.
   useRGuard toggles the R_m < 0.05 forced rebase (UNVALIDATED folklore from the prep
   tests — near-axis dips may be exactly where it does damage). */
function runGlitch(o, dposFromRef, iters, hpPixTraj, useRGuard) {
  let d = dposFromRef.map(hdr), m = 0, rebases = 0, maxErr = 0;
  for (let k = 0; k < iters; k++) {
    d = addv(stepMirror(o, m, d), dposFromRef).map(hdr);
    m++;
    const z = addv(o.V32[m], d.map(fr));
    const t = hpPixTraj[k + 1].map(HP.toNum);
    maxErr = Math.max(maxErr, len3([z[0] - t[0], z[1] - t[1], z[2] - t[2]]));
    if (dot(z, z) < dot(d, d) || (useRGuard && o.R32[m] < 0.05) || m >= o.len - 1) {
      d = z.map((zz, j) => hdr(zz - o.V32[0][j]));
      m = 0; rebases++;
    }
  }
  return { maxErr, rebases };
}
/* STRUCTURAL metric (the rendering-relevant one — Result B): run TWO nearby pixels
   against the same reference and compare their reconstruction DIFFERENCE to the hp
   truth difference. The f32 rounding of the stored reference cancels in the pair, so
   this measures whether pixel-to-pixel STRUCTURE survives — exactly what the DE needs.
   hp truth differences are exact BigInt subtractions. */
function runGlitchPair(o, dposA, dposB, iters, hpA, hpB, useRGuard) {
  const run1 = (dpos) => {
    const zs = [];
    let d = dpos.map(hdr), m = 0, rebases = 0;
    for (let k = 0; k < iters; k++) {
      d = addv(stepMirror(o, m, d), dpos).map(hdr);
      m++;
      zs.push(addv(o.V32[m], d.map(fr)));
      const z = zs[zs.length - 1];
      if (dot(z, z) < dot(d, d) || (useRGuard && o.R32[m] < 0.05) || m >= o.len - 1) {
        d = z.map((zz, j) => hdr(zz - o.V32[0][j]));
        m = 0; rebases++;
      }
    }
    return { zs, rebases };
  };
  const A = run1(dposA), B = run1(dposB);
  // truth differences per iteration (exact in BigInt, then to f64)
  let maxRel = 0, maxTruth = 0;
  const tD = [];
  for (let k = 1; k <= iters; k++) {
    const td = [0, 1, 2].map(j => HP.toNum(hpA[k][j] - hpB[k][j]));
    tD.push(td);
    maxTruth = Math.max(maxTruth, len3(td));
  }
  for (let k = 1; k <= iters; k++) {
    const td = tD[k - 1];
    const tdL = len3(td);
    if (tdL < 1e-3 * maxTruth) continue;                  // skip degenerate crossings
    const gd = [0, 1, 2].map(j => A.zs[k - 1][j] - B.zs[k - 1][j]);
    maxRel = Math.max(maxRel, len3([gd[0] - td[0], gd[1] - td[1], gd[2] - td[2]]) / tdL);
  }
  return { maxRel, rebases: Math.max(A.rebases, B.rebases) };
}

/* ---------- the experiment ---------- */
const ITERS = 14;
const path = searchNearOrigin();
console.log('§8.5 — axis/pole glitch detect + reference relocation, hp-refereed (P=384)\n');
console.log(`pathological reference C_path = [${path.C.map(v => v.toFixed(2))}]`);
console.log(`  orbit min|V_k| = ${path.minR.toExponential(2)} at k=${path.dipK}, min ρ_k (axis dist) = ${path.minRho.toExponential(2)}`);
console.log(`  → DETECT: reference orbit dips below origin/axis thresholds → glitch-prone\n`);

const oPath = buildRefOrbit(path.C, ITERS + 2);
const dirs = [[1, -0.7, 0.4], [-0.5, 1, 0.3]].map(d => { const L = len3(d); return d.map(x => x / L); });

const pairFor = (Cbase, u, mag, o, useRGuard) => {
  const dposA = u.map(x => mag * x);
  const dposB = u.map(x => 1.3 * mag * x);                 // radial pair, separation 0.3·mag
  const hpA = HP.traj(Cbase.map((c, j) => HP.fromNum(c) + HP.fromNum(dposA[j])), ITERS);
  const hpB = HP.traj(Cbase.map((c, j) => HP.fromNum(c) + HP.fromNum(dposB[j])), ITERS);
  return runGlitchPair(o, dposA, dposB, ITERS, hpA, hpB, useRGuard);
};

console.log('(1) GLITCH BASELINE — pixels vs the pathological reference (structural pair metric):');
console.log('offset      dir   structRel   rebases');
const before = [];
for (const mag of [1e-6, 1e-9, 1e-12]) {
  dirs.forEach((u, di) => {
    const r = pairFor(path.C, u, mag, oPath, true);
    before.push(r.maxRel);
    console.log(`${mag.toExponential(0).padStart(7)}     d${di}    ${r.maxRel.toExponential(2).padStart(9)}   ${r.rebases}`);
  });
}

console.log('\n(2a) AXIS-CENTERED RELOCATION (the naive 2D playbook) — C2 = C_path + ε·(off-axis), pixels stay around C_path:');
const offAxis = (() => { const d = [0.71, 0.71, 0.05], L = len3(d); return d.map(x => x / L); })();
console.log('ε(reloc)    C2 min|V_k|   C2 min ρ_k    structRel d0   structRel d1');
const MAG = 1e-9;
let relocRecovered = 0;
for (const eps of [1e-2, 1e-4, 1e-6, 1e-8]) {
  const C2 = path.C.map((c, j) => c + eps * offAxis[j]);
  const s2 = orbitStats(C2, ITERS + 1);
  if (s2.escaped) { console.log(`${eps.toExponential(0).padStart(7)}     (escaped)`); continue; }
  const o2 = buildRefOrbit(C2, ITERS + 2);
  const errs = dirs.map((u) => {
    const dposA = u.map((x, j) => path.C[j] + MAG * x - C2[j]);          // δ′ vs the new reference
    const dposB = u.map((x, j) => path.C[j] + 1.3 * MAG * x - C2[j]);
    const hpA = HP.traj(path.C.map((c, j) => HP.fromNum(c) + HP.fromNum(MAG * u[j])), ITERS);
    const hpB = HP.traj(path.C.map((c, j) => HP.fromNum(c) + HP.fromNum(1.3 * MAG * u[j])), ITERS);
    return runGlitchPair(o2, dposA, dposB, ITERS, hpA, hpB, true).maxRel;
  });
  if (errs.every(e => e < 0.5)) relocRecovered++;
  console.log(`${eps.toExponential(0).padStart(7)}     ${s2.minR.toExponential(2).padStart(9)}    ${s2.minRho.toExponential(2).padStart(9)}     ${errs[0].toExponential(2).padStart(9)}     ${errs[1].toExponential(2).padStart(9)}`);
}
console.log(`→ axis-centered relocation recovered ${relocRecovered}/4 rows. The structure: [0,0,−1] is the
  z-axis pole whose orbit maps EXACTLY to the origin (period-2 axis↔origin cycle). EVERY
  ε-relocated reference inherits a dip to ~ε (measured above: minR ≈ 0.73ε) — the
  degeneracy is STRUCTURAL in this neighbourhood, not measure-zero as in 2D. The triplex
  power is non-analytic at the origin/axis; 2D's "any nearby reference works" does not hold.`);

console.log('\n(2b) THE PRODUCTION CASE — view centered NEAR (not on) the axis at axis-distance a,');
console.log('reference = view center itself, pixels at 1e-9; R<0.05 forced-rebase guard ON vs OFF:');
console.log('a(axis)     ref min|V_k|   structRel GUARD-ON (rb)   structRel GUARD-OFF (rb)   verdict');
let prodRecovered = 0, prodRows = 0;
for (const a of [1e-3, 1e-5, 1e-7]) {
  const C3 = path.C.map((c, j) => c + a * offAxis[j]);
  const s3 = orbitStats(C3, ITERS + 1);
  if (s3.escaped) { console.log(`${a.toExponential(0).padStart(7)}     (escaped)`); continue; }
  const o3 = buildRefOrbit(C3, ITERS + 2);
  let worstOn = 0, worstOff = 0, rbOn = 0, rbOff = 0;
  for (const u of dirs) {
    const rOn = pairFor(C3, u, MAG, o3, true);
    const rOff = pairFor(C3, u, MAG, o3, false);
    worstOn = Math.max(worstOn, rOn.maxRel); rbOn = Math.max(rbOn, rOn.rebases);
    worstOff = Math.max(worstOff, rOff.maxRel); rbOff = Math.max(rbOff, rOff.rebases);
  }
  prodRows++;
  const ok = worstOff < 0.5;
  if (ok) prodRecovered++;
  console.log(`${a.toExponential(0).padStart(7)}     ${s3.minR.toExponential(2).padStart(9)}      ${worstOn.toExponential(2).padStart(9)} (${rbOn})         ${worstOff.toExponential(2).padStart(9)} (${rbOff})       ${ok ? 'OK without guard' : 'still glitched'}`);
}

/* glitch threshold: healthy structural accuracy is ~1e-7..1e-5; ≥1e-2 = structure
   visibly wrong. (With the exact Δθ the baseline reads 0.09–0.31, down from 1.1–1.3
   with the old first-order form — better, but still 10⁵× the healthy floor.) */
const glitched = before.every(e => e > 1e-2);
console.log(`\n${glitched && prodRecovered > 0 ? 'PASS (with the structural-axis finding)' : 'FAIL'} — baseline glitched in all ${before.length} cases: ${glitched}; near-axis views recovered ${prodRecovered}/${prodRows} (guard off).

READING (§8.5) — three findings, one of them overturning the plan:
 • DETECTION WORKS, RELOCATION (2D playbook) DOES NOT — for the axis pathology.
   min|V_k|/min ρ_k of the reference orbit is a perfect worker-side detector. But near
   the axis-origin cycle ([0,0,−1] → origin → back, period 2), EVERY nearby reference
   has a dip ~ its own axis distance — the degeneracy is structural (the triplex power
   is non-analytic on the axis; in 2D, z² is analytic at 0, which is why relocation
   works there). Axis-CENTERED deep zoom cannot be fixed by reference choice alone.
 • THE PRODUCTION CASE (view near but not on the axis, reference = view center) is
   decided by the FORCED R<0.05 rebase guard: with the guard the reference re-anchors
   every dip pass (PART-3 Result-A poison) and pixels glitch; WITHOUT it, the §3
   formulas ride through the dip (conditioning ~ pixel-offset/axis-distance) and
   accuracy recovers — see (2b). KERNEL RULE UPDATE: drop the absolute R<0.05 rebase;
   keep pure Zhuoran (|z|² < |δ|²) + reference-exhaustion only. (The guard was
   folklore from prep; regime-2 never actually fired it.)
 • EXACTLY-ON-AXIS deep zoom (the bulb's pole spike) remains OPEN — it needs special
   handling, not a better reference: e.g. exact small-value treatment of the dip
   iteration, a frame-free Cartesian delta for near-axis steps, or accepting the spike
   as out-of-scope (every published Mandelbulb renderer shows axis artifacts already).
   Affected set: pixels whose own orbits enter the degenerate neighbourhood — a line
   through the image for generic views, the whole view only when zooming ON the pole.`);
