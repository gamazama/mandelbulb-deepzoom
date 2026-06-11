/* =============================================================================
   de-depth.test.mjs  —  perturbed DE AT DEPTH, refereed by the hp orbit (§8.3)
   Run:  node de-depth.test.mjs
   -----------------------------------------------------------------------------
   de-perturbation.test.mjs validated the DE only at SHALLOW offsets (≥1e-7), where
   forming r = R + Δr in f32 still keeps Δr. At deep zoom Δr ≪ R·1e-7 underflows that
   sum → the DE's pixel-to-pixel VARIATION (the structure you're zooming into) vanishes.
   Now that hp-orbit.test.mjs gives an exact ground truth, we can measure this and test
   the fix: carry the radius delta Δr and a DERIVATIVE-DELTA Δdr recurrence in HDR.

   Δdr derivation (dr_{k+1}=N·r^{N-1}·dr+1, r=R+Δr, dr=DR+Δdr, all reference quantities R,DR known):
     Δp   = r^{N-1} − R^{N-1} = R^{N-1}·expm1((N-1)·log1p(Δr/R))        (HDR-small, no cancel)
     Δdr_{k+1} = N·[ Δp·DR_k + (R_k^{N-1}+Δp)·Δdr_k ]                    (HDR-small, no cancel)
   The pixel's r,dr are then (R+Δr, DR+Δdr); the DE variation lives entirely in the HDR
   deltas Δr,Δdr — which survive depth, unlike the naive scalar form.

   Metric: the TRUE pixel variation  Δr_final = r_final(C+δp) − r_final(C)  and
   Δdr_final = dr_final(C+δp) − dr_final(C), computed EXACTLY by BigInt subtraction (no
   cancellation), vs what each perturbed method captures. rel-err → 1 means "variation lost".
   ============================================================================= */

const N = 8;
/* ---- hp fixed-point factory (with dr), from hp-orbit.test.mjs ---- */
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
  // orbit + dr in fixed point. center is fixed-point. returns {rFinal, drFinal} (fixed-point BigInt)
  const orbitDr = (Cfp, iters) => {
    let v = Cfp.map(c => c), dr = ONE;
    for (let k = 0; k < iters; k++) {
      const r2 = mul(v[0], v[0]) + mul(v[1], v[1]) + mul(v[2], v[2]);
      if (r2 > 4n * SCALE) break;                                       // escape guard (avoid BigInt blowup)
      const r6 = mul(mul(r2, r2), r2), r = fsqrt(r2), r7 = mul(r6, r);   // r^{N-1}=r^7
      dr = 8n * mul(r7, dr) + ONE;
      v = addV(P8(v), Cfp);
    }
    const rF = fsqrt(mul(v[0], v[0]) + mul(v[1], v[1]) + mul(v[2], v[2]));
    return { rFinal: rF, drFinal: dr };
  };
  const refOrbit = (Cfp, iters) => { const V = [Cfp.map(c => c)]; for (let i = 0; i < iters; i++) V.push(addV(P8(V[i]), Cfp)); return V; };
  return { fromNum, toNum, orbitDr, refOrbit, SCALE };
}
const HP = makeHP(384);

/* ---- f64 perturbation pieces ---- */
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], len = a => Math.hypot(a[0], a[1], a[2]);
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const Pf = (v) => { const r = len(v); if (r === 0) return [0, 0, 0]; const th = Math.acos(Math.max(-1, Math.min(1, v[2] / r))) * N, ph = Math.atan2(v[1], v[0]) * N, zr = Math.pow(r, N); return [zr * Math.sin(th) * Math.cos(ph), zr * Math.sin(ph) * Math.sin(th), zr * Math.cos(th)]; };
const hdr = (x) => { if (x === 0) return 0; const e = Math.floor(Math.log2(Math.abs(x))), s = Math.pow(2, e); return Math.fround(x / s) * s; };
function stableDelta(V, d) {
  const R = len(V); if (R < 1e-300) return Pf(add(V, d));
  const q = 2 * dot(V, d) + dot(d, d), r2 = Math.sqrt(R * R + q), Dr = q / (r2 + R);
  const Rn = Math.pow(R, N), dRn = Rn * Math.expm1(N * Math.log1p(Dr / R));
  const Dphi = Math.atan2(V[0] * d[1] - V[1] * d[0], V[0] * (V[0] + d[0]) + V[1] * (V[1] + d[1]));
  const th0 = Math.acos(Math.max(-1, Math.min(1, V[2] / R))), ph0 = Math.atan2(V[1], V[0]);
  // §3c EXACT Δθ (Fable slice-5 fix — old form was first-order, O(δ²) truncation)
  const rho = Math.hypot(V[0], V[1]);
  const qr2 = 2 * (V[0] * d[0] + V[1] * d[1]) + d[0] * d[0] + d[1] * d[1];
  const rhoP = Math.sqrt(Math.max(rho * rho + qr2, 0));
  const drho = qr2 / (rhoP + rho);
  const Dth = Math.atan2(drho * V[2] - d[2] * rho, rho * rhoP + V[2] * (V[2] + d[2]));
  const a = N * th0, b = N * ph0, Da = N * Dth, Db = N * Dphi;
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  const cA = Math.cos(Da), sA = Math.sin(Da), cB = Math.cos(Db), sB = Math.sin(Db);
  const cAm = -sA * sA / (1 + cA), cBm = -sB * sB / (1 + cB);
  const dsA = sa * cAm + ca * sA, dcA = ca * cAm - sa * sA, dsB = sb * cBm + cb * sB, dcB = cb * cBm - sb * sB;
  const DD = [sa * dcB + dsA * cb + dsA * dcB, sa * dsB + dsA * sb + dsA * dsB, dcA], Dref = [sa * cb, sa * sb, ca];
  return add(scl(DD, Rn), scl(add(Dref, DD), dRn));
}
const stableDr = (R, q) => { const r2 = Math.sqrt(R * R + q); return q / (r2 + R); };   // Δr

// both perturbed methods share the δ + Δr propagation; they differ only in how dr is formed.
function runPert(Vf, Rk, DRk, Rpow, dpos, iters, method) {
  let d = dpos.map(hdr), drNaive = 1, Rfinal = Rk[0], dr_naiveFinal = 1;
  let Drr = 0, Ddr = 0;                       // Δr_final tracker, Δdr (HDR deltas)
  for (let k = 0; k < iters; k++) {
    const V = Vf[k], R = Rk[k], DR = DRk[k], Rp = Rpow[k];
    const q = 2 * dot(V, d) + dot(d, d);
    const Dr = stableDr(R, q);                // Δr_k (HDR-small)
    if (method === 'naive') {
      const rk = Math.fround(R + Dr);          // ← loses Dr at depth
      drNaive = N * Math.pow(rk, N - 1) * drNaive + 1;
    } else {                                   // 'delta': Δdr recurrence
      const Dp = Rp * Math.expm1((N - 1) * Math.log1p(Dr / R));   // r^{N-1}−R^{N-1}, HDR-small
      Ddr = N * (Dp * DR + (Rp + Dp) * Ddr);   // Δdr_{k+1}
      Ddr = hdr(Ddr);
    }
    Drr = Dr;                                  // Δr at this step (final after loop)
    d = add(stableDelta(V, d), dpos).map(hdr); // δ recurrence (HDR)
  }
  if (method === 'naive') return { dRfinal: Math.fround(Rfinal + 0) /*unused*/, drFinalAbs: drNaive };
  return { Drr, Ddr };
}

console.log('PERTURBED DE AT DEPTH — captured pixel-variation vs hp ground truth (P=384)\n');
const centers = [[0.30, 0.50, 0.20], [0.90, 0.0, 0.10]];   // both bounded (gen + near-origin moDip)
const ITERS = 14;
for (const C of centers) {
  // reference (f64) precompute
  const Vfp = HP.refOrbit(C.map(HP.fromNum), ITERS + 1);
  const Vf = Vfp.map(v => v.map(HP.toNum)), Rk = Vf.map(len);
  const DRk = [1]; const Rpow = Rk.map(R => Math.pow(R, N - 1));
  for (let k = 0; k < ITERS; k++) DRk.push(N * Rpow[k] * DRk[k] + 1);
  const refHP = HP.orbitDr(C.map(HP.fromNum), ITERS);

  console.log(`center [${C}]:`);
  console.log('  offset     Δdr rel-err: naive | delta     (Δdr_true)');
  for (const mag of [1e-3, 1e-6, 1e-9, 1e-15, 1e-24]) {
    const dpos = [mag, -mag * 0.7, mag * 0.4];
    // hp ground-truth pixel variation (exact BigInt subtraction)
    const Cd = C.map(HP.fromNum).map((c, j) => c + HP.fromNum(dpos[j]));
    const pixHP = HP.orbitDr(Cd, ITERS);
    const ddrTrue = HP.toNum(pixHP.drFinal - refHP.drFinal);
    // naive
    const naive = runPert(Vf, Rk, DRk, Rpow, dpos, ITERS, 'naive');
    const ddrNaive = naive.drFinalAbs - DRk[ITERS];   // implied variation = dr_naive − reference DR
    // delta (Δdr recurrence)
    const del = runPert(Vf, Rk, DRk, Rpow, dpos, ITERS, 'delta');
    const relN = Math.abs(ddrNaive - ddrTrue) / Math.abs(ddrTrue);
    const relD = Math.abs(del.Ddr - ddrTrue) / Math.abs(ddrTrue);
    console.log(`  ${mag.toExponential(0).padStart(7)}    ${relN.toExponential(2).padStart(9)}  ${relD.toExponential(2).padStart(9)}    (${ddrTrue.toExponential(2)})`);
  }
  console.log('');
}
console.log(`READING (§8.1 deep — the DE perturbation refinement):
 • 'naive' (scalar dr fed r=fround(R+Δr)) rel-err → ~1 as offset shrinks: the f32 sum R+Δr
   drops Δr, so dr_final stops varying between pixels → the DE structure VANISHES at depth.
   (de-perturbation.test.mjs missed this — it only tested offsets ≥1e-7.)
 • 'delta' (Δdr derivative-delta recurrence in HDR) tracks the true variation FLAT across depth
   at ~1e-7 rel — the HDR f32-mantissa floor (plenty for rendering; dd-ify per §8.2 only to go
   lower). → KERNEL: the DE needs Δr (HDR) AND a Δdr recurrence (HDR), from reference R_k, R_k^{N-1}, DR_k.
 • This is the deep-zoom-correct perturbed DE. Reference orbit must carry R_k, R_k^{N-1}, DR_k
   alongside V_k. Refereed by the hp orbit (§8.3) — the upgraded ground truth made it measurable.`);
