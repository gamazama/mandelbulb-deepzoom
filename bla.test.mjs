/* =============================================================================
   bla.test.mjs — BLA/LA SKIP TABLES for triplex perturbation: CPU validation
   Run from this folder:  node bla.test.mjs
   -----------------------------------------------------------------------------
   THE PERF PRIZE (brief step 6): skip RUNS of perturbation iterations through a
   table of composed linear operators. Per-step linearization is the exact §2
   Jacobian  J·δ = 8R⁷[(r̂·δ)r̂′ + (θ̂·δ)θ̂′ + s(φ̂·δ)φ̂′],  s = U₇(cosθ),
   plus the Δdr channel  Δdr′ ≈ 8R⁷·Δdr + 56R⁶·DR·(r̂·δ).
   Runs compose:  δ' = A·δ + B·δc,  Δdr' = g·δ + a·Δdr + h·δc   (A,B 3×3; g,h rows).
   Validity (conservative, baked at build):  apply run iff |δ| ≤ rA − rB·|δc|, with
     base   rA = τ·R_k·min(1,|s|)/8  (remainder O(δ²·8²R⁶) ≤ τ·σmin·|δ|),  rB = 0,
            rA = 0 if R_k > 1.5  (escape-safety: pixel can't cross bailout in-run),
     merge  rA = min(rA1, rA2/‖A1‖),  rB = max(rB1, (rB2+‖B1‖)/‖A1‖)   (Frobenius).
   Within validity no Zhuoran rebase can fire in-run (fires at |δ|≳R ≫ τR/8).
   VALIDATION (the law): BLA path vs exact §3 single-step path vs the BigInt
   DIRECT referee (zero shared math). Sweeps τ; reports skip fraction.
   ============================================================================= */
const N = 8;

/* ---------- BigInt fixed-point factory (hp-orbit.test.mjs) ---------- */
function makeHP(P) {
  const Pb = BigInt(P), SCALE = 1n << Pb;
  const mul = (a, b) => (a * b) >> Pb;
  const fdiv = (a, b) => (a << Pb) / b;
  const isqrt = (n) => { if (n < 2n) return n < 0n ? 0n : n; let x = n, y = (x + 1n) >> 1n; while (y < x) { x = y; y = (x + n / x) >> 1n; } return x; };
  const fsqrt = (a) => isqrt(a << Pb);
  const SMALL = SCALE >> 90n;
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
  const toNum = (v) => Number(v) / Number(SCALE);
  const ONE = SCALE;
  const chebT = (x, n) => { let a = ONE, b = x; if (n === 0) return a; if (n === 1) return b; for (let k = 2; k <= n; k++) { const t = 2n * mul(x, b) - a; a = b; b = t; } return b; };
  const chebU = (x, n) => { let a = ONE, b = 2n * x; if (n === 0) return a; if (n === 1) return b; for (let k = 2; k <= n; k++) { const u = 2n * mul(x, b) - a; a = b; b = u; } return b; };
  const P8 = (v) => {
    const [x, y, z] = v;
    const x2 = mul(x, x), y2 = mul(y, y), z2 = mul(z, z);
    const r2 = x2 + y2 + z2, rho2 = x2 + y2;
    const r = fsqrt(r2), rho = fsqrt(rho2);
    const r4 = mul(r2, r2), r8 = mul(r4, r4);
    if (r === 0n) return [0n, 0n, 0n];
    const cth = fdiv(z, r), sth = fdiv(rho, r);
    let cph, sph;
    if (rho > SMALL) { cph = fdiv(x, rho); sph = fdiv(y, rho); } else { cph = ONE; sph = 0n; }
    const c8th = chebT(cth, N), s8th = mul(sth, chebU(cth, N - 1));
    const c8ph = chebT(cph, N), s8ph = mul(sph, chebU(cph, N - 1));
    return [mul(mul(r8, s8th), c8ph), mul(mul(r8, s8th), s8ph), mul(r8, c8th)];
  };
  const addV = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  return { P, fromNum, toNum, P8, addV, SCALE };
}
const hp = makeHP(256);

/* ---------- f64 helpers + reference orbit (mirrors harness buildRefOrbit) ---------- */
const dot3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2], len3=a=>Math.hypot(a[0],a[1],a[2]);
const add3=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]], sub3=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], scl3=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const powm1_7 = u => u*(7+u*(21+u*(35+u*(35+u*(21+u*(7+u))))));
function buildRefOrbit(CHP, lenMax) {
  const V = [CHP.slice()], Vf = [CHP.map(hp.toNum)];
  for (let k = 0; k < lenMax - 1; k++) {
    const nv = hp.addV(hp.P8(V[k]), CHP);
    V.push(nv);
    const vf = nv.map(hp.toNum);
    Vf.push(vf);
    if (Math.hypot(...vf) > 100) break;
  }
  const len = V.length, Rk = [], DRk = [];
  let DR = 1;
  for (let k = 0; k < len; k++) {
    const R = Math.hypot(...Vf[k]) || 1e-300;
    Rk.push(R); DRk.push(DR);
    DR = N * Math.pow(R, N - 1) * DR + 1;
  }
  return { len, Vf, Rk, DRk };
}

/* ---------- exact §3 single-step machinery (verbatim harness mirrors) ---------- */
function stableDeltaJS(V,d){
  const R=len3(V); if(R<1e-300) return [0,0,0];
  const q=2*dot3(V,d)+dot3(d,d), r2=Math.sqrt(Math.max(R*R+q,0)), Dr=q/(r2+R);
  const Rn=Math.pow(R,N), dRn=Rn*Math.expm1(N*Math.log1p(Dr/R));
  const Dphi=Math.atan2(V[0]*d[1]-V[1]*d[0], V[0]*(V[0]+d[0])+V[1]*(V[1]+d[1]));
  const th0=Math.acos(Math.max(-1,Math.min(1,V[2]/R))), ph0=Math.atan2(V[1],V[0]);
  const rho=Math.hypot(V[0],V[1]);
  const qr2=2*(V[0]*d[0]+V[1]*d[1])+d[0]*d[0]+d[1]*d[1];
  const rhoP=Math.sqrt(Math.max(rho*rho+qr2,0));
  const drho=qr2/(rhoP+rho);
  const Dth=Math.atan2(drho*V[2]-d[2]*rho, rho*rhoP+V[2]*(V[2]+d[2]));
  const a=N*th0,b=N*ph0,Da=N*Dth,Db=N*Dphi;
  const ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);
  const cA=Math.cos(Da),sA=Math.sin(Da),cB=Math.cos(Db),sB=Math.sin(Db);
  const cAm=-sA*sA/(1+cA), cBm=-sB*sB/(1+cB);
  const dsA=sa*cAm+ca*sA, dcA=ca*cAm-sa*sA;
  const dsB=sb*cBm+cb*sB, dcB=cb*cBm-sb*sB;
  const DD=[sa*dcB+dsA*cb+dsA*dcB, sa*dsB+dsA*sb+dsA*dsB, dcA], Dref=[sa*cb,sa*sb,ca];
  return add3(scl3(DD,Rn), scl3(add3(Dref,DD),dRn));
}
function exactStep(O, m, d, Ddr){           // one §3+Δdr step at index m → {d', Ddr'}
  const V=O.Vf[m], R=O.Rk[m];
  const q=2*dot3(V,d)+dot3(d,d), r2=Math.sqrt(Math.max(R*R+q,0)), Dr=q/(r2+R);
  const u=Dr/R, Rp=Math.pow(R,7);
  const Dp=Rp*powm1_7(u);
  const nDdr=8*(Dp*O.DRk[m]+(Rp+Dp)*Ddr);
  return { d: stableDeltaJS(V,d), Ddr: nDdr };
}
function pertOrbitJS(O, dc, maxIter){       // exact reference implementation
  let d=dc.slice(), Ddr=0, m=0, escaped=false;
  for(let k=0;k<maxIter;k++){
    const st=exactStep(O,m,d,Ddr);
    d=add3(st.d,dc); Ddr=st.Ddr;
    m++;
    const z=add3(O.Vf[m],d);
    if(dot3(z,z)>4){ escaped=true; break; }
    if(dot3(z,z)<dot3(d,d) || m>=O.len-1){ d=sub3(z,O.Vf[0]); Ddr=(O.DRk[m]-1)+Ddr; m=0; }
  }
  const V=O.Vf[m], R=O.Rk[m];
  const q=2*dot3(V,d)+dot3(d,d), r2=Math.sqrt(Math.max(R*R+q,1e-300));
  return { de: 0.5*r2*Math.log(Math.max(r2,1e-300))/(O.DRk[m]+Ddr), escaped };
}
function deCPU(p, Nit){
  let [zx,zy,zz]=p, dr=1, r=0;
  for(let i=0;i<Nit;i++){
    r=Math.hypot(zx,zy,zz);
    if(r>2) break;
    const rp1=Math.pow(r,7);
    dr=rp1*8*dr+1;
    const theta=Math.acos(Math.max(-1,Math.min(1,zz/r)))*8, phi=Math.atan2(zy,zx)*8;
    const zr=rp1*r;
    zx=zr*Math.sin(theta)*Math.cos(phi)+p[0];
    zy=zr*Math.sin(phi)*Math.sin(theta)+p[1];
    zz=zr*Math.cos(theta)+p[2];
  }
  return 0.5*Math.log(Math.max(r,1e-8))*r/Math.max(dr,1e-20);
}
/* BigInt DIRECT referee (zero shared math) — verbatim harness hpDirectDE */
function hpDirectDE(posFp, maxIter){
  const Pb = BigInt(hp.P), SCALE = hp.SCALE, ONE = SCALE;
  const mul = (a,b) => (a*b) >> Pb;
  const isqrt = (n) => { if (n < 2n) return n < 0n ? 0n : n; let x = n, y = (x + 1n) >> 1n; while (y < x) { x = y; y = (x + n / x) >> 1n; } return x; };
  const fsqrt = (a) => isqrt(a << Pb);
  let v = posFp.slice(), dr = ONE, escaped = false, r2 = 0n;
  for (let i = 0; i < maxIter; i++) {
    r2 = mul(v[0],v[0]) + mul(v[1],v[1]) + mul(v[2],v[2]);
    if (r2 > 4n * SCALE) { escaped = true; break; }
    const r = fsqrt(r2), r6 = mul(mul(r2,r2),r2), r7 = mul(r6,r);
    dr = 8n * mul(r7, dr) + ONE;
    v = hp.addV(hp.P8(v), posFp);
  }
  if (!escaped) {
    r2 = mul(v[0],v[0]) + mul(v[1],v[1]) + mul(v[2],v[2]);
    escaped = r2 > 4n * SCALE;
  }
  const r = Number(fsqrt(r2)) / Number(SCALE);
  const drF = Number(dr) / Number(SCALE);
  return { de: 0.5 * r * Math.log(Math.max(r, 1e-300)) / drF, escaped };
}

/* ---------- CPU dive (harness __dive logic, no GPU) ---------- */
let camHP;
const moveAlong = (dir, dist) => { for(let j=0;j<3;j++) camHP[j] += hp.fromNum(dir[j]*dist); };
function diveHopJS(O, rd, maxIter, hint){
  const probe = (tv) => pertOrbitJS(O, scl3(rd, tv), maxIter);
  let t = 0, r0 = probe(0);
  if (!r0.escaped || r0.de <= 0) return { inside: true };
  let de = r0.de, lo = -1, hi = -1;
  let cap = Math.max((hint || 1e-5) * 0.01, 1e-300);
  for (let s = 0; s < 2000; s++) {
    const tn = t + Math.min(Math.max(de, 1e-300) * 0.5, cap);
    const rn = probe(tn);
    if (!rn.escaped || rn.de <= 0) { lo = t; hi = tn; break; }
    t = tn; de = rn.de; cap *= 1.25;
    if (t > 8) return { noSurface: true };
  }
  if (hi < 0) return { t };
  for (let b = 0; b < 100; b++) {
    const mid = 0.5 * (lo + hi);
    const rm = probe(mid);
    if (!rm.escaped || rm.de <= 0) hi = mid; else lo = mid;
  }
  return { t: lo };
}
function dive(depth, iters, ray){
  camHP = ray.pos.map(hp.fromNum);
  const fwd = ray.fwd;
  for (let s = 0; s < 4000; s++) {
    const cp = camHP.map(hp.toNum);
    const de = deCPU(cp, iters);
    if (de <= 1e-5) break;
    moveAlong(fwd, de * 0.6);
    if (Math.hypot(...camHP.map(hp.toNum)) > 8) return { error: 'left scene' };
  }
  let back = 1e-5, placed = false, verified = false;
  for (let hop = 0; hop < 600; hop++) {
    const O = buildRefOrbit(camHP, iters + 2);
    const r = diveHopJS(O, fwd, iters, back);
    if (r.noSurface) return { error: 'no surface' };
    if (r.inside) { moveAlong(fwd, -back); back *= 2; placed = false; continue; }
    if (placed && r.t >= depth * 0.34 && r.t <= depth * 3) { verified = true; break; }
    const target = Math.max(depth, r.t * 1e-3);
    if (r.t - target > 0) moveAlong(fwd, r.t - target);
    back = target;
    placed = (target <= depth * 1.001);
  }
  return verified ? {} : { error: 'placement not verified' };
}

/* ════════════════ BLA TABLE ════════════════ */
const mat3mul=(A,B)=>{ const C=new Array(9); for(let i=0;i<3;i++)for(let j=0;j<3;j++){let s=0;for(let k=0;k<3;k++)s+=A[i*3+k]*B[k*3+j];C[i*3+j]=s;} return C; };
const mat3vec=(A,v)=>[A[0]*v[0]+A[1]*v[1]+A[2]*v[2], A[3]*v[0]+A[4]*v[1]+A[5]*v[2], A[6]*v[0]+A[7]*v[1]+A[8]*v[2]];
const rowMat3=(g,A)=>[g[0]*A[0]+g[1]*A[3]+g[2]*A[6], g[0]*A[1]+g[1]*A[4]+g[2]*A[7], g[0]*A[2]+g[1]*A[5]+g[2]*A[8]];
const frob=A=>{let s=0;for(const x of A)s+=x*x;return Math.sqrt(s);};
const frob3=g=>Math.hypot(g[0],g[1],g[2]);
const chebU7=x=>{ let a=1,b=2*x; for(let k=2;k<=7;k++){const u=2*x*b-a;a=b;b=u;} return b; };  // U_7(x)
function blaBuild(O, tau){
  const usable = O.len - 1;                 // runs must end at index ≤ len-1
  const base = [];
  for (let k = 0; k < usable; k++) {
    const V=O.Vf[k], R=O.Rk[k], rho=Math.hypot(V[0],V[1]);
    const cth=V[2]/R, sth=rho/R;
    let cp=1, sp=0; if (rho>1e-300){ cp=V[0]/rho; sp=V[1]/rho; }
    const th=Math.acos(Math.max(-1,Math.min(1,cth))), ph=Math.atan2(V[1],V[0]);
    const a8=8*th, b8=8*ph;
    const rhat=[sth*cp,sth*sp,cth], that=[cth*cp,cth*sp,-sth], phat=[-sp,cp,0];
    const sa=Math.sin(a8),caa=Math.cos(a8),sb=Math.sin(b8),cbb=Math.cos(b8);
    const rhp=[sa*cbb,sa*sb,caa], thp=[caa*cbb,caa*sb,-sa], php=[-sb,cbb,0];
    const s = chebU7(cth);                  // sin8θ/sinθ — exact, pole-safe
    const Rp=Math.pow(R,7), gJ=8*Rp;
    const A=new Array(9);
    for(let i=0;i<3;i++)for(let j=0;j<3;j++)
      A[i*3+j]=gJ*(rhp[i]*rhat[j] + thp[i]*that[j] + s*php[i]*phat[j]);
    const g = scl3(rhat, 56*Math.pow(R,6)*O.DRk[k]);
    const B=[1,0,0,0,1,0,0,0,1];
    const rA = R > 1.5 ? 0 : tau*R*Math.min(1,Math.abs(s))/8;
    base.push({A, B, g, a:gJ, h:[0,0,0], rA, rB:0, nA:frob(A), nB:Math.sqrt(3)});
  }
  const lv=[base];
  let maxLvl=0;
  for(let l=1; (1<<l)<=usable; l++){
    const prev=lv[l-1], cur=[];
    for(let i=0; 2*i+1<prev.length; i++){
      const X=prev[2*i], Y=prev[2*i+1];     // X first, then Y
      const A=mat3mul(Y.A,X.A);
      const B=mat3mul(Y.A,X.B); for(let j=0;j<9;j++)B[j]+=Y.B[j];
      const g=rowMat3(Y.g,X.A); for(let j=0;j<3;j++)g[j]+=Y.a*X.g[j];
      const a=Y.a*X.a;
      const gB=rowMat3(Y.g,X.B);
      const h=[Y.h[0]+gB[0]+Y.a*X.h[0], Y.h[1]+gB[1]+Y.a*X.h[1], Y.h[2]+gB[2]+Y.a*X.h[2]];
      const rA = (X.rA<=0||Y.rA<=0) ? 0 : Math.min(X.rA, Y.rA/Math.max(X.nA,1e-300));
      const rB = Math.max(X.rB, (Y.rB + X.nB)/Math.max(X.nA,1e-300));
      cur.push({A,B,g,a,h,rA,rB,nA:frob(A),nB:frob(B)});
    }
    if(!cur.length) break;
    lv.push(cur); maxLvl=l;
  }
  return {lv, maxLvl, usable};
}
function pertOrbitBLA(O, T, dc, maxIter){
  let d=dc.slice(), Ddr=0, m=0, escaped=false;
  const dcMag=Math.hypot(...dc);
  let k=0, singles=0, runs=0, skipped=0;
  while(k<maxIter){
    let applied=false;
    const dMag=Math.hypot(...d);
    let lvlMax = m===0 ? T.maxLvl : Math.min(T.maxLvl, 31-Math.clz32(m&-m));
    for(let lvl=lvlMax; lvl>=1; lvl--){
      if(m % (1<<lvl)) continue;
      const idx=m>>lvl;
      if(lvl>=T.lv.length || idx>=T.lv[lvl].length) continue;
      const run=T.lv[lvl][idx];
      if(run.rA<=0) continue;
      if(m+(1<<lvl) > T.usable) continue;
      if(k+(1<<lvl) > maxIter) continue;
      if(dMag > run.rA - run.rB*dcMag) continue;
      const nd=mat3vec(run.A,d);
      const bd=mat3vec(run.B,dc);
      Ddr = dot3(run.g,d) + run.a*Ddr + dot3(run.h,dc);
      d=[nd[0]+bd[0], nd[1]+bd[1], nd[2]+bd[2]];
      m+=1<<lvl; k+=1<<lvl; runs++; skipped+=1<<lvl;
      applied=true; break;
    }
    if(!applied){
      const st=exactStep(O,m,d,Ddr);
      d=add3(st.d,dc); Ddr=st.Ddr;
      m++; k++; singles++;
    }
    const z=add3(O.Vf[m],d);
    if(dot3(z,z)>4){ escaped=true; break; }
    if(dot3(z,z)<dot3(d,d) || m>=O.len-1){ d=sub3(z,O.Vf[0]); Ddr=(O.DRk[m]-1)+Ddr; m=0; }
  }
  const V=O.Vf[m], R=O.Rk[m];
  const q=2*dot3(V,d)+dot3(d,d), r2=Math.sqrt(Math.max(R*R+q,1e-300));
  return { de: 0.5*r2*Math.log(Math.max(r2,1e-300))/(O.DRk[m]+Ddr), escaped, singles, runs, skipped };
}

/* ════════════════ MAIN ════════════════ */
const RAYS = [
  { pos: [0.1, 0.15, 2.6], fwd: [0,0,-1] },
  { pos: [0.4, 0.3, 2.5],  fwd: [0,0,-1] },
  { pos: [-0.3, 0.2, 2.6], fwd: [0,0,-1] },
];
const DEPTHS = [1e-9, 1e-20, 1e-30, 1e-40, 1e-50];
const TAUS = [1e-3, 1e-4, 1e-5];

for (const depth of DEPTHS) {
  const iters = Math.min(120, 32 + Math.ceil(1.35 * -Math.log10(depth)));
  let placed = false, fwd;
  for (const ray of RAYS) {
    const r = dive(depth, iters, ray);
    if (!r.error) { placed = true; fwd = ray.fwd; break; }
  }
  if (!placed) { console.log(`depth ${depth.toExponential(0).padStart(6)}  SKIP (placement failed on all rays)`); continue; }
  const O = buildRefOrbit(camHP, iters + 2);
  console.log(`\ndepth ${depth.toExponential(0)} it=${iters} orbitLen=${O.len}`);
  for (const t of [depth*0.3, depth*0.7]) {
    const dc = scl3(fwd, t);
    const ex = pertOrbitJS(O, dc, iters);
    const pos = camHP.map((c,j) => c + hp.fromNum(t*fwd[j]));
    const tr = hpDirectDE(pos, iters);
    const exRel = Math.abs(ex.de-tr.de)/Math.max(Math.abs(tr.de),1e-300);
    let line = `  t=${t.toExponential(1)}  exact: rel=${exRel.toExponential(1)} esc=${ex.escaped}/${tr.escaped} |`;
    for (const tau of TAUS) {
      const T = blaBuild(O, tau);
      const bl = pertOrbitBLA(O, T, dc, iters);
      const blRel = Math.abs(bl.de-tr.de)/Math.max(Math.abs(tr.de),1e-300);
      const blVsEx = Math.abs(bl.de-ex.de)/Math.max(Math.abs(ex.de),1e-300);
      const total = bl.singles + bl.skipped;
      line += `  τ=${tau.toExponential(0)}: rel=${blRel.toExponential(1)} vsEx=${blVsEx.toExponential(1)} esc=${bl.escaped===tr.escaped?'✓':'✗'} skip=${bl.skipped}/${total}(${bl.runs}r,${bl.singles}s)`;
    }
    console.log(line);
  }
}
console.log(`\nREADING: "rel" = vs the BigInt DIRECT referee (non-circular). "vsEx" = BLA-induced
deviation from the exact §3 path. skip=a/b = iterations skipped via runs / total advanced.
GREEN LIGHT for the GPU build = vsEx ≲ 1e-6-ish at some τ with skip fraction ≳ 70%.`);
