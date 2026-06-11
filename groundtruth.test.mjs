/* =============================================================================
   groundtruth.test.mjs  —  numeric validation of the triplex perturbation
   recurrence BEFORE any GPU code is written.   Run:  node groundtruth.test.mjs
   -----------------------------------------------------------------------------
   It compares three ways of advancing a small delta δ against a high-precision
   reference orbit V, all measured against a double-precision "ground truth"
   actual orbit:
      (A) NAIVE-f32   : z = fround(V+δ); δ' = fround(P(z)-P(V)) + δc   ← what a
                        careless shader does. Demonstrates catastrophic cancellation.
      (B) JACOBIAN    : δ' = J_P(V)·δ + δc                 (math note §2, truncated)
      (C) EXACT-STABLE: δ' = stableDelta(V,δ) + δc         (math note §3, candidate)
   δ is rounded to f32 between iterations to model GPU storage; the reference V
   stays double (it stands in for the real double-double/BigInt orbit).

   GOAL FOR FABLE: (C) should track ground truth to ~f32 relative error for many
   iterations and tiny offsets where (A) collapses. If it doesn't, the printed
   per-iteration error tells you which term diverges — fix that before the shader.
   This file is a SCAFFOLD: (C) is implemented from the math note but UNVERIFIED.
   ============================================================================= */

const f32 = Math.fround;
const f32v = v => v.map(f32);
const N_POWER = 8;

/* ---------- vector helpers ---------- */
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const scl=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const len=a=>Math.hypot(a[0],a[1],a[2]);

/* ---------- the triplex power P(v), double precision ---------- */
function P(v){
  const r=len(v); if(r===0) return [0,0,0];
  const rp1=Math.pow(r,N_POWER-1);
  const theta=Math.acos(Math.max(-1,Math.min(1,v[2]/r)))*N_POWER;
  const phi=Math.atan2(v[1],v[0])*N_POWER;
  const zr=rp1*r;
  return [zr*Math.sin(theta)*Math.cos(phi), zr*Math.sin(phi)*Math.sin(theta), zr*Math.cos(theta)];
}

/* ---------- (B) exact Jacobian J_P(V)·δ  (math note §2) ---------- */
function jacApply(V, d){
  const R=len(V); if(R<1e-300) return [0,0,0];
  const rh=scl(V,1/R);
  const theta=Math.acos(Math.max(-1,Math.min(1,V[2]/R)));
  const phi=Math.atan2(V[1],V[0]);
  const ct=Math.cos(theta),st=Math.sin(theta),cp=Math.cos(phi),sp=Math.sin(phi);
  const th=[ct*cp,ct*sp,-st];          // θ̂
  const ph=[-sp,cp,0];                  // φ̂
  const a=N_POWER*theta, b=N_POWER*phi;
  const ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);
  const rhp=[sa*cb,sa*sb,ca], thp=[ca*cb,ca*sb,-sa], php=[-sb,cb,0]; // output frame
  const gain=N_POWER*Math.pow(R,N_POWER-1);
  const ratio = Math.abs(st)<1e-12 ? N_POWER*(Math.cos(a)/Math.max(1e-12,Math.cos(theta))) // limit
                                   : Math.sin(a)/st;
  const cr=dot(rh,d), cthc=dot(th,d), cfc=dot(ph,d);
  return scl( add(add(scl(rhp,cr), scl(thp,cthc)), scl(php, ratio*cfc)), gain);
}

/* ---------- (C) exact cancellation-free delta  (math note §3) ---------- */
const log1p=Math.log1p, expm1=Math.expm1;
function stableDelta(V, d){
  const R=len(V); if(R<1e-300) return P(add(V,d));   // near-origin: fall back (would rebase)
  // §3a radius
  const q = 2*dot(V,d) + dot(d,d);
  const r2 = Math.sqrt(R*R + q);
  const Dr = q/(r2+R);
  // §3b radial power
  const uu = Dr/R;
  const Rn = Math.pow(R,N_POWER);
  const dRn = Rn*expm1(N_POWER*log1p(uu));
  // §3c angle deltas — Dphi numerator V0(V1+d1)-V1(V0+d0) is EXACTLY V0·d1-V1·d0
  // (each product O(δ); the parenthesised form subtracted O(1) terms → cancellation). [Fable fix]
  const Dphi = Math.atan2(V[0]*d[1] - V[1]*d[0],
                          V[0]*(V[0]+d[0]) + V[1]*(V[1]+d[1]));
  const theta=Math.acos(Math.max(-1,Math.min(1,V[2]/R)));
  // §3c EXACT Δθ (Fable slice-5 fix — the old atan2(θ̂·δ, R+r̂·δ) was FIRST-ORDER;
  // its O(δ²) truncation chaos-amplified into the user-visible surface mangling)
  const rho=Math.hypot(V[0],V[1]);
  const qr2=2*(V[0]*d[0]+V[1]*d[1])+d[0]*d[0]+d[1]*d[1];
  const rhoP=Math.sqrt(Math.max(rho*rho+qr2,0));
  const drho=qr2/(rhoP+rho);
  const Dtheta=Math.atan2(drho*V[2]-d[2]*rho, rho*rhoP+V[2]*(V[2]+d[2]));
  // §3d output angles
  const a=N_POWER*theta, b=N_POWER*Math.atan2(V[1],V[0]);
  const Da=N_POWER*Dtheta, Db=N_POWER*Dphi;
  const ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);
  const cA=Math.cos(Da),sA=Math.sin(Da),cB=Math.cos(Db),sB=Math.sin(Db);
  // §3e ΔD via angle-addition RESIDUALS (cancellation-free): cos(x)-1 = -sin²x/(1+cos x);
  // ds*=sin(a+Da)-sin(a), dc*=cos(a+Da)-cos(a). The old sAp·cBp-sa·cb subtracted O(1)
  // values and was the test's real ~1e-13 floor. [Fable fix]
  const cAm=-sA*sA/(1+cA), cBm=-sB*sB/(1+cB);
  const dsA=sa*cAm+ca*sA, dcA=ca*cAm-sa*sA;
  const dsB=sb*cBm+cb*sB, dcB=cb*cBm-sb*sB;
  const DD=[ sa*dcB+dsA*cb+dsA*dcB, sa*dsB+dsA*sb+dsA*dsB, dcA ];
  const Dref=[sa*cb, sa*sb, ca];
  // §3f assemble  Δ = Rⁿ·ΔD + Δ(rⁿ)·(Dref+ΔD)
  return add( scl(DD,Rn), scl(add(Dref,DD), dRn) );
}

/* ---------- run a comparison for one pixel offset ---------- */
function run(C, dpos, iters){
  let V=[...C];                  // reference orbit (double)
  let A=add(C,dpos);             // actual orbit  (double ground truth)
  let dJ=[...dpos], dC=[...dpos]; // jacobian / exact deltas
  let dN=f32v(dpos);             // naive-f32 delta
  const dcF=f32v(dpos);
  const err={naive:0,jac:0,exact:0};
  for(let k=0;k<iters;k++){
    // ground truth advance
    const Vn=add(P(V),C);
    const An=add(P(A),add(C,dpos));
    const trueDelta=sub(An,Vn);
    // (A) naive f32
    {
      const z=f32v(add(V,dN));
      dN=f32v(add(sub(P(z),P(V)), dcF));
    }
    // (B) jacobian (δ rounded to f32 between steps), δc = dpos
    dJ=f32v(add(jacApply(V,dJ), dpos));
    // (C) exact stable, δc = dpos
    dC=f32v(add(stableDelta(V,dC), dpos));
    // advance refs
    V=Vn; A=An;
    err.naive=Math.max(err.naive, len(sub(dN,trueDelta)));
    err.jac  =Math.max(err.jac,   len(sub(dJ,trueDelta)));
    err.exact=Math.max(err.exact, len(sub(dC,trueDelta)));
  }
  return err;
}

/* ---------- report ---------- */
// A center known to stay bounded for a good few iterations (near the bulb surface).
const C=[0.30, 0.50, 0.20];
console.log(`triplex perturbation ground-truth test  (n=${N_POWER}, center=${C})`);
console.log(`offset      iters   naive-f32      jacobian(§2)   exact(§3)`);
for(const mag of [1e-2,1e-4,1e-6,1e-9,1e-12]){
  const dpos=[mag, -mag*0.7, mag*0.4];
  const e=run(C,dpos,12);
  const fmt=x=>x.toExponential(2).padStart(12);
  console.log(`${mag.toExponential(0).padStart(8)}    12    ${fmt(e.naive)}  ${fmt(e.jac)}  ${fmt(e.exact)}`);
}
console.log(`
INTERPRETATION (for Fable) — what the prep run actually showed:
 • naive-f32 PLATEAUS (~4e-9 here) and never improves as the offset shrinks: it
   physically cannot resolve a δ smaller than f32-epsilon-of-|V|. THIS is the
   deep-zoom wall (the renders break ~1e-6 for the same reason).
 • exact(§3) keeps IMPROVING as δ shrinks (≈1e-5 → 1e-9 → 1e-13). At offset 1e-6
   it was ~4 orders more accurate than naive — that headroom is the whole point.
 • jacobian(§2) works but is ~10-20× worse than §3 at every scale and grows like
   O(|δ|²·gain) per step → it needs aggressive rebasing to survive depth.
 • VERDICT from prep: §3 is the path. It is validated for a BOUNDED orbit only.
   NOT YET tested where the reference passes near origin (R→0, math note §4.1) —
   that is the real stress test and the most likely failure mode. ADD a center
   whose orbit visits small |V| and confirm rebasing (§5) rescues it BEFORE
   trusting the GPU kernel. A clean §3 column there = green light for the shader.`);

/* =============================================================================
   PART 2 — the R→0 stress test + rebasing  (math note §4.1, §5)
   Done in the Opus prep session (step 1 of the ladder). Finds a reference center
   whose Mandelbulb orbit dips near the origin, then compares §3 WITHOUT vs WITH
   rebasing. Metric = max |reconstructed_actual − true_actual| over the run.
   ============================================================================= */

function orbitMinRadius(C, iters){
  let V=[...C], minR=Infinity;
  for(let k=0;k<iters;k++){
    V=add(P(V),C);
    const r=len(V);
    if(r>2.0) return {minR:Infinity, escaped:true};
    if(k>0) minR=Math.min(minR,r);
  }
  return {minR, escaped:false};
}
function searchNearOrigin(){
  let best=null;
  for(let x=-1.2;x<=1.2;x+=0.1)
   for(let y=-1.2;y<=1.2;y+=0.1)
    for(let z=-1.2;z<=1.2;z+=0.1){
      if(Math.hypot(x,y,z)<0.35) continue;             // exclude the trivial origin fixed point
      const {minR,escaped}=orbitMinRadius([x,y,z],16);
      if(escaped) continue;
      if(!best||minR<best.minR) best={C:[x,y,z],minR};
    }
  return best;
}

// §3 perturbation with reference-index tracking + optional rebasing
function runRebase(C, dpos, iters, {rebase, originEps=0.05}){
  const V=[[...C]]; for(let k=0;k<iters;k++) V.push(add(P(V[k]),C));
  const cpix=add(C,dpos);
  const A=[[...cpix]]; for(let k=0;k<iters;k++) A.push(add(P(A[k]),cpix));
  let m=0, d=f32v(dpos), maxErr=0, rebases=0;
  for(let k=0;k<iters;k++){
    d=f32v(add(stableDelta(V[m], d), dpos));   // δc = dpos
    m+=1;
    let aRec=add(V[m], d);
    if(rebase && (dot(aRec,aRec) < dot(d,d) || len(V[m])<originEps || m>=iters)){
      d=f32v(sub(aRec, V[0]));  m=0;  rebases++;   // re-anchor to V0
      aRec=add(V[0], d);
    }
    maxErr=Math.max(maxErr, len(sub(aRec, A[k+1])));
  }
  return {maxErr, rebases};
}

// a generic, OFF-AXIS near-origin dip (the realistic hard case, not the pathology)
function searchModerate(){
  let best=null;
  for(let x=-1.2;x<=1.2;x+=0.1)
   for(let y=-1.2;y<=1.2;y+=0.1)
    for(let z=-1.2;z<=1.2;z+=0.1){
      if(Math.hypot(x,y)<0.4) continue;                // keep off the z-axis (avoid poles)
      const {minR,escaped}=orbitMinRadius([x,y,z],16);
      if(escaped||minR<0.03) continue;                 // exclude exact-origin pathologies
      if(!best||minR<best.minR) best={C:[x,y,z],minR};
    }
  return best;
}

const near=searchNearOrigin();
console.log(`\n\nPART 2 — R→0 stress (step 1):  nearest-origin bounded center = `+
            `[${near.C.map(v=>v.toFixed(2))}], min|V_k| = ${near.minR.toExponential(2)}`);
console.log(`offset      no-rebase     err/offset    with-rebase  (#rebases)`);
for(const mag of [1e-3,1e-6,1e-9,1e-12]){
  const dpos=[mag,-mag*0.7,mag*0.4];
  const off=runRebase(near.C,dpos,16,{rebase:false});
  const on =runRebase(near.C,dpos,16,{rebase:true});
  const f=x=>x.toExponential(2).padStart(11);
  console.log(`${mag.toExponential(0).padStart(8)}   ${f(off.maxErr)}   ${(off.maxErr/mag).toFixed(1).padStart(7)}x   ${f(on.maxErr)}  (${on.rebases})`);
}
const mod=searchModerate();
console.log(`\nGeneric OFF-AXIS dip:  center = [${mod.C.map(v=>v.toFixed(2))}], min|V_k| = ${mod.minR.toExponential(2)}`);
console.log(`offset      err           err/offset`);
for(const mag of [1e-3,1e-6,1e-9,1e-12]){
  const dpos=[mag,-mag*0.7,mag*0.4];
  const e=runRebase(mod.C,dpos,16,{rebase:true});
  console.log(`${mag.toExponential(0).padStart(8)}   ${e.maxErr.toExponential(2).padStart(11)}   ${(e.maxErr/mag).toExponential(2).padStart(10)}`);
}
console.log(`
PART 2 READING — three regimes (re-characterized 2026-06-10 with the FIXED stableDelta):
 RE-RUN NOTE: porting Fable's cancellation-free residual forms left regimes 2 & 3
   UNCHANGED. That is the correct, informative result: Part 2's runRebase carries δ in
   f32, so its floors are limited by the f32 MANTISSA (and, in regime 3, the axis/origin
   degeneracy) — NOT by the O(1) cancellation the fix removed. The fix only lowers floors
   when δ has >f32 mantissa bits (precision-ladder.test.mjs's dd path: generic→1e-18).
   CONSEQUENCE for the kernel: GPU HDR-δ also has an f32 mantissa (≈7 digits) + extended
   exponent — so HDR does NOT lower these floors either. The depth lever for HDR-δ is
   REBASING CADENCE (keep relative δ inside the f32 mantissa window), measured in the
   rebasing sweep, not the cancellation fix. (The fix is still MANDATORY: without it f32
   hits the cancellation at ~1e-6, destroying accuracy even at the operating scale.)
 (1) GENERIC bounded orbit (Part 1): §3 excellent, err ≪ offset. Fully validated.
 (2) MODERATE off-axis dip (minR~0.06): with f32-mantissa δ, floors ~1e-9 near the dip —
     this is the f32-mantissa relative limit, now CONFIRMED (precision-ladder: dd δ lowers
     it to ~2e-16, so it's mantissa not formula). For HDR-δ (f32 mantissa) the lever is
     rebasing frequency, not extra mantissa.
 (3) PATHOLOGICAL axis+origin (minR~1e-15, pole): §3 degrades to err≈10x·offset; rebasing
     does NOT help. Measure-zero GLITCH PIXELS — detect (err/offset ~ O(1)) and relocate
     the reference (auto-reference) or local fallback, exactly as 2D handles glitches.
 NET: §3 is the right recurrence; the residual forms are mandatory; HDR-δ's depth is set by
 rebasing cadence (next: measure it). Axis pixels are glitch-relocate territory.`);
