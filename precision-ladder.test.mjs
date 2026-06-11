/* =============================================================================
   precision-ladder.test.mjs  —  prep experiments #1 and #2  (Opus session)
   Run:  node precision-ladder.test.mjs
   -----------------------------------------------------------------------------
   Answers two questions the whole 1e-50 goal hinges on:

   #2  Can we build a HIGH-PRECISION Mandelbulb reference orbit?  2D is easy
       (z²+c is polynomial). The Mandelbulb power normally needs acos/atan2/sin/cos
       — transcendentals, which are painful in extended precision. KEY FINDING
       (proven below): for INTEGER power n the triplex map can be written with ONLY
       +,−,×,÷,sqrt via Chebyshev multiple-angle polynomials. So a double-double
       (dd, ~106-bit) reference orbit needs no transcendentals — just dd sqrt. We
       build one and verify it against the f64 reference.

   #1  Does higher-precision δ lower the regime-2 error floor (~1e-9 with f32 δ)?
       We carry δ at k f32 "limbs" (k=1 ≈ f32, k=2 ≈ GPU double-single ~44-bit,
       k=3 ≈ ~68-bit) against the dd ground-truth orbit, through a near-origin dip.
       If the floor drops with k, δ precision is the depth lever (as claimed).
   ============================================================================= */

const N=8;                                   // integer power (Chebyshev form below is n=8 specific via recurrence)

/* ---------- double-double (dd) scalar arithmetic, base = f64 ~106-bit ---------- */
const SPLT=134217729;                         // 2^27+1
const twoSum=(a,b)=>{const s=a+b,v=s-a,e=(a-(s-v))+(b-v);return[s,e];};
const qTwoSum=(a,b)=>{const s=a+b,e=b-(s-a);return[s,e];};
const splt=a=>{const c=SPLT*a,h=c-(c-a);return[h,a-h];};
const twoProd=(a,b)=>{const p=a*b,[ah,al]=splt(a),[bh,bl]=splt(b);return[p,((ah*bh-p)+ah*bl+al*bh)+al*bl];};
const ddAdd=(a,b)=>{let[s,e]=twoSum(a[0],b[0]);e+=a[1]+b[1];return qTwoSum(s,e);};
const ddSub=(a,b)=>ddAdd(a,[-b[0],-b[1]]);
const ddMul=(a,b)=>{let[p,e]=twoProd(a[0],b[0]);e+=a[0]*b[1]+a[1]*b[0];return qTwoSum(p,e);};
const ddF=x=>[x,0];
const f=a=>a[0]+a[1];
function ddDiv(a,b){
  const q1=a[0]/b[0]; let r=ddSub(a,ddMul(b,[q1,0]));
  const q2=r[0]/b[0]; r=ddSub(r,ddMul(b,[q2,0]));
  const q3=r[0]/b[0]; return ddAdd(qTwoSum(q1,q2),[q3,0]);
}
function ddSqrt(a){ if(a[0]<=0)return[0,0]; const x=Math.sqrt(a[0]);
  const d=ddSub(a,ddMul([x,0],[x,0])); return ddAdd([x,0],[d[0]/(2*x),0]); }

/* ---------- Chebyshev T_n, U_n in dd (the transcendental-free machinery) ---------- */
function chebT(x,n){ let a=[1,0],b=[x[0],x[1]]; if(n===0)return a; if(n===1)return b;
  for(let k=2;k<=n;k++){const t=ddSub(ddMul([2,0],ddMul(x,b)),a);a=b;b=t;} return b; }
function chebU(x,n){ let a=[1,0],b=ddMul([2,0],x); if(n===0)return a; if(n===1)return b;
  for(let k=2;k<=n;k++){const u=ddSub(ddMul([2,0],ddMul(x,b)),a);a=b;b=u;} return b; }

/* ---------- triplex power P (dd), integer n=8, NO transcendentals ---------- */
// cos(nθ)=T_n(cosθ), sin(nθ)=sinθ·U_{n-1}(cosθ); cosθ=z/r, sinθ=ρ/r; same for φ in xy.
function Pdd(v){
  const [x,y,z]=v;
  const x2=ddMul(x,x),y2=ddMul(y,y),z2=ddMul(z,z);
  const r2=ddAdd(ddAdd(x2,y2),z2), rho2=ddAdd(x2,y2);
  const r=ddSqrt(r2), rho=ddSqrt(rho2);
  const r4=ddMul(r2,r2), r8=ddMul(r4,r4);                 // r^8 = (r²)^4
  const cth=ddDiv(z,r), sth=ddDiv(rho,r);
  let cph,sph;
  if(rho[0]>1e-180){ cph=ddDiv(x,rho); sph=ddDiv(y,rho); } else { cph=[1,0]; sph=[0,0]; }
  const c8th=chebT(cth,N), s8th=ddMul(sth,chebU(cth,N-1));
  const c8ph=chebT(cph,N), s8ph=ddMul(sph,chebU(cph,N-1));
  return [ ddMul(ddMul(r8,s8th),c8ph), ddMul(ddMul(r8,s8th),s8ph), ddMul(r8,c8th) ];
}
const addV=(a,b)=>[ddAdd(a[0],b[0]),ddAdd(a[1],b[1]),ddAdd(a[2],b[2])];

/* ---------- f64 triplex P (for cross-check) ---------- */
function Pf(v){ const r=Math.hypot(...v); if(r===0)return[0,0,0];
  const th=Math.acos(Math.max(-1,Math.min(1,v[2]/r)))*N, ph=Math.atan2(v[1],v[0])*N, zr=Math.pow(r,N);
  return[zr*Math.sin(th)*Math.cos(ph), zr*Math.sin(ph)*Math.sin(th), zr*Math.cos(th)]; }

/* ---------- f64 stable delta (from groundtruth.test.mjs §3) ---------- */
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2], len=a=>Math.hypot(a[0],a[1],a[2]);
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]], scl=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
function stableDelta(V,d){
  const R=len(V); if(R<1e-300) return Pf(add(V,d));
  const q=2*dot(V,d)+dot(d,d), r2=Math.sqrt(R*R+q), Dr=q/(r2+R);
  const Rn=Math.pow(R,N), dRn=Rn*Math.expm1(N*Math.log1p(Dr/R));
  // Dphi numerator: V0(V1+d1)-V1(V0+d0) expands EXACTLY to V0·d1-V1·d0 — compute
  // that form (each product is O(δ); the original subtracts O(1) terms → cancellation).
  const Dphi=Math.atan2(V[0]*d[1]-V[1]*d[0], V[0]*(V[0]+d[0])+V[1]*(V[1]+d[1]));
  const th0=Math.acos(Math.max(-1,Math.min(1,V[2]/R))), ph0=Math.atan2(V[1],V[0]);
  // §3c EXACT Δθ (Fable slice-5 fix — old form was first-order, O(δ²) truncation)
  const rho=Math.hypot(V[0],V[1]);
  const qr2=2*(V[0]*d[0]+V[1]*d[1])+d[0]*d[0]+d[1]*d[1];
  const rhoP=Math.sqrt(Math.max(rho*rho+qr2,0));
  const drho=qr2/(rhoP+rho);
  const Dth=Math.atan2(drho*V[2]-d[2]*rho, rho*rhoP+V[2]*(V[2]+d[2]));
  const a=N*th0,b=N*ph0,Da=N*Dth,Db=N*Dphi;
  const ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);
  const cA=Math.cos(Da),sA=Math.sin(Da),cB=Math.cos(Db),sB=Math.sin(Db);
  // DD via angle-addition RESIDUALS (cancellation-free): cos(x)-1 = -sin²x/(1+cos x),
  // dsA = sin(a+Da)-sin(a) = sa·(cA-1)+ca·sA, etc. Every term is O(Da)/O(Db) — the
  // original sAp*cBp-sa*cb subtracted O(1) values and was the test's real ~1e-13 floor.
  const cAm=-sA*sA/(1+cA), cBm=-sB*sB/(1+cB);
  const dsA=sa*cAm+ca*sA, dcA=ca*cAm-sa*sA;
  const dsB=sb*cBm+cb*sB, dcB=cb*cBm-sb*sB;
  const DD=[sa*dcB+dsA*cb+dsA*dcB, sa*dsB+dsA*sb+dsA*dsB, dcA], Dref=[sa*cb,sa*sb,ca];
  return add(scl(DD,Rn), scl(add(Dref,DD),dRn));
}

/* =====================  #2 — verify dd triplex orbit  ===================== */
console.log('#2  dd transcendental-free triplex vs f64 (should agree ~1e-14):');
for(const v of [[0.3,0.5,0.2],[0.9,-0.1,0.4],[-0.2,0.7,-0.6]]){
  const a=Pdd(v.map(ddF)).map(f), b=Pf(v);
  console.log(`   v=[${v}]  |P_dd - P_f64| = ${len(sub(a,b)).toExponential(2)}`);
}
// build a dd reference orbit and confirm it stays consistent
function ddOrbit(C, iters){ const V=[C.map(ddF)]; for(let i=0;i<iters;i++) V.push(addV(Pdd(V[i]),C.map(ddF))); return V; }

/* =====================  #1 — δ precision lever  ===================== */
// k f32 limbs of a dd value → dd (models GPU storage precision)
function roundLimbs(dd,k){ let acc=[0,0],r=[dd[0],dd[1]]; for(let i=0;i<k;i++){const h=Math.fround(r[0]+r[1]);acc=ddAdd(acc,[h,0]);r=ddSub(r,[h,0]);} return acc; }

function runLadder(C, dpos, iters, k){
  const V=ddOrbit(C,iters+1);
  const cpix=C.map((c,j)=>[c+dpos[j],0]);
  const A=[cpix.map(c=>[...c])]; for(let i=0;i<iters;i++) A.push(addV(Pdd(A[i]),cpix));
  let m=0, d=dpos.map(x=>roundLimbs([x,0],k)); const dcK=dpos.map(x=>roundLimbs([x,0],k));
  let maxErr=0;
  for(let it=0;it<iters;it++){
    const Vf=V[m].map(f), df=d.map(f);
    const sd=stableDelta(Vf,df);                                   // f64 delta update (the ceiling)
    d=[0,1,2].map(j=>roundLimbs(ddAdd([sd[j],0],dcK[j]),k));       // δ = sd + δc, stored at k limbs
    m+=1;
    let aRec=[0,1,2].map(j=>ddAdd(V[m][j],d[j]));                  // reconstruct actual in dd
    const aR2=aRec.reduce((s,c)=>s+f(c)**2,0), dM2=d.reduce((s,c)=>s+f(c)**2,0), Rm=Math.hypot(...V[m].map(f));
    if(aR2<dM2 || Rm<0.05 || m>=iters){ d=[0,1,2].map(j=>roundLimbs(ddSub(aRec[j],V[0][j]),k)); m=0; aRec=[0,1,2].map(j=>ddAdd(V[0][j],d[j])); }
    let e2=0; for(let j=0;j<3;j++){const dif=f(ddSub(aRec[j],A[it+1][j]));e2+=dif*dif;}
    maxErr=Math.max(maxErr,Math.sqrt(e2));
  }
  return maxErr;
}

// centers: generic, moderate off-axis dip (the regime-2 case from groundtruth Part 2)
const centers={ generic:[0.30,0.50,0.20], moDip:[0.90,0.0,0.10] };
console.log('\n#1  δ-precision floor sweep (err vs dd ground truth, 16 iters):');
console.log('center      offset      k=1(f32)     k=2(ds)      k=3');
for(const [name,C] of Object.entries(centers)){
  for(const mag of [1e-6,1e-9,1e-12,1e-15,1e-18]){
    const dpos=[mag,-mag*0.7,mag*0.4];
    const e1=runLadder(C,dpos,16,1), e2=runLadder(C,dpos,16,2), e3=runLadder(C,dpos,16,3);
    const p=x=>x.toExponential(2).padStart(11);
    console.log(`${name.padEnd(10)}  ${mag.toExponential(0).padStart(7)}   ${p(e1)}  ${p(e2)}  ${p(e3)}`);
  }
}
console.log(`
READING:
 #2 PROVEN: dd triplex matches f64 → the Chebyshev (transcendental-free) form is
    correct, so a dd / arbitrary-precision Mandelbulb REFERENCE ORBIT is feasible
    with only +,−,×,÷,sqrt. This removes the biggest worry about high-precision
    reference orbits for the bulb. (Bonus: no acos-domain clamp needed.)
 #1: if k=2/k=3 columns drop BELOW the k=1 (~1e-9) floor → δ precision is the depth
    lever; higher-precision δ storage is what buys orders past the f32 wall.
 UPDATE (2026-06-10, Fable): the old ~1e-13 ceiling was NOT "stableDelta is f64" —
    it was two O(1) cancellations in the implementation (Dphi numerator formed
    V0(V1+d1)-V1(V0+d0); DD formed sAp·cBp-sa·cb). Both now rewritten in exact
    cancellation-free residual form (V0·d1-V1·d0; angle-addition residuals with
    cos(x)-1 = -sin²x/(1+cos x)). Result: floors now track δ-storage precision —
    generic k=2 → ~2e-17, k=3 → ~1e-18 (they SEPARATE at offset 1e-18); moDip
    k=2 → ~2e-16 (was 6e-15). Remaining floor ≈ f64 trig's 1e-16 ABSOLUTE error —
    the genuine f64 limit, finally reached. To measure below it (toward 1e-50),
    NOW §8.2 applies: make the δ-update dd (Chebyshev, like Pdd). The same
    residual forms above are mandatory in the GLSL kernel (f32 hits the identical
    cancellation at ~1e-6 otherwise). groundtruth.test.mjs still has the OLD
    cancelling forms — port this fix there too.`);

/* ====================================================================================
   PART 3 — HDR-δ + REBASING CADENCE SWEEP  (Opus, 2026-06-10; handoff item 2)
   The GPU uses HDR-δ (f32 mantissa + extended exponent; double-single is dead on NVIDIA).
   HDR removes UNDERFLOW (δ can be 1e-50) but mantissa stays f32 (~7 relative digits). So
   the depth question is: does HDR-δ + rebasing keep δ's RELATIVE error small (err/offset
   ≪ 1, i.e. the pixel resolves) as the zoom deepens, and how often must we rebase?
   Metric = max |reconstructed_actual − true_actual| / |offset|  (resolvability).
   Ground truth is dd (~1e-32 abs) → at offset 1e-24 the measurement floor is ~1e-8; deeper
   offsets need item §8.3 (triple-double/BigInt orbit). We sweep within dd headroom.
   ==================================================================================== */
// HDR storage: f32 mantissa, unbounded binary exponent (no underflow). Models GPU HDR-δ.
function roundHDR(dd){ const v=dd[0]+dd[1]; if(v===0)return[0,0];
  const e=Math.floor(Math.log2(Math.abs(v))), s=Math.pow(2,e);
  return [Math.fround(v/s)*s, 0]; }

function runHDR(C, dpos, iters, rebaseEveryN){
  const V=ddOrbit(C,iters+3);                          // long enough that m never spuriously exhausts
  const cpix=C.map((c,j)=>[c+dpos[j],0]);
  const A=[cpix.map(c=>[...c])]; for(let i=0;i<iters;i++) A.push(addV(Pdd(A[i]),cpix));
  let m=0, d=dpos.map(x=>roundHDR([x,0])); const dcH=dpos.map(x=>roundHDR([x,0]));
  let maxRel=0, rebases=0; const off=Math.hypot(...dpos);
  for(let it=0;it<iters;it++){
    const Vf=V[m].map(f), df=d.map(f);
    const sd=stableDelta(Vf,df);
    d=[0,1,2].map(j=>roundHDR(ddAdd([sd[j],0],dcH[j])));
    m+=1;
    let aRec=[0,1,2].map(j=>ddAdd(V[m][j],d[j]));
    const aR2=aRec.reduce((s,c)=>s+f(c)**2,0), dM2=d.reduce((s,c)=>s+f(c)**2,0), Rm=Math.hypot(...V[m].map(f));
    const forced = rebaseEveryN>0 && ((it+1)%rebaseEveryN===0);
    if(aR2<dM2 || Rm<0.05 || m>=V.length-1 || forced){    // real Zhuoran/exhaustion only (+forced)
      d=[0,1,2].map(j=>roundHDR(ddSub(aRec[j],V[0][j]))); m=0; rebases++;
      aRec=[0,1,2].map(j=>ddAdd(V[0][j],d[j]));
    }
    let e2=0; for(let j=0;j<3;j++){const dif=f(ddSub(aRec[j],A[it+1][j]));e2+=dif*dif;}
    maxRel=Math.max(maxRel, Math.sqrt(e2)/off);
  }
  return {maxRel, rebases};
}

console.log('\n\nPART 3 — HDR-δ resolvability (err/offset) vs rebasing cadence, 16 iters:');
console.log('(HDR-δ = f32 mantissa + free exponent; lower err/offset = pixel better resolved)');
const cadences=[['cond-only',0],['every8',8],['every4',4],['every2',2],['every1',1]];
for(const [name,C] of Object.entries(centers)){
  console.log(`\ncenter ${name} [${C}]:`);
  console.log('  offset      '+cadences.map(c=>c[0].padStart(10)).join('  '));
  for(const mag of [1e-6,1e-12,1e-18,1e-24]){
    const dpos=[mag,-mag*0.7,mag*0.4];
    const cells=cadences.map(([,n])=>runHDR(C,dpos,16,n).maxRel.toExponential(1).padStart(10));
    console.log(`  ${mag.toExponential(0).padStart(7)}    ${cells.join('  ')}`);
  }
}
console.log(`
PART 3 READING — measured 2026-06-10 (Opus). Three results, one of them a correction:

 RESULT A (solid, actionable): FREQUENT REBASING IS POISON. cond-only ≈ 1e-7 at 1e-6 while
   every-N forced rebasing is ~3e0 at EVERY depth (catastrophic). Reason: rebasing re-anchors
   δ to the O(1) base reference (δ_new = actual − V0 ~ O(1)), and HDR's f32 mantissa rounds
   that O(1) value to ~1e-7 ABSOLUTE — destroying a deep δ. → KERNEL RULE: rebase ONLY on the
   Zhuoran condition (|V+δ|²<|δ|² ∨ R small ∨ ref-exhausted), NEVER on a fixed cadence. "The
   number" is: cadence = conditional-only.

 RESULT B (a CORRECTION — this metric is too strict): the err/offset climb (1e-7→~1 by 1e-18)
   is ABSOLUTE position-reconstruction error, which is NOT the rendering requirement. As in 2D
   deep zoom, the deep precision lives in the CPU REFERENCE orbit; the GPU δ needs only ~f32
   RELATIVE precision, and it's only ever consumed at O(1) scale (the escape test |V+δ|>2 and
   the DE, where δ's tiny absolute error is irrelevant). So "HDR-δ floors at ~1e-18" is an
   artifact of measuring the wrong quantity. The right metric is DE / escape-iteration relative
   error (de-perturbation.test.mjs), which tolerates f32-relative δ.

 RESULT C (the binding constraint): EVERY metric here now walls at the dd GROUND-TRUTH precision
   (~1e-13–1e-18 abs). We literally cannot measure HDR-δ's true deep behaviour — or build the
   correct DE-difference metric — without a higher-precision reference. → §8.3 (triple-double /
   BigInt orbit + a perturbed-DE-DIFFERENCE formulation that avoids the O(1) DE subtraction) is
   now the GATE for all deeper validation, not an optional refinement. Do it before claiming
   anything past ~1e-15.`);
