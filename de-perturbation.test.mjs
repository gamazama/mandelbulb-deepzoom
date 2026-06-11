/* =============================================================================
   de-perturbation.test.mjs  —  shortfall §8.1: the PERTURBED DISTANCE ESTIMATOR
   Run:  node de-perturbation.test.mjs
   -----------------------------------------------------------------------------
   The render needs the DE, not just the orbit position. The bulb DE uses a scalar
   running derivative:   dr_{k+1} = n·r_k^{n-1}·dr_k + 1,   DE = 0.5·ln(r)·r/dr.
   r_k = |z_k| = |V_k + δ_k|.  CLAIM (to test): dr has NO subtraction (always +,
   monotone growing) → no catastrophic cancellation → we can propagate it per-pixel
   directly, PROVIDED r_k is the accurate perturbed radius (which §3a gives us
   stably). So "perturbed DE" = "ordinary dr recurrence fed the stable perturbed r"
   — NOT a second delta problem. This test checks DE_pert vs DE_true (double).
   If it holds, §8.1 is closed: the DE needs no new perturbation math, just |V+δ|.
   ============================================================================= */

const N=8;
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2], len=a=>Math.hypot(a[0],a[1],a[2]);
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]], sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], scl=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
const f32=Math.fround, f32v=v=>v.map(f32);

function Pf(v){ const r=len(v); if(r===0)return[0,0,0];
  const th=Math.acos(Math.max(-1,Math.min(1,v[2]/r)))*N, ph=Math.atan2(v[1],v[0])*N, zr=Math.pow(r,N);
  return[zr*Math.sin(th)*Math.cos(ph), zr*Math.sin(ph)*Math.sin(th), zr*Math.cos(th)]; }

function stableDelta(V,d){
  const R=len(V); if(R<1e-300) return Pf(add(V,d));
  const q=2*dot(V,d)+dot(d,d), r2=Math.sqrt(R*R+q), Dr=q/(r2+R);
  const Rn=Math.pow(R,N), dRn=Rn*Math.expm1(N*Math.log1p(Dr/R));
  const Dphi=Math.atan2(V[0]*(V[1]+d[1])-V[1]*(V[0]+d[0]), V[0]*(V[0]+d[0])+V[1]*(V[1]+d[1]));
  const th0=Math.acos(Math.max(-1,Math.min(1,V[2]/R))), ph0=Math.atan2(V[1],V[0]);
  const ct=Math.cos(th0),st=Math.sin(th0),cp=Math.cos(ph0),sp=Math.sin(ph0);
  const thv=[ct*cp,ct*sp,-st], rh=scl(V,1/R);
  const Dth=Math.atan2(dot(thv,d), R+dot(rh,d));
  const a=N*th0,b=N*ph0,Da=N*Dth,Db=N*Dphi;
  const ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);
  const cA=Math.cos(Da),sA=Math.sin(Da),cB=Math.cos(Db),sB=Math.sin(Db);
  const sAp=sa*cA+ca*sA,cAp=ca*cA-sa*sA,cBp=cb*cB-sb*sB,sBp=sb*cB+cb*sB;
  const DD=[sAp*cBp-sa*cb, sAp*sBp-sa*sb, cAp-ca], Dref=[sa*cb,sa*sb,ca];
  return add(scl(DD,Rn), scl(add(Dref,DD),dRn));
}
// stable perturbed radius |V+δ| (§3a) — no cancellation
function stableR(V,d){ const R=len(V); const q=2*dot(V,d)+dot(d,d); return Math.sqrt(R*R+q); }

// ground-truth DE at a point p (full double orbit)
function deTrue(p, maxIter){
  let z=[...p], dr=1, r=len(z);
  for(let i=0;i<maxIter;i++){
    r=len(z); if(r>2) break;
    dr=N*Math.pow(r,N-1)*dr+1;
    z=add(Pf(z),p);
  }
  return {de:0.5*Math.log(Math.max(r,1e-12))*r/Math.max(dr,1e-20), iters:Math.min(maxIter,maxIter), rFinal:r};
}

// perturbed DE: reference orbit of C + δ-perturbation; dr fed the STABLE perturbed r.
// fp = δ storage precision (1 = f32, 0 = exact double) to also probe precision.
function dePert(C, dp, maxIter, useF32){
  // reference orbit
  const V=[[...C]]; for(let k=0;k<maxIter+2;k++) V.push(add(Pf(V[k]),C));
  let m=0, d=useF32?f32v(dp):[...dp], dr=1, r=0;
  const dcK=useF32?f32v(dp):[...dp];
  for(let i=0;i<maxIter;i++){
    r=stableR(V[m], d);                          // accurate perturbed radius
    if(r>2) break;
    dr=N*Math.pow(r,N-1)*dr+1;                    // ordinary dr recurrence, perturbed r
    let dn=add(stableDelta(V[m],d), dcK); if(useF32) dn=f32v(dn); d=dn;
    m++;
    // rebase if ref about to exhaust (keep concept simple; orbits here don't)
    if(m>=V.length-1){ const aRec=add(V[m],d); d=useF32?f32v(sub(aRec,V[0])):sub(aRec,V[0]); m=0; }
  }
  return 0.5*Math.log(Math.max(r,1e-12))*r/Math.max(dr,1e-20);
}

console.log('PERTURBED DE vs ground-truth DE  (n=8, 48 iters)');
console.log('relative error |DE_pert - DE_true| / DE_true\n');
const centers=[[0.40,0.35,0.30],[0.90,0.0,0.10],[0.20,0.60,0.55]];
for(const C of centers){
  const base=deTrue(C,48);
  console.log(`center [${C}]   DE_true=${base.de.toExponential(3)}`);
  console.log(`  offset      DE-rel-err (double δ)   DE-rel-err (f32 δ)`);
  for(const mag of [1e-3,1e-5,1e-7,1e-9]){
    const dp=[mag,-mag*0.6,mag*0.5];
    const t=deTrue(add(C,dp),48).de;
    const pd=dePert(C,dp,48,false), pf=dePert(C,dp,48,true);
    const e=(x)=>(Math.abs(x-t)/Math.abs(t)).toExponential(2).padStart(11);
    console.log(`  ${mag.toExponential(0).padStart(7)}     ${e(pd)}            ${e(pf)}`);
  }
  console.log('');
}
console.log(`READING (§8.1):
 • If 'double δ' rel-err is tiny (≪1) → the DE perturbs correctly with NO new math:
   feed the ordinary dr recurrence the stable perturbed radius |V+δ|. §8.1 closed in
   principle; the dr line ports straight into the GPU kernel next to the δ update.
 • 'f32 δ' column shows how δ storage precision feeds through to DE accuracy (the same
   lever as precision-ladder.test.mjs) — DE inherits the orbit's precision floor.
 • CAVEAT still open: (a) dr magnitude can overflow f32 at depth (it grows ~∏ n·r^{n-1});
   the kernel may need dr in log-space or rescaled. (b) escape/bailout mismatch between
   pixel and reference (§8.6) not stressed here — points chosen to stay bounded.`);
