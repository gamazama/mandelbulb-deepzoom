# Triplex Perturbation Math — the cancellation-free delta recurrence

> The derivation behind this renderer: the exact 3×3 triplex Jacobian (§2) and an **exact,
> cancellation-free delta recurrence** for the non-analytic triplex power (§3). A literature sweep
> (June 2026) found no prior published cancellation-free expansion of the triplex power, so §3 is, to
> our knowledge, new. It has since been **numerically verified** against arbitrary-precision ground
> truth and certified end-to-end on the GPU — see `groundtruth.test.mjs`, `direct-check.test.mjs`
> (11/11 vs an independent BigInt referee, 1e-5 → 1e-50), and `power-cert.mjs`. Notation matches the
> renderer's GLSL so the two stay consistent.

---

## 0. The map and convention

We match GMT's `formula_Mandelbulb` convention exactly (so the harness GLSL lift is consistent):

```
v = (x, y, z) ∈ ℝ³
r     = |v| = sqrt(x²+y²+z²)
theta = acos(clamp(z/r, -1, 1))      # polar angle from +z,  θ ∈ [0,π]
phi   = atan2(y, x)                  # azimuth,               φ ∈ (-π,π]

P(v)  = r^n · ( sin(nθ)cos(nφ), sin(nθ)sin(nφ), cos(nθ) )      # n = power, typically 8
```

Iteration: `v_{k+1} = P(v_k) + c`. Phase offsets (GMT's `uVec2A`) just add constants to `nθ, nφ`
*after* multiplication; they do not change any derivative below, so we omit them. Include them in
code by replacing `nθ → nθ + φ_offset_theta`, `nφ → nφ + φ_offset_phi` wherever an OUTPUT angle appears.

**Why 2D is easy and this is not.** For `z²`, `(Z+δ)² − Z² = 2Zδ + δ²` is *exact and finite* — two
terms, no cancellation, no truncation. `P` is non-analytic (routes through `|v|`, `acos`, `atan2`,
trig-of-8×-angle), so its Taylor series in δ is **infinite**. Any Jacobian-only scheme (§2) therefore
**drifts** unless rebased; the exact reorganisation (§3) avoids the drift but costs more per iteration.

---

## 1. Spherical frame (used throughout)

At a point with angles (θ, φ), the orthonormal spherical basis is:

```
r̂ = ( sinθ cosφ,  sinθ sinφ,  cosθ )
θ̂ = ( cosθ cosφ,  cosθ sinφ, -sinθ )
φ̂ = (-sinφ,        cosφ,       0     )
```

Cartesian→spherical gradients (standard):  `∇r = r̂`,  `∇θ = θ̂/r`,  `∇φ = φ̂/(r sinθ)`.

---

## 2. The exact Jacobian J_P (linear perturbation)

Derived via chain rule `P = (spherical→cartesian) ∘ (scale: r↦rⁿ, θ↦nθ, φ↦nφ) ∘ (cartesian→spherical)`.
Let primed basis vectors `r̂', θ̂', φ̂'` be evaluated at the OUTPUT angles `(nθ, nφ)`, and unprimed at
the INPUT angles `(θ, φ)`. Then for any cartesian δ:

```
                 ┌                                                            ┐
J_P(V)·δ = n·Rⁿ⁻¹ │ (r̂·δ) r̂'  +  (θ̂·δ) θ̂'  +  (sin(nθ)/sinθ)·(φ̂·δ) φ̂' │
                 └                                                            ┘
        with  R = |V|,  and  θ,φ  the spherical angles of V.
```

**Reading it:** the scalar gain `n·Rⁿ⁻¹` is the exact analog of 2D's `|n·Zⁿ⁻¹|`. The map takes δ's
components in the *input* frame {r̂,θ̂,φ̂}, rotates them into the *output* frame {r̂',θ̂',φ̂'} (the angle
multiplication is a frame rotation), and additionally scales the azimuthal component by
`sin(nθ)/sinθ = U_{n-1}(cosθ)` (a Chebyshev-U ratio).

**Pole behaviour:** `sin(nθ)/sinθ → ±n` as `θ→0,π` (finite, bounded by n), so the Jacobian *magnitude*
is well-behaved at the poles. What degenerates there is the *decomposition* of δ into θ̂/φ̂ (the frame
itself is undefined on the axis). See §4.

The first-order perturbation recurrence is then:

```
δ_{k+1} = J_P(V_k)·δ_k  +  R_k  +  δc
```

where `R_k` is the higher-order remainder (what 2D gets exactly as `δ²`). **Linear-only = truncated**;
`R_k = O(|δ|²·n²·Rⁿ⁻²)`. Usable ONLY with aggressive rebasing (§5) to keep |δ|≪|V|. This is the cheap
path: implement it first to get *a* picture moving, expect glitches at depth.

---

## 3. ★ Exact cancellation-free delta (the promising path) ★

Goal: compute `Δ = P(V+δ) − P(V)` in f32, given the reference `V` (and its precomputed `R=|V|`,
angles, `Rⁿ`) to high precision and `δ` small — **without ever subtracting two nearby large numbers.**
Every step below is exact (no series truncation); the trick is purely reorganisation so the small
quantity is produced *directly* rather than as a difference of big ones.

### 3a. Stable radius delta
```
q   = 2·(V·δ) + |δ|²                  # exact; small relative to R²
r₂  = sqrt(R² + q)                    # = |V+δ|
Δr  = q / (r₂ + R)                    # STABLE: no cancellation (rationalised difference)
```

### 3b. Stable radial-power delta  (Δ(rⁿ) = r₂ⁿ − Rⁿ)
```
u        = Δr / R                     # small
Δrn_over_Rn = expm1(n · log1p(u))     # = (1+u)ⁿ − 1, STABLE for small u
Δ(rⁿ)    = Rⁿ · Δrn_over_Rn
r₂ⁿ      = Rⁿ + Δ(rⁿ)
```
(`expm1`/`log1p` are not GLSL built-ins — provide them: `log1p(u)=log(1+u)` is fine for |u|≳1e-4,
else use the series `u - u²/2 + u³/3`; `expm1(x)=exp(x)-1`, series `x + x²/2 + ...` for small x.)

### 3c. Stable angle deltas (Δθ, Δφ) via atan2-of-(cross,dot) — BOTH EXACT
> ⚠️ HISTORY (Fable slice 5, 2026-06-10): the original Δθ here was FIRST-ORDER
> (`atan2(θ̂·δ, R + r̂·δ)`). Its O(δ²) per-step truncation chaos-amplified over the
> 30-90-iteration near-critical orbits the renderer marches and produced visibly
> mangled surfaces past 1e-4 — caught by the USER's eye + a non-circular direct
> referee after passing three implementation-agreement suites. The form below is
> exact, same construction as Δφ. Do not regress this.

Compute the small rotation directly, never `θ₂ − θ_V`. Both angles use the exact
difference-of-atan2 identity `atan2(s₁,c₁) − atan2(s₂,c₂) = atan2(s₁c₂−c₁s₂, c₁c₂+s₁s₂)`
with all small quantities produced as cancellation-free residuals:

```
# azimuth: project onto xy-plane
Δφ = atan2( V.x·δ.y − V.y·δ.x ,  V.x·(V.x+δ.x) + V.y·(V.y+δ.y) )    # EXACT, stable

# polar: θ = atan2(ρ, z) with ρ = √(x²+y²).  Δθ = atan2(ρ', z') − atan2(ρ, z):
qρ  = 2·(V.x·δ.x + V.y·δ.y) + δ.x² + δ.y²        # exact (= ρ'² − ρ²)
ρ'  = sqrt(ρ² + qρ)
Δρ  = qρ / (ρ' + ρ)                               # rationalised — no cancellation
Δθ  = atan2( Δρ·V.z − δ.z·ρ ,  ρ·ρ' + V.z·(V.z+δ.z) )              # EXACT, stable
```
The numerator `ρ'z − z'ρ = (ρ+Δρ)z − (z+δz)ρ = Δρ·z − δz·ρ` — every term carries a
δ-residual factor, no big−big subtraction. Validated end-to-end: escape-bailout.test.mjs
22/22 to offset 1e-2, direct-check.test.mjs 11/11 vs the BigInt direct referee to 1e-50.

### 3d. Output angles and their deltas
```
Δa = n·Δθ           # output polar delta   (a = nθ)
Δb = n·Δφ           # output azimuth delta  (b = nφ)
a  = nθ_V , b = nφ_V                       # reference output angles (from the precomputed orbit)
```

### 3e. Stable direction delta  ΔD = D(a+Δa, b+Δb) − D(a, b),  D(a,b)=(sin a cos b, sin a sin b, cos a)
Expand with angle-sum identities, grouping so every term carries a `sinΔ` or `(cosΔ−1)` factor
(both → 0 with the delta, no cancellation). With `ca=cos a, sa=sin a, cb=cos b, sb=sin b`,
`cA=cos Δa, sA=sin Δa, cB=cos Δb, sB=sin Δb`, and using `cosΔ−1 = −2 sin²(Δ/2)` (stable):

```
sin(a+Δa) = sa·cA + ca·sA
cos(a+Δa) = ca·cA − sa·sA
cos(b+Δb) = cb·cB − sb·sB
sin(b+Δb) = sb·cB + cb·sB

ΔD.x = sin(a+Δa)cos(b+Δb) − sa·cb
ΔD.y = sin(a+Δa)sin(b+Δb) − sa·sb
ΔD.z = cos(a+Δa) − ca
```
To make ΔD itself cancellation-free, substitute the expansions and factor `(cA−1)=−2sin²(Δa/2)` etc.
so no bare `big − big` remains. (The substitution is mechanical; the testbench checks it.) For modest
δ the direct form above is already accurate because `a,b` come from the *high-precision* reference, so
`sa,ca,sb,cb` are exact and only the small `sA,sB,(cA−1),(cB−1)` carry δ.

### 3f. Assemble the exact delta
```
D_ref = ( sa·cb, sa·sb, ca )                       # = P(V)/Rⁿ, the reference direction
Δ = Rⁿ · ΔD  +  Δ(rⁿ) · ( D_ref + ΔD )             # product-rule split, both factors are deltas
  = Rⁿ·ΔD  +  Δ(rⁿ)·D_ref  +  Δ(rⁿ)·ΔD
```
Then the perturbation step:  **`δ_{k+1} = Δ + δc`** , with `δc` = this pixel's offset from the
reference c (carried in double-single / HDR exactly as GMT's 2D `dc`).

**This is exact** — no truncated series. The whole game was producing `Δr, Δ(rⁿ), Δθ, Δφ, ΔD`
*directly* instead of by subtracting nearby large values.

---

## 4. Where it breaks (the genuinely hard spots — budget Fable's time here)

1. **Reference passes near the origin** (`R = |V_k| → 0`): `Rⁿ⁻¹`, `1/R`, `1/(R sinθ)` blow up; the
   spherical frame is undefined at `v=0`. The Mandelbulb orbit *does* visit small-|v| points. Mitigation:
   detect `R < ε_origin` and **rebase** (§5) or fall back to a direct double-single evaluation for that
   iteration. This is the triplex analog of 2D's `|Z|→0` glitch and is the #1 expected failure mode.
2. **Poles** (`sinθ → 0`, i.e. v near ±z axis): `φ` and `φ̂` ill-conditioned; `Δφ` via atan2 stays
   finite but loses meaning. `sin(nθ)/sinθ` is fine (→n). Watch the φ-channel specifically.
3. **Azimuth wrap** (`φ` crossing ±π): the atan2-of-(cross,dot) form in §3c is *immune* (it never forms
   `φ₂−φ_V` directly), which is exactly why it's preferred over naive angle subtraction.
4. **f32 underflow of δ at depth** (the 2D problem too): at 1e-50, `δ` underflows f32 (`~1e-38`). You
   **must rescale** — carry δ as `(mantissa·2^e)` HDR pairs, or periodically renormalise. GMT's 2D HDR
   float helpers (`hdrFromFloat/hdrMul/...` in `fractalKernel.ts`) port directly.

---

## 5. Rebasing (Zhuoran) — the safety net, already in GMT

Identical in spirit to the 2D code. When `|δ_k| > κ·|V_k|` (κ ≈ 0.5–1) OR `|V_k|` is tiny OR the ref
index hits the orbit end: **reset** `v_actual = V_k + δ_k`, restart the reference index at 0, set
`δ ← v_actual` relative to `V_0`. This bounds the truncation error of the linear scheme (§2) and rescues
the near-origin glitch of the exact scheme (§3). GMT's 2D glitch-rebase
(`fractalKernel.ts` ~lines 708-719) and period-modulo-wrap (`uRefPeriod`) are the template — port the
control flow verbatim, swap the 2D `dz` update for §3's `Δ`.

---

## 6. Recommended attack order for Fable

1. **Validate the math numerically FIRST** (`groundtruth.test.mjs`): does §3's exact delta, evaluated
   with f32-simulated δ over a double-precision reference orbit, match a direct double evaluation to
   ~f32 epsilon for many iterations and many pixel offsets? If yes → the algorithm is sound, build the
   GPU kernel. If no → the failure mode (which term diverges, at which iteration) tells you exactly
   what to fix before spending GPU time. **Do not write the shader until this passes.**
2. **CPU reference orbit** in double-double (port GMT `dd.ts` + `referenceOrbit.ts` pattern; iteration
   body swapped to the triplex power). Store per-iter `(Vx,Vy,Vz, R, nθ, nφ)` — precompute the angles so
   the kernel needn't call acos/atan2 on the reference.
3. **GPU delta kernel** = §3 in GLSL, with §5 rebasing and §4.4 HDR rescaling. Reuse GMT's orbit-texture
   upload + double-single center.
4. **Only then** optimise (linear-Jacobian §2 fast path with denser rebasing; LA/BLA-style skip tables
   are a *stretch* — they need the operator-norm bound on J_P, doable from §2's `n·Rⁿ⁻¹` gain but unproven).

See `FABLE_BRIEF.md` for how this slots into the staged plan and `2d-perturbation-pattern.md` for the
GMT scaffold to mirror.
