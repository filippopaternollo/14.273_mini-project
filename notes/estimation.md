# Estimation of the 2-period regional model

Companion to `notes/simulated_data.md`. Describes the two-step estimator
implemented in `code/src/estimate.jl` and driven by
`code/scripts/estimate.jl`.

## What we observe

`data/simulated_data.csv` has one row per firm × period with columns
`market_id, period, firm_id, region, firm_type, q_old, q_new, profit,
action`. Per-market initial and terminal states are reconstructed by
counting firms of each type × region within each period — this is the
exact inverse of the roster construction in `simulate.jl`.

## What we estimate

Unknown parameters:
- `κ` — innovation cost (`old → both` transition).
- `φ` — entry cost (`pe → new` transition).
- `γ = (γ₁, γ₂, γ₃)` — per-region agglomeration spillover:
  `c_{n,r}(s) = max(0, c_n0 − γ_r · (n_b[r] + n_n[r]))`.

Known (fixed at `default_params()`): demand `(A, B, M, ρ)`, old-tech
cost `c_o`, baseline new-tech cost `c_n0`, discount factor `β`, EVT1
scale `σ`, state cap `N_max`.

## Identification

Cournot quantities are **deterministic** given the state and `γ`
(`cournot_quantities_regional`): there is no measurement error on `q`,
so `γ` is pinned exactly by any observed quantity. Conditional on `γ`,
the period-1 discrete actions are logit draws from the sequential-move
solver, so `κ, φ` are identified from the CCPs via standard MLE.

This suggests the two-step structure:

1. **NLS on quantities → γ̂.**  Minimize
   `Σ (q_obs − q_hat(γ))²` over all (market, period, region, slot)
   cells. With simulated, noiseless data the minimum is machine zero
   and `γ̂` equals the truth. This step doubles as a sanity check that
   state reconstruction is correct.

2. **MLE on actions → (κ̂, φ̂) | γ̂.**  Maximize
   `Σ_m ℓ_m(κ, φ; γ̂)` where `ℓ_m` is the per-market period-1 action
   log-likelihood built by *replaying* the stage order the simulator
   uses.

## Stage replay

`solve_state` returns **marginal** CCPs integrated over the forward
traversal, whereas `simulate_market` draws each firm's action at the
*realized* intermediate sub-state (after earlier regions and earlier
stages have updated the state). Evaluating the marginal CCPs at `s₀`
would therefore not match the DGP.

The likelihood replays the stage order used in `simulate.jl:99–172`:

```
OLD  stage: for r in 1..R, call solve_old_region(s, r, …)  at current s
BOTH stage: for r in 1..R, call solve_both_region(s, r, …) at current s
NEW  stage: for r in 1..R, call solve_new_region(s, r, …)  at current s
PE   stage: for r in 1..R, call solve_pe_region(s, r, …)   at current s
```

After each region's decisions are observed, the sub-state `s` is
updated exactly as in the simulator before moving to the next region.
The likelihood contribution of an individual firm uses the stage CCP at
the sub-state it actually faced.

Per-market contribution:

```
ℓ_m = Σ_r [ k_so·log p_so(s_OLD_r) + k_io·log p_io(s_OLD_r)
                                   + k_eo·log(1−p_so−p_io) ]
    + Σ_r [ k_sb·log p_sb(s_BOTH_r) + k_eb·log(1−p_sb) ]
    + Σ_r [ k_sn·log p_sn(s_NEW_r)  + k_en·log(1−p_sn)  ]
    + Σ_r [ k_ep·log p_ep(s_PE_r)   + k_op·log(1−p_ep)  ]
```

where `k_*` are counts of each action outcome in region `r`, recovered
from the CSV period-1 rows.

## Optimization

Both steps use `Optim.NelderMead`. γ is left **unconstrained** so we can
see whether the data want a negative spillover (they don't, but the
point is to let the optimizer decide).

Starting values are intentionally far from the truth:

- NLS: `γ⁰ = (0, 0, 0)` (true `γ = (0.05, 0.05, 0.05)`).
- MLE: `(κ⁰, φ⁰) = (1.0, 1.0)` (true `(0.3, 0.2)`).

This is a deliberate diagnostic: if the optimizer lands near the truth
starting from a point well away from it, we know the likelihood is
well-behaved and that the solver is actually being exercised.

## Standard errors

BHHH (outer-product-of-gradients) standard errors for `(κ̂, φ̂)`:

```
B = Σ_m g_m g_m',   g_m = ∇ ℓ_m(κ̂, φ̂),   V = B⁻¹
```

Gradients are computed by central finite differences on per-market
log-likelihoods (step `1e-4`). γ is held fixed at `γ̂`, so reported
SEs reflect step-2 sampling variation only.

## Results (500 markets)

| Parameter | Truth   | Estimate | S.E.    |
| --------- | ------- | -------- | ------- |
| κ         | 0.3000  | 0.3470   | 0.0773  |
| φ         | 0.2000  | 0.2307   | 0.0545  |
| γ₁        | 0.0500  | 0.0500   |  —      |
| γ₂        | 0.0500  | 0.0500   |  —      |
| γ₃        | 0.0500  | 0.0500   |  —      |

NLS SSR at `γ̂`: ~1e-13 (machine zero, as expected for noiseless
data). Log-likelihood at the optimum: ≈ −3181.4.  Both `κ̂` and `φ̂`
are within ~1 standard error of their true values; with only 500
markets and handful of deciders per market this is about the precision
we should expect.

The same numbers are available as LaTeX macros in
`output/estimates/estimation.txt` and as a tabular in
`output/tables/estimation.tex`.

## Regenerating

```bash
cd code
julia --project=. scripts/estimate.jl
```

Outputs are overwritten each run. To change starting values, edit the
top of `scripts/estimate.jl`; to change the data set, rerun
`scripts/simulate_data.jl` first.
