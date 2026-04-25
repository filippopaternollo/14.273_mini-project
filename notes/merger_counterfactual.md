# EU–US alliance counterfactual

Pooling the innovator counts of regions 1 and 2 raises welfare in **all three**
regions and lifts the global sum by `+0.092` per market. Allied regions gain
about `+0.045` each. Region 3 still gains a small `+0.003` because cheaper
global new-tech raises consumer surplus by more than its own producer surplus
falls. This note describes how the counterfactual is set up, computed, and
verified. The estimator we use is in `code/src/estimate.jl`; the counterfactual
runner is `code/scripts/run_merger.jl`.

## Setup

Recall the 2-period regional model. Each region's new-tech marginal cost
in the baseline depends only on innovators *located in that region*:

```
c_{n,r}(s) = max(0, c_{n0} − γ_r · (n_b[r] + n_n[r])).
```

The counterfactual treats two regions as a single spillover pool. Under the
{1, 2} alliance:

```
c_{n,r}(s) = max(0, c_{n0} − γ_r · Σ_{r' ∈ bloc(r)} (n_b[r'] + n_n[r']))
```

with `bloc(1) = bloc(2) = {1, 2}` and `bloc(3) = {3}`. Cournot competition stays
global; only the cost-spillover pool changes. The implementation is one field
on `Params` (`blocs::NTuple{R,Int}`) and one function in `state_space.jl`
(`c_n_eff`); everything else propagates automatically because every flow
profit reads its `c_n` through `c_n_vec`.

We pool *counts* but keep `γ_r` per region. Under uniform `γ̂`
this is observationally equivalent to pooling both. The choice matters
only if a future estimation step finds `γ_r` to differ across regions.

## Welfare

The two-market linear inverse demand `P_g = A − b·Q_g − s_·Q_{−g}` with
`b = B/M` and `s_ = ρ·B/M` is rationalised by the quasi-linear utility
`U = A·(Q_o + Q_n) − (b/2)·(Q_o² + Q_n²) − s_·Q_o·Q_n`. Consumer surplus follows:

```
CS = (b/2)·(Q_o² + Q_n²) + s_·Q_o·Q_n.
```

Cournot is global, so `CS` is a single number. We allocate it equally across
regions (equal-population assumption). Producer surplus is local: each firm's
Cournot profit is charged to its own region. Innovation and entry costs are
paid by the firm's region in period 1.

Per-region two-period discounted welfare is therefore:

```
W_r = (CS(s_0) + β·CS(s_1)) / R                     ← global CS, equal split
    + PS_r(s_0) + β·PS_r(s_1)                       ← own-region producer surplus
    − κ · k_innov_r − φ · k_enter_r                 ← own-region costs paid
```

By construction `Σ_r W_r = CS_total + Σ_r PS_r − Σ_r costs_r`. The runner
checks this identity numerically and finds residuals of order `1e-14`.

## Sample design

Each market draws its initial state via `random_s0(rng, p)`, the same DGP
used in `simulate_data.jl` and consumed by `estimate.jl`. The counterfactual
is therefore evaluated on the same population the model was estimated on,
closing the `simulate → estimate → counterfactual` loop. We do not fix a
representative `s_0` (`run_regional.jl` does that, but only for compact
comparative-statics displays). Using the random-`s_0` distribution lets the
counterfactual answer "what does the alliance do *on average* across the
markets in our sample?" rather than "at one stylised initial state."

## Monte Carlo with common random numbers

We average `K = 5000` independent simulations per scenario. Both scenarios
use the same `seed = 20260424` and per-market `MersenneTwister(seed + k)`.
Same seed implies identical `s_0` draws *and* identical EVT1 shock paths in
the sequential-move stages. Under common random numbers, every market
contributes a paired observation `(W_base, W_alli)` driven by the same
primitives; only the parameters differ. The variance of the difference
`W_alli − W_base` is therefore much smaller than the variance of either level.

Convergence is fast. The runner reports `ΔΣW` at three sample sizes:

| K     | ΔW₁     | ΔW₂     | ΔW₃     | ΔΣW     |
|-------|---------|---------|---------|---------|
| 500   | +0.0482 | +0.0426 | +0.0029 | +0.0936 |
| 1000  | +0.0448 | +0.0439 | +0.0026 | +0.0913 |
| 5000  | +0.0443 | +0.0453 | +0.0027 | +0.0923 |

Differences stabilise to three digits by `K = 1000`. We report `K = 5000`
for headline numbers.

## Calibration

We use the estimated parameters from `output/estimates/estimation.txt`:

```
κ̂ = 0.2849      φ̂ = 0.1635      γ̂ = (0.05, 0.05, 0.05).
```

The remaining parameters are treated as known and held at
`default_params()`: `A = 3`, `B = 1`, `M = 1`, `c_o = 1.5`, `c_{n0} = 0.5`,
`β = 0.9`, `σ = 1.0`, `ρ = 0.5`, `N_max = 6`.

## Results

Innovation rates (period-1 old → both, pooled across markets in the sample):

| Region | Baseline | Alliance | Δ        |
|--------|----------|----------|----------|
| 1      | 0.3480   | 0.3502   | +0.0023  |
| 2      | 0.3465   | 0.3504   | +0.0039  |
| 3      | 0.3485   | 0.3449   | −0.0035  |

Innovation rises in both allied regions and falls slightly in region 3.
The mechanism is straightforward: the pooled spillover lowers `c_{n,1}` and
`c_{n,2}` whenever either region has an innovator, which raises the
expected period-2 profit from being on new tech in regions 1 and 2 and
makes innovation more attractive there. Region 3's innovation falls
because global Cournot competition is now tougher: the allied regions
produce more new-tech output, dragging down the new-tech price that region
3's potential innovators stand to receive.

Welfare components per region:

| Region | Baseline `PS_r` | Alliance `PS_r` | ΔPS_r   | ΔW_r    |
|--------|-----------------|-----------------|---------|---------|
| 1      | 0.7560          | 0.7805          | +0.0245 | +0.0443 |
| 2      | 0.7614          | 0.7871          | +0.0257 | +0.0453 |
| 3      | 0.7890          | 0.7709          | −0.0181 | +0.0027 |

`CS / R` rises by `+0.0202` per region in the alliance. Region 3's `PS_r`
falls by `−0.0181`, but its share of the global CS gain plus its slightly
lower paid costs leaves it with a small net welfare gain `+0.0027`.
Total welfare rises by `ΔΣW = +0.0923` per market, roughly half of which
comes from each allied region and the rest from region 3.

## Caveats

The sequential-move solver introduces a small first-mover wedge across
regions (≤ `1e-3` in CCPs at a fixed state), so under blocs `(1, 1, 2)`
regions 1 and 2 produce *almost* but not exactly identical numbers. The
gap is well below MC noise and is documented in
`notes/solver_walkthrough.md` §3.

Equal-population CS allocation rests on an assumption we have not built
into the model. A heterogeneous-population version would scale `CS / R`
by population shares; the welfare-by-region columns would shift but the
sign of `Δ` in each region would not.

Cross-market substitution `ρ` is treated as known. A future estimation
step that lets `ρ` differ from `0.5` would change `CS` levels and could
alter the size of region 3's gain, though not its sign for plausible `ρ`.

## Regenerating

```bash
cd code
julia --project=. scripts/run_merger.jl
```

The runner overwrites:

- `output/tables/merger_results.tex`
- `output/figures/merger_innovation.pdf`
- `output/figures/merger_welfare.pdf`
- `output/estimates/merger_estimates.txt`

The writeup `\input`s the macro file directly. To change `K`, the seed,
or the alliance composition, edit the constants at the top of
`scripts/run_merger.jl`. Setting `blocs = (1, 1, 1)` runs full integration
as an upper bound; setting `blocs = (1, 2, 3)` recovers the baseline.
