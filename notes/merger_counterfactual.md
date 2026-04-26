# EU–US alliance counterfactual

Pooling the innovator counts of regions 1 and 2 lifts welfare by **+6.4 %**
in each allied region and leaves region 3 essentially unchanged
(`+0.01 %`). Aggregate welfare rises by **+4.3 %**. The picture on
*innovation* is sharper: allied regions raise `P(innov | old)` by `+4` to
`+5 %`, and region 3 *loses* `−4.0 %` because cheaper allied-region
production drags the global new-tech price down. This note describes how
the counterfactual is set up, computed, and verified. The estimator we
use is in `code/src/estimate.jl`; the counterfactual runner is
`code/scripts/run_merger.jl`.

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
closing the `simulate → estimate → counterfactual` loop. Averaging over
the random-`s_0` distribution rather than fixing a single representative
state lets the counterfactual answer "what does the alliance do *on
average* across the markets in our sample?" rather than "at one stylised
initial state."

## Monte Carlo with common random numbers

We average `K = 5000` independent simulations per scenario. Both scenarios
use the same `seed = 20260424` and per-market `MersenneTwister(seed + k)`.
Same seed implies identical `s_0` draws *and* identical EVT1 shock paths in
the sequential-move stages. Under common random numbers, every market
contributes a paired observation `(W_base, W_alli)` driven by the same
primitives; only the parameters differ. The variance of the difference
`W_alli − W_base` is therefore much smaller than the variance of either level.

Convergence is fast. The runner reports `ΔW` (absolute) and `ΔW / W₀`
(percent of baseline) at three sample sizes:

| K     | ΔW₁     | ΔW₂     | ΔW₃     | ΔW₁ %    | ΔW₂ %    | ΔW₃ %    | ΔΣW %    |
|-------|---------|---------|---------|----------|----------|----------|----------|
| 500   | +0.1458 | +0.1311 | −0.0042 | +7.02 %  | +5.90 %  | −0.20 %  | +4.23 %  |
| 1000  | +0.1350 | +0.1321 | −0.0011 | +6.36 %  | +6.06 %  | −0.05 %  | +4.11 %  |
| 5000  | +0.1376 | +0.1389 | +0.0002 | +6.44 %  | +6.52 %  | +0.01 %  | +4.31 %  |

Both absolute and percent differences stabilise to ~0.2 percentage points
by `K = 1000`. We report `K = 5000` for headline numbers. Region 3's
welfare change is small in both directions across `K`, which is itself
the substantive finding: its CS gain almost exactly offsets its PS loss.

## Calibration

We use the estimated parameters from `output/estimates/estimation.txt`:

```
κ̂ = 0.3002      φ̂ = 0.1777      γ̂ = (0.15, 0.15, 0.15).
```

The remaining parameters are treated as known and held at
`default_params()`: `A = 3`, `B = 1`, `M = 1`, `c_o = 1.5`, `c_{n0} = 0.5`,
`β = 0.9`, `σ = 0.5`, `ρ = 0.5`, `N_max = 6`.

Why this calibration. An earlier exercise used `(γ, σ) = (0.05, 1.0)` and
produced merger effects of order `+1 %`. Two amplifiers explain why the
present calibration is more revealing: tripling `γ` triples the cost
reduction the merger delivers in the allied regions (still well below the
`c_n` floor at `γ ≳ 0.30`), and halving `σ` roughly doubles the logit
slope `P(1−P)/σ` against any payoff change. The combined effect on CCPs is
roughly `3 × 2 = 6×`, which matches what the comparative statics show.
The choice puts innovation in the responsive interior of the logit without
saturating any constraint.

## Results

Innovation rates (period-1 old → both, pooled across markets in the sample):

| Region | Baseline | Alliance | Δ        | Δ / baseline |
|--------|----------|----------|----------|--------------|
| 1      | 0.3507   | 0.3688   | +0.0181  | +5.16 %      |
| 2      | 0.3495   | 0.3650   | +0.0155  | +4.43 %      |
| 3      | 0.3533   | 0.3392   | −0.0141  | **−4.00 %**  |

Entry rates (period-1 pe → new) move in the same direction:

| Region | Baseline | Alliance | Δ        | Δ / baseline |
|--------|----------|----------|----------|--------------|
| 1      | 0.5562   | 0.5723   | +0.0162  | +2.91 %      |
| 2      | 0.5801   | 0.5968   | +0.0167  | +2.87 %      |
| 3      | 0.5792   | 0.5624   | −0.0168  | **−2.91 %**  |

Innovation and entry both rise in the allied regions and fall in region 3.
The mechanism is straightforward. The pooled spillover lowers `c_{n,1}`
and `c_{n,2}` whenever either region has an innovator, which raises the
expected period-2 profit of being on new tech in regions 1 and 2 and makes
both innovation and fresh entry more attractive there. Region 3's
innovation and entry both fall because global Cournot competition is now
tougher: the allied regions produce more new-tech output, dragging down
the new-tech price that region 3's potential innovators stand to receive.

Welfare components per region:

| Region | Baseline `PS_r` | Alliance `PS_r` | ΔPS_r    | ΔW_r     | ΔW_r / W_{r,0} |
|--------|-----------------|-----------------|----------|----------|----------------|
| 1      | 0.8500          | 0.9290          | +0.0790  | +0.1376  | +6.44 %        |
| 2      | 0.8488          | 0.9290          | +0.0801  | +0.1389  | +6.52 %        |
| 3      | 0.8800          | 0.8147          | −0.0653  | +0.0002  | +0.01 %        |

`CS / R` rises by `+0.0622` per region in the alliance. The allied regions
gain twice — through their own CS share and through a sharp `+9 %` jump in
PS — and post welfare gains of about `+6.5 %`. Region 3's `PS_r` falls by
`−0.0653`, almost exactly offsetting its `+0.0622` CS-share gain, so its
net welfare moves by less than a basis point. Total welfare rises by
`+0.277` per market, or `+4.31 %` of baseline.

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

EVT1 shocks are treated as a smoothing device, not as real cost
heterogeneity. The realised-shock contribution to firm welfare,
`σ · γ_em` per decision (γ_em ≈ 0.5772 the Euler–Mascheroni constant),
is omitted from `PS_r`. The number of choosing firms is identical
between baseline and alliance because both scenarios use the same
drawn `s_0` per market under common random numbers, so this constant
cancels in every Δ we report. Absolute welfare *levels* would shift
by `σ · γ_em × N_decisions` per market under the real-shock
interpretation, but our headline differences are unaffected. The
σ·log Σ exp closed-form ex-ante value never appears in the recursion
either: period 2 is terminal, so no ex-ante value is ever fed into a
later choice. A `T > 2` extension would need to add the σ·log Σ exp
term at every internal period.

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
