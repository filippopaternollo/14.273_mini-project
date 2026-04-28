# Region-3 innovation subsidy counterfactual

A region-3 innovation subsidy of size τ shifts the private cost of
innovation in region 3 from κ̂ to (κ̂ − τ) while leaving the social
resource cost unchanged at κ̂. Each region is treated as a sovereign
country: its own government funds its own subsidy from its own taxpayers,
so the τ·k_innov,3 transfer cancels exactly inside region 3's welfare and
never crosses borders. Sweeping τ over the grid τ/κ̂ ∈ {0, 0.1, …, 0.5}
shows that **aggregate welfare is weakly decreasing in the subsidy
across the entire grid**, with the welfare-maximising point at τ = 0.
Region 3 raises P(innov | old) by **+18.73 %** at the largest grid point
and its own welfare moves by **+0.06 %**, while regions 1 and 2 lose
**−0.08 %** and **−0.09 %** respectively. The estimator we use is in
`code/src/estimate.jl`; the counterfactual runner is
`code/scripts/run_subsidy.jl`.

## Setup

Recall the 2-period regional model. In the baseline, an old-tech
incumbent in region r pays κ to switch to "both" technology, and a
potential entrant pays φ to enter as "new". A *region-r innovation
subsidy* of size τ_r ≥ 0 changes only the **private** innovation payoff:

```
u^innov_r = π^both_r(s) − (κ − τ_r) + β · EV^old→both_r,
```

leaving the entry decision and the rest of the game unchanged. The
implementation is one field on `Params` (`subsidy::NTuple{R,Float64}`)
and one line in `solver.jl` (`kappa_priv = p.kappa - p.subsidy[r]`);
everything else propagates automatically because every CCP reads its κ
through that local variable.

We sweep a one-region subsidy `(0, 0, τ)` rather than a global one. This
isolates the question "what does targeted industrial policy in region 3
do?" from the question "should the planet subsidise R&D at all?". The
spillover spec stays at the local-only baseline `c_{n,r} = c_{n0} −
γ_r·(n_b[r] + n_n[r])`; pooling spillovers across regions is the
*alliance* counterfactual (`notes/merger_counterfactual.md`).

## Welfare under sovereign funding

Consumer surplus is global, allocated equally across regions; producer
surplus is local; innovation and entry costs are paid in the firm's
region in period 1. The novelty here is the **transfer**: τ_r·k_innov,r
flows from region r's taxpayers (a flat lump sum on its consumers) to
region r's innovating firms. Under the sovereign-funding interpretation,
that transfer never leaves region r. In welfare terms,

```
W_r = (CS(s_0) + β·CS(s_1)) / R                ← global CS, equal split
    + PS_r(s_0) + β·PS_r(s_1)                  ← own-region producer surplus
    − κ · k_innov_r − φ · k_enter_r            ← own-region resource cost (full κ)
    + τ_r · k_innov_r                          ← transfer received by firms
    − τ_r · k_innov_r                          ← transfer paid by taxpayers
```

The last two cancel inside region r. The social resource cost remains
the full κ even when the firm only pays (κ − τ); the subsidy reduces
the *private* burden, not the *real* labour and materials used to
produce a new-technology blueprint. With this accounting, identity

```
Σ_r W_r = CS_total + Σ_r PS_r − Σ_r (κ·k_innov_r + φ·k_enter_r)
```

holds exactly in baseline and in every grid scenario. The runner
verifies it numerically and finds residuals at machine precision
(`+8e-15` in baseline).

This is the right accounting if one believes that the relevant unit of
analysis is a single sovereign region whose government balances its own
budget. It is *not* the right accounting for a federal system where a
central authority uses general taxation to fund region-3's subsidy out
of every region's taxpayers' money: in that setting one would charge
each region τ·k_innov,3 / R as a federal tax.

## Sample design

Each market draws its initial state via `random_s0(rng, p)`, the same
DGP used in `simulate_data.jl` and consumed by `estimate.jl`. The
counterfactual is therefore evaluated on the same population the model
was estimated on. Averaging over the random-`s_0` distribution rather
than fixing a single representative state lets the counterfactual answer
"what does the subsidy do *on average* across the markets in our
sample?" rather than "at one stylised initial state."

## Monte Carlo with common random numbers

We average `K = 5000` independent simulations per scenario. The
baseline and every grid point use the same `seed = 20260424` and
per-market `MersenneTwister(seed + k)`. Same seed implies identical
`s_0` draws *and* identical EVT1 shock paths in the sequential-move
stages. Under common random numbers, every market contributes a paired
observation `(W_base, W_τ)` driven by the same primitives; only the
parameters differ. The variance of the difference `W_τ − W_base` is
therefore much smaller than the variance of either level.

K-stability at the largest grid point (τ/κ̂ = 0.5):

| K     | ΔW₁     | ΔW₂     | ΔW₃     | ΔW₁ %    | ΔW₂ %    | ΔW₃ %    | ΔΣW %    |
|-------|---------|---------|---------|----------|----------|----------|----------|
| 500   | −0.0024 | +0.0008 | +0.0020 | −0.12 %  | +0.03 %  | +0.09 %  | +0.01 %  |
| 1000  | −0.0020 | −0.0008 | +0.0018 | −0.09 %  | −0.04 %  | +0.08 %  | −0.02 %  |
| 5000  | −0.0017 | −0.0018 | +0.0013 | −0.08 %  | −0.09 %  | +0.06 %  | −0.04 %  |

Headline numbers stabilise to roughly 0.05 percentage points by `K =
1000`. We report `K = 5000` for the headline.

## Calibration

We use the estimated parameters from `output/estimates/estimation.txt`:

```
κ̂ = 0.3002      φ̂ = 0.1777      γ̂ = (0.15, 0.15, 0.15).
```

The remaining parameters are held at `default_params()`:
`A = 3`, `B = 1`, `M = 1`, `c_o = 1.5`, `c_{n0} = 0.5`,
`β = 0.9`, `σ = 0.5`, `ρ = 0.5`, `N_max = 6`. The grid is
`τ/κ̂ ∈ {0, 0.10, 0.20, 0.30, 0.40, 0.50}`; the largest point cuts the
private cost of innovation in half.

## Results

Innovation rates (period-1 old → both, pooled across markets):

| τ / κ̂ | τ      | P_innov,1 | P_innov,2 | P_innov,3 |
|-------|--------|-----------|-----------|-----------|
| 0.00  | 0.0000 | 0.3507    | 0.3495    | 0.3533    |
| 0.10  | 0.0300 | 0.3507    | 0.3495    | 0.3630    |
| 0.20  | 0.0600 | 0.3484    | 0.3495    | 0.3776    |
| 0.30  | 0.0901 | 0.3484    | 0.3487    | 0.3886    |
| 0.40  | 0.1201 | 0.3480    | 0.3482    | 0.3996    |
| 0.50  | 0.1501 | 0.3475    | 0.3482    | 0.4195    |

Region 3's P(innov) climbs steadily from `0.3533` at τ = 0 to `0.4195`
at the largest grid point, a `+18.73 %` lift. Regions 1 and 2 dip
slightly; this reflects competitive feedback through the global Cournot
stage and through the period-2 state distribution (more region-3
innovators today depresses the new-tech price tomorrow, which makes
period-1 innovation in regions 1 and 2 a slightly worse bet).

Welfare deltas vs. baseline:

| τ / κ̂ | ΔW₁     | ΔW₂     | ΔW₃     | ΔΣW     | ΔΣW %  |
|-------|---------|---------|---------|---------|--------|
| 0.00  | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.00 %|
| 0.10  | −0.0003 | −0.0004 | +0.0001 | −0.0006 | −0.01 %|
| 0.20  | −0.0007 | −0.0009 | +0.0007 | −0.0008 | −0.01 %|
| 0.30  | −0.0009 | −0.0011 | +0.0007 | −0.0013 | −0.02 %|
| 0.40  | −0.0010 | −0.0015 | +0.0009 | −0.0016 | −0.03 %|
| 0.50  | −0.0017 | −0.0018 | +0.0013 | −0.0023 | −0.04 %|

Aggregate welfare is **weakly decreasing in τ across the entire grid**;
the welfare-maximising grid point is τ = 0. Region 3's own welfare
gains are an order of magnitude smaller than its innovation-rate gain,
because the subsidy mostly redistributes within region 3 (taxpayers →
firms) and only a small slice of the innovation gain reaches consumers
or augments long-run profits net of full κ. Regions 1 and 2 lose a
small but consistent amount: cheaper region-3 production drags the
new-technology price down globally, so non-treated firms see lower
profits, and their CS gain from cheaper goods is too small to compensate.

The mechanism is the textbook one but in reverse. A subsidy is welfare-
improving only if there is a positive externality from the subsidised
margin that the firm fails to internalise. In our calibration the
agglomeration externality `γ̂ = 0.15` is *local* — region 3's innovation
helps only region 3's own future cost. So the externality the firm
fails to internalise (its effect on its *own* future cost via state
transitions) is small relative to the subsidy's distortion of its
period-1 cost. Cournot competition across regions, by contrast, makes
region-3 expansion a *negative* externality on regions 1 and 2, which
the planner does internalise. Both forces push the optimum toward
τ = 0.

## Caveats

The sequential-move solver introduces a small first-mover wedge across
regions (≤ `1e-3` in CCPs at a fixed state), which is below MC noise
and is documented in `notes/solver_walkthrough.md` §3.

Equal-population CS allocation rests on an assumption we have not built
into the model. A heterogeneous-population version would scale `CS / R`
by population shares.

The local-spillover specification `c_{n,r} = c_{n0} − γ_r·(n_b[r] +
n_n[r])` is what makes the optimum τ = 0. Under the alliance spec
(`notes/merger_counterfactual.md`) where a region-3 subsidy *also*
lowered partner regions' costs, the planner would internalise more of
the benefit and the optimum would shift right.

EVT1 shocks are a smoothing device. The realised-shock contribution to
firm welfare, `σ · γ_em` per decision, is omitted from `PS_r` for the
same reasons documented in `notes/merger_counterfactual.md`; the
constant cancels in every Δ we report.

## Regenerating

```bash
cd code
julia --project=. scripts/run_subsidy.jl
```

The runner overwrites:

- `output/tables/subsidy_results.tex`
- `output/figures/subsidy_innovation.pdf`
- `output/figures/subsidy_grid.pdf`
- `output/estimates/subsidy_estimates.txt`

The writeup `\input`s the macro file directly. To change `K`, the seed,
or the grid, edit the constants at the top of `scripts/run_subsidy.jl`.
Setting `subsidy = (τ, τ, τ)` would run a uniform innovation subsidy as
a comparator.
