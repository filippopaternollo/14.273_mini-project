# Region-3 entry subsidy counterfactual

A region-3 entry subsidy of size ψ shifts the private cost of entry in
region 3 from φ̂ to (φ̂ − ψ) while leaving the social resource cost
unchanged at φ̂. Each region is treated as a sovereign country: its own
government funds its own subsidy from its own taxpayers, so the
ψ·k_enter,3 transfer cancels exactly inside region 3's welfare and never
crosses borders. Sweeping ψ over the grid ψ/φ̂ ∈ {0, 0.1, …, 0.5} shows
that **aggregate welfare is weakly increasing in the entry subsidy**,
with the welfare-maximising grid point at the largest ψ (`+0.01 %` of
baseline). Region 3 raises P(enter | pe) by **+6.71 %** and gains
**+0.14 %** of its own welfare; regions 1 and 2 lose **−0.07 %** and
**−0.05 %** respectively. The estimator is in `code/src/estimate.jl`;
the counterfactual runner is `code/scripts/run_entry_subsidy.jl`.

The mechanism mirrors the innovation-subsidy counterfactual
(`notes/subsidy_counterfactual.md`) but the sign of ΔΣW flips. Because
entry brings a *new* product to the new-tech market — rather than
shifting an existing producer between technologies — it expands global
new-tech output more directly, and the consumer-surplus gain is
slightly larger than the producer-surplus losses borne by competitors.
The aggregate effect is small but positive.

## Setup

Recall the 2-period regional model. In the baseline, a potential
entrant in region r pays φ to enter as a "new"-type firm. A *region-r
entry subsidy* of size ψ_r ≥ 0 changes only the **private** entry
payoff:

```
u^enter_r = −(φ − ψ_r) + β · EV^pe→new_r,
```

leaving the innovation decision and the rest of the game unchanged.
The implementation is one field on `Params`
(`entry_subsidy::NTuple{R,Float64}`) and one line in `solver.jl`
(`phi_priv = p.phi - p.entry_subsidy[r]`); everything else propagates
automatically because the PE entry payoff reads φ through that local
variable.

We sweep a one-region subsidy `(0, 0, ψ)` rather than a global one. This
isolates the question "what does targeted entry policy in region 3 do?"
from the question "should the planet subsidise market entry at all?".
The spillover spec stays at the local-only baseline.

## Welfare under sovereign funding

Consumer surplus is global, allocated equally across regions; producer
surplus is local; innovation and entry costs are paid in the firm's
region in period 1. The entry subsidy is a transfer:
ψ_r·k_enter,r flows from region r's taxpayers to region r's entrants.
Under sovereign funding, that transfer never leaves region r:

```
W_r = (CS(s_0) + β·CS(s_1)) / R                ← global CS, equal split
    + PS_r(s_0) + β·PS_r(s_1)                  ← own-region producer surplus
    − κ · k_innov_r − φ · k_enter_r            ← own-region resource cost (full φ)
    + ψ_r · k_enter_r                          ← transfer received by entrants
    − ψ_r · k_enter_r                          ← transfer paid by taxpayers
```

The last two cancel inside region r. The social resource cost of an
entry remains the full φ even when the entrant only pays (φ − ψ); the
subsidy reduces the *private* burden, not the *real* setup cost. With
this accounting,

```
Σ_r W_r = CS_total + Σ_r PS_r − Σ_r (κ·k_innov_r + φ·k_enter_r)
```

holds exactly in baseline and in every grid scenario. The runner
verifies it numerically and finds residuals at machine precision
(`+8e-15` in baseline).

## Sample design

Each market draws its initial state via `random_s0(rng, p)`, the same
DGP used in `simulate_data.jl` and consumed by `estimate.jl`. The
counterfactual is therefore evaluated on the same population the model
was estimated on.

## Monte Carlo with common random numbers

We average `K = 5000` independent simulations per scenario, with the
same `seed = 20260424` and per-market `MersenneTwister(seed + k)`
across baseline and every grid point.

K-stability at the largest grid point (ψ/φ̂ = 0.5):

| K     | ΔW₁     | ΔW₂     | ΔW₃     | ΔW₁ %    | ΔW₂ %    | ΔW₃ %    | ΔΣW %    |
|-------|---------|---------|---------|----------|----------|----------|----------|
| 500   | −0.0037 | −0.0021 | +0.0059 | −0.18 %  | −0.10 %  | +0.28 %  | +0.00 %  |
| 1000  | −0.0025 | −0.0012 | +0.0041 | −0.12 %  | −0.06 %  | +0.19 %  | +0.01 %  |
| 5000  | −0.0016 | −0.0011 | +0.0031 | −0.07 %  | −0.05 %  | +0.14 %  | +0.01 %  |

Headline numbers stabilise to roughly 0.05 percentage points by `K =
1000`. The aggregate welfare effect is small (about a basis point);
its sign is robust across `K`.

## Calibration

We use the estimated parameters from `output/estimates/estimation.txt`:

```
κ̂ = 0.3002      φ̂ = 0.1777      γ̂ = (0.15, 0.15, 0.15).
```

The remaining parameters are held at `default_params()`. The grid is
`ψ/φ̂ ∈ {0, 0.10, 0.20, 0.30, 0.40, 0.50}`; the largest point cuts the
private cost of entry in half, from `φ̂ = 0.1777` to `0.0889`.

## Results

Entry rates (period-1 pe → new, pooled across markets):

| ψ / φ̂ | ψ      | P_enter,1 | P_enter,2 | P_enter,3 |
|-------|--------|-----------|-----------|-----------|
| 0.00  | 0.0000 | 0.5562    | 0.5801    | 0.5792    |
| 0.10  | 0.0178 | 0.5557    | 0.5801    | 0.5891    |
| 0.20  | 0.0355 | 0.5557    | 0.5801    | 0.5982    |
| 0.30  | 0.0533 | 0.5557    | 0.5796    | 0.6081    |
| 0.40  | 0.0711 | 0.5553    | 0.5796    | 0.6116    |
| 0.50  | 0.0888 | 0.5548    | 0.5796    | 0.6180    |

Region 3's P(enter) climbs steadily from `0.5792` at ψ = 0 to `0.6180`
at the largest grid point, a `+6.71 %` lift. Regions 1 and 2 dip very
slightly through the same equilibrium-feedback channel as in the
innovation-subsidy counterfactual.

Welfare deltas vs. baseline:

| ψ / φ̂ | ΔW₁     | ΔW₂     | ΔW₃     | ΔΣW     | ΔΣW %  |
|-------|---------|---------|---------|---------|--------|
| 0.00  | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.00 %|
| 0.10  | −0.0004 | −0.0004 | +0.0009 | +0.0001 | +0.00 %|
| 0.20  | −0.0007 | −0.0008 | +0.0018 | +0.0003 | +0.00 %|
| 0.30  | −0.0012 | −0.0010 | +0.0025 | +0.0003 | +0.00 %|
| 0.40  | −0.0014 | −0.0011 | +0.0028 | +0.0003 | +0.00 %|
| 0.50  | −0.0016 | −0.0011 | +0.0031 | +0.0003 | +0.01 %|

Aggregate welfare is **weakly increasing in ψ across the grid**, and
the welfare-maximising grid point is the largest one. The magnitude is
small (about `+0.01 %` of baseline at ψ/φ̂ = 0.5) but signed in the
opposite direction to the innovation subsidy.

The qualitative difference from the innovation subsidy is driven by
how each policy reshapes the new-tech market. Innovating firms (old →
both) are *already producing* old-tech output that they cannibalise as
they expand into new-tech; their marginal contribution to total
new-tech output is partially offset by their reduction in old-tech
output. New entrants (pe → new) bring a *fresh* competitor to the
new-tech market without that cannibalisation drag, so each marginal
entry adds more total new-tech output to the global market. The
consumer-surplus gain from the additional new-tech output is therefore
slightly larger relative to the producer-surplus losses, and the net
effect on Σ_r W_r is positive (though small in absolute terms).

This is consistent with the standard intuition that *entry subsidies*
correct an under-entry distortion in oligopoly more cleanly than
*technology-switching subsidies*, which have to net out the
cannibalisation that the switching firm would otherwise impose on
itself.

## Caveats

The sequential-move solver introduces a small first-mover wedge across
regions (≤ `1e-3` in CCPs at a fixed state); below MC noise. The
welfare gain from the entry subsidy is small enough that it is on the
edge of the noise floor at K = 500, but stabilises at K ≥ 1000.

Equal-population CS allocation rests on an assumption we have not
built into the model.

EVT1 shocks are a smoothing device; the realised-shock contribution to
firm welfare is omitted from `PS_r` for the same reasons documented in
`notes/merger_counterfactual.md`. The constant cancels in every Δ.

## Regenerating

```bash
cd code
julia --project=. scripts/run_entry_subsidy.jl
```

The runner overwrites:

- `output/tables/entry_subsidy_results.tex`
- `output/figures/entry_subsidy_entry.pdf`
- `output/figures/entry_subsidy_grid.pdf`
- `output/estimates/entry_subsidy_estimates.txt`

The writeup `\input`s the macro file directly. To change `K`, the seed,
or the grid, edit the constants at the top of
`scripts/run_entry_subsidy.jl`. Setting `entry_subsidy = (ψ, ψ, ψ)`
would run a uniform entry subsidy as a comparator.
