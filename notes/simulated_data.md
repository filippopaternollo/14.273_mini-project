# Simulated firm-level dataset (`data/simulated_data.csv`)

## Purpose

A synthetic "industry dataset" generated from the 2-period regional
agglomeration model. It is meant to look like something a researcher
could plausibly scrape together for an industry — firms, regions, outputs,
profits, and realized discrete actions — so that a future estimation
exercise can try to recover the structural parameters (γ, κ, φ, …) that
were used to generate it. Those parameters are deliberately **not** stored
in the CSV.

See [`estimation.md`](estimation.md) for the two-step estimator that
consumes this dataset and recovers `(γ, κ, φ)`.

## Data-generating process

The 2-period regional model:

- `code/src/parameters.jl` — `Params`, `default_params()`
- `code/src/cournot.jl`    — regional two-market Cournot equilibrium
- `code/src/state_space.jl` — regional state, log-factorial helpers
- `code/src/solver.jl`     — sequential-move backward induction
- `code/src/simulate.jl`   — per-firm action sampling and panel assembly

Within each period, firms move sequentially by stage (old → both → new →
pe) and within each stage by region (r = 1 → 2 → 3). At every sub-stage
the simulator calls the same `solve_*_region` routines the solver uses
and draws each firm's realized action from the returned CCP. Period-1
and period-2 Cournot quantities/profits come from
`cournot_quantities_regional` / `cournot_profits_regional`.

## True parameters (ground truth, not in CSV)

`default_params()` at `code/src/parameters.jl`:

| symbol  | value | meaning                                    |
|---------|-------|--------------------------------------------|
| A       | 3.0   | demand intercept (both markets)            |
| B       | 1.0   | own-market slope                           |
| M       | 1.0   | market size                                |
| c_o     | 1.5   | old-tech marginal cost                     |
| c_n0    | 0.5   | new-tech baseline marginal cost            |
| β       | 0.9   | discount factor                            |
| κ       | 0.3   | innovation cost (old → both)               |
| φ       | 0.2   | entry cost (pe → new)                      |
| σ       | 1.0   | EVT1 shock scale                           |
| γ_r     | 0.05  | regional agglomeration (scalar, uniform)   |
| ρ       | 0.5   | cross-market substitution                  |
| N_max   | 6     | state-space firm cap                       |

## Initial state

Each market draws its own s₀ independently. For every region we sample
`n_o, n_b, n_n, n_pe` iid from `{0, 1, 2}`, reject the draw if the total
number of active firms exceeds `N_max = 6` or if the market is empty, and
otherwise accept. This spreads markets over a range of local cluster
sizes so the estimator sees choice variation across states.

## Sampling scheme

- `n_markets = 500` independent replications, each with its own random s₀.
- Fixed seed `20260414`, `MersenneTwister` RNG.
- "`market_id`" is a simulation id: the model has one global product
  market with three regional cost types. Replications share the same
  parameters and equilibrium but differ in their drawn s₀ and in the
  realized private EVT1 shocks.

## Schema

One row per (market_id, period, firm_id).

| column       | type    | description                                                   |
|--------------|---------|---------------------------------------------------------------|
| `market_id`  | Int     | 1..500 (simulation replication id)                            |
| `period`     | Int     | 1 (decision period) or 2 (terminal Cournot)                   |
| `firm_id`    | Int     | stable within a market across periods                        |
| `region`     | Int     | 1..3                                                          |
| `firm_type`  | String  | `"old"` / `"both"` / `"new"` / `"pe"` (pe only in period 1)   |
| `q_old`      | Float64 | old-gen quantity produced (0 if none)                         |
| `q_new`      | Float64 | new-gen quantity produced (0 if none)                         |
| `profit`     | Float64 | per-firm Cournot profit that period                           |
| `action`     | String  | period 1: `stay`/`innovate`/`exit`/`enter`/`stay_out`; period 2: `""` |

Notes on rows:

- Period-1 rows cover every firm on the roster (including potential
  entrants that don't enter, and incumbents that exit). These firms
  still get a period-1 row so the panel is self-contained, but no
  period-2 row.
- "Both" firms may have `q_old = 0` at some states — that's a corner
  solution where cannibalization kills old-gen for them.
- A period-1 old firm that chooses `innovate` appears in period 2
  with `firm_type = "both"` (same `firm_id`).
- A period-1 potential entrant that chooses `enter` appears in period
  2 with `firm_type = "new"` (same `firm_id`).

## Sanity checks the script prints

`code/scripts/simulate_data.jl` picks the first drawn market as a
reference, solves it via `solve_initial(s0, p)`, and compares the
empirical `P(innovate | old, r)` among that market's period-1 old firms
to the solver marginal `ccps.p_io[r]`. With only a handful of old firms
per market the per-region check is noisy; it is a smoke test, not a
precise calibration.

## Regenerating

```bash
cd mini-project/code
julia --project=. scripts/simulate_data.jl
```

Output overwrites `data/simulated_data.csv`. To change the number of
replications or the seed, edit the constants at the top of
`scripts/simulate_data.jl`.
