# Constrained social planner counterfactual

A third counterfactual alongside the EU–US alliance (`run_merger.jl`) and
the region-1 innovation subsidy (`run_subsidy.jl`). Here a constrained
social planner replaces firm-level decisions, internalising the
agglomeration spillover (and, in the full variant, also the static
markup). At the estimated calibration the **full first best lifts
aggregate welfare by +8.8 %**; almost all of that gain is the static
markup correction (+10.7 %), while the dynamic-only cut actually falls
−1.9 % below equilibrium. The split is informative on its own — see
*Interpretation* below. The planner solver is in `code/src/planner.jl`;
the runner is `code/scripts/run_planner.jl`.

## Two cuts

```
1. Dynamic-only first best — planner CCPs for innovation/entry, but
                              period-by-period production stays Cournot.
                              Isolates the spillover (dynamic) channel.
2. Full first best          — planner CCPs PLUS competitive (P = MC)
                              production in both periods. Adds the
                              static markup correction.
```

Both cuts share the same backward-induction machinery; they differ only
in the static welfare convention used to build the period-2 terminal
value table `V1_social[s]` (and to evaluate period-1 welfare in the
aggregator).

## Constrained-planner CCP

At every sub-stage decision node the planner uses the **same logit form
firms use**, but with the social marginal value substituted for the
private one:

```
π_planner = Λ((ΔW_social − κ) / σ)        (innovation, OLD stage)
π_planner = Λ((ΔW_social − φ) / σ)        (entry, PE stage)
π_planner = Λ(ΔW_social / σ)              (innovation/exit, NEW & BOTH)

ΔW_social = E[Σ_r W_r | own acts] − E[Σ_r W_r | own does NOT act]
```

The planner inherits the firm's ε shock structure (same `σ`), so this is
a *constrained* planner — it cannot perfectly select; it picks each
option with logit probability based on social value. Continuation values
come from backward induction on the same OLD → BOTH → NEW → PE sub-stage
tree as `solver.jl`, with regions sequenced 1 → 2 → 3 inside each stage.

**Key simplification.** Period-1 production is computed at `s_0`
regardless of period-1 decisions (see `simulate.jl:91-96`). Period-1
social welfare is therefore **independent of every stage decision** and
cancels in every ΔW_social. The FP machinery only needs the period-2
terminal value `V1_social[s] = CS(s) + Σ_r PS_r(s)` (or `CS(s)` alone
under `competitive_static = true`).

## Why no Monte Carlo on the inner shocks

The planner's CCPs and value function are deterministic given the
state. Backward induction on the same sub-stage tree yields
`E[Σ W_r | s_0]` in closed form (binomial / multinomial weights are
already used in `solver.jl` for the equilibrium continuation). The only
sampling we keep is at the `s_0` level, so `random_s0` calls are
CRN-matched against the existing equilibrium baseline.

For each market `k = 1..K`:

```
rng  = MersenneTwister(seed + k)        # same seeds as expected_welfare_mc
s_0  = random_s0(rng, p)
w    = planner_welfare_at(s_0, p, V1_social; competitive_static)
```

Output is the same NamedTuple shape as `expected_welfare_mc` so the run
script treats Eq / DynOnly / FullFB uniformly.

## Files

```
code/src/planner.jl              solver + traversal + driver
code/scripts/run_planner.jl      run script + outputs
```

`planner.jl` contents (mirroring `solver.jl`):

- `competitive_outcome(s, p)` — closed form P=MC quantities. The 2×2
  inverse-demand system at `P_o = c_o`, `P_n = c_n_min` pins down
  `(Q_o, Q_n)`. Old-gen output is split symmetrically across all active
  old- and both-tech firms (they share `c_o`); new-gen output goes
  entirely to the lowest-`c_n` active region.
- `social_welfare_static(s, p; competitive)` — picks Cournot or
  competitive surpluses depending on the flag.
- `compute_terminal_planner_values` — builds the `V1_social[s]` table.
- Sub-stage planner solvers (`solve_pe_region_planner`,
  `solve_new_region_planner`, `solve_both_region_planner`,
  `solve_old_region_planner`) — same FPs as the equilibrium versions
  with social value substituted everywhere. The OLD stage is 3-way
  (stay / innovate / exit); NEW and BOTH are 2-way (stay / exit); PE is
  2-way (enter / not). Unlike the equilibrium, *all* branches have
  non-zero continuations because exit doesn't zero out the rest of the
  market under social welfare.
- `W_after_*_region_planner` — analogous to `ev_after_*_region` but
  returns a single Float64 (sum over regions) instead of an `EV` triple.
- `traverse_*_planner!` — forward traversal accumulating per-region
  CCPs, expected innovation/entry counts, and period-2 CS / PS_r.
- `planner_welfare_at(s_0, ...)` and `expected_planner_welfare(p; ...)`
  — drivers; output shape matches `welfare_for_market` /
  `expected_welfare_mc`.

Cache layout mirrors `SolveCaches`: PE caches are global (the PE problem
depends only on `(s, r, V1_social, p)`); NEW/BOTH/OLD caches are
per-`s_orig` (BOTH and OLD use `s_orig.n_b[r]` / `s_orig.n_o[r]` as the
decider count).

## Sanity checks (from a recent run, K = 5000, seed = 20260424)

```
Σ_r W_r identity (equilibrium): diff = +7.99e-15
Σ_r W_r identity (dynamic FB ): diff = +2.04e-14
Σ_r W_r identity (full FB    ): diff = +2.56e-13
```

The identity `Σ_r W_r = CS_total + Σ PS_r − Σ costs_r` holds at machine
precision for all three scenarios (under full FB, `Σ PS = 0` exactly).
K-stability of the headline `ΔΣW (Full FB − Eq)` across
`K ∈ {500, 1000, 5000}` is well within 0.5 % of itself.

## Headline numbers

```
              Σ_r W_r   ΔΣW vs Eq   P_innov (avg)   P_enter (avg)
Equilibrium    6.4257                   0.351           0.572
Dynamic-only   6.3065     −1.85 %       0.301           0.555
Full FB        6.9930     +8.83 %       0.276           0.516
                                      static slice = +10.68 %
```

Per-region innovation, entry, PS, costs, and welfare are in
`output/tables/planner_results.tex`; the full set of LaTeX macros for
the writeup is in `output/estimates/planner_estimates.txt`. The grouped
bar chart of W_r across the three scenarios is at
`output/figures/planner_welfare.pdf`.

## Interpretation

The dynamic-only cut falls *below* equilibrium even though γ > 0. This
is not a bug — the welfare identity passes at machine precision and the
full first best gain is large and positive. Two forces are at play:

1. **Spillover.** Each innovator lowers `c_n[r]` by `γ` for every
   new-gen producer in its bloc — the externality the planner is
   nominally there to internalise. This pushes equilibrium innovation
   *below* the social optimum.
2. **Cournot welfare convexity (Mankiw–Whinston excess entry).** With
   constant marginal cost and a fixed innovation cost `κ`, the marginal
   social value of one more new-gen producer is *decreasing* in the
   number already producing — adding the first kills a large
   deadweight-loss triangle, the fifth barely makes a dent. The
   firm's *private* gain from innovating, by contrast, is roughly
   `π_b − π_o` per period and does not fall in the same way. So firms
   keep innovating long after the social marginal benefit has dropped
   below `κ`.

Diagnostic at the calibration: in a sparse state, e.g.
`(n_o, n_b, n_n) = (2,1,1)/(0,0,0)/(0,0,0)`, `β·ΔV1_social = 1.52 ≫ κ`
— the planner badly wants innovation. In a fuller state,
`(1,0,0)/(0,1,0)/(0,0,1)`, `β·ΔV1_social = 0.13 < κ`. Averaged over the
`random_s0` distribution, the saturation-rich states dominate, so the
planner ends up innovating less than equilibrium.

The decomposition cleanly separates the two channels:

- **Static slice (Full FB − Dyn-only): +10.68 %.** Pure markup
  correction — moving from Cournot to P = MC. Doesn't depend on the
  spillover at all.
- **Dynamic slice (Dyn-only − Eq): −1.85 %.** Net of the two forces
  above. At this calibration, business-stealing wins.

## Caveats

- **Plug-in CCP.** Using the firm's `σ` directly on social welfare
  differences ignores that ΔW_social and ΔU_private live on different
  scales, and that ΔW_social varies non-linearly with the number of
  symmetric deciders at one node (welfare is convex in `c_n`). The
  convexity correction would be at most second-order at this
  calibration.
- **Full-FB PS = 0 by construction.** Under P = MC each region's
  reported PS is zero. The "winner" of new-gen production is the
  lowest-`c_n` active region, an artefact of constant MC + no capacity
  constraint. `Σ_r W_r` is invariant to that choice.
- **Subsidies are not in scope.** The planner's reported
  `subsidy_received_by_region` and `gov_outlay_total` are zero by
  construction.
