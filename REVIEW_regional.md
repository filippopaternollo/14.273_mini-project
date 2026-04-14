# Review: `regional-agglomeration` branch

Two commits on top of `main`:
1. `ee0bb9b` — R=3 regional extension with joint Jacobi FP per stage.
2. (this commit) — swap joint FP for **sequential region moves** within each
   stage, per the regional analogue of Igami's cross-type sequencing.

## TL;DR

Extended the 2-period Igami model to **R = 3 regions** with global Cournot
competition and *region-local* agglomeration spillovers. Regions move
sequentially within each stage (r=1 → 2 → 3), so every sub-stage is a
single-agent problem with a **scalar** fixed point instead of a joint FP
over `(p₁, p₂, p₃)`. CCPs are cached by sub-state `(State, r)`, and
per-region CCPs reported in the writeup are marginalised over earlier
regions' realised play via a forward traversal.

```
Parameters:  R = 3, γ ∈ ℝ³, N_max = 6
Runtime:     ~20ms per solve_initial (post-compile)
Solver size: ~460 lines (down from ~700)
```

## What changed relative to commit 1 (regional-agglomeration baseline)

### `code/src/solver.jl` — full rewrite (~460 lines)

Each stage now has a pair `(solve_<stage>_region, ev_after_<stage>_region)`,
processed in reverse region order. The PE stage is at the bottom of the
recursion; OLD is at the top.

| Function                       | Does                                                                 |
|--------------------------------|----------------------------------------------------------------------|
| `solve_pe_region(s, r, ...)`   | Scalar FP for PE region r at sub-state (s, r)                        |
| `ev_after_pe_region(s, r, ...)`| Integrates over regions r..R of PE, recurses into r+1; caches result |
| `solve_new_region(s, r, ...)`  | Scalar FP for NEW region r                                           |
| `ev_after_new_region(s, r, ...)`| Hands off to `ev_after_pe_region(s, 1, ...)` at r > R               |
| `solve_both_region(s, r, ...)` | Scalar FP for BOTH region r                                          |
| `ev_after_both_region(s, r, ...)`| Hands off to `ev_after_new_region(s, 1, ...)` at r > R             |
| `solve_old_region(s, r, ...)`  | 2-dim FP over `(p_stay, p_innov)` for OLD region r                   |
| `ev_after_old_region(s, r, ...)`| Hands off to `ev_after_both_region(s, 1, ...)` at r > R             |

The cross-region joint FP loops are gone. Each solver call does at most a
few dozen scalar FP iterations; downstream continuations are cache lookups.

Caches, bundled into `SolveCaches`:

| Cache        | Key              | Scope            |
|--------------|------------------|------------------|
| `pe_ccp`     | `(State, r)`     | **global**       |
| `ev_pe`      | `(State, r)`     | **global**       |
| `new_ccp`    | `(State, r)`     | per `s_orig`     |
| `ev_new`     | `(State, r)`     | per `s_orig`     |
| `both_ccp`   | `(State, r)`     | per `s_orig`     |
| `ev_both`    | `(State, r)`     | per `s_orig`     |
| `old_ccp`    | `(State, r)` → `(p_s, p_i)` | per `s_orig` |
| `ev_old`     | `(State, r)`     | per `s_orig`     |

PE remains globally cached because its problem depends only on the
post-new-stage state and V1. The other three depend on flow profits at
`s_orig` and are rebuilt per call to `solve_state`.

### Forward traversal for marginal CCPs

Under sequential moves the CCP at `(stage, r)` is path-dependent on the
realised outcomes of regions 1..r-1. `solve_state` computes the
*marginalised* per-region CCP by walking the equilibrium sub-stage tree from
`s_orig` with the four `traverse_<stage>!` helpers, accumulating
`weight · p_r` at each node. This replaces the old per-`s_orig` forward
multinomial enumeration and keeps the `StateCCPs` output interface
unchanged.

### Doc updates

- `code/src/state_space.jl`: header comment rewritten to describe sequential
  region moves instead of joint FP.
- `code/src/solver.jl`: module docstring rewritten.
- `README.md`: solution section now describes the sequential sub-stage
  structure and the scalar fixed points.
- `writeup/progress_agglomeration.tex`: §5 "Solution with Regional States"
  rewritten to explain the sequential-region convention, and a paragraph
  noting the resulting first-mover asymmetry (see below).

## Verification

All three plan checks still pass in `run_regional.jl`:

1. **Cournot sanity.** `cournot_profits_regional` with all firms in region 1
   reproduces the single-region `cournot_profits` bit-for-bit (unchanged).
2. **Near-symmetry at uniform γ.** Baseline CCPs are
   `p_io = (0.3333, 0.3330, 0.3326)`. *Not* identical across regions, unlike
   the Jacobi version (which gave exactly 0.3331 for all three) — this is
   the **sequential-move first-mover asymmetry**. Region 1 commits before
   observing any other region; region 3 best-responds to realised history.
   The gap is $\le 10^{-3}$.
3. **Monotonicity in γ₁.** Subsidising region 1 (γ=(0.15,0.05,0.05)) raises
   `p_io[1]` from 0.333 → 0.359 and drags `p_io[2], p_io[3]` down to ≈ 0.326
   through the global Cournot channel. Unchanged qualitatively.

End-to-end runs:
- `julia --project=code code/scripts/run_regional.jl` — ~450ms first solve
  (compilation), ~20ms subsequent. Same as the Jacobi version.
- `cd writeup && latexmk -pdf progress_agglomeration.tex` — compiles cleanly.

## Counterfactual comparison (sequential vs. Jacobi)

| Scenario          | r1 (seq / jac)  | r2 (seq / jac)  | r3 (seq / jac)  |
|-------------------|-----------------|-----------------|-----------------|
| Baseline          | 0.3333 / 0.3331 | 0.3330 / 0.3331 | 0.3326 / 0.3331 |
| CF-A: γ₁ = 0.15   | 0.3594 / 0.3594 | 0.3263 / 0.3263 | 0.3260 / 0.3260 |
| CF-B: cluster r1  | 0.3374 / 0.3370 | 0.3236 / 0.3240 | 0.3234 / 0.3240 |

The treatment effects are identical within the resolution that matters for
the writeup (4 decimals). All the economics of the counterfactuals carries
over.

## Things worth double-checking

1. **First-mover asymmetry.** The $\le 10^{-3}$ wedge in the baseline is the
   cleanest sign that sequential moves are working (and not a bug). But it
   *is* an economic choice that r=1 moves first; if you want it symmetric,
   averaging solutions over random region orderings would be the textbook
   fix (not implemented).
2. **Marginal CCPs.** Under sequential play, region 2's and region 3's CCPs
   are functions of the realised history, so the single number reported per
   region in `StateCCPs` is the *expectation* over the equilibrium
   distribution over histories. This matches the semantics you'd want when
   comparing across counterfactuals but is worth noting when looking at the
   raw numbers.
3. **State caps.** Still `N_max = 6`. Reachable states within a period
   strictly shrink in total firms (old exits, both exits, new exits) or stay
   the same (PE entry just reclassifies firms), so as long as the initial
   state has total ≤ 6, every reachable state is in V1. `get(V1, s, ZERO_EV)`
   is a safety net that should never fire in practice.
4. **Per-region gamma plumbing.** Unchanged. `default_params` still accepts
   scalar or `NTuple{3,Float64}`.

## Not in this branch (still deliberately)

- Welfare accounting (consumer surplus, fiscal cost of subsidy).
- A T > 2 version.
- Replacing the illustrative calibration with Igami Table 3.
- Endogenous region choice at entry.
- Averaging over random region orderings to kill the first-mover wedge.
