# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Extension of Igami (2017) "Making Oligopoly" (HDD industry dynamic oligopoly) for 14.273 (MIT). Adds **agglomeration spillovers** to a 2-period dynamic oligopoly with two technology generations (old / new) and four firm types (`old`, `both`, `new`, `pe`). Two branches share the same module layout but differ in state space and solver:

- `main` — non-spatial baseline. Scalar state `(n_o, n_b, n_n, n_pe)`, single global new-tech cost `c_n,t = c_n0 − γ·(n_b + n_n)`.
- `regional-agglomeration` (and downstream branches like `writeup_edits`, `merger-counterfactual`) — `R = 3` regions, each firm count is an `NTuple{3,Int}`, **global** Cournot competition. The new-tech cost is bloc-pooled: `c_n,r = c_n0 − γ_r · Σ_{r' ∈ bloc(r)} (n_b[r'] + n_n[r'])`, where `bloc(r) = {r' : p.blocs[r'] == p.blocs[r]}`. The default `Params.blocs == (1, 2, 3)` puts every region in its own pool and recovers the purely local spec. Setting equal bloc ids (e.g. `blocs == (1, 1, 2)`) merges those regions into a shared spillover pool — the mechanism behind the alliance counterfactual on `merger-counterfactual`.

Calibration is illustrative throughout (`default_params()` in `code/src/parameters.jl`); replacing it with Igami's Table 3 estimates is on the to-do list.

## Common commands

Always activate the project from `code/`:

```bash
cd code

# Baseline (non-spatial) — only on main branch
julia --project=. scripts/run_2period.jl

# Regional extension + counterfactuals (uniform γ baseline, region-1 subsidy, clustered s₀)
julia --project=. scripts/run_regional.jl

# Estimation pipeline (regional branch)
julia --project=. scripts/simulate_data.jl   # writes data/simulated_data.csv (500 markets)
julia --project=. scripts/estimate.jl        # two-step estimator + writes macros & table
```

Compile the writeup (after re-running scripts that updated `output/`):

```bash
cd writeup && latexmk -pdf progress_agglomeration.tex
```

There is no test suite. The "tests" are the sanity checks printed by `run_regional.jl` (single-region collapse, near-symmetry under uniform γ, monotonicity in γ₁) and the smoke check in `simulate_data.jl` that compares empirical innovation rates to solver CCPs.

## Architecture you should know before editing the solver

### Sequential moves are the central trick

Within each period, decisions are ordered **stage** (old → both → new → pe) and within each stage **region** (r=1 → 2 → 3). Each firm's problem is single-agent given the realized history, so every sub-stage reduces to a **scalar fixed point** over its own region's mixing probability — not a joint FP across regions. Region 1 commits before observing anyone; region 3 best-responds to the realized prefix. This produces a small (~10⁻³) first-mover asymmetry in baseline CCPs that is **not a bug**.

### `solver.jl` is bottom-up

The file is intentionally ordered so Julia parses dependencies linearly: PE stage at the bottom of the recursion, OLD at the top. Each stage exposes a pair:

- `solve_<stage>_region(s, r, ...)` — scalar (or 2-d for OLD) FP returning the CCP for region `r`.
- `ev_after_<stage>_region(s, r, ...)` — expected continuation **after** regions `r..R` of the current stage have played, used as the continuation in earlier sub-stages' FP.

The base case is `ev_after_pe_region(s, r > R, ...) = V1[s]`, where `V1` is the pre-computed period-1 Cournot terminal value table.

### Sub-state semantics — easy to get wrong

The same `State` struct represents states at every sub-stage. Mid-stage, some counts represent "locked-in from earlier sub-stages" and others "about to play." See `notes/solver_walkthrough.md` §5 for the full rules. The two non-obvious ones:

- **At BOTH stage region `r`**, the *deciders* are `ctx.s_orig.n_b[r]`, **not** `s.n_b[r]`. Current `s.n_b[r]` includes locked-in innovators from the OLD stage who don't choose again at BOTH.
- **`solve_<stage>_region` uses `n − 1` peers** (own firm is one of the `n` deciders), but **`ev_after_<stage>_region` uses `n` peers** (the observer is not among the deciders).

### Cache scope — critical for correctness

`SolveCaches` bundles eight `Dict`s. Two are **global** (PE: `pe_ccp`, `ev_pe`); the other six are **per-`s_orig`** and rebuilt inside `solve_state`. The PE problem depends only on `(s, r, V1, p)` — no flow-profit term — so its caches are safe to share across initial states. The other stages' continuations all use `ctx.flow_*` from `s_orig`'s Cournot outcome; mixing those caches across initial states would silently return wrong answers.

### Marginal CCPs come from a forward traversal

Under sequential moves, the CCP at `(stage, r)` is path-dependent. The single number per region in `StateCCPs` is the **expectation** over the equilibrium distribution of histories, computed by `traverse_old! → traverse_both! → traverse_new! → traverse_pe!` walking the sub-stage tree from `s_orig`. The traversal hits the same caches as backward induction, so it's effectively free.

### Cournot block (`cournot.jl`)

Two markets (old-gen, new-gen) linked by ρ ∈ [0, 1). "Both"-type firms internalize cannibalization — their FOC has `(n_b[r]+1)` on the *cross*-market own-slot, not just the own-market own-slot. Corner handling drops slots whose interior `q*` ≤ 0 and re-solves; the loop terminates in ≤ `4R+2` iterations. Full derivation: `notes/cournot_derivation.md`.

### Estimation (`estimate.jl`)

Two-step:
1. **NLS** on Cournot quantities → `γ̂`. Quantities are deterministic given state and γ, so noiseless data pins γ exactly (SSR ≈ 1e-13).
2. **MLE** on period-1 actions → `(κ̂, φ̂) | γ̂`. Crucial detail: `solve_state` returns *marginal* CCPs, but the simulator draws each action at the realized intermediate sub-state. The likelihood **replays the stage order** (`simulate.jl:99–172`) and evaluates each firm's contribution at the sub-state it actually faced. Evaluating marginal CCPs at `s₀` would not match the DGP. BHHH standard errors at the optimum.

## Outputs and writeup workflow

Standard for this user: scripts write `\newcommand` macros to `output/estimates/*.txt` and bare `tabular` environments to `output/tables/*.tex`; figures go to `output/figures/*.pdf`. The writeup at `writeup/progress_agglomeration.tex` `\input`s these directly — never inline numbers in the .tex by hand.

`data/raw/` and `data/processed/` are gitignored. `data/simulated_data.csv` (regenerated by `scripts/simulate_data.jl`, seed `20260414`) is the simulated dataset consumed by estimation.

## Documentation map

When working on a piece of the code, read its companion note first:

- `notes/cournot_derivation.md` — every term in `cournot_profits_regional`, including why "both"-type FOCs have two `+1`'s.
- `notes/solver_walkthrough.md` — full tour of `solver.jl` (sub-state semantics, cache scope, traversal).
- `notes/estimation.md` — identification argument and stage-replay likelihood.
- `notes/simulated_data.md` — DGP and CSV schema for `data/simulated_data.csv`.
- `notes/REVIEW_regional.md` — design notes for the sequential-region rewrite vs. the earlier joint-FP version.
- `README.md` — model summary, key parameters, key results.
