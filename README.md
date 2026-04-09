# 14.273 Mini Project — Igami (2017) Extension with Agglomeration

Extension of Igami (2017) "Making Oligopoly: Entry and Innovation in the Hard Disk Drive Industry" to study innovation spillovers / agglomeration effects in a dynamic oligopoly.

## Current Status

**Implemented:** 2-period model with non-spatial agglomeration (linear cost spillover).

## Structure

```
mini-project/
├── code/                      Julia project (activate here)
│   ├── Project.toml
│   ├── src/
│   │   ├── Igami2017.jl       Module entry point
│   │   ├── parameters.jl      Params struct + default calibration
│   │   ├── cournot.jl         Two-market multi-product Cournot (cannibalization)
│   │   ├── state_space.jl     State, CCPs, transition distribution
│   │   └── solver.jl          Backward induction solver
│   └── scripts/
│       └── run_2period.jl     Main script (solve + comparative statics)
├── output/
│   ├── estimates/             LaTeX \newcommand macros
│   ├── tables/                Bare tabular .tex environments
│   └── figures/               PDF plots
└── writeup/
    ├── progress_agglomeration.tex   Progress note (\input output files)
    └── 14_273_Mini_project_simulation_writeup.pdf
```

## Running the Code

```bash
cd code
julia --project=. scripts/run_2period.jl
```

Outputs are written to `output/`. The writeup in `writeup/progress_agglomeration.tex` `\input`s them directly.

## Model

**State:** `(n_o, n_b, n_n, n_pe)` — old incumbents, both-tech incumbents, new entrants, potential entrants.

**Demand (two-market):** Consumers substitute between old-gen and new-gen products:
```
P_o = A - B*(Q_o/M) - ρ*B*(Q_n/M)
P_n = A - B*(Q_n/M) - ρ*B*(Q_o/M)
```
ρ ∈ [0,1) is the cross-market substitution parameter. ρ=0 → independent markets; ρ→1 → homogeneous goods.

**Cannibalization:** `n_b` firms ("both") sell in *both* markets and internalize that their new-gen production reduces demand for their own old-gen product. This is the mechanism that deters innovation by incumbents.

**Agglomeration:**
```
c_n,t = c_n0 - γ * (n_b,t + n_n,t)
```
More firms on new tech → lower marginal cost for all new-tech firms. `γ=0` recovers the baseline.

**Solution:** One-pass backward induction. Private EVT1 shocks make each firm's problem single-agent, yielding closed-form logit CCPs. A within-period fixed-point iteration handles the simultaneity among firms of the same type.

## Key Parameters (Illustrative Calibration)

| Parameter | Value | Description |
|-----------|-------|-------------|
| A=3, B=1, M=1 | | Linear demand (own-market slope) |
| c_o=1.5, c_n0=0.5 | | Old / new-tech costs |
| β=0.9, κ=0.3, φ=0.2 | | Discount, innovation cost, entry cost |
| σ=1.0 | | EVT1 scale |
| γ=0.05 | | Agglomeration (comparative statics) |
| ρ=0.5 | | Cross-market substitution |
| s₀ = (4,1,1,2) | | Initial state of interest |

## Key Results

At s₀ = (4,1,1,2), ρ=0.5:
- Baseline (γ=0): P(innovate|old) ≈ 0.313, P(enter|pe) ≈ 0.507
- Agglomeration (γ=0.05): P(innovate|old) ≈ 0.320, P(enter|pe) ≈ 0.515
- Cannibalization: pi_b(ρ=0.5) ≈ 0.327 vs pi_b(ρ=0) ≈ 0.531 (ρ reduces "both" firm profits)
- Effect saturates near γ≈0.2 due to EVT1 / logit structure

## Dependencies

Julia packages (all in `Project.toml`):
- `Distributions`, `SpecialFunctions`, `Plots`, `DataFrames`, `LaTeXStrings`

## Next Steps

1. Replace illustrative calibration with Igami (2017) structural estimates
2. Extend to T > 2 periods
3. Add spatial agglomeration (cost depends on cluster-level N_inn)
4. Welfare analysis and industrial policy experiments
