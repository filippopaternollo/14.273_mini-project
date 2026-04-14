# 14.273 Mini Project — Igami (2017) Extension with Agglomeration

Extension of Igami (2017) "Making Oligopoly: Entry and Innovation in the Hard Disk Drive Industry" to study innovation spillovers / agglomeration effects in a dynamic oligopoly.

## Current Status

**Implemented:**
- 2-period model with non-spatial agglomeration (linear cost spillover) — `main` branch.
- **Regional extension** with $R=3$ regions, global Cournot competition, and
  *local* cost spillovers — `regional-agglomeration` branch.

## Structure

```
mini-project/
├── code/                      Julia project (activate here)
│   ├── Project.toml
│   ├── src/
│   │   ├── Igami2017.jl       Module entry point
│   │   ├── parameters.jl      Params struct (regional γ) + default calibration
│   │   ├── cournot.jl         Two-market Cournot (symmetric + regional asymmetric)
│   │   ├── state_space.jl     Regional State, CCPs, probability helpers
│   │   └── solver.jl          Sequential-move backward induction solver
│   └── scripts/
│       ├── run_2period.jl     Baseline agglomeration script
│       └── run_regional.jl    Regional extension: baseline + counterfactuals
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
julia --project=. scripts/run_2period.jl   # baseline (non-spatial) agglomeration
julia --project=. scripts/run_regional.jl  # regional extension + counterfactuals
```

Outputs are written to `output/`. The writeup in `writeup/progress_agglomeration.tex` `\input`s them directly.

## Model

**State (baseline, `main` branch):** `(n_o, n_b, n_n, n_pe)` — old incumbents, both-tech incumbents, new entrants, potential entrants.

**State (regional, `regional-agglomeration` branch):** each of the four firm
counts is an `NTuple{3,Int}` indexed by region $r \in \{1,2,3\}$. Firms are
pinned to a region for life; potential entrants sit in per-region pools.

**Demand (two-market):** Consumers substitute between old-gen and new-gen products:
```
P_o = A - B*(Q_o/M) - ρ*B*(Q_n/M)
P_n = A - B*(Q_n/M) - ρ*B*(Q_o/M)
```
ρ ∈ [0,1) is the cross-market substitution parameter. ρ=0 → independent markets; ρ→1 → homogeneous goods.

**Cannibalization:** `n_b` firms ("both") sell in *both* markets and internalize that their new-gen production reduces demand for their own old-gen product. This is the mechanism that deters innovation by incumbents.

**Agglomeration:**
```
baseline: c_n,t    = c_n0 - γ   · (n_b,t   + n_n,t)        # global spillover
regional: c_n,r,t  = c_n0 - γ_r · (n_b,r,t + n_n,r,t)      # local spillover
```
More firms on new tech → lower marginal cost for all new-tech firms (in the
same region, in the regional version). `γ=0` recovers the baseline. In the
regional branch Cournot competition is still global, so `c_n,r` enters an
asymmetric FOC system with per-region marginal costs on new-gen.

**Solution:** One-pass backward induction. Private EVT1 shocks make each
firm's problem single-agent, yielding closed-form logit CCPs at each stage
(old → both → new → pe). In the regional version each stage is solved via a
joint Jacobi fixed point over the three regions, cascaded through layered
caches (`ev_after_pe`, `ev_after_new`, `ev_after_both`).

## Key Parameters (Illustrative Calibration)

| Parameter | Value | Description |
|-----------|-------|-------------|
| A=3, B=1, M=1 | | Linear demand (own-market slope) |
| c_o=1.5, c_n0=0.5 | | Old / new-tech costs |
| β=0.9, κ=0.3, φ=0.2 | | Discount, innovation cost, entry cost |
| σ=1.0 | | EVT1 scale |
| γ=0.05 | | Agglomeration (scalar or per-region NTuple) |
| ρ=0.5 | | Cross-market substitution |
| N_max | 8 baseline / 6 regional | State-space firm cap |
| s₀ baseline | (4,1,1,2) | Initial state (baseline script) |
| s₀ regional (sym.) | ((1,1,1),(1,1,1),(0,0,0),(0,0,0)) | Symmetric initial state |
| s₀ regional (clust.) | ((1,1,1),(3,0,0),(0,0,0),(0,0,0)) | Innovators clustered in region 1 |

## Key Results

**Baseline agglomeration** (s₀ = (4,1,1,2), ρ=0.5):
- γ=0: P(innovate|old) ≈ 0.313, P(enter|pe) ≈ 0.507
- γ=0.05: P(innovate|old) ≈ 0.320, P(enter|pe) ≈ 0.515
- Cannibalization: pi_b(ρ=0.5) ≈ 0.327 vs pi_b(ρ=0) ≈ 0.531 (ρ reduces "both" firm profits)
- Effect saturates near γ≈0.2 due to EVT1 / logit structure

**Regional extension** (symmetric s₀, uniform γ=0.05 unless stated):
- Baseline: P(innovate|old, r) = 0.3331 for all r (symmetry check)
- Subsidize region 1 (γ=(0.15, 0.05, 0.05)): r1 rises to 0.359, r2/r3 fall to 0.326 via global-Cournot drag
- Clustered innovators (3 in r1): r1 barely moves (0.337 — spillover saturated), r2/r3 fall to 0.323

## Dependencies

Julia packages (all in `Project.toml`):
- `Distributions`, `SpecialFunctions`, `Plots`, `DataFrames`, `LaTeXStrings`

## Next Steps

1. Replace illustrative calibration with Igami (2017) structural estimates
2. Extend to T > 2 periods
3. Richer counterfactuals on the regional branch: bilateral spillover pools,
   endogenous region choice at entry, welfare accounting (CS + industry profit)
4. Welfare analysis and industrial policy experiments
