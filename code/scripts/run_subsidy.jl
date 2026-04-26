"""
run_subsidy.jl — Region-1 innovation-subsidy counterfactual (grid sweep).

Compares the baseline (no subsidy) against a grid of region-1 innovation
subsidies of size τ = frac · κ̂. Firms innovating in region 1 face private
cost (κ̂ − τ); the full κ̂ remains the social resource cost of innovation.
Each region is treated as a sovereign country: its own government funds
its own subsidy from its own taxpayers, so the τ_r·k_innov_r transfer
cancels exactly inside region r's welfare and never crosses borders.

Welfare and innovation rates are evaluated by Monte Carlo over `random_s0`
markets, using the same DGP as `simulate_data.jl`. Common random numbers
(same `seed`, same per-market `MersenneTwister(seed + k)`) are used across
the baseline and every grid point so the welfare *differences* are
essentially noise-free.

Calibration is the **estimated** parameter vector from
`output/estimates/estimation.txt`. The grid spans τ/κ̂ ∈ {0, 0.1, …, 0.5}.

Outputs:
  - output/tables/subsidy_results.tex        (booktabs table by τ)
  - output/figures/subsidy_innovation.pdf    (P(innov | old, r) vs τ)
  - output/figures/subsidy_grid.pdf          (ΔΣW and ΔW_r vs τ)
  - output/estimates/subsidy_estimates.txt   (LaTeX macros for the writeup)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "../src/MiniProject.jl"))
using .MiniProject
using Plots, Printf, Random

# ── Output paths ────────────────────────────────────────────────────────────
const OUTPUT_DIR = joinpath(@__DIR__, "../../output")
const OUT_TAB    = joinpath(OUTPUT_DIR, "tables")
const OUT_FIG    = joinpath(OUTPUT_DIR, "figures")
const OUT_EST    = joinpath(OUTPUT_DIR, "estimates")
mkpath(OUT_TAB); mkpath(OUT_FIG); mkpath(OUT_EST)

# ── Calibration: estimated parameters ───────────────────────────────────────
const EST_PATH = joinpath(OUTPUT_DIR, "estimates", "estimation.txt")

"""
    read_macro(path, name) → Float64

Parse a `\\newcommand{\\<name>}{<value>}` line from a LaTeX macros file.
"""
function read_macro(path::String, name::String)
    pat = Regex("\\\\newcommand\\{\\\\$name\\}\\{([^}]+)\\}")
    for line in eachline(path)
        m = match(pat, line)
        m !== nothing && return parse(Float64, m.captures[1])
    end
    error("Macro \\$name not found in $path")
end

const KAPPA_HAT = read_macro(EST_PATH, "InnovCostHat")
const PHI_HAT   = read_macro(EST_PATH, "EntryCostHat")
const GAMMA_HAT = (read_macro(EST_PATH, "SpilloverOneHat"),
                   read_macro(EST_PATH, "SpilloverTwoHat"),
                   read_macro(EST_PATH, "SpilloverThreeHat"))

# ── Subsidy grid: τ = frac · κ̂ ──────────────────────────────────────────────
const SUBSIDY_GRID_FRAC = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

# ── MC configuration ────────────────────────────────────────────────────────
const N_MARKETS = 5000
const SEED      = 20260424

# ── Baseline parameters ─────────────────────────────────────────────────────
p_base = default_params(; gamma = GAMMA_HAT, kappa = KAPPA_HAT, phi = PHI_HAT,
                          subsidy = (0.0, 0.0, 0.0))

@printf("=== Region-1 innovation subsidy: grid sweep ===\n")
@printf("  Calibration: κ̂ = %.4f,  φ̂ = %.4f,  γ̂ = (%.4f, %.4f, %.4f)\n",
        KAPPA_HAT, PHI_HAT, GAMMA_HAT...)
@printf("  Grid: τ/κ̂ ∈ %s\n", SUBSIDY_GRID_FRAC)
@printf("  Markets per scenario: K = %d   |   seed = %d   (CRN)\n",
        N_MARKETS, SEED)

# ── Run baseline once ───────────────────────────────────────────────────────
println("\nBaseline (subsidy = (0, 0, 0))…")
@time w_base = expected_welfare_mc(p_base; n_markets = N_MARKETS, seed = SEED)

# Sanity: Σ_r W_r = CS_total + Σ PS_r − Σ costs_r (transfers cancel by design)
function check_sum_identity(w, label, beta)
    cs_disc = w.cs_p1 + beta * w.cs_p2
    rhs = cs_disc + sum(w.ps_by_region) - sum(w.costs_by_region)
    err = w.total_welfare - rhs
    @printf("  Σ_r W_r identity (%-8s): Σ = %.6f   direct = %.6f   diff = %+.2e\n",
            label, w.total_welfare, rhs, err)
end
println("\n=== Sanity (transfers cancel under sovereign-funding accounting) ===")
check_sum_identity(w_base, "baseline", p_base.beta)

# ── Loop over grid ──────────────────────────────────────────────────────────
println("\n=== Grid sweep over τ/κ̂ (CRN) ===")
@printf("  %6s %8s | %8s %8s %8s | %10s %10s %10s | %10s %10s\n",
        "τ/κ̂", "τ", "P_innov₁", "P_innov₂", "P_innov₃",
        "ΔW₁", "ΔW₂", "ΔW₃", "ΔΣW", "ΔΣW %")

w_grid           = Vector{Any}(undef, length(SUBSIDY_GRID_FRAC))
grid_innov_r1    = Float64[]; grid_innov_r2 = Float64[]; grid_innov_r3 = Float64[]
grid_dW1         = Float64[]; grid_dW2 = Float64[]; grid_dW3 = Float64[]
grid_dWtot       = Float64[]; grid_dWtot_pct = Float64[]
grid_outlay      = Float64[]

for (i, frac) in enumerate(SUBSIDY_GRID_FRAC)
    τ_g = frac * KAPPA_HAT
    p_g = default_params(; gamma = GAMMA_HAT, kappa = KAPPA_HAT, phi = PHI_HAT,
                           subsidy = (τ_g, 0.0, 0.0))
    w_g = expected_welfare_mc(p_g; n_markets = N_MARKETS, seed = SEED)
    w_grid[i] = w_g
    dW    = ntuple(r -> w_g.welfare_by_region[r] - w_base.welfare_by_region[r], R)
    dWtot = w_g.total_welfare - w_base.total_welfare
    pctTot = 100.0 * dWtot / w_base.total_welfare
    push!(grid_innov_r1, w_g.innov_rate_by_region[1])
    push!(grid_innov_r2, w_g.innov_rate_by_region[2])
    push!(grid_innov_r3, w_g.innov_rate_by_region[3])
    push!(grid_dW1, dW[1]); push!(grid_dW2, dW[2]); push!(grid_dW3, dW[3])
    push!(grid_dWtot, dWtot); push!(grid_dWtot_pct, pctTot)
    push!(grid_outlay, w_g.gov_outlay_total)
    @printf("  %6.2f %8.4f | %8.4f %8.4f %8.4f | %+10.4f %+10.4f %+10.4f | %+10.4f %+9.2f%%\n",
            frac, τ_g,
            w_g.innov_rate_by_region...,
            dW..., dWtot, pctTot)
end

i_best = argmax(grid_dWtot)
@printf("\n  Welfare-maximising grid point: τ/κ̂ = %.2f  (τ = %.4f)   ΔΣW = %+.4f   (%+.2f%%)\n",
        SUBSIDY_GRID_FRAC[i_best],
        SUBSIDY_GRID_FRAC[i_best] * KAPPA_HAT,
        grid_dWtot[i_best],
        100.0 * grid_dWtot[i_best] / w_base.total_welfare)

# ── K-stability at the largest τ (representative non-zero point) ────────────
i_max = lastindex(SUBSIDY_GRID_FRAC)
τ_max = SUBSIDY_GRID_FRAC[i_max] * KAPPA_HAT
p_max = default_params(; gamma = GAMMA_HAT, kappa = KAPPA_HAT, phi = PHI_HAT,
                         subsidy = (τ_max, 0.0, 0.0))

println("\n=== K-stability at τ/κ̂ = $(SUBSIDY_GRID_FRAC[i_max]) (CRN) ===")
@printf("  %5s | %10s %10s %10s | %10s %10s %10s | %10s\n",
        "K", "ΔW₁", "ΔW₂", "ΔW₃", "ΔW₁ %", "ΔW₂ %", "ΔW₃ %", "ΔΣW %")
for k_test in (500, 1000, 5000)
    wb = expected_welfare_mc(p_base; n_markets = k_test, seed = SEED)
    ws = expected_welfare_mc(p_max;  n_markets = k_test, seed = SEED)
    Δw = ntuple(r -> ws.welfare_by_region[r] - wb.welfare_by_region[r], R)
    pw = ntuple(r -> 100.0 * Δw[r] / wb.welfare_by_region[r], R)
    Δs = ws.total_welfare - wb.total_welfare
    ps = 100.0 * Δs / wb.total_welfare
    @printf("  %5d | %+10.4f %+10.4f %+10.4f | %+9.2f%% %+9.2f%% %+9.2f%% | %+9.2f%%\n",
            k_test, Δw..., pw..., ps)
end

# ── Plot 1: P(innov | old, r) vs τ ──────────────────────────────────────────
const COL_R1 = colorant"#009E73"   # treated region — green
const COL_R2 = colorant"#0072B2"   # blue
const COL_R3 = colorant"#D55E00"   # vermilion

plt_innov = plot(SUBSIDY_GRID_FRAC, grid_innov_r1;
                 lw = 2.2, marker = :utriangle, ms = 5, color = COL_R1,
                 label = "Region 1 (treated)",
                 xlabel = "Subsidy fraction τ / κ̂",
                 ylabel = "P(innovate | old, r)",
                 title  = "Innovation rate by region across the subsidy grid",
                 legend = :outerbottom, legend_columns = 3,
                 foreground_color_legend = nothing,
                 background_color_legend = nothing,
                 framestyle = :semi, grid = :y, gridalpha = 0.25,
                 size = (720, 460),
                 titlefontsize = 12, guidefontsize = 10,
                 tickfontsize = 9, legendfontsize = 9,
                 left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
                 top_margin = 3Plots.mm)
plot!(plt_innov, SUBSIDY_GRID_FRAC, grid_innov_r2;
      lw = 2.2, marker = :diamond, ms = 5, color = COL_R2, label = "Region 2")
plot!(plt_innov, SUBSIDY_GRID_FRAC, grid_innov_r3;
      lw = 2.2, marker = :rect, ms = 5, color = COL_R3, label = "Region 3")
fig_innov_path = joinpath(OUT_FIG, "subsidy_innovation.pdf")
savefig(plt_innov, fig_innov_path)
println("\nSaved figure: $fig_innov_path")

# ── Plot 2: ΔΣW and ΔW_r vs τ ──────────────────────────────────────────────
plt_grid = plot(SUBSIDY_GRID_FRAC, grid_dWtot;
                lw = 2.5, marker = :circle, ms = 5,
                color = colorant"#000000",
                label = "ΔΣW (total)",
                xlabel = "Subsidy fraction τ / κ̂",
                ylabel = "Δ welfare (vs baseline)",
                title  = "Welfare effect of the region-1 subsidy",
                legend = :outerbottom, legend_columns = 4,
                foreground_color_legend = nothing,
                background_color_legend = nothing,
                framestyle = :semi, grid = :y, gridalpha = 0.25,
                size = (720, 460),
                titlefontsize = 12, guidefontsize = 10,
                tickfontsize = 9, legendfontsize = 9,
                left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
                top_margin = 3Plots.mm)
plot!(plt_grid, SUBSIDY_GRID_FRAC, grid_dW1; lw = 1.8, marker = :utriangle,
      ms = 4, color = COL_R1, label = "ΔW₁ (treated)")
plot!(plt_grid, SUBSIDY_GRID_FRAC, grid_dW2; lw = 1.8, marker = :diamond,
      ms = 4, color = COL_R2, label = "ΔW₂")
plot!(plt_grid, SUBSIDY_GRID_FRAC, grid_dW3; lw = 1.8, marker = :rect,
      ms = 4, color = COL_R3, label = "ΔW₃")
hline!(plt_grid, [0.0]; color = :gray, ls = :dash, lw = 1, label = "")
fig_grid_path = joinpath(OUT_FIG, "subsidy_grid.pdf")
savefig(plt_grid, fig_grid_path)
println("Saved figure: $fig_grid_path")

# ── Results table (booktabs, by τ) ──────────────────────────────────────────
fmt(x) = @sprintf("%.4f", x)
fmt2(x) = @sprintf("%.2f", x)
sgn(x)  = x ≥ 0 ? @sprintf("%+.4f", x) : @sprintf("%.4f", x)
sgn2(x) = x ≥ 0 ? @sprintf("%+.2f", x) : @sprintf("%.2f", x)

global tex
tex = """\\begin{tabular}{cccccccccc}
\\toprule
\$\\tau/\\widehat{\\kappa}\$ & \$\\tau\$ &
\$P_{\\text{innov},1}\$ & \$P_{\\text{innov},2}\$ & \$P_{\\text{innov},3}\$ &
\$\\Delta W_1\$ & \$\\Delta W_2\$ & \$\\Delta W_3\$ &
\$\\Delta \\Sigma W\$ & \$\\Delta \\Sigma W\\,(\\%)\$ \\\\
\\midrule
"""
for (i, frac) in enumerate(SUBSIDY_GRID_FRAC)
    global tex
    τ_g = frac * KAPPA_HAT
    tex *= @sprintf("%.2f & %.4f & %.4f & %.4f & %.4f & %s & %s & %s & %s & %s \\\\\n",
                    frac, τ_g,
                    grid_innov_r1[i], grid_innov_r2[i], grid_innov_r3[i],
                    sgn(grid_dW1[i]), sgn(grid_dW2[i]), sgn(grid_dW3[i]),
                    sgn(grid_dWtot[i]), sgn2(grid_dWtot_pct[i]) * "\\%")
end
tex *= "\\bottomrule\n\\end{tabular}\n"

table_path = joinpath(OUT_TAB, "subsidy_results.tex")
open(table_path, "w") do io; write(io, tex); end
println("Saved table: $table_path")

# ── LaTeX macros (config + headline τ_max + grid headline numbers) ─────────
i_h    = lastindex(SUBSIDY_GRID_FRAC)         # headline = largest grid point
frac_h = SUBSIDY_GRID_FRAC[i_h]
τ_h    = frac_h * KAPPA_HAT
w_h    = w_grid[i_h]

macros = """% Auto-generated by code/scripts/run_subsidy.jl
% Region-1 innovation-subsidy grid sweep (sovereign funding accounting).
% Grid τ/κ̂ ∈ $(SUBSIDY_GRID_FRAC); K = $N_MARKETS markets per τ; CRN seed = $SEED.
\\newcommand{\\SubsidyNMarkets}{$N_MARKETS}
\\newcommand{\\SubsidySeed}{$SEED}
\\newcommand{\\SubsidyKappaHat}{$(fmt(KAPPA_HAT))}
\\newcommand{\\SubsidyPhiHat}{$(fmt(PHI_HAT))}
\\newcommand{\\SubsidyGammaHat}{$(fmt(GAMMA_HAT[1]))}
\\newcommand{\\SubsidyGridLength}{$(length(SUBSIDY_GRID_FRAC))}
\\newcommand{\\SubsidyMaxFrac}{$(fmt2(frac_h))}
\\newcommand{\\SubsidyMaxTau}{$(fmt(τ_h))}
% Baseline (τ = 0) per-region innovation rate
\\newcommand{\\SubsidyBaseInnovROne}{$(fmt(w_base.innov_rate_by_region[1]))}
\\newcommand{\\SubsidyBaseInnovRTwo}{$(fmt(w_base.innov_rate_by_region[2]))}
\\newcommand{\\SubsidyBaseInnovRThree}{$(fmt(w_base.innov_rate_by_region[3]))}
% Headline τ = τ_max per-region innovation rate
\\newcommand{\\SubsidyMaxInnovROne}{$(fmt(w_h.innov_rate_by_region[1]))}
\\newcommand{\\SubsidyMaxInnovRTwo}{$(fmt(w_h.innov_rate_by_region[2]))}
\\newcommand{\\SubsidyMaxInnovRThree}{$(fmt(w_h.innov_rate_by_region[3]))}
\\newcommand{\\SubsidyMaxInnovPctROne}{$(sgn2(100.0 * (w_h.innov_rate_by_region[1] - w_base.innov_rate_by_region[1]) / w_base.innov_rate_by_region[1]))}
\\newcommand{\\SubsidyMaxInnovPctRTwo}{$(sgn2(100.0 * (w_h.innov_rate_by_region[2] - w_base.innov_rate_by_region[2]) / w_base.innov_rate_by_region[2]))}
\\newcommand{\\SubsidyMaxInnovPctRThree}{$(sgn2(100.0 * (w_h.innov_rate_by_region[3] - w_base.innov_rate_by_region[3]) / w_base.innov_rate_by_region[3]))}
% Headline τ welfare deltas (absolute and percent)
\\newcommand{\\SubsidyMaxDWROne}{$(sgn(grid_dW1[i_h]))}
\\newcommand{\\SubsidyMaxDWRTwo}{$(sgn(grid_dW2[i_h]))}
\\newcommand{\\SubsidyMaxDWRThree}{$(sgn(grid_dW3[i_h]))}
\\newcommand{\\SubsidyMaxDWTotal}{$(sgn(grid_dWtot[i_h]))}
\\newcommand{\\SubsidyMaxDWPctTotal}{$(sgn2(grid_dWtot_pct[i_h]))}
\\newcommand{\\SubsidyMaxDWPctROne}{$(sgn2(100.0 * grid_dW1[i_h] / w_base.welfare_by_region[1]))}
\\newcommand{\\SubsidyMaxDWPctRTwo}{$(sgn2(100.0 * grid_dW2[i_h] / w_base.welfare_by_region[2]))}
\\newcommand{\\SubsidyMaxDWPctRThree}{$(sgn2(100.0 * grid_dW3[i_h] / w_base.welfare_by_region[3]))}
% Government outlay at headline τ (per market)
\\newcommand{\\SubsidyMaxGovOutlay}{$(fmt(w_h.gov_outlay_total))}
% Welfare-maximising grid point
\\newcommand{\\SubsidyOptimumFrac}{$(fmt2(SUBSIDY_GRID_FRAC[i_best]))}
\\newcommand{\\SubsidyOptimumDWTotal}{$(sgn(grid_dWtot[i_best]))}
\\newcommand{\\SubsidyOptimumDWPct}{$(sgn2(grid_dWtot_pct[i_best]))}
"""
macro_path = joinpath(OUT_EST, "subsidy_estimates.txt")
open(macro_path, "w") do io; write(io, macros); end
println("Saved macros: $macro_path")

println("\nDone.")
