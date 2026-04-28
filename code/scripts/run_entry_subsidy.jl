"""
run_entry_subsidy.jl — Region-3 entry-subsidy counterfactual (grid sweep).

Compares the baseline (no subsidy) against a grid of region-3 entry
subsidies of size ψ = frac · φ̂. Potential entrants in region 3 face
private cost (φ̂ − ψ); the full φ̂ remains the social resource cost of
entry. Each region is treated as a sovereign country: its own government
funds its own subsidy from its own taxpayers, so the ψ_r·k_enter_r
transfer cancels exactly inside region r's welfare and never crosses
borders.

Welfare and entry rates are evaluated by Monte Carlo over `random_s0`
markets, using the same DGP as `simulate_data.jl`. Common random numbers
(same `seed`, same per-market `MersenneTwister(seed + k)`) are used across
the baseline and every grid point so the welfare *differences* are
essentially noise-free.

Calibration is the **estimated** parameter vector from
`output/estimates/estimation.txt`. The grid spans ψ/φ̂ ∈ {0, 0.1, …, 0.5}.

Outputs:
  - output/tables/entry_subsidy_results.tex      (booktabs table by ψ)
  - output/figures/entry_subsidy_entry.pdf       (P(enter | pe, r) vs ψ)
  - output/figures/entry_subsidy_grid.pdf        (ΔΣW and ΔW_r vs ψ)
  - output/estimates/entry_subsidy_estimates.txt (LaTeX macros for the writeup)
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

# ── Subsidy grid: ψ = frac · φ̂ ─────────────────────────────────────────────
const SUBSIDY_GRID_FRAC = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

# ── MC configuration ────────────────────────────────────────────────────────
const N_MARKETS = 5000
const SEED      = 20260424

# ── Baseline parameters ─────────────────────────────────────────────────────
p_base = default_params(; gamma = GAMMA_HAT, kappa = KAPPA_HAT, phi = PHI_HAT,
                          entry_subsidy = (0.0, 0.0, 0.0))

@printf("=== Region-3 entry subsidy: grid sweep ===\n")
@printf("  Calibration: κ̂ = %.4f,  φ̂ = %.4f,  γ̂ = (%.4f, %.4f, %.4f)\n",
        KAPPA_HAT, PHI_HAT, GAMMA_HAT...)
@printf("  Grid: ψ/φ̂ ∈ %s\n", SUBSIDY_GRID_FRAC)
@printf("  Markets per scenario: K = %d   |   seed = %d   (CRN)\n",
        N_MARKETS, SEED)

# ── Run baseline once ───────────────────────────────────────────────────────
println("\nBaseline (entry_subsidy = (0, 0, 0))…")
@time w_base = expected_welfare_mc(p_base; n_markets = N_MARKETS, seed = SEED)

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
println("\n=== Grid sweep over ψ/φ̂ (CRN) ===")
@printf("  %6s %8s | %8s %8s %8s | %10s %10s %10s | %10s %10s\n",
        "ψ/φ̂", "ψ", "P_enter₁", "P_enter₂", "P_enter₃",
        "ΔW₁", "ΔW₂", "ΔW₃", "ΔΣW", "ΔΣW %")

w_grid           = Vector{typeof(w_base)}(undef, length(SUBSIDY_GRID_FRAC))
grid_enter_r1    = Float64[]; grid_enter_r2 = Float64[]; grid_enter_r3 = Float64[]
grid_dW1         = Float64[]; grid_dW2 = Float64[]; grid_dW3 = Float64[]
grid_dWtot       = Float64[]; grid_dWtot_pct = Float64[]
grid_outlay      = Float64[]

for (i, frac) in enumerate(SUBSIDY_GRID_FRAC)
    ψ_g = frac * PHI_HAT
    p_g = default_params(; gamma = GAMMA_HAT, kappa = KAPPA_HAT, phi = PHI_HAT,
                           entry_subsidy = (0.0, 0.0, ψ_g))
    w_g = expected_welfare_mc(p_g; n_markets = N_MARKETS, seed = SEED)
    w_grid[i] = w_g
    dW    = ntuple(r -> w_g.welfare_by_region[r] - w_base.welfare_by_region[r], R)
    dWtot = w_g.total_welfare - w_base.total_welfare
    pctTot = 100.0 * dWtot / w_base.total_welfare
    push!(grid_enter_r1, w_g.enter_rate_by_region[1])
    push!(grid_enter_r2, w_g.enter_rate_by_region[2])
    push!(grid_enter_r3, w_g.enter_rate_by_region[3])
    push!(grid_dW1, dW[1]); push!(grid_dW2, dW[2]); push!(grid_dW3, dW[3])
    push!(grid_dWtot, dWtot); push!(grid_dWtot_pct, pctTot)
    push!(grid_outlay, w_g.gov_outlay_total)
    @printf("  %6.2f %8.4f | %8.4f %8.4f %8.4f | %+10.4f %+10.4f %+10.4f | %+10.4f %+9.2f%%\n",
            frac, ψ_g,
            w_g.enter_rate_by_region...,
            dW..., dWtot, pctTot)
end

i_best = argmax(grid_dWtot)
@printf("\n  Welfare-maximising grid point: ψ/φ̂ = %.2f  (ψ = %.4f)   ΔΣW = %+.4f   (%+.2f%%)\n",
        SUBSIDY_GRID_FRAC[i_best],
        SUBSIDY_GRID_FRAC[i_best] * PHI_HAT,
        grid_dWtot[i_best],
        100.0 * grid_dWtot[i_best] / w_base.total_welfare)

# ── K-stability at the largest ψ ───────────────────────────────────────────
i_max = lastindex(SUBSIDY_GRID_FRAC)
ψ_max = SUBSIDY_GRID_FRAC[i_max] * PHI_HAT
p_max = default_params(; gamma = GAMMA_HAT, kappa = KAPPA_HAT, phi = PHI_HAT,
                         entry_subsidy = (0.0, 0.0, ψ_max))

println("\n=== K-stability at ψ/φ̂ = $(SUBSIDY_GRID_FRAC[i_max]) (CRN) ===")
@printf("  %5s | %10s %10s %10s | %10s %10s %10s | %10s\n",
        "K", "ΔW₁", "ΔW₂", "ΔW₃", "ΔW₁ %", "ΔW₂ %", "ΔW₃ %", "ΔΣW %")
for k_test in (500, 1000, 5000)
    if k_test == N_MARKETS
        wb = w_base
        ws = w_grid[i_max]
    else
        wb = expected_welfare_mc(p_base; n_markets = k_test, seed = SEED)
        ws = expected_welfare_mc(p_max;  n_markets = k_test, seed = SEED)
    end
    Δw = ntuple(r -> ws.welfare_by_region[r] - wb.welfare_by_region[r], R)
    pw = ntuple(r -> 100.0 * Δw[r] / wb.welfare_by_region[r], R)
    Δs = ws.total_welfare - wb.total_welfare
    ps = 100.0 * Δs / wb.total_welfare
    @printf("  %5d | %+10.4f %+10.4f %+10.4f | %+9.2f%% %+9.2f%% %+9.2f%% | %+9.2f%%\n",
            k_test, Δw..., pw..., ps)
end

# ── Plot 1: P(enter | pe, r) vs ψ ──────────────────────────────────────────
const COL_R1 = colorant"#009E73"   # green
const COL_R2 = colorant"#0072B2"   # blue
const COL_R3 = colorant"#D55E00"   # vermilion (treated)

plt_enter = plot(SUBSIDY_GRID_FRAC, grid_enter_r1;
                 lw = 2.2, marker = :utriangle, ms = 5, color = COL_R1,
                 label = "Region 1",
                 xlabel = "Entry-subsidy fraction ψ / φ̂",
                 ylabel = "P(enter | pe, r)",
                 title  = "Entry rate by region across the entry-subsidy grid",
                 legend = :outerbottom, legend_columns = 3,
                 foreground_color_legend = nothing,
                 background_color_legend = nothing,
                 framestyle = :semi, grid = :y, gridalpha = 0.25,
                 size = (720, 460),
                 titlefontsize = 12, guidefontsize = 10,
                 tickfontsize = 9, legendfontsize = 9,
                 left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
                 top_margin = 3Plots.mm)
plot!(plt_enter, SUBSIDY_GRID_FRAC, grid_enter_r2;
      lw = 2.2, marker = :diamond, ms = 5, color = COL_R2, label = "Region 2")
plot!(plt_enter, SUBSIDY_GRID_FRAC, grid_enter_r3;
      lw = 2.2, marker = :rect, ms = 5, color = COL_R3, label = "Region 3 (treated)")
fig_enter_path = joinpath(OUT_FIG, "entry_subsidy_entry.pdf")
savefig(plt_enter, fig_enter_path)
println("\nSaved figure: $fig_enter_path")

# ── Plot 2: ΔΣW and ΔW_r vs ψ ─────────────────────────────────────────────
plt_grid = plot(SUBSIDY_GRID_FRAC, grid_dWtot;
                lw = 2.5, marker = :circle, ms = 5,
                color = colorant"#000000",
                label = "ΔΣW (total)",
                xlabel = "Entry-subsidy fraction ψ / φ̂",
                ylabel = "Δ welfare (vs baseline)",
                title  = "Welfare effect of the region-3 entry subsidy",
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
      ms = 4, color = COL_R1, label = "ΔW₁")
plot!(plt_grid, SUBSIDY_GRID_FRAC, grid_dW2; lw = 1.8, marker = :diamond,
      ms = 4, color = COL_R2, label = "ΔW₂")
plot!(plt_grid, SUBSIDY_GRID_FRAC, grid_dW3; lw = 1.8, marker = :rect,
      ms = 4, color = COL_R3, label = "ΔW₃ (treated)")
hline!(plt_grid, [0.0]; color = :gray, ls = :dash, lw = 1, label = "")
fig_grid_path = joinpath(OUT_FIG, "entry_subsidy_grid.pdf")
savefig(plt_grid, fig_grid_path)
println("Saved figure: $fig_grid_path")

# ── Results table (booktabs, by ψ) ─────────────────────────────────────────
fmt(x) = @sprintf("%.4f", x)
fmt2(x) = @sprintf("%.2f", x)
sgn(x)  = x ≥ 0 ? @sprintf("%+.4f", x) : @sprintf("%.4f", x)
sgn2(x) = x ≥ 0 ? @sprintf("%+.2f", x) : @sprintf("%.2f", x)

global tex
tex = """\\begin{tabular}{cccccccccc}
\\toprule
\$\\psi/\\widehat{\\phi}\$ & \$\\psi\$ &
\$P_{\\text{enter},1}\$ & \$P_{\\text{enter},2}\$ & \$P_{\\text{enter},3}\$ &
\$\\Delta W_1\$ & \$\\Delta W_2\$ & \$\\Delta W_3\$ &
\$\\Delta \\Sigma W\$ & \$\\Delta \\Sigma W\\,(\\%)\$ \\\\
\\midrule
"""
for (i, frac) in enumerate(SUBSIDY_GRID_FRAC)
    global tex
    ψ_g = frac * PHI_HAT
    tex *= @sprintf("%.2f & %.4f & %.4f & %.4f & %.4f & %s & %s & %s & %s & %s \\\\\n",
                    frac, ψ_g,
                    grid_enter_r1[i], grid_enter_r2[i], grid_enter_r3[i],
                    sgn(grid_dW1[i]), sgn(grid_dW2[i]), sgn(grid_dW3[i]),
                    sgn(grid_dWtot[i]), sgn2(grid_dWtot_pct[i]) * "\\%")
end
tex *= "\\bottomrule\n\\end{tabular}\n"

table_path = joinpath(OUT_TAB, "entry_subsidy_results.tex")
open(table_path, "w") do io; write(io, tex); end
println("Saved table: $table_path")

# ── LaTeX macros (config + headline ψ_max + grid headline numbers) ────────
i_h    = lastindex(SUBSIDY_GRID_FRAC)
frac_h = SUBSIDY_GRID_FRAC[i_h]
ψ_h    = frac_h * PHI_HAT
w_h    = w_grid[i_h]

macros = """% Auto-generated by code/scripts/run_entry_subsidy.jl
% Region-3 entry-subsidy grid sweep (sovereign funding accounting).
% Grid ψ/φ̂ ∈ $(SUBSIDY_GRID_FRAC); K = $N_MARKETS markets per ψ; CRN seed = $SEED.
\\newcommand{\\EntrySubsidyNMarkets}{$N_MARKETS}
\\newcommand{\\EntrySubsidySeed}{$SEED}
\\newcommand{\\EntrySubsidyKappaHat}{$(fmt(KAPPA_HAT))}
\\newcommand{\\EntrySubsidyPhiHat}{$(fmt(PHI_HAT))}
\\newcommand{\\EntrySubsidyGammaHat}{$(fmt(GAMMA_HAT[1]))}
\\newcommand{\\EntrySubsidyGridLength}{$(length(SUBSIDY_GRID_FRAC))}
\\newcommand{\\EntrySubsidyMaxFrac}{$(fmt2(frac_h))}
\\newcommand{\\EntrySubsidyMaxPsi}{$(fmt(ψ_h))}
% Baseline (ψ = 0) per-region entry rate
\\newcommand{\\EntrySubsidyBaseEnterROne}{$(fmt(w_base.enter_rate_by_region[1]))}
\\newcommand{\\EntrySubsidyBaseEnterRTwo}{$(fmt(w_base.enter_rate_by_region[2]))}
\\newcommand{\\EntrySubsidyBaseEnterRThree}{$(fmt(w_base.enter_rate_by_region[3]))}
% Headline ψ = ψ_max per-region entry rate
\\newcommand{\\EntrySubsidyMaxEnterROne}{$(fmt(w_h.enter_rate_by_region[1]))}
\\newcommand{\\EntrySubsidyMaxEnterRTwo}{$(fmt(w_h.enter_rate_by_region[2]))}
\\newcommand{\\EntrySubsidyMaxEnterRThree}{$(fmt(w_h.enter_rate_by_region[3]))}
\\newcommand{\\EntrySubsidyMaxEnterPctROne}{$(sgn2(100.0 * (w_h.enter_rate_by_region[1] - w_base.enter_rate_by_region[1]) / w_base.enter_rate_by_region[1]))}
\\newcommand{\\EntrySubsidyMaxEnterPctRTwo}{$(sgn2(100.0 * (w_h.enter_rate_by_region[2] - w_base.enter_rate_by_region[2]) / w_base.enter_rate_by_region[2]))}
\\newcommand{\\EntrySubsidyMaxEnterPctRThree}{$(sgn2(100.0 * (w_h.enter_rate_by_region[3] - w_base.enter_rate_by_region[3]) / w_base.enter_rate_by_region[3]))}
% Headline ψ welfare deltas (absolute and percent)
\\newcommand{\\EntrySubsidyMaxDWROne}{$(sgn(grid_dW1[i_h]))}
\\newcommand{\\EntrySubsidyMaxDWRTwo}{$(sgn(grid_dW2[i_h]))}
\\newcommand{\\EntrySubsidyMaxDWRThree}{$(sgn(grid_dW3[i_h]))}
\\newcommand{\\EntrySubsidyMaxDWTotal}{$(sgn(grid_dWtot[i_h]))}
\\newcommand{\\EntrySubsidyMaxDWPctTotal}{$(sgn2(grid_dWtot_pct[i_h]))}
\\newcommand{\\EntrySubsidyMaxDWPctROne}{$(sgn2(100.0 * grid_dW1[i_h] / w_base.welfare_by_region[1]))}
\\newcommand{\\EntrySubsidyMaxDWPctRTwo}{$(sgn2(100.0 * grid_dW2[i_h] / w_base.welfare_by_region[2]))}
\\newcommand{\\EntrySubsidyMaxDWPctRThree}{$(sgn2(100.0 * grid_dW3[i_h] / w_base.welfare_by_region[3]))}
% Government outlay at headline ψ (per market)
\\newcommand{\\EntrySubsidyMaxGovOutlay}{$(fmt(w_h.gov_outlay_total))}
% Welfare-maximising grid point
\\newcommand{\\EntrySubsidyOptimumFrac}{$(fmt2(SUBSIDY_GRID_FRAC[i_best]))}
\\newcommand{\\EntrySubsidyOptimumDWTotal}{$(sgn(grid_dWtot[i_best]))}
\\newcommand{\\EntrySubsidyOptimumDWPct}{$(sgn2(grid_dWtot_pct[i_best]))}
"""
macro_path = joinpath(OUT_EST, "entry_subsidy_estimates.txt")
open(macro_path, "w") do io; write(io, macros); end
println("Saved macros: $macro_path")

println("\nDone.")
