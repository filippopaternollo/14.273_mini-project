"""
run_joint_subsidy.jl — Joint region-3 entry + innovation subsidy grid sweep.

For each (τ, ψ) on the grid {0, 0.1, …, 0.5} × {0, 0.1, …, 0.5}, where
τ = frac_τ · κ̂ is the region-3 innovation subsidy and ψ = frac_ψ · φ̂ is
the region-3 entry subsidy, we evaluate aggregate welfare under the same
sovereign-funding accounting used by `run_subsidy.jl` and
`run_entry_subsidy.jl`. Common random numbers (same seed and per-market
RNG) are used across baseline and every grid cell.

We run two parallel sweeps to isolate the role of agglomeration:
  (i) the headline calibration γ = γ̂; and
  (ii) a counterfactual with γ = (0, 0, 0).
The latter strips the local agglomeration spillover, leaving Cournot
markup correction and cross-region competition as the only forces
shaping the planner's optimal subsidy.

Outputs:
  - output/tables/joint_subsidy_dWpct.tex            (ΔΣW % grid, γ = γ̂)
  - output/figures/joint_subsidy_heatmap.pdf         (ΔΣW % heatmap, γ = γ̂,
                                                      with both planner argmaxes
                                                      under γ̂ and γ = 0)
  - output/figures/joint_subsidy_heatmap_r3.pdf      (ΔW₃ % heatmap, γ = γ̂)
  - output/estimates/joint_subsidy_estimates.txt     (LaTeX macros, both runs)
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

# ── Joint grid: (τ/κ̂, ψ/φ̂) ────────────────────────────────────────────────
# 11×11 grid in steps of 0.05. A 3×3 box average is applied before
# plotting (see `box_smooth` below) to reduce per-cell MC noise; the
# argmax cell is computed from the raw matrix.
const TAU_FRAC = collect(0.0:0.05:0.50)
const PSI_FRAC = collect(0.0:0.05:0.50)

# ── MC configuration ────────────────────────────────────────────────────────
const N_MARKETS = 5000
const SEED      = 20260424

# ── Baseline ────────────────────────────────────────────────────────────────
p_base = default_params(; gamma = GAMMA_HAT, kappa = KAPPA_HAT, phi = PHI_HAT,
                          subsidy = (0.0, 0.0, 0.0),
                          entry_subsidy = (0.0, 0.0, 0.0))

@printf("=== Region-3 joint (innovation + entry) subsidy: %d×%d grid ===\n",
        length(TAU_FRAC), length(PSI_FRAC))
@printf("  Calibration: κ̂ = %.4f,  φ̂ = %.4f,  γ̂ = (%.4f, %.4f, %.4f)\n",
        KAPPA_HAT, PHI_HAT, GAMMA_HAT...)
@printf("  τ/κ̂ ∈ %s\n", TAU_FRAC)
@printf("  ψ/φ̂ ∈ %s\n", PSI_FRAC)
@printf("  Markets per cell: K = %d   |   seed = %d   (CRN)\n",
        N_MARKETS, SEED)

println("\nBaseline (no subsidy)…")
@time w_base = expected_welfare_mc(p_base; n_markets = N_MARKETS, seed = SEED)

# ── Sweep ───────────────────────────────────────────────────────────────────
nT = length(TAU_FRAC); nP = length(PSI_FRAC)
dWtot      = Matrix{Float64}(undef, nT, nP)
dWtot_pct  = Matrix{Float64}(undef, nT, nP)
dW_r3_pct  = Matrix{Float64}(undef, nT, nP)
dW_r1_pct  = Matrix{Float64}(undef, nT, nP)
dW_r2_pct  = Matrix{Float64}(undef, nT, nP)

println("\n=== Grid sweep (CRN) ===")
@printf("  %6s %6s | %10s %10s | %+9s %+9s %+9s\n",
        "τ/κ̂", "ψ/φ̂", "ΔΣW", "ΔΣW %", "ΔW₁ %", "ΔW₂ %", "ΔW₃ %")

for (i, fT) in enumerate(TAU_FRAC), (j, fP) in enumerate(PSI_FRAC)
    τ = fT * KAPPA_HAT
    ψ = fP * PHI_HAT
    p_g = default_params(; gamma = GAMMA_HAT, kappa = KAPPA_HAT, phi = PHI_HAT,
                           subsidy       = (0.0, 0.0, τ),
                           entry_subsidy = (0.0, 0.0, ψ))
    w_g = expected_welfare_mc(p_g; n_markets = N_MARKETS, seed = SEED)
    Δ   = w_g.total_welfare - w_base.total_welfare
    Δp  = 100.0 * Δ / w_base.total_welfare
    dWtot[i,j]     = Δ
    dWtot_pct[i,j] = Δp
    dW_r1_pct[i,j] = 100.0 * (w_g.welfare_by_region[1] - w_base.welfare_by_region[1]) / w_base.welfare_by_region[1]
    dW_r2_pct[i,j] = 100.0 * (w_g.welfare_by_region[2] - w_base.welfare_by_region[2]) / w_base.welfare_by_region[2]
    dW_r3_pct[i,j] = 100.0 * (w_g.welfare_by_region[3] - w_base.welfare_by_region[3]) / w_base.welfare_by_region[3]
    @printf("  %6.2f %6.2f | %+10.4f %+9.2f%% | %+8.2f%% %+8.2f%% %+8.2f%%\n",
            fT, fP, Δ, Δp, dW_r1_pct[i,j], dW_r2_pct[i,j], dW_r3_pct[i,j])
end

# ── Optimum cell ────────────────────────────────────────────────────────────
i_best, j_best = Tuple(argmax(dWtot))
@printf("\n  Welfare-max cell: τ/κ̂ = %.2f,  ψ/φ̂ = %.2f   ΔΣW = %+.4f  (%+.2f%%)\n",
        TAU_FRAC[i_best], PSI_FRAC[j_best],
        dWtot[i_best,j_best], dWtot_pct[i_best,j_best])

# ── Counterfactual sweep: γ = (0, 0, 0) ───────────────────────────────────
# Same grid, no agglomeration spillover. Used to isolate how much of the
# planner's optimal subsidy is driven by the agglomeration externality
# vs.\ the Cournot markup / cross-region competition channels.
const GAMMA_ZERO = (0.0, 0.0, 0.0)

p_base_zg = default_params(; gamma = GAMMA_ZERO, kappa = KAPPA_HAT, phi = PHI_HAT,
                             subsidy = (0.0, 0.0, 0.0),
                             entry_subsidy = (0.0, 0.0, 0.0))

@printf("\n=== Counterfactual: same %d×%d grid with γ = (0, 0, 0) ===\n",
        length(TAU_FRAC), length(PSI_FRAC))

println("Baseline (no agglomeration, no subsidy)…")
@time w_base_zg = expected_welfare_mc(p_base_zg; n_markets = N_MARKETS, seed = SEED)

dWtot_zg     = Matrix{Float64}(undef, nT, nP)
dWtot_pct_zg = Matrix{Float64}(undef, nT, nP)
dW_r3_pct_zg = Matrix{Float64}(undef, nT, nP)

@printf("\n  %6s %6s | %10s %10s | %+9s\n",
        "τ/κ̂", "ψ/φ̂", "ΔΣW", "ΔΣW %", "ΔW₃ %")
for (i, fT) in enumerate(TAU_FRAC), (j, fP) in enumerate(PSI_FRAC)
    τ = fT * KAPPA_HAT
    ψ = fP * PHI_HAT
    p_g = default_params(; gamma = GAMMA_ZERO, kappa = KAPPA_HAT, phi = PHI_HAT,
                           subsidy       = (0.0, 0.0, τ),
                           entry_subsidy = (0.0, 0.0, ψ))
    w_g = expected_welfare_mc(p_g; n_markets = N_MARKETS, seed = SEED)
    Δ   = w_g.total_welfare - w_base_zg.total_welfare
    Δp  = 100.0 * Δ / w_base_zg.total_welfare
    dWtot_zg[i,j]     = Δ
    dWtot_pct_zg[i,j] = Δp
    dW_r3_pct_zg[i,j] = 100.0 * (w_g.welfare_by_region[3] - w_base_zg.welfare_by_region[3]) / w_base_zg.welfare_by_region[3]
    @printf("  %6.2f %6.2f | %+10.4f %+9.2f%% | %+8.2f%%\n",
            fT, fP, Δ, Δp, dW_r3_pct_zg[i,j])
end

i_best_zg, j_best_zg = Tuple(argmax(dWtot_zg))
@printf("\n  γ = 0 welfare-max cell: τ/κ̂ = %.2f,  ψ/φ̂ = %.2f   ΔΣW = %+.4f  (%+.2f%%)\n",
        TAU_FRAC[i_best_zg], PSI_FRAC[j_best_zg],
        dWtot_zg[i_best_zg,j_best_zg], dWtot_pct_zg[i_best_zg,j_best_zg])

# Visual smoothing: a 3×3 box average tames per-cell MC noise without
# moving the argmax cell (computed above from the raw matrix). Used
# only for the figure; macros and the booktabs table use raw data.
function box_smooth(M::Matrix{Float64})
    n, m = size(M)
    S = similar(M)
    @inbounds for i in 1:n, j in 1:m
        s = 0.0; c = 0
        for di in -1:1, dj in -1:1
            ii = i + di; jj = j + dj
            (ii < 1 || ii > n || jj < 1 || jj > m) && continue
            s += M[ii, jj]; c += 1
        end
        S[i, j] = s / c
    end
    return S
end

# ── Heatmap of ΔΣW % under γ = γ̂, with both planner argmaxes overlaid ─────
dWtot_pct_smooth = box_smooth(dWtot_pct)
hmax = maximum(abs, dWtot_pct_smooth)
plt = heatmap(PSI_FRAC, TAU_FRAC, dWtot_pct_smooth;
              xlabel = "Entry-subsidy fraction  ψ / φ̂",
              ylabel = "Innovation-subsidy fraction  τ / κ̂",
              title  = "Aggregate welfare change ΔΣW (% of baseline)",
              c = cgrad(:RdBu, rev = true),
              clims = (-hmax, hmax),
              colorbar_title = " ΔΣW (%)",
              size = (760, 560),
              dpi = 200,
              titlefontsize = 13, guidefontsize = 11,
              tickfontsize = 10, colorbar_titlefontsize = 11,
              xticks = 0.0:0.10:0.50, yticks = 0.0:0.10:0.50,
              framestyle = :box,
              left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
              right_margin = 9Plots.mm, top_margin = 4Plots.mm)
scatter!(plt, [PSI_FRAC[j_best]], [TAU_FRAC[i_best]];
         marker = :star5, ms = 14, color = :gold,
         markerstrokecolor = :black, markerstrokewidth = 1.5,
         label = "Planner argmax  (γ = γ̂)",
         legend = :topright, foreground_color_legend = nothing,
         background_color_legend = RGBA(1.0, 1.0, 1.0, 0.85))
scatter!(plt, [PSI_FRAC[j_best_zg]], [TAU_FRAC[i_best_zg]];
         marker = :circle, ms = 9, color = :black,
         markerstrokecolor = :white, markerstrokewidth = 1.5,
         label = "Planner argmax  (γ = 0)")
fig_path = joinpath(OUT_FIG, "joint_subsidy_heatmap.pdf")
savefig(plt, fig_path)
println("\nSaved figure: $fig_path")

# ── Heatmap of ΔW₃ % (treated region) ─────────────────────────────────────
i_best_r3, j_best_r3 = Tuple(argmax(dW_r3_pct))
dW_r3_pct_smooth = box_smooth(dW_r3_pct)
hmax_r3 = maximum(abs, dW_r3_pct_smooth)
plt3 = heatmap(PSI_FRAC, TAU_FRAC, dW_r3_pct_smooth;
               xlabel = "Entry-subsidy fraction  ψ / φ̂",
               ylabel = "Innovation-subsidy fraction  τ / κ̂",
               title  = "Region-3 welfare change ΔW₃ (% of baseline)",
               c = cgrad(:RdBu, rev = true),
               clims = (-hmax_r3, hmax_r3),
               colorbar_title = " ΔW₃ (%)",
               size = (760, 560),
               dpi = 200,
               titlefontsize = 13, guidefontsize = 11,
               tickfontsize = 10, colorbar_titlefontsize = 11,
               xticks = 0.0:0.10:0.50, yticks = 0.0:0.10:0.50,
               framestyle = :box,
               left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
               right_margin = 9Plots.mm, top_margin = 4Plots.mm)
scatter!(plt3, [PSI_FRAC[j_best_r3]], [TAU_FRAC[i_best_r3]];
         marker = :star5, ms = 14, color = :gold,
         markerstrokecolor = :black, markerstrokewidth = 1.5,
         label = "Region-3 argmax",
         legend = :topright, foreground_color_legend = nothing,
         background_color_legend = RGBA(1.0, 1.0, 1.0, 0.85))
fig_path_r3 = joinpath(OUT_FIG, "joint_subsidy_heatmap_r3.pdf")
savefig(plt3, fig_path_r3)
println("Saved figure: $fig_path_r3")

# ── Table: ΔΣW % grid ──────────────────────────────────────────────────────
sgn2(x) = x ≥ 0 ? @sprintf("%+.2f", x) : @sprintf("%.2f", x)

global tex
tex = "\\begin{tabular}{c|" * "c"^nP * "}\n\\toprule\n"
tex *= " & \\multicolumn{$nP}{c}{\$\\psi/\\widehat{\\phi}\$} \\\\\n"
tex *= "\$\\tau/\\widehat{\\kappa}\$ & "
tex *= join([@sprintf("%.2f", fP) for fP in PSI_FRAC], " & ") * " \\\\\n"
tex *= "\\midrule\n"
for (i, fT) in enumerate(TAU_FRAC)
    global tex
    cells = [sgn2(dWtot_pct[i,j]) * "\\%" for j in 1:nP]
    tex *= @sprintf("%.2f & %s \\\\\n", fT, join(cells, " & "))
end
tex *= "\\bottomrule\n\\end{tabular}\n"

table_path = joinpath(OUT_TAB, "joint_subsidy_dWpct.tex")
open(table_path, "w") do io; write(io, tex); end
println("Saved table: $table_path")

# ── LaTeX macros ───────────────────────────────────────────────────────────
fmt(x)  = @sprintf("%.4f", x)
fmt2(x) = @sprintf("%.2f", x)
sgn(x)  = x ≥ 0 ? @sprintf("%+.4f", x) : @sprintf("%.4f", x)

macros = """% Auto-generated by code/scripts/run_joint_subsidy.jl
% Joint region-3 (innovation × entry) subsidy grid sweep.
% τ/κ̂ ∈ $(TAU_FRAC); ψ/φ̂ ∈ $(PSI_FRAC); K = $N_MARKETS; CRN seed = $SEED.
\\newcommand{\\JointSubsidyNMarkets}{$N_MARKETS}
\\newcommand{\\JointSubsidySeed}{$SEED}
\\newcommand{\\JointSubsidyGridLengthTau}{$(length(TAU_FRAC))}
\\newcommand{\\JointSubsidyGridLengthPsi}{$(length(PSI_FRAC))}
\\newcommand{\\JointSubsidyOptTauFrac}{$(fmt2(TAU_FRAC[i_best]))}
\\newcommand{\\JointSubsidyOptPsiFrac}{$(fmt2(PSI_FRAC[j_best]))}
\\newcommand{\\JointSubsidyOptTau}{$(fmt(TAU_FRAC[i_best]*KAPPA_HAT))}
\\newcommand{\\JointSubsidyOptPsi}{$(fmt(PSI_FRAC[j_best]*PHI_HAT))}
\\newcommand{\\JointSubsidyOptDWTotal}{$(sgn(dWtot[i_best,j_best]))}
\\newcommand{\\JointSubsidyOptDWPct}{$(sgn2(dWtot_pct[i_best,j_best]))}
\\newcommand{\\JointSubsidyOptDWPctROne}{$(sgn2(dW_r1_pct[i_best,j_best]))}
\\newcommand{\\JointSubsidyOptDWPctRTwo}{$(sgn2(dW_r2_pct[i_best,j_best]))}
\\newcommand{\\JointSubsidyOptDWPctRThree}{$(sgn2(dW_r3_pct[i_best,j_best]))}
% Corner cells (innov-only at largest τ; entry-only at largest ψ; both at max)
\\newcommand{\\JointSubsidyInnovOnlyMaxDWPct}{$(sgn2(dWtot_pct[end,1]))}
\\newcommand{\\JointSubsidyEntryOnlyMaxDWPct}{$(sgn2(dWtot_pct[1,end]))}
\\newcommand{\\JointSubsidyBothMaxDWPct}{$(sgn2(dWtot_pct[end,end]))}
% Region-3 welfare-max cell
\\newcommand{\\JointSubsidyROptTauFrac}{$(fmt2(TAU_FRAC[i_best_r3]))}
\\newcommand{\\JointSubsidyROptPsiFrac}{$(fmt2(PSI_FRAC[j_best_r3]))}
\\newcommand{\\JointSubsidyROptDWPctRThree}{$(sgn2(dW_r3_pct[i_best_r3,j_best_r3]))}
"""

macros *= """% No-agglomeration counterfactual: γ = (0, 0, 0).
\\newcommand{\\JointSubsidyNoAgglomOptTauFrac}{$(fmt2(TAU_FRAC[i_best_zg]))}
\\newcommand{\\JointSubsidyNoAgglomOptPsiFrac}{$(fmt2(PSI_FRAC[j_best_zg]))}
\\newcommand{\\JointSubsidyNoAgglomOptTau}{$(fmt(TAU_FRAC[i_best_zg]*KAPPA_HAT))}
\\newcommand{\\JointSubsidyNoAgglomOptPsi}{$(fmt(PSI_FRAC[j_best_zg]*PHI_HAT))}
\\newcommand{\\JointSubsidyNoAgglomOptDWTotal}{$(sgn(dWtot_zg[i_best_zg,j_best_zg]))}
\\newcommand{\\JointSubsidyNoAgglomOptDWPct}{$(sgn2(dWtot_pct_zg[i_best_zg,j_best_zg]))}
% Corner cells under γ = 0
\\newcommand{\\JointSubsidyNoAgglomInnovOnlyMaxDWPct}{$(sgn2(dWtot_pct_zg[end,1]))}
\\newcommand{\\JointSubsidyNoAgglomEntryOnlyMaxDWPct}{$(sgn2(dWtot_pct_zg[1,end]))}
\\newcommand{\\JointSubsidyNoAgglomBothMaxDWPct}{$(sgn2(dWtot_pct_zg[end,end]))}
"""

macro_path = joinpath(OUT_EST, "joint_subsidy_estimates.txt")
open(macro_path, "w") do io; write(io, macros); end
println("Saved macros: $macro_path")

println("\nDone.")
