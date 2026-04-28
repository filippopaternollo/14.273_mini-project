"""
run_planner.jl — Constrained social planner counterfactual.

Compares the equilibrium baseline against two planner cuts:

  - Dynamic-only first best  : planner CCPs internalize the spillover,
                               but production stays Cournot (isolates the
                               dynamic, externality channel).
  - Full first best          : planner CCPs *plus* P=MC production in
                               both periods (adds the static markup
                               correction).

The planner is *constrained* — it picks each option with the same logit
form firms use, but with social welfare differences in place of private
ones (`notes/planner_counterfactual.md`, design choice (a)).  At this
calibration the dynamic-only cut can fall *below* equilibrium, because:
the social ΔW between innovate and stay is nonlinear in the number of
already-innovated firms (welfare is convex in c_n) and saturates once
spillovers are large; with σ tuned to private-value scale, the planner's
plug-in CCPs are not "decisive" enough to internalize the spillover when
its marginal contribution to ΣW is small relative to κ.  The static
slice (Full FB − Dynamic-only) cleanly isolates the markup-correction
gain.

Welfare is evaluated by Monte Carlo over `random_s0` markets, the same
DGP `simulate_data.jl` uses.  All three scenarios share `seed = SEED`
and per-market `MersenneTwister(seed + k)`, so the welfare *differences*
across scenarios are essentially noise-free (CRN).

Calibration is the **estimated** parameter vector from
`output/estimates/estimation.txt`.

Outputs:
  - output/tables/planner_results.tex        (booktabs table by scenario × region)
  - output/tables/planner_decomposition.tex  (ΔW dyn / full / static slice)
  - output/figures/planner_welfare.pdf       (W_r grouped bars across scenarios)
  - output/estimates/planner_estimates.txt   (LaTeX macros for the writeup)
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

# ── MC configuration ────────────────────────────────────────────────────────
const N_MARKETS = 5000
const SEED      = 20260424

# ── Build calibration ───────────────────────────────────────────────────────
p_base = default_params(; gamma = GAMMA_HAT, blocs = (1, 2, 3),
                          kappa = KAPPA_HAT, phi = PHI_HAT)

@printf("=== Constrained social planner counterfactual ===\n")
@printf("  Calibration: κ̂ = %.4f,  φ̂ = %.4f,  γ̂ = (%.4f, %.4f, %.4f)\n",
        KAPPA_HAT, PHI_HAT, GAMMA_HAT...)
@printf("  Markets per scenario: K = %d   |   seed = %d   (CRN)\n",
        N_MARKETS, SEED)

# ── Run three scenarios under common random numbers ────────────────────────
println("\nEquilibrium (firms decide)…")
@time w_eq  = expected_welfare_mc(p_base; n_markets = N_MARKETS, seed = SEED)
println("Dynamic-only first best (planner CCPs, Cournot static)…")
@time w_dyn = expected_planner_welfare(p_base; n_markets = N_MARKETS, seed = SEED,
                                        competitive_static = false)
println("Full first best (planner CCPs, P=MC static)…")
@time w_fb  = expected_planner_welfare(p_base; n_markets = N_MARKETS, seed = SEED,
                                        competitive_static = true)

# ── Sanity: Σ_r W_r identity ────────────────────────────────────────────────
function check_sum_identity(w, label, beta)
    cs_disc = w.cs_p1 + beta * w.cs_p2
    rhs = cs_disc + sum(w.ps_by_region) - sum(w.costs_by_region)
    err = w.total_welfare - rhs
    @printf("  Σ_r W_r identity (%-12s): Σ = %.6f   direct = %.6f   diff = %+.2e\n",
            label, w.total_welfare, rhs, err)
end
println("\n=== Sanity: Σ_r W_r = CS_total + Σ PS_r − Σ costs_r ===")
check_sum_identity(w_eq,  "equilibrium", p_base.beta)
check_sum_identity(w_dyn, "dynamic FB",  p_base.beta)
check_sum_identity(w_fb,  "full FB",     p_base.beta)

# Sign checks (informational; we expect W_FB > W_Eq, but DynOnly under
# the plug-in approximation can fall short of Eq when κ ≈ private benefit
# but well below the social benefit at low n_b — the planner over-innovates
# in absolute terms relative to its own σ-shock optimum, costing more κ
# than the spillover gain).
println("\n=== Sign checks (informational) ===")
@printf("  ΣW: Eq = %.4f, DynOnly = %.4f, FullFB = %.4f\n",
        w_eq.total_welfare, w_dyn.total_welfare, w_fb.total_welfare)
@printf("  Innov rates: Eq = (%.4f, %.4f, %.4f); DynOnly = (%.4f, %.4f, %.4f); FullFB = (%.4f, %.4f, %.4f)\n",
        w_eq.innov_rate_by_region..., w_dyn.innov_rate_by_region..., w_fb.innov_rate_by_region...)
@printf("  Enter rates: Eq = (%.4f, %.4f, %.4f); DynOnly = (%.4f, %.4f, %.4f); FullFB = (%.4f, %.4f, %.4f)\n",
        w_eq.enter_rate_by_region..., w_dyn.enter_rate_by_region..., w_fb.enter_rate_by_region...)

# ── Display headline numbers ────────────────────────────────────────────────
function print_scenario(label, w, beta)
    println("\n--- $label ---")
    @printf("  Innov rate (per region): (%.4f, %.4f, %.4f)\n", w.innov_rate_by_region...)
    @printf("  Enter rate (per region): (%.4f, %.4f, %.4f)\n", w.enter_rate_by_region...)
    @printf("  CS / R per region      : %.4f\n", (w.cs_p1 + beta * w.cs_p2) / R)
    @printf("  PS_r per region        : (%.4f, %.4f, %.4f)\n", w.ps_by_region...)
    @printf("  Costs_r per region     : (%.4f, %.4f, %.4f)\n", w.costs_by_region...)
    @printf("  Welfare_r per region   : (%.4f, %.4f, %.4f)\n", w.welfare_by_region...)
    @printf("  Σ_r W_r                : %.4f\n", w.total_welfare)
end
print_scenario("Equilibrium",        w_eq,  p_base.beta)
print_scenario("Dynamic-only FB",    w_dyn, p_base.beta)
print_scenario("Full first best",    w_fb,  p_base.beta)

# ── Decomposition ──────────────────────────────────────────────────────────
ΔW_dyn      = w_dyn.total_welfare - w_eq.total_welfare       # dynamic slice
ΔW_fb       = w_fb.total_welfare  - w_eq.total_welfare       # total
ΔW_static   = w_fb.total_welfare  - w_dyn.total_welfare      # static slice

pct_dyn    = 100.0 * ΔW_dyn    / w_eq.total_welfare
pct_fb     = 100.0 * ΔW_fb     / w_eq.total_welfare
pct_static = 100.0 * ΔW_static / w_eq.total_welfare

println("\n=== Decomposition (planner vs. equilibrium) ===")
@printf("  ΔΣW (DynOnly  − Eq)   = %+.4f   (%+.2f%% of Eq)\n",  ΔW_dyn,    pct_dyn)
@printf("  ΔΣW (Full FB  − Eq)   = %+.4f   (%+.2f%% of Eq)\n",  ΔW_fb,     pct_fb)
@printf("  ΔΣW (static slice)    = %+.4f   (%+.2f%% of Eq)\n",  ΔW_static, pct_static)

# ── K-stability on the Full-FB delta (mirrors run_merger.jl) ───────────────
println("\n=== K-stability of ΔΣW (Full FB − Eq), seed-matched, CRN ===")
@printf("  %5s | %12s %12s | %10s\n", "K", "ΣW Eq", "ΣW FB", "ΔΣW%")
for k_test in (500, 1000, 5000)
    w_eq_k = k_test == N_MARKETS ? w_eq :
             expected_welfare_mc(p_base; n_markets = k_test, seed = SEED)
    w_fb_k = k_test == N_MARKETS ? w_fb :
             expected_planner_welfare(p_base; n_markets = k_test, seed = SEED,
                                       competitive_static = true)
    Δ_k = w_fb_k.total_welfare - w_eq_k.total_welfare
    pct = 100.0 * Δ_k / w_eq_k.total_welfare
    @printf("  %5d | %12.4f %12.4f | %+9.2f%%\n",
            k_test, w_eq_k.total_welfare, w_fb_k.total_welfare, pct)
end

# ── Plot: per-region welfare across the three scenarios ───────────────────
const REGION_LABELS = ["Region 1", "Region 2", "Region 3"]
const COL_EQ  = colorant"#0072B2"   # Wong blue       — equilibrium
const COL_DYN = colorant"#009E73"   # Wong green      — dynamic-only FB
const COL_FB  = colorant"#D55E00"   # Wong vermilion  — full FB
const X_EQ  = [0.78, 1.78, 2.78]
const X_DYN = [1.00, 2.00, 3.00]
const X_FB  = [1.22, 2.22, 3.22]
const BW    = 0.20

function planner_grouped_bar(values_eq, values_dyn, values_fb;
                             ylabel, title, fig_path)
    ymax = max(maximum(values_eq), maximum(values_dyn), maximum(values_fb))
    pad  = 0.14 * ymax

    plt = bar(X_EQ, values_eq;
              bar_width      = BW,
              label          = "Equilibrium",
              color          = COL_EQ,
              linecolor      = COL_EQ,
              xticks         = (1:3, REGION_LABELS),
              xlims          = (0.4, 3.6),
              ylims          = (0, ymax + pad),
              ylabel         = ylabel,
              title          = title,
              legend         = :outerbottom,
              legend_columns = 3,
              foreground_color_legend = nothing,
              background_color_legend = nothing,
              framestyle     = :semi,
              grid           = :y,
              gridalpha      = 0.25,
              gridlinewidth  = 0.5,
              tick_direction = :out,
              titlefontsize  = 12,
              guidefontsize  = 10,
              tickfontsize   = 9,
              legendfontsize = 10,
              size           = (760, 480),
              left_margin    = 5Plots.mm,
              bottom_margin  = 5Plots.mm,
              top_margin     = 3Plots.mm)
    bar!(plt, X_DYN, values_dyn;
         bar_width = BW, label = "Dynamic-only FB",
         color = COL_DYN, linecolor = COL_DYN)
    bar!(plt, X_FB, values_fb;
         bar_width = BW, label = "Full FB",
         color = COL_FB, linecolor = COL_FB)

    label_offset = 0.025 * ymax
    for r in 1:3
        annotate!(plt, X_EQ[r],  values_eq[r]  + label_offset,
                  Plots.text(@sprintf("%.3f", values_eq[r]),
                             7, COL_EQ, :center, :bottom))
        annotate!(plt, X_DYN[r], values_dyn[r] + label_offset,
                  Plots.text(@sprintf("%.3f", values_dyn[r]),
                             7, COL_DYN, :center, :bottom))
        annotate!(plt, X_FB[r],  values_fb[r]  + label_offset,
                  Plots.text(@sprintf("%.3f", values_fb[r]),
                             7, COL_FB, :center, :bottom))
    end

    savefig(plt, fig_path)
    return fig_path
end

fig_welfare_path = planner_grouped_bar(
    collect(w_eq.welfare_by_region),
    collect(w_dyn.welfare_by_region),
    collect(w_fb.welfare_by_region);
    ylabel   = "Discounted per-region welfare W_r",
    title    = "Welfare by region: equilibrium vs. constrained planner",
    fig_path = joinpath(OUT_FIG, "planner_welfare.pdf"),
)
println("\nSaved figure: $fig_welfare_path")

# ── Results table (booktabs) ───────────────────────────────────────────────
fmt(x) = @sprintf("%.4f", x)
sgn(x) = x ≥ 0 ? @sprintf("%+.4f", x) : @sprintf("%.4f", x)
fmt2(x) = @sprintf("%.2f", x)
sgn2(x) = x ≥ 0 ? @sprintf("%+.2f", x) : @sprintf("%.2f", x)

cs_per_region(w, beta) = (w.cs_p1 + beta * w.cs_p2) / R

cs_eq  = cs_per_region(w_eq,  p_base.beta)
cs_dyn = cs_per_region(w_dyn, p_base.beta)
cs_fb  = cs_per_region(w_fb,  p_base.beta)

# Table layout: rows grouped by scenario; columns = R1, R2, R3, total/avg.
# 5 columns total (label + 3 regions + total).
function row(name, vals; total = "—")
    return @sprintf("%s & %s & %s & %s & %s \\\\\n",
                    name, fmt(vals[1]), fmt(vals[2]), fmt(vals[3]), total)
end

tex = """\\begin{tabular}{lrrrr}
\\toprule
                                  & Region 1 & Region 2 & Region 3 & Total/avg. \\\\
\\midrule
\\multicolumn{5}{l}{\\textit{Equilibrium}} \\\\
"""
tex *= row("\\quad \$P(\\text{innov}\\,|\\,\\text{old})\$",
           w_eq.innov_rate_by_region; total = "—")
tex *= row("\\quad \$P(\\text{enter}\\,|\\,\\text{pe})\$",
           w_eq.enter_rate_by_region; total = "—")
tex *= row("\\quad \$\\mathrm{PS}_r\$",
           w_eq.ps_by_region; total = fmt(sum(w_eq.ps_by_region)))
tex *= row("\\quad Costs paid",
           w_eq.costs_by_region; total = fmt(sum(w_eq.costs_by_region)))
tex *= row("\\quad \$W_r\$",
           w_eq.welfare_by_region; total = fmt(w_eq.total_welfare))
tex *= "\\midrule\n\\multicolumn{5}{l}{\\textit{Dynamic-only first best}} \\\\\n"
tex *= row("\\quad \$P(\\text{innov}\\,|\\,\\text{old})\$",
           w_dyn.innov_rate_by_region; total = "—")
tex *= row("\\quad \$P(\\text{enter}\\,|\\,\\text{pe})\$",
           w_dyn.enter_rate_by_region; total = "—")
tex *= row("\\quad \$\\mathrm{PS}_r\$",
           w_dyn.ps_by_region; total = fmt(sum(w_dyn.ps_by_region)))
tex *= row("\\quad Costs paid",
           w_dyn.costs_by_region; total = fmt(sum(w_dyn.costs_by_region)))
tex *= row("\\quad \$W_r\$",
           w_dyn.welfare_by_region; total = fmt(w_dyn.total_welfare))
tex *= "\\midrule\n\\multicolumn{5}{l}{\\textit{Full first best (P = MC)}} \\\\\n"
tex *= row("\\quad \$P(\\text{innov}\\,|\\,\\text{old})\$",
           w_fb.innov_rate_by_region; total = "—")
tex *= row("\\quad \$P(\\text{enter}\\,|\\,\\text{pe})\$",
           w_fb.enter_rate_by_region; total = "—")
tex *= row("\\quad \$\\mathrm{PS}_r\$",
           w_fb.ps_by_region; total = fmt(sum(w_fb.ps_by_region)))
tex *= row("\\quad Costs paid",
           w_fb.costs_by_region; total = fmt(sum(w_fb.costs_by_region)))
tex *= row("\\quad \$W_r\$",
           w_fb.welfare_by_region; total = fmt(w_fb.total_welfare))
tex *= "\\bottomrule\n\\end{tabular}\n"

table_path = joinpath(OUT_TAB, "planner_results.tex")
open(table_path, "w") do io; write(io, tex); end
println("Saved table: $table_path")

# ── Decomposition table (booktabs) ─────────────────────────────────────────
dtex = """\\begin{tabular}{lrr}
\\toprule
                       & \$\\Delta \\Sigma W\$ & \$\\Delta \\Sigma W / \\Sigma W_{\\text{eq}}\$ \\\\
\\midrule
"""
dtex *= @sprintf("Dynamic-only FB \$-\$ Eq.   & %s & %s\\%% \\\\\n", sgn(ΔW_dyn),    sgn2(pct_dyn))
dtex *= @sprintf("Full FB \$-\$ Eq.           & %s & %s\\%% \\\\\n", sgn(ΔW_fb),     sgn2(pct_fb))
dtex *= "\\midrule\n"
dtex *= @sprintf("Static slice (Full FB \$-\$ Dyn) & %s & %s\\%% \\\\\n",
                 sgn(ΔW_static), sgn2(pct_static))
dtex *= "\\bottomrule\n\\end{tabular}\n"

decomp_path = joinpath(OUT_TAB, "planner_decomposition.tex")
open(decomp_path, "w") do io; write(io, dtex); end
println("Saved table: $decomp_path")

# ── LaTeX macros for the writeup ───────────────────────────────────────────
macros = """% Auto-generated by code/scripts/run_planner.jl
% Constrained social planner counterfactual.
% Three scenarios: equilibrium, dynamic-only first best (Cournot static),
%                  full first best (P = MC static).
% K = $N_MARKETS markets per scenario, common-random-numbers seed = $SEED.
\\newcommand{\\PlannerNMarkets}{$N_MARKETS}
\\newcommand{\\PlannerSeed}{$SEED}
\\newcommand{\\PlannerKappaHat}{$(fmt(KAPPA_HAT))}
\\newcommand{\\PlannerPhiHat}{$(fmt(PHI_HAT))}
\\newcommand{\\PlannerGammaHat}{$(fmt(GAMMA_HAT[1]))}
% Total welfare under each scenario
\\newcommand{\\PlannerWelfEq}{$(fmt(w_eq.total_welfare))}
\\newcommand{\\PlannerWelfDyn}{$(fmt(w_dyn.total_welfare))}
\\newcommand{\\PlannerWelfFB}{$(fmt(w_fb.total_welfare))}
% Welfare deltas
\\newcommand{\\PlannerDeltaDynAbs}{$(sgn(ΔW_dyn))}
\\newcommand{\\PlannerDeltaFBAbs}{$(sgn(ΔW_fb))}
\\newcommand{\\PlannerDeltaStaticAbs}{$(sgn(ΔW_static))}
\\newcommand{\\PlannerDeltaDynPct}{$(sgn2(pct_dyn))}
\\newcommand{\\PlannerDeltaFBPct}{$(sgn2(pct_fb))}
\\newcommand{\\PlannerDeltaStaticPct}{$(sgn2(pct_static))}
% Per-region innovation rate, equilibrium
\\newcommand{\\PlannerEqInnovROne}{$(fmt(w_eq.innov_rate_by_region[1]))}
\\newcommand{\\PlannerEqInnovRTwo}{$(fmt(w_eq.innov_rate_by_region[2]))}
\\newcommand{\\PlannerEqInnovRThree}{$(fmt(w_eq.innov_rate_by_region[3]))}
% Per-region innovation rate, dynamic-only FB
\\newcommand{\\PlannerDynInnovROne}{$(fmt(w_dyn.innov_rate_by_region[1]))}
\\newcommand{\\PlannerDynInnovRTwo}{$(fmt(w_dyn.innov_rate_by_region[2]))}
\\newcommand{\\PlannerDynInnovRThree}{$(fmt(w_dyn.innov_rate_by_region[3]))}
% Per-region innovation rate, full FB
\\newcommand{\\PlannerFBInnovROne}{$(fmt(w_fb.innov_rate_by_region[1]))}
\\newcommand{\\PlannerFBInnovRTwo}{$(fmt(w_fb.innov_rate_by_region[2]))}
\\newcommand{\\PlannerFBInnovRThree}{$(fmt(w_fb.innov_rate_by_region[3]))}
% Per-region entry rate, equilibrium
\\newcommand{\\PlannerEqEnterROne}{$(fmt(w_eq.enter_rate_by_region[1]))}
\\newcommand{\\PlannerEqEnterRTwo}{$(fmt(w_eq.enter_rate_by_region[2]))}
\\newcommand{\\PlannerEqEnterRThree}{$(fmt(w_eq.enter_rate_by_region[3]))}
% Per-region entry rate, dynamic-only FB
\\newcommand{\\PlannerDynEnterROne}{$(fmt(w_dyn.enter_rate_by_region[1]))}
\\newcommand{\\PlannerDynEnterRTwo}{$(fmt(w_dyn.enter_rate_by_region[2]))}
\\newcommand{\\PlannerDynEnterRThree}{$(fmt(w_dyn.enter_rate_by_region[3]))}
% Per-region entry rate, full FB
\\newcommand{\\PlannerFBEnterROne}{$(fmt(w_fb.enter_rate_by_region[1]))}
\\newcommand{\\PlannerFBEnterRTwo}{$(fmt(w_fb.enter_rate_by_region[2]))}
\\newcommand{\\PlannerFBEnterRThree}{$(fmt(w_fb.enter_rate_by_region[3]))}
% Per-region welfare, equilibrium
\\newcommand{\\PlannerEqWROne}{$(fmt(w_eq.welfare_by_region[1]))}
\\newcommand{\\PlannerEqWRTwo}{$(fmt(w_eq.welfare_by_region[2]))}
\\newcommand{\\PlannerEqWRThree}{$(fmt(w_eq.welfare_by_region[3]))}
% Per-region welfare, dynamic-only FB
\\newcommand{\\PlannerDynWROne}{$(fmt(w_dyn.welfare_by_region[1]))}
\\newcommand{\\PlannerDynWRTwo}{$(fmt(w_dyn.welfare_by_region[2]))}
\\newcommand{\\PlannerDynWRThree}{$(fmt(w_dyn.welfare_by_region[3]))}
% Per-region welfare, full FB
\\newcommand{\\PlannerFBWROne}{$(fmt(w_fb.welfare_by_region[1]))}
\\newcommand{\\PlannerFBWRTwo}{$(fmt(w_fb.welfare_by_region[2]))}
\\newcommand{\\PlannerFBWRThree}{$(fmt(w_fb.welfare_by_region[3]))}
"""
macro_path = joinpath(OUT_EST, "planner_estimates.txt")
open(macro_path, "w") do io; write(io, macros); end
println("Saved macros: $macro_path")

println("\nDone.")
