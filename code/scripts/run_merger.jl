"""
run_merger.jl — EU–US alliance counterfactual.

Compares the baseline (no merger; each region's spillover pool is itself)
against an alliance of regions {1, 2} (regions 1 and 2 share an innovator
pool; region 3 stays alone).  Welfare and innovation rates are evaluated
by Monte Carlo over `random_s0` markets — the same DGP used in
`simulate_data.jl` and consumed by `estimate.jl`.  Common random numbers
(same seed, same per-market `MersenneTwister(seed + k)`) are used across
the two scenarios so the welfare *difference* is essentially noise-free.

Calibration is the **estimated** parameter vector from
`output/estimates/estimation.txt`.  Other parameters (A, B, M, c_o, c_n0,
β, σ, ρ, N_max) are treated as known.

Outputs:
  - output/tables/merger_results.tex        (booktabs table by region)
  - output/figures/merger_innovation.pdf    (innovation rate, baseline vs alliance)
  - output/figures/merger_welfare.pdf       (welfare components, baseline vs alliance)
  - output/estimates/merger_estimates.txt   (LaTeX macros for the writeup)
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
# From output/estimates/estimation.txt at the (γ, σ) = (0.15, 0.5) calibration:
#   κ̂ = 0.3002,  φ̂ = 0.1777,  γ̂ = (0.15, 0.15, 0.15)
const KAPPA_HAT = 0.3002
const PHI_HAT   = 0.1777
const GAMMA_HAT = (0.15, 0.15, 0.15)

# ── MC configuration ────────────────────────────────────────────────────────
const N_MARKETS = 5000
const SEED      = 20260424

# ── Build scenarios ─────────────────────────────────────────────────────────
p_base = default_params(; gamma = GAMMA_HAT, blocs = (1, 2, 3),
                          kappa = KAPPA_HAT, phi = PHI_HAT)
p_alli = default_params(; gamma = GAMMA_HAT, blocs = (1, 1, 2),
                          kappa = KAPPA_HAT, phi = PHI_HAT)

@printf("=== EU–US alliance counterfactual ===\n")
@printf("  Calibration: κ̂ = %.4f,  φ̂ = %.4f,  γ̂ = (%.4f, %.4f, %.4f)\n",
        KAPPA_HAT, PHI_HAT, GAMMA_HAT...)
@printf("  Markets per scenario: K = %d   |   seed = %d   (CRN)\n",
        N_MARKETS, SEED)

# ── Run baseline + alliance under common random numbers ────────────────────
println("\nBaseline (blocs = (1, 2, 3))...")
@time w_base = expected_welfare_mc(p_base; n_markets = N_MARKETS, seed = SEED)
println("Alliance (blocs = (1, 1, 2))...")
@time w_alli = expected_welfare_mc(p_alli; n_markets = N_MARKETS, seed = SEED)

# Verification: Σ_r W_r equals CS_total + Σ PS_r − Σ costs_r
function check_sum_identity(w, label, beta)
    cs_disc = w.cs_p1 + beta * w.cs_p2
    rhs = cs_disc + sum(w.ps_by_region) - sum(w.costs_by_region)
    err = w.total_welfare - rhs
    @printf("  Σ_r W_r identity (%s): Σ = %.6f   direct = %.6f   diff = %+.2e\n",
            label, w.total_welfare, rhs, err)
end
println("\n=== Sanity: Σ_r W_r = CS_total + Σ PS_r − Σ costs_r ===")
check_sum_identity(w_base, "baseline", p_base.beta)
check_sum_identity(w_alli, "alliance", p_alli.beta)

# ── Display headline numbers ────────────────────────────────────────────────
function print_scenario(label, w, beta)
    println("\n--- $label ---")
    @printf("  Innov rate (per region): (%.4f, %.4f, %.4f)\n",
            w.innov_rate_by_region...)
    @printf("  Enter rate (per region): (%.4f, %.4f, %.4f)\n",
            w.enter_rate_by_region...)
    @printf("  CS / R per region      : %.4f\n",
            (w.cs_p1 + beta * w.cs_p2) / R)
    @printf("  PS_r per region        : (%.4f, %.4f, %.4f)\n",
            w.ps_by_region...)
    @printf("  Costs_r per region     : (%.4f, %.4f, %.4f)\n",
            w.costs_by_region...)
    @printf("  Welfare_r per region   : (%.4f, %.4f, %.4f)\n",
            w.welfare_by_region...)
    @printf("  Σ_r W_r                : %.4f\n", w.total_welfare)
end

print_scenario("Baseline", w_base, p_base.beta)
print_scenario("Alliance {1, 2}", w_alli, p_alli.beta)

println("\n=== Δ (alliance − baseline) ===")
Δ_innov   = ntuple(r -> w_alli.innov_rate_by_region[r] - w_base.innov_rate_by_region[r], R)
Δ_enter   = ntuple(r -> w_alli.enter_rate_by_region[r] - w_base.enter_rate_by_region[r], R)
Δ_ps      = ntuple(r -> w_alli.ps_by_region[r]      - w_base.ps_by_region[r], R)
Δ_costs   = ntuple(r -> w_alli.costs_by_region[r]   - w_base.costs_by_region[r], R)
Δ_welfare = ntuple(r -> w_alli.welfare_by_region[r] - w_base.welfare_by_region[r], R)
Δ_total   = w_alli.total_welfare - w_base.total_welfare
Δ_cs_per  = ((w_alli.cs_p1 + p_alli.beta * w_alli.cs_p2) - (w_base.cs_p1 + p_base.beta * w_base.cs_p2)) / R

# Percent changes relative to baseline (guard against /0 for innov/enter rates)
safepct(num, denom) = denom > 0 ? 100.0 * num / denom : 0.0
pct_innov = ntuple(r -> safepct(Δ_innov[r], w_base.innov_rate_by_region[r]), R)
pct_enter = ntuple(r -> safepct(Δ_enter[r], w_base.enter_rate_by_region[r]), R)
pct_W     = ntuple(r -> safepct(Δ_welfare[r], w_base.welfare_by_region[r]), R)
pct_ΣW    = safepct(Δ_total, w_base.total_welfare)

@printf("  Δ innov rate    : (%+.4f, %+.4f, %+.4f)   [abs.]\n", Δ_innov...)
@printf("  Δ innov / base  : (%+.2f%%, %+.2f%%, %+.2f%%)   [pct]\n", pct_innov...)
@printf("  Δ enter rate    : (%+.4f, %+.4f, %+.4f)   [abs.]\n", Δ_enter...)
@printf("  Δ enter / base  : (%+.2f%%, %+.2f%%, %+.2f%%)   [pct]\n", pct_enter...)
@printf("  Δ CS / R        : %+.4f\n", Δ_cs_per)
@printf("  Δ PS_r          : (%+.4f, %+.4f, %+.4f)\n", Δ_ps...)
@printf("  Δ costs_r       : (%+.4f, %+.4f, %+.4f)\n", Δ_costs...)
@printf("  Δ W_r           : (%+.4f, %+.4f, %+.4f)   [abs.]\n", Δ_welfare...)
@printf("  Δ W_r / W_r0    : (%+.2f%%, %+.2f%%, %+.2f%%)   [pct]\n", pct_W...)
@printf("  Δ ΣW            : %+.4f   (%+.2f%% of baseline ΣW)\n", Δ_total, pct_ΣW)

# ── K-stability diagnostic ─────────────────────────────────────────────────
println("\n=== K-stability of Δ welfare (seed-matched, CRN) ===")
@printf("  %5s | %10s %10s %10s | %10s %10s %10s | %10s\n",
        "K", "ΔW₁", "ΔW₂", "ΔW₃", "ΔW₁ %", "ΔW₂ %", "ΔW₃ %", "ΔΣW %")
for k_test in (500, 1000, 5000)
    wb = expected_welfare_mc(p_base; n_markets = k_test, seed = SEED)
    wa = expected_welfare_mc(p_alli; n_markets = k_test, seed = SEED)
    Δw = ntuple(r -> wa.welfare_by_region[r] - wb.welfare_by_region[r], R)
    pw = ntuple(r -> 100.0 * Δw[r] / wb.welfare_by_region[r], R)
    Δs = wa.total_welfare - wb.total_welfare
    ps = 100.0 * Δs / wb.total_welfare
    @printf("  %5d | %+10.4f %+10.4f %+10.4f | %+9.2f%% %+9.2f%% %+9.2f%% | %+9.2f%%\n",
            k_test, Δw..., pw..., ps)
end

# ── Plot helpers: side-by-side grouped bars at offset x-coords ──────────────
# Plots.bar with a matrix overlays the series rather than dodging them, so we
# place the two series at hand-tuned x-offsets and use bar_width to keep them
# from touching.
const REGION_LABELS = ["Region 1", "Region 2", "Region 3"]
const X_BASE = [0.8, 1.8, 2.8]
const X_ALLI = [1.2, 2.2, 3.2]
const BW     = 0.36

function grouped_bar(values_base, values_alli; ylabel, title, fig_path)
    plt = bar(X_BASE, values_base;
              bar_width = BW, label = "Baseline",
              color = :steelblue, linecolor = :steelblue,
              xlabel = "Region", ylabel = ylabel, title = title,
              xticks = (1:3, REGION_LABELS), xlims = (0.4, 3.6),
              legend = :topright, grid = :y, size = (720, 430))
    bar!(plt, X_ALLI, values_alli;
         bar_width = BW, label = "Alliance \\{1, 2\\}",
         color = :tomato, linecolor = :tomato)
    savefig(plt, fig_path)
    return fig_path
end

# ── Plot 1: innovation rate by region ───────────────────────────────────────
fig_innov_path = grouped_bar(
    collect(w_base.innov_rate_by_region),
    collect(w_alli.innov_rate_by_region);
    ylabel   = "P(innovate | old)",
    title    = "Innovation rate by region under EU–US alliance",
    fig_path = joinpath(OUT_FIG, "merger_innovation.pdf"),
)
println("\nSaved figure: $fig_innov_path")

# ── Plot 2: per-region welfare ──────────────────────────────────────────────
fig_welfare_path = grouped_bar(
    collect(w_base.welfare_by_region),
    collect(w_alli.welfare_by_region);
    ylabel   = "Discounted per-region welfare W_r",
    title    = "Welfare by region under EU–US alliance",
    fig_path = joinpath(OUT_FIG, "merger_welfare.pdf"),
)
println("Saved figure: $fig_welfare_path")

# ── Results table (booktabs) ────────────────────────────────────────────────
fmt(x) = @sprintf("%.4f", x)
sgn(x) = x ≥ 0 ? @sprintf("%+.4f", x) : @sprintf("%.4f", x)

tex = """\\begin{tabular}{lrrrrrr}
\\toprule
& \\multicolumn{2}{c}{Region 1 (US)} & \\multicolumn{2}{c}{Region 2 (EU)} & \\multicolumn{2}{c}{Region 3 (RoW)} \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}
Quantity & Baseline & Alliance & Baseline & Alliance & Baseline & Alliance \\\\
\\midrule
"""

function tex_row(name, base_tup, alli_tup)
    return @sprintf("%s & %s & %s & %s & %s & %s & %s \\\\\n",
                    name,
                    fmt(base_tup[1]), fmt(alli_tup[1]),
                    fmt(base_tup[2]), fmt(alli_tup[2]),
                    fmt(base_tup[3]), fmt(alli_tup[3]))
end

# Helper for Δ% rows (one cell per region, spanning Baseline+Alliance columns)
function tex_pct_row(name, pct_tup)
    return @sprintf("%s & \\multicolumn{2}{c}{%+.2f\\%%} & \\multicolumn{2}{c}{%+.2f\\%%} & \\multicolumn{2}{c}{%+.2f\\%%} \\\\\n",
                    name, pct_tup...)
end

tex *= tex_row("\$P(\\text{innov}\\,|\\,\\text{old})\$",
               w_base.innov_rate_by_region, w_alli.innov_rate_by_region)
tex *= tex_pct_row("\\quad \$\\Delta P / P_0\$", pct_innov)
tex *= tex_row("\$P(\\text{enter}\\,|\\,\\text{pe})\$",
               w_base.enter_rate_by_region, w_alli.enter_rate_by_region)
tex *= tex_pct_row("\\quad \$\\Delta P / P_0\$", pct_enter)
tex *= tex_row("\$\\mathrm{PS}_r\$",
               w_base.ps_by_region, w_alli.ps_by_region)
tex *= tex_row("Costs paid",
               w_base.costs_by_region, w_alli.costs_by_region)
tex *= "\\midrule\n"
tex *= tex_row("\$W_r\$",
               w_base.welfare_by_region, w_alli.welfare_by_region)
tex *= tex_pct_row("\\quad \$\\Delta W_r / W_{r,0}\$", pct_W)
tex *= "\\bottomrule\n\\end{tabular}\n"

table_path = joinpath(OUT_TAB, "merger_results.tex")
open(table_path, "w") do io; write(io, tex); end
println("Saved table: $table_path")

# ── LaTeX macros ────────────────────────────────────────────────────────────
macros = """% Auto-generated by code/scripts/run_merger.jl
% EU–US alliance counterfactual (blocs = (1,1,2)) vs baseline (blocs = (1,2,3))
% K = $N_MARKETS markets, common-random-numbers seed = $SEED
\\newcommand{\\MergerNMarkets}{$N_MARKETS}
\\newcommand{\\MergerSeed}{$SEED}
\\newcommand{\\MergerKappaHat}{$(fmt(KAPPA_HAT))}
\\newcommand{\\MergerPhiHat}{$(fmt(PHI_HAT))}
\\newcommand{\\MergerGammaHat}{$(fmt(GAMMA_HAT[1]))}
% Per-region innovation rate
\\newcommand{\\MergerInnovBaselineROne}{$(fmt(w_base.innov_rate_by_region[1]))}
\\newcommand{\\MergerInnovBaselineRTwo}{$(fmt(w_base.innov_rate_by_region[2]))}
\\newcommand{\\MergerInnovBaselineRThree}{$(fmt(w_base.innov_rate_by_region[3]))}
\\newcommand{\\MergerInnovAllianceROne}{$(fmt(w_alli.innov_rate_by_region[1]))}
\\newcommand{\\MergerInnovAllianceRTwo}{$(fmt(w_alli.innov_rate_by_region[2]))}
\\newcommand{\\MergerInnovAllianceRThree}{$(fmt(w_alli.innov_rate_by_region[3]))}
\\newcommand{\\MergerInnovDeltaROne}{$(sgn(Δ_innov[1]))}
\\newcommand{\\MergerInnovDeltaRTwo}{$(sgn(Δ_innov[2]))}
\\newcommand{\\MergerInnovDeltaRThree}{$(sgn(Δ_innov[3]))}
% Per-region innovation rate, percent change relative to baseline
\\newcommand{\\MergerInnovPctROne}{$(@sprintf("%+.2f", pct_innov[1]))}
\\newcommand{\\MergerInnovPctRTwo}{$(@sprintf("%+.2f", pct_innov[2]))}
\\newcommand{\\MergerInnovPctRThree}{$(@sprintf("%+.2f", pct_innov[3]))}
% Per-region entry rate, percent change relative to baseline
\\newcommand{\\MergerEnterPctROne}{$(@sprintf("%+.2f", pct_enter[1]))}
\\newcommand{\\MergerEnterPctRTwo}{$(@sprintf("%+.2f", pct_enter[2]))}
\\newcommand{\\MergerEnterPctRThree}{$(@sprintf("%+.2f", pct_enter[3]))}
% Per-region welfare
\\newcommand{\\MergerWelfBaselineROne}{$(fmt(w_base.welfare_by_region[1]))}
\\newcommand{\\MergerWelfBaselineRTwo}{$(fmt(w_base.welfare_by_region[2]))}
\\newcommand{\\MergerWelfBaselineRThree}{$(fmt(w_base.welfare_by_region[3]))}
\\newcommand{\\MergerWelfAllianceROne}{$(fmt(w_alli.welfare_by_region[1]))}
\\newcommand{\\MergerWelfAllianceRTwo}{$(fmt(w_alli.welfare_by_region[2]))}
\\newcommand{\\MergerWelfAllianceRThree}{$(fmt(w_alli.welfare_by_region[3]))}
\\newcommand{\\MergerWelfDeltaROne}{$(sgn(Δ_welfare[1]))}
\\newcommand{\\MergerWelfDeltaRTwo}{$(sgn(Δ_welfare[2]))}
\\newcommand{\\MergerWelfDeltaRThree}{$(sgn(Δ_welfare[3]))}
\\newcommand{\\MergerWelfDeltaTotal}{$(sgn(Δ_total))}
% Welfare percent change relative to baseline (formatted with leading sign)
\\newcommand{\\MergerWelfPctROne}{$(@sprintf("%+.2f", pct_W[1]))}
\\newcommand{\\MergerWelfPctRTwo}{$(@sprintf("%+.2f", pct_W[2]))}
\\newcommand{\\MergerWelfPctRThree}{$(@sprintf("%+.2f", pct_W[3]))}
\\newcommand{\\MergerWelfPctTotal}{$(@sprintf("%+.2f", pct_ΣW))}
"""
macro_path = joinpath(OUT_EST, "merger_estimates.txt")
open(macro_path, "w") do io; write(io, macros); end
println("Saved macros: $macro_path")

println("\nDone.")
