"""
run_2period.jl — Solve the 2-period Igami model with agglomeration.

Outputs:
  - CCPs at initial state s0 = (4,1,1,2) for baseline (γ=0) and γ=0.05
  - Comparative statics: P(innovate|old, s0) vs γ  →  output/figures/
  - Key scalars exported as LaTeX macros              →  output/estimates/
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
include(joinpath(@__DIR__, "../src/Igami2017.jl"))
using .Igami2017
using Plots, Printf, DataFrames

# ── Directories ──────────────────────────────────────────────────────────────
const OUTPUT_DIR = joinpath(@__DIR__, "../../output")
mkpath(joinpath(OUTPUT_DIR, "figures"))
mkpath(joinpath(OUTPUT_DIR, "estimates"))
mkpath(joinpath(OUTPUT_DIR, "tables"))

# ── Initial state of interest ────────────────────────────────────────────────
# Inspired by early HDD market: several old incumbents, a few innovative firms
const S0 = State(4, 1, 1, 2)   # (n_o, n_b, n_n, n_pe)

# ── Verification: Cournot check ───────────────────────────────────────────────
println("=== Cournot verification ===")
# rho=0: independent markets, so each market solves standard symmetric Cournot
p_check = default_params(gamma=0.0, rho=0.0)
# Symmetric N-firm new-gen only: q = M*(A-c)/(B*(N+1)), π = B/M * q^2
N_sym = 4; c_sym = 0.5
q_sym_formula = p_check.M * (p_check.A - c_sym) / (p_check.B * (N_sym + 1))
pi_sym_formula = (p_check.B / p_check.M) * q_sym_formula^2
_, _, pi_sym_code = cournot_profits(0, 0, N_sym, p_check.c_o, c_sym, p_check)
@printf("  N=%d new-gen only (c=%.2f, rho=0): formula=%.4f, code=%.4f  [%s]\n",
        N_sym, c_sym, pi_sym_formula, pi_sym_code,
        abs(pi_sym_formula - pi_sym_code) < 1e-10 ? "PASS" : "FAIL")

# Cannibalization check: pi_b(rho=0.5) < pi_b(rho=0) for a "both" firm
p_rho0  = default_params(gamma=0.0, rho=0.0)
p_rho05 = default_params(gamma=0.0, rho=0.5)
_, pi_b_rho0,  _ = cournot_profits(2, 1, 2, p_rho0.c_o,  p_rho0.c_n0,  p_rho0)
_, pi_b_rho05, _ = cournot_profits(2, 1, 2, p_rho05.c_o, p_rho05.c_n0, p_rho05)
@printf("  Cannibalization: pi_b(rho=0)=%.4f, pi_b(rho=0.5)=%.4f  [%s]\n",
        pi_b_rho0, pi_b_rho05,
        pi_b_rho05 < pi_b_rho0 ? "PASS (cannibalization reduces pi_b)" : "FAIL")

# ── Solve baseline (γ = 0, ρ = 0.5) ─────────────────────────────────────────
println("\n=== Baseline model (γ = 0, ρ = 0.5) ===")
p0   = default_params(gamma=0.0, rho=0.5)
V1_0, ccps_0 = solve_2period(p0)
c0   = ccps_0[S0]
@printf("  s0 = (%d,%d,%d,%d)\n", S0.n_o, S0.n_b, S0.n_n, S0.n_pe)
@printf("  P(innovate|old) = %.4f\n", c0.p_io)
@printf("  P(stay|old)     = %.4f\n", c0.p_so)
@printf("  P(enter|pe)     = %.4f\n", c0.p_ep)

# ── Solve agglomeration model (γ = 0.05, ρ = 0.5) ────────────────────────────
println("\n=== Agglomeration model (γ = 0.05, ρ = 0.5) ===")
p1   = default_params(gamma=0.05, rho=0.5)
V1_1, ccps_1 = solve_2period(p1)
c1   = ccps_1[S0]
@printf("  P(innovate|old) = %.4f\n", c1.p_io)
@printf("  P(stay|old)     = %.4f\n", c1.p_so)
@printf("  P(enter|pe)     = %.4f\n", c1.p_ep)

# ── Comparative statics: P(innovate|old, s0) vs γ ────────────────────────────
println("\n=== Comparative statics ===")
gamma_grid = 0.0:0.01:0.3
p_innov = Float64[]
p_enter = Float64[]

for γ in gamma_grid
    p_γ = default_params(gamma=γ, rho=0.5)
    _, ccps_γ = solve_2period(p_γ)
    c_γ = ccps_γ[S0]
    push!(p_innov, c_γ.p_io)
    push!(p_enter, c_γ.p_ep)
end

# ── Plot ──────────────────────────────────────────────────────────────────────
plt = plot(
    collect(gamma_grid), [p_innov p_enter],
    label  = ["P(innovate | old)" "P(enter | pe)"],
    xlabel = "Agglomeration parameter γ",
    ylabel = "Probability",
    title  = "CCPs vs Agglomeration (s₀ = (4,1,1,2))",
    lw     = 2,
    legend = :right,
    grid   = true
)

fig_path = joinpath(OUTPUT_DIR, "figures", "comp_stats_agglomeration.pdf")
savefig(plt, fig_path)
println("  Figure saved to: $fig_path")

# ── Comparative statics: P(innovate|old, s0) vs ρ (cannibalization) ──────────
println("\n=== Comparative statics vs ρ (cannibalization) ===")
rho_grid   = 0.0:0.05:0.95
p_innov_rho = Float64[]
p_enter_rho = Float64[]

for ρ in rho_grid
    p_ρ = default_params(gamma=0.05, rho=ρ)
    _, ccps_ρ = solve_2period(p_ρ)
    c_ρ = ccps_ρ[S0]
    push!(p_innov_rho, c_ρ.p_io)
    push!(p_enter_rho, c_ρ.p_ep)
end

plt_rho = plot(
    collect(rho_grid), [p_innov_rho p_enter_rho],
    label  = ["P(innovate | old)" "P(enter | pe)"],
    xlabel = "Substitution parameter ρ",
    ylabel = "Probability",
    title  = "CCPs vs Cannibalization (s₀ = (4,1,1,2), γ = 0.05)",
    lw     = 2,
    legend = :right,
    grid   = true
)

fig_rho_path = joinpath(OUTPUT_DIR, "figures", "comp_stats_cannibalization.pdf")
savefig(plt_rho, fig_rho_path)
println("  Figure saved to: $fig_rho_path")

# ── Summary table ─────────────────────────────────────────────────────────────
df = DataFrame(
    gamma     = collect(gamma_grid)[1:5:end],
    p_innov   = p_innov[1:5:end],
    p_enter   = p_enter[1:5:end]
)
println("\n  Sample of comparative statics:")
println("  ", df)

# Export LaTeX table
tex_table = """\\begin{tabular}{ccc}
\\hline
\$\\gamma\$ & \$P(\\text{innovate} | \\text{old})\$ & \$P(\\text{enter} | \\text{pe})\$ \\\\
\\hline
"""
for row in eachrow(df)
    global tex_table *= @sprintf("%.2f & %.4f & %.4f \\\\\n", row.gamma, row.p_innov, row.p_enter)
end
tex_table *= "\\hline\n\\end{tabular}"

table_path = joinpath(OUTPUT_DIR, "tables", "comp_stats_agglomeration.tex")
open(table_path, "w") do f
    write(f, tex_table)
end
println("\n  Table saved to: $table_path")

# ── Export key scalars as LaTeX macros ────────────────────────────────────────
macros = """% 2-period Igami model with agglomeration — key estimates
% Generated by run_2period.jl
\\newcommand{\\pInnovatBaseline}{$(round(ccps_0[S0].p_io, digits=4))}
\\newcommand{\\pInnovatAgglom}{$(round(ccps_1[S0].p_io, digits=4))}
\\newcommand{\\pEnterBaseline}{$(round(ccps_0[S0].p_ep, digits=4))}
\\newcommand{\\pEnterAgglom}{$(round(ccps_1[S0].p_ep, digits=4))}
\\newcommand{\\gammaAgglom}{0.05}
"""

macro_path = joinpath(OUTPUT_DIR, "estimates", "2period_estimates.txt")
open(macro_path, "w") do f
    write(f, macros)
end
println("  Macros saved to: $macro_path")

println("\nDone.")
