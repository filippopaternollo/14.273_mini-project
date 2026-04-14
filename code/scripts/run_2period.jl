"""
run_2period.jl — Solve the 2-period Igami model with agglomeration.

Outputs:
  - CCPs at initial state s0 = (4,1,1,2) for baseline (γ=0) and γ=0.05
  - Comparative statics: P(innovate|old, s0) vs γ  →  output/figures/
  - Key scalars exported as LaTeX macros              →  output/estimates/
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
include(joinpath(@__DIR__, "../src/Igami2017.jl"))
using .Igami2017
import .Igami2017: State, Params, StateCCPs, default_params, solve_2period, cournot_profits  # ← add this
using Plots, Plots.PlotMeasures, Printf, DataFrames

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
# ── Figure 1: CCPs vs γ — one subplot per probability, lines by ρ ─────────────
rho_vals   = [0.0, 0.25, 0.5, 0.75]
rho_colors = [:steelblue, :darkorange, :green, :crimson]
rho_labels = ["ρ = 0.00" "ρ = 0.25" "ρ = 0.50" "ρ = 0.75"]

gamma_grid = 0.0:0.01:0.3
# Matrix: rows = γ values, cols = ρ values
innov_by_rho = zeros(length(gamma_grid), length(rho_vals))
enter_by_rho = zeros(length(gamma_grid), length(rho_vals))

for (j, ρ) in enumerate(rho_vals)
    for (i, γ) in enumerate(gamma_grid)
        _, ccps_γρ = solve_2period(default_params(gamma=γ, rho=ρ))
        c = ccps_γρ[S0]
        innov_by_rho[i, j] = c.p_io
        enter_by_rho[i, j] = c.p_ep
    end
end

γ_vec = collect(gamma_grid)

ax1 = plot(xlabel="Agglomeration γ", ylabel="Probability",
           title="P(innovate | old)", legend=:topright, grid=true)
ax2 = plot(xlabel="Agglomeration γ", ylabel="",
           title="P(enter | pe)", legend=:topright, grid=true)

for (j, ρ) in enumerate(rho_vals)
    plot!(ax1, γ_vec, innov_by_rho[:, j],
          label="ρ = $(rho_vals[j])", color=rho_colors[j], lw=2)
    plot!(ax2, γ_vec, enter_by_rho[:, j],
          label="ρ = $(rho_vals[j])", color=rho_colors[j], lw=2)
end

fig1 = plot(ax1, ax2,
    layout=(1, 2), size=(900, 380),
    plot_title="CCPs vs Agglomeration (s₀ = (4,1,1,2))",
    left_margin=10Plots.mm, bottom_margin=8Plots.mm)


fig1_path = joinpath(OUTPUT_DIR, "figures", "comp_stats_agglomeration.pdf")
savefig(fig1, fig1_path)
println("  Figure saved to: $fig1_path")

# ── Figure 2: CCPs vs ρ — one subplot per probability, lines by γ ─────────────
gamma_vals   = [0.0, 0.05, 0.10, 0.20]
gamma_colors = [:steelblue, :darkorange, :green, :crimson]
gamma_labels = ["γ = 0.00" "γ = 0.05" "γ = 0.10" "γ = 0.20"]

rho_grid = 0.0:0.05:0.95
innov_by_gamma = zeros(length(rho_grid), length(gamma_vals))
enter_by_gamma = zeros(length(rho_grid), length(gamma_vals))

for (j, γ) in enumerate(gamma_vals)
    for (i, ρ) in enumerate(rho_grid)
        _, ccps_ργ = solve_2period(default_params(gamma=γ, rho=ρ))
        c = ccps_ργ[S0]
        innov_by_gamma[i, j] = c.p_io
        enter_by_gamma[i, j] = c.p_ep
    end
end

ρ_vec = collect(rho_grid)

ax3 = plot(xlabel="Substitution ρ", ylabel="Probability",
           title="P(innovate | old)", legend=:topright, grid=true)
ax4 = plot(xlabel="Substitution ρ", ylabel="",
           title="P(enter | pe)", legend=:topright, grid=true)

for (j, γ) in enumerate(gamma_vals)
    plot!(ax3, ρ_vec, innov_by_gamma[:, j],
          label="γ = $(gamma_vals[j])", color=gamma_colors[j], lw=2)
    plot!(ax4, ρ_vec, enter_by_gamma[:, j],
          label="γ = $(gamma_vals[j])", color=gamma_colors[j], lw=2)
end

fig2 = plot(ax3, ax4,
    layout=(1, 2), size=(900, 380),
    plot_title="CCPs vs Cannibalization (s₀ = (4,1,1,2))",
    left_margin=10Plots.mm, bottom_margin=8Plots.mm)
    
fig2_path = joinpath(OUTPUT_DIR, "figures", "comp_stats_cannibalization.pdf")
savefig(fig2, fig2_path)
println("  Figure saved to: $fig2_path")

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
