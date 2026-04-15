"""
simulate_data.jl — Generate a firm-level panel from the 2-period regional
model under `default_params()` and write it to `data/simulated_panel.csv`.

The CSV is intended as synthetic "industry data" for a later estimation
exercise.  Structural parameters (γ, κ, φ, …) are the ground-truth DGP
used to draw actions and are NOT stored in the CSV.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "../src/MiniProject.jl"))
using .MiniProject
using CSV, DataFrames, Printf

# ── Configuration ────────────────────────────────────────────────────────────
const N_MARKETS = 500
const SEED      = 20260414
const S0        = State((1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1))

const DATA_DIR = joinpath(@__DIR__, "../../data")
mkpath(DATA_DIR)
const CSV_PATH = joinpath(DATA_DIR, "simulated_panel.csv")

# ── Simulate ─────────────────────────────────────────────────────────────────
p = default_params()
println("Simulating $N_MARKETS markets from s0 = $(S0)")
@time df = simulate_panel(S0, p; n_markets = N_MARKETS, seed = SEED)

CSV.write(CSV_PATH, df)
println("Wrote $(nrow(df)) rows to $CSV_PATH")

# ── Summary ──────────────────────────────────────────────────────────────────
println("\n=== Summary ===")
println("Firms per period:")
for g in groupby(df, :period)
    @printf("  period %d: %d rows\n", g.period[1], nrow(g))
end

println("\nMean profit by (period, firm_type):")
for g in groupby(sort(df, [:period, :firm_type]), [:period, :firm_type])
    @printf("  period %d  %-8s  n=%4d  mean profit = %.4f\n",
            g.period[1], g.firm_type[1], nrow(g), sum(g.profit)/nrow(g))
end

# ── Sanity check vs solver marginals ─────────────────────────────────────────
println("\n=== Sanity check: innovation rate vs solver marginals ===")
_, ccps0 = solve_initial(S0, p)
period1_old = df[(df.period .== 1) .& (df.firm_type .== "old"), :]
for r in 1:3
    mask = period1_old.region .== r
    n = count(mask)
    emp = n == 0 ? 0.0 : count(period1_old.action[mask] .== "innovate") / n
    thy = ccps0.p_io[r]
    @printf("  region %d: empirical P(innov|old) = %.4f   solver = %.4f   diff = %+.4f\n",
            r, emp, thy, emp - thy)
end

println("\nDone.")
