"""
simulate_data.jl — Generate a firm-level dataset from the 2-period regional
model under `default_params()` and write it to `data/simulated_data.csv`.

Each market draws its own initial state s₀ uniformly over small region-level
counts so the panel spans a variety of local cluster sizes. Structural
parameters (γ, κ, φ, …) are the ground-truth DGP used to draw actions and
are NOT stored in the CSV.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "../src/MiniProject.jl"))
using .MiniProject
using CSV, DataFrames, Printf, Random

# ── Configuration ────────────────────────────────────────────────────────────
const N_MARKETS = 500
const SEED      = 20260414

const DATA_DIR = joinpath(@__DIR__, "../../data")
mkpath(DATA_DIR)
const CSV_PATH = joinpath(DATA_DIR, "simulated_data.csv")

# ── Simulate ─────────────────────────────────────────────────────────────────
p = default_params()
rng = MersenneTwister(SEED)

states = all_states(p.N_max)
V1 = compute_terminal_values(states, p)
pe_ccp_cache      = Dict{Tuple{State,Int}, Float64}()
ev_after_pe_cache = Dict{Tuple{State,Int}, EV}()

println("Simulating $N_MARKETS markets with randomized initial states")
all_rows = NamedTuple[]
s0_by_market = Dict{Int, State}()
@time for m in 1:N_MARKETS
    s0_m = random_s0(rng, p)
    s0_by_market[m] = s0_m
    rows = simulate_market(s0_m, p, rng, V1, pe_ccp_cache, ev_after_pe_cache;
                           market_id = m)
    append!(all_rows, rows)
end
df = DataFrame(all_rows)

CSV.write(CSV_PATH, df)
println("Wrote $(nrow(df)) rows to $CSV_PATH")

# ── Summary ──────────────────────────────────────────────────────────────────
println("\n=== Summary ===")
println("Rows per period:")
for g in groupby(df, :period)
    @printf("  period %d: %d rows\n", g.period[1], nrow(g))
end

println("\nMean profit by (period, firm_type):")
for g in groupby(sort(df, [:period, :firm_type]), [:period, :firm_type])
    @printf("  period %d  %-8s  n=%4d  mean profit = %.4f\n",
            g.period[1], g.firm_type[1], nrow(g), sum(g.profit)/nrow(g))
end

# ── Sanity check vs solver marginals on a representative market ──────────────
println("\n=== Sanity check: innovation rate vs solver marginals ===")
ref_m  = 1
ref_s0 = s0_by_market[ref_m]
println("Reference market $ref_m, s0 = $ref_s0")
_, ccps_ref = solve_initial(ref_s0, p)
ref_old = df[(df.market_id .== ref_m) .& (df.period .== 1) .&
             (df.firm_type .== "old"), :]
for r in 1:R
    mask = ref_old.region .== r
    n = count(mask)
    if n == 0
        @printf("  region %d: no old firms in this market\n", r)
        continue
    end
    emp = count(ref_old.action[mask] .== "innovate") / n
    thy = ccps_ref.p_io[r]
    @printf("  region %d: empirical P(innov|old) = %.4f   solver = %.4f   diff = %+.4f  (n=%d)\n",
            r, emp, thy, emp - thy, n)
end

println("\nDone.")
