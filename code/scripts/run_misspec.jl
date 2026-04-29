"""
run_misspec.jl — Misspecified-model strawman: fit a region intercept on the
new-tech marginal cost while imposing γ ≡ 0, then re-estimate (κ, φ) by MLE.

The "wrong-model" analyst attributes the cross-region cost differences they
see in the Cournot data to exogenous regional fundamentals (factor prices,
geography), not to the agglomeration spillover. NLS fits a region offset
δ_r on top of `c_n0`; with γ ≡ 0 the misspecified Cournot model is
state-independent within a region, so it cannot fit the true model's
within-region cross-state variation — δ̃_r lands near
`−γ̂_r · E[n_b[r] + n_n[r]]`.

The model is then dynamically wrong: continuation values do not respond to
peer density, so MLE produces biased (κ̃, φ̃) and the implied period-1
CCPs miss the agglomeration channel.

The script reports parameter estimates and two sets of moments — innovation
rates by firm type (the "innovator's dilemma" gap) and Lerner-index
markups by generation — averaged over the period-0 states observed in
`data/simulated_data.csv`. Outputs are written to
`output/estimates/misspec_estimates.txt` and
`output/tables/misspec_comparison.tex`.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "../src/MiniProject.jl"))
using .MiniProject
using CSV, DataFrames, Printf, Optim

const DATA_PATH = joinpath(@__DIR__, "../../data/simulated_data.csv")
const EST_PATH  = joinpath(@__DIR__, "../../output/estimates/estimation.txt")
const OUT_EST   = joinpath(@__DIR__, "../../output/estimates")
const OUT_TAB   = joinpath(@__DIR__, "../../output/tables")
mkpath(OUT_EST); mkpath(OUT_TAB)

# ── Read the true estimates (κ̂, φ̂, γ̂) ────────────────────────────────────
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

# ── Load data + base params ───────────────────────────────────────────────
println("Loading $DATA_PATH")
df = CSV.read(DATA_PATH, DataFrame)
markets = recover_markets(df)
println("  $(length(markets)) markets")

p_base = default_params()
@printf("True estimates from %s:\n  κ̂ = %.4f, φ̂ = %.4f, γ̂ = (%.4f, %.4f, %.4f)\n",
        EST_PATH, KAPPA_HAT, PHI_HAT, GAMMA_HAT...)

# ─────────────────────────────────────────────────────────────────────────
# Step 1 (misspec): NLS for δ_r with γ ≡ 0.
# Same Cournot SSR as `nls_gamma`, but the free parameters are the three
# region MC shifters and γ is frozen at zero.
# ─────────────────────────────────────────────────────────────────────────
println("\n=== Misspec Step 1: NLS for δ_r (γ ≡ 0; start δ⁰ = (0,0,0)) ===")

function nls_offset(markets::Vector{MiniProject.MarketData}, p0::Params;
                    delta0::NTuple{3,Float64} = (0.0, 0.0, 0.0))
    function obj(δv::Vector{Float64})
        p = MiniProject._with_params(p0;
                                     gamma = (0.0, 0.0, 0.0),
                                     c_n0_offset = (δv[1], δv[2], δv[3]))
        ss = 0.0
        for m in markets, (t, s) in ((1, m.s0), (2, m.s1))
            cn = c_n_vec(s, p)
            q_oo, q_bo, q_bn, q_nn =
                cournot_quantities_regional(s.n_o, s.n_b, s.n_n, p.c_o, cn, p)
            for r in 1:3
                isnan(m.q_oo[t][r]) || (ss += (m.q_oo[t][r] - q_oo[r])^2)
                isnan(m.q_bo[t][r]) || (ss += (m.q_bo[t][r] - q_bo[r])^2)
                isnan(m.q_bn[t][r]) || (ss += (m.q_bn[t][r] - q_bn[r])^2)
                isnan(m.q_nn[t][r]) || (ss += (m.q_nn[t][r] - q_nn[r])^2)
            end
        end
        return ss
    end
    opts = Optim.Options(show_trace = false, iterations = 2000, g_tol = 1e-12)
    res  = optimize(obj, [delta0[1], delta0[2], delta0[3]], NelderMead(), opts)
    δv   = Optim.minimizer(res)
    return ((δv[1], δv[2], δv[3]), Optim.minimum(res))
end

δ̃, ssr_misspec = nls_offset(markets, p_base)
@printf("  δ̃ = (%.6f, %.6f, %.6f)\n", δ̃...)
@printf("  min SSR = %.3e   (γ̂ NLS at truth ≈ 0; misspec leaves residuals)\n", ssr_misspec)

# ─────────────────────────────────────────────────────────────────────────
# Step 2 (misspec): MLE for (κ, φ) with γ ≡ 0 and δ̃ plugged in.
# ─────────────────────────────────────────────────────────────────────────
println("\n=== Misspec Step 2: MLE for (κ, φ) given γ ≡ 0, δ̃ ===")

p_base_misspec = MiniProject._with_params(p_base; c_n0_offset = δ̃)
κ̃, φ̃, ℓ̃, _ = mle_kappa_phi(markets, (0.0, 0.0, 0.0), p_base_misspec;
                              kappa0 = 1.0, phi0 = 1.0, trace = false)
@printf("  κ̃ = %.4f    φ̃ = %.4f    ℓ̃ = %.3f\n", κ̃, φ̃, ℓ̃)

# ─────────────────────────────────────────────────────────────────────────
# Moments at period-0 states under each model.
#
# True   model: γ = γ̂,  δ = 0,    κ = κ̂, φ = φ̂   (matches the DGP)
# Misspec model: γ = 0,  δ = δ̃,    κ = κ̃, φ = φ̃
# ─────────────────────────────────────────────────────────────────────────
p_true = MiniProject._with_params(p_base;
                                  gamma = GAMMA_HAT,
                                  kappa = KAPPA_HAT, phi = PHI_HAT,
                                  c_n0_offset = (0.0, 0.0, 0.0))
p_miss = MiniProject._with_params(p_base;
                                  gamma = (0.0, 0.0, 0.0),
                                  kappa = κ̃, phi = φ̃,
                                  c_n0_offset = δ̃)

"""
    moments(p, markets) -> NamedTuple

For every market, solve at `m.s0` and compute:
  • Innovation/entry CCPs (firm-weighted across (market, region));
  • Per-market Lerner indices `(P − MC)/P` for old and new generations,
    quantity-weighted across regions within a market.

Returns the firm-weighted innovation/entry rates, the across-market
*mean* and *standard deviation* of each Lerner index, and the comparative
static `Δp_io[1]` from a `+1 BOTH` shock to region 1 — the response of
region-1 OLD firms' innovation rate to a unit increase in same-region
peer density. Markets where the perturbed state would exceed `N_max` are
dropped (state space is bounded).
"""
function moments(p::Params, markets::Vector{MiniProject.MarketData})
    states = all_states(p.N_max)
    V1 = compute_terminal_values(states, p)
    pe_ccp = Dict{Tuple{State,Int},Float64}()
    ev_pe  = Dict{Tuple{State,Int},EV}()

    sum_io = 0.0; n_old_total = 0
    sum_ep = 0.0; n_pe_total  = 0

    lerner_old_per_mkt = Float64[]
    lerner_new_per_mkt = Float64[]

    sum_dpio = 0.0; n_dpio_old = 0
    n_perturb_skipped = 0

    A, B, M, c_o, ρ = p.A, p.B, p.M, p.c_o, p.rho

    for m in markets
        s = m.s0
        ccps = solve_state(s, V1, pe_ccp, ev_pe, p)
        for r in 1:3
            sum_io += s.n_o[r]  * ccps.p_io[r]
            n_old_total += s.n_o[r]
            sum_ep += s.n_pe[r] * ccps.p_ep[r]
            n_pe_total  += s.n_pe[r]
        end

        cn = c_n_vec(s, p)
        q_oo, q_bo, q_bn, q_nn =
            cournot_quantities_regional(s.n_o, s.n_b, s.n_n, p.c_o, cn, p)

        Q_old = 0.0; Q_new = 0.0
        for r in 1:3
            Q_old += s.n_o[r] * q_oo[r] + s.n_b[r] * q_bo[r]
            Q_new += s.n_b[r] * q_bn[r] + s.n_n[r] * q_nn[r]
        end
        P_old = A - B * (Q_old / M) - ρ * B * (Q_new / M)
        P_new = A - B * (Q_new / M) - ρ * B * (Q_old / M)

        if Q_old > 0 && P_old > 0
            num = 0.0
            for r in 1:3
                qo = s.n_o[r] * q_oo[r] + s.n_b[r] * q_bo[r]
                num += qo * (P_old - c_o) / P_old
            end
            push!(lerner_old_per_mkt, num / Q_old)
        end
        if Q_new > 0 && P_new > 0
            num = 0.0
            for r in 1:3
                qn = s.n_b[r] * q_bn[r] + s.n_n[r] * q_nn[r]
                num += qn * (P_new - cn[r]) / P_new
            end
            push!(lerner_new_per_mkt, num / Q_new)
        end

        # Comparative static: +1 BOTH firm in region 1 → response of p_io[1].
        # Region-1 OLD firms see lower c_n,1 under the true model and
        # nothing under misspec. Skip when region 1 has no OLD firms (no
        # one to innovate) or perturbation would exceed N_max.
        if s.n_o[1] >= 1
            tot = sum(s.n_o) + sum(s.n_b) + sum(s.n_n) + sum(s.n_pe)
            if tot + 1 <= p.N_max
                s_p = State(s.n_o,
                            MiniProject.set_i(s.n_b, 1, s.n_b[1] + 1),
                            s.n_n, s.n_pe)
                ccps_p = solve_state(s_p, V1, pe_ccp, ev_pe, p)
                Δ = ccps_p.p_io[1] - ccps.p_io[1]
                sum_dpio   += s.n_o[1] * Δ
                n_dpio_old += s.n_o[1]
            else
                n_perturb_skipped += 1
            end
        end
    end

    mean_(v) = isempty(v) ? NaN : sum(v) / length(v)
    sd_(v)   = isempty(v) ? NaN :
               sqrt(sum((x - mean_(v))^2 for x in v) / length(v))

    return (
        p_io = n_old_total > 0 ? sum_io / n_old_total : NaN,
        p_ep = n_pe_total  > 0 ? sum_ep / n_pe_total  : NaN,
        lerner_old      = mean_(lerner_old_per_mkt),
        lerner_old_sd   = sd_(lerner_old_per_mkt),
        lerner_new      = mean_(lerner_new_per_mkt),
        lerner_new_sd   = sd_(lerner_new_per_mkt),
        dpio_r1         = n_dpio_old > 0 ? sum_dpio / n_dpio_old : NaN,
        n_perturb_skipped = n_perturb_skipped,
    )
end

println("\nComputing moments under each model …")
mom_true = moments(p_true, markets)
mom_miss = moments(p_miss, markets)
gap_true = mom_true.p_ep - mom_true.p_io
gap_miss = mom_miss.p_ep - mom_miss.p_io

println("\n=== Moments at period-0 states ===")
@printf("%-32s %10s %10s\n", "moment", "true", "misspec")
@printf("%-32s %10.4f %10.4f\n", "P(innov | old)",                  mom_true.p_io, mom_miss.p_io)
@printf("%-32s %10.4f %10.4f\n", "P(enter | pe)",                   mom_true.p_ep, mom_miss.p_ep)
@printf("%-32s %10.4f %10.4f\n", "gap = P(enter)−P(innov)",         gap_true,      gap_miss)
@printf("%-32s %10.4f %10.4f\n", "Lerner, old gen. (mean)",         mom_true.lerner_old,    mom_miss.lerner_old)
@printf("%-32s %10.4f %10.4f\n", "Lerner, old gen. (s.d.)",         mom_true.lerner_old_sd, mom_miss.lerner_old_sd)
@printf("%-32s %10.4f %10.4f\n", "Lerner, new gen. (mean)",         mom_true.lerner_new,    mom_miss.lerner_new)
@printf("%-32s %10.4f %10.4f\n", "Lerner, new gen. (s.d.)",         mom_true.lerner_new_sd, mom_miss.lerner_new_sd)
@printf("%-32s %10.4f %10.4f\n", "Δ P(innov | old, r=1) from +1 BOTH r=1",
        mom_true.dpio_r1, mom_miss.dpio_r1)
@printf("  (perturbation skipped in %d / %d markets due to N_max)\n",
        mom_true.n_perturb_skipped, length(markets))

# ─────────────────────────────────────────────────────────────────────────
# LaTeX macros + comparison table
# ─────────────────────────────────────────────────────────────────────────
fmt(x)  = @sprintf("%.4f", x)
fmt2(x) = @sprintf("%.2f", x)

open(joinpath(OUT_EST, "misspec_estimates.txt"), "w") do io
    println(io, "% Auto-generated by code/scripts/run_misspec.jl")
    @printf(io, "\\newcommand{\\MisspecDeltaOne}{%s}\n",   fmt(δ̃[1]))
    @printf(io, "\\newcommand{\\MisspecDeltaTwo}{%s}\n",   fmt(δ̃[2]))
    @printf(io, "\\newcommand{\\MisspecDeltaThree}{%s}\n", fmt(δ̃[3]))
    @printf(io, "\\newcommand{\\MisspecKappa}{%s}\n",      fmt(κ̃))
    @printf(io, "\\newcommand{\\MisspecPhi}{%s}\n",        fmt(φ̃))
    @printf(io, "\\newcommand{\\MisspecLogLik}{%.3f}\n",   ℓ̃)
    @printf(io, "\\newcommand{\\MisspecNlsSsr}{%.2f}\n",   ssr_misspec)
    @printf(io, "\\newcommand{\\TrueInnovRate}{%s}\n",     fmt(mom_true.p_io))
    @printf(io, "\\newcommand{\\MisspecInnovRate}{%s}\n",  fmt(mom_miss.p_io))
    @printf(io, "\\newcommand{\\TrueEntryRate}{%s}\n",     fmt(mom_true.p_ep))
    @printf(io, "\\newcommand{\\MisspecEntryRate}{%s}\n",  fmt(mom_miss.p_ep))
    @printf(io, "\\newcommand{\\TrueDilemmaGap}{%s}\n",    fmt(gap_true))
    @printf(io, "\\newcommand{\\MisspecDilemmaGap}{%s}\n", fmt(gap_miss))
    @printf(io, "\\newcommand{\\TrueLernerOld}{%s}\n",      fmt(mom_true.lerner_old))
    @printf(io, "\\newcommand{\\MisspecLernerOld}{%s}\n",   fmt(mom_miss.lerner_old))
    @printf(io, "\\newcommand{\\TrueLernerOldSd}{%s}\n",    fmt(mom_true.lerner_old_sd))
    @printf(io, "\\newcommand{\\MisspecLernerOldSd}{%s}\n", fmt(mom_miss.lerner_old_sd))
    @printf(io, "\\newcommand{\\TrueLernerNew}{%s}\n",      fmt(mom_true.lerner_new))
    @printf(io, "\\newcommand{\\MisspecLernerNew}{%s}\n",   fmt(mom_miss.lerner_new))
    @printf(io, "\\newcommand{\\TrueLernerNewSd}{%s}\n",    fmt(mom_true.lerner_new_sd))
    @printf(io, "\\newcommand{\\MisspecLernerNewSd}{%s}\n", fmt(mom_miss.lerner_new_sd))
    @printf(io, "\\newcommand{\\TrueDpioROne}{%s}\n",       fmt(mom_true.dpio_r1))
    @printf(io, "\\newcommand{\\MisspecDpioROne}{%s}\n",    fmt(mom_miss.dpio_r1))
end
println("\nWrote $(joinpath(OUT_EST, "misspec_estimates.txt"))")

open(joinpath(OUT_TAB, "misspec_comparison.tex"), "w") do io
    println(io, "% Auto-generated by code/scripts/run_misspec.jl")
    println(io, "\\begin{tabular}{lcc}")
    println(io, "\\toprule")
    println(io, " & True (\$\\widehat{\\gamma}>0\$) & Misspec.\\ (\$\\gamma\\equiv 0\$) \\\\")
    println(io, "\\midrule")
    println(io, "\\multicolumn{3}{l}{\\emph{Estimated parameters}} \\\\")
    @printf(io, "\$\\widehat{\\kappa}\$ & %.4f & %.4f \\\\\n", KAPPA_HAT, κ̃)
    @printf(io, "\$\\widehat{\\phi}\$   & %.4f & %.4f \\\\\n", PHI_HAT,   φ̃)
    @printf(io, "\$\\widehat{\\gamma}_1\$ & %.4f & --- \\\\\n", GAMMA_HAT[1])
    @printf(io, "\$\\widehat{\\gamma}_2\$ & %.4f & --- \\\\\n", GAMMA_HAT[2])
    @printf(io, "\$\\widehat{\\gamma}_3\$ & %.4f & --- \\\\\n", GAMMA_HAT[3])
    @printf(io, "\$\\widehat{\\delta}_1\$ & --- & %.4f \\\\\n", δ̃[1])
    @printf(io, "\$\\widehat{\\delta}_2\$ & --- & %.4f \\\\\n", δ̃[2])
    @printf(io, "\$\\widehat{\\delta}_3\$ & --- & %.4f \\\\\n", δ̃[3])
    println(io, "\\midrule")
    println(io, "\\multicolumn{3}{l}{\\emph{Period-1 actions} (firm-weighted across period-0 states)} \\\\")
    @printf(io, "\$P(\\text{innov}\\mid\\text{old})\$        & %.4f & %.4f \\\\\n",
            mom_true.p_io, mom_miss.p_io)
    @printf(io, "\$P(\\text{enter}\\mid\\text{pe})\$         & %.4f & %.4f \\\\\n",
            mom_true.p_ep, mom_miss.p_ep)
    @printf(io, "\\,\\,Innovator's-dilemma gap                 & %.4f & %.4f \\\\\n",
            gap_true, gap_miss)
    println(io, "\\midrule")
    println(io, "\\multicolumn{3}{l}{\\emph{Lerner index} \$(P-MC)/P\$ \\emph{(across-market mean / s.d.)}} \\\\")
    @printf(io, "Old generation & %.4f \\, / \\, %.4f & %.4f \\, / \\, %.4f \\\\\n",
            mom_true.lerner_old, mom_true.lerner_old_sd,
            mom_miss.lerner_old, mom_miss.lerner_old_sd)
    @printf(io, "New generation & %.4f \\, / \\, %.4f & %.4f \\, / \\, %.4f \\\\\n",
            mom_true.lerner_new, mom_true.lerner_new_sd,
            mom_miss.lerner_new, mom_miss.lerner_new_sd)
    println(io, "\\midrule")
    println(io, "\\multicolumn{3}{l}{\\emph{Comparative static}: response to } \\\\")
    println(io, "\\multicolumn{3}{l}{\\emph{a +1 BOTH-firm shock to region 1 (firm-weighted)}} \\\\")
    @printf(io, "\$\\Delta P(\\text{innov}\\mid\\text{old}, r=1)\$ & %.4f & %.4f \\\\\n",
            mom_true.dpio_r1, mom_miss.dpio_r1)
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end
println("Wrote $(joinpath(OUT_TAB, "misspec_comparison.tex"))")

println("\nDone.")
