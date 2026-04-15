"""
Two-step estimator for the 2-period regional Igami model.

Step 1 (NLS): estimate γ from deterministic Cournot quantities.
Step 2 (MLE): estimate (κ,φ) from period-1 discrete actions, replaying the
              sequential stages (old → both → new → pe, regions 1..R within
              each stage) exactly as `simulate_market` does so that every
              decision is evaluated at the sub-state its firm actually faced.

Entry points:
  recover_markets(df)           -> Vector{MarketData}
  nls_gamma(markets, p_base)    -> (γ̂, objective)
  mle_kappa_phi(markets, γ̂, p_base) -> (κ̂, φ̂, ℓ̂, ses, result)

A `MarketData` object stores the inferred s0, s1, per-region action counts,
and the observed mean quantities by (period, region, slot).
"""

using DataFrames
using Optim
using LinearAlgebra

# ---------------------------------------------------------------------------
# Per-market data container
# ---------------------------------------------------------------------------
struct MarketData
    s0::State
    s1::State
    # Period-1 action counts by region (length R)
    k_so::NTuple{R,Int}  # stay | old
    k_io::NTuple{R,Int}  # innovate | old
    k_eo::NTuple{R,Int}  # exit | old
    k_sb::NTuple{R,Int}  # stay | both
    k_eb::NTuple{R,Int}  # exit | both
    k_sn::NTuple{R,Int}  # stay | new
    k_en::NTuple{R,Int}  # exit | new
    k_ep::NTuple{R,Int}  # enter | pe
    k_op::NTuple{R,Int}  # stay_out | pe
    # Observed mean quantities by period, region, slot (NaN if empty cell)
    q_oo::NTuple{2,NTuple{R,Float64}}  # old firms, old-gen
    q_bo::NTuple{2,NTuple{R,Float64}}  # both firms, old-gen
    q_bn::NTuple{2,NTuple{R,Float64}}  # both firms, new-gen
    q_nn::NTuple{2,NTuple{R,Float64}}  # new firms, new-gen
end

# ---------------------------------------------------------------------------
# State & action recovery from the CSV
# ---------------------------------------------------------------------------
function _counts_by_region(sub::AbstractDataFrame, key::Symbol, value::String)
    c = [0, 0, 0]
    for row in eachrow(sub)
        if getproperty(row, key) == value
            c[row.region] += 1
        end
    end
    return (c[1], c[2], c[3])
end

function _mean_q_by_region(sub::AbstractDataFrame, ftype::String, qcol::Symbol)
    s = [0.0, 0.0, 0.0]; n = [0, 0, 0]
    for row in eachrow(sub)
        row.firm_type == ftype || continue
        r = row.region
        s[r] += getproperty(row, qcol)
        n[r] += 1
    end
    return ntuple(r -> n[r] == 0 ? NaN : s[r] / n[r], R)
end

function recover_markets(df::DataFrame)::Vector{MarketData}
    markets = MarketData[]
    for sub in groupby(df, :market_id)
        p1 = sub[sub.period .== 1, :]
        p2 = sub[sub.period .== 2, :]

        n_o  = _counts_by_region(p1, :firm_type, "old")
        n_b  = _counts_by_region(p1, :firm_type, "both")
        n_n  = _counts_by_region(p1, :firm_type, "new")
        n_pe = _counts_by_region(p1, :firm_type, "pe")
        s0 = State(n_o, n_b, n_n, n_pe)

        n_o1 = _counts_by_region(p2, :firm_type, "old")
        n_b1 = _counts_by_region(p2, :firm_type, "both")
        n_n1 = _counts_by_region(p2, :firm_type, "new")
        s1 = State(n_o1, n_b1, n_n1, (0, 0, 0))

        # Action counts (period 1 only)
        p1_old  = p1[p1.firm_type .== "old",  :]
        p1_both = p1[p1.firm_type .== "both", :]
        p1_new  = p1[p1.firm_type .== "new",  :]
        p1_pe   = p1[p1.firm_type .== "pe",   :]

        k_so = _counts_by_region(p1_old,  :action, "stay")
        k_io = _counts_by_region(p1_old,  :action, "innovate")
        k_eo = _counts_by_region(p1_old,  :action, "exit")
        k_sb = _counts_by_region(p1_both, :action, "stay")
        k_eb = _counts_by_region(p1_both, :action, "exit")
        k_sn = _counts_by_region(p1_new,  :action, "stay")
        k_en = _counts_by_region(p1_new,  :action, "exit")
        k_ep = _counts_by_region(p1_pe,   :action, "enter")
        k_op = _counts_by_region(p1_pe,   :action, "stay_out")

        q_oo = (_mean_q_by_region(p1, "old",  :q_old),
                _mean_q_by_region(p2, "old",  :q_old))
        q_bo = (_mean_q_by_region(p1, "both", :q_old),
                _mean_q_by_region(p2, "both", :q_old))
        q_bn = (_mean_q_by_region(p1, "both", :q_new),
                _mean_q_by_region(p2, "both", :q_new))
        q_nn = (_mean_q_by_region(p1, "new",  :q_new),
                _mean_q_by_region(p2, "new",  :q_new))

        push!(markets, MarketData(s0, s1,
            k_so, k_io, k_eo, k_sb, k_eb, k_sn, k_en, k_ep, k_op,
            q_oo, q_bo, q_bn, q_nn))
    end
    return markets
end

# ---------------------------------------------------------------------------
# Helper: build a Params from a base + fresh (κ, φ, γ)
# ---------------------------------------------------------------------------
function _with_params(p::Params; kappa = p.kappa, phi = p.phi,
                      gamma::NTuple{R,Float64} = p.gamma)
    return Params(p.A, p.B, p.M, p.c_o, p.c_n0, p.beta,
                  kappa, phi, p.sigma, gamma, p.rho, p.N_max)
end

# ---------------------------------------------------------------------------
# Step 1: NLS for γ
# ---------------------------------------------------------------------------
"""
    nls_gamma(markets, p_base; gamma0 = (0.0,0.0,0.0))

Minimize Σ (q_obs − q_hat(γ))² over all market/period/region/slot cells with
observations. Returns `(γ̂, objective_value)`.
"""
function nls_gamma(markets::Vector{MarketData}, p_base::Params;
                   gamma0::NTuple{R,Float64} = (0.0, 0.0, 0.0),
                   trace::Bool = true)

    function _predict(s::State, p::Params)
        cn = c_n_vec(s, p)
        return cournot_quantities_regional(s.n_o, s.n_b, s.n_n, p.c_o, cn, p)
    end

    function obj(gv::Vector{Float64})
        p = _with_params(p_base; gamma = (gv[1], gv[2], gv[3]))
        ss = 0.0
        for m in markets
            for (t, s) in ((1, m.s0), (2, m.s1))
                q_oo, q_bo, q_bn, q_nn = _predict(s, p)
                for r in 1:R
                    if !isnan(m.q_oo[t][r])
                        ss += (m.q_oo[t][r] - q_oo[r])^2
                    end
                    if !isnan(m.q_bo[t][r])
                        ss += (m.q_bo[t][r] - q_bo[r])^2
                    end
                    if !isnan(m.q_bn[t][r])
                        ss += (m.q_bn[t][r] - q_bn[r])^2
                    end
                    if !isnan(m.q_nn[t][r])
                        ss += (m.q_nn[t][r] - q_nn[r])^2
                    end
                end
            end
        end
        return ss
    end

    opts = Optim.Options(show_trace = trace, iterations = 2000, g_tol = 1e-12)
    res = optimize(obj, [gamma0[1], gamma0[2], gamma0[3]], NelderMead(), opts)
    γv = Optim.minimizer(res)
    return ((γv[1], γv[2], γv[3]), Optim.minimum(res))
end

# ---------------------------------------------------------------------------
# Step 2: per-market log-likelihood (stage replay)
# ---------------------------------------------------------------------------
"""
    loglik_market(m, p, V1)

Evaluate the period-1 action log-likelihood for a single market by replaying
the stage order used by `simulate_market`. `V1` must be a terminal-value
dictionary already built under the same γ as `p`. Fresh per-market caches
are created (only the PE-stage caches are safe to share across markets, but
we rebuild them here for simplicity).
"""
function loglik_market(m::MarketData, p::Params, V1::Dict{State,EV})
    s = m.s0
    cn = c_n_vec(s, p)
    pi_o, pi_b, pi_n =
        cournot_profits_regional(s.n_o, s.n_b, s.n_n, p.c_o, cn, p)
    ctx = SolveContext(s, pi_o, pi_b, pi_n)
    C = SolveCaches(
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, EV}(),
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, EV}(),
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, EV}(),
        Dict{Tuple{State,Int}, Tuple{Float64,Float64}}(),
        Dict{Tuple{State,Int}, EV}(),
    )

    ll = 0.0
    # Clamp probabilities into [eps, 1 - eps] before taking logs so the
    # likelihood stays well-scaled (log(eps()) ≈ -36) and finite-difference
    # gradients remain numerically meaningful even when a stage CCP hits a
    # boundary. This avoids the magic-constant penalty of a hand-coded
    # sentinel and keeps BHHH SEs well-defined.
    _safelog(x) = log(clamp(x, eps(Float64), 1.0 - eps(Float64)))

    # OLD stage
    for r in 1:R
        s.n_o[r] == 0 && continue
        p_s, p_i = solve_old_region(s, r, ctx, V1, p, C)
        p_e = max(0.0, 1.0 - p_s - p_i)
        ll += m.k_so[r] * _safelog(p_s)
        ll += m.k_io[r] * _safelog(p_i)
        ll += m.k_eo[r] * _safelog(p_e)
        s = State(set_i(s.n_o, r, m.k_so[r]),
                  add_i(s.n_b, r, m.k_io[r]),
                  s.n_n, s.n_pe)
    end

    # BOTH stage (original both firms only — uses ctx.s_orig.n_b[r])
    for r in 1:R
        ctx.s_orig.n_b[r] == 0 && continue
        p_sb = solve_both_region(s, r, ctx, V1, p, C)
        k_sb = m.k_sb[r]; k_eb = m.k_eb[r]
        ll += k_sb * _safelog(p_sb)
        ll += k_eb * _safelog(1.0 - p_sb)
        exiters = k_eb
        s = State(s.n_o, add_i(s.n_b, r, -exiters), s.n_n, s.n_pe)
    end

    # NEW stage (original new firms only — uses sub-state n_n[r])
    for r in 1:R
        s.n_n[r] == 0 && continue
        p_sn = solve_new_region(s, r, ctx, V1, p, C)
        k_sn = m.k_sn[r]; k_en = m.k_en[r]
        ll += k_sn * _safelog(p_sn)
        ll += k_en * _safelog(1.0 - p_sn)
        s = State(s.n_o, s.n_b, set_i(s.n_n, r, k_sn), s.n_pe)
    end

    # PE stage
    for r in 1:R
        s.n_pe[r] == 0 && continue
        p_ep = solve_pe_region(s, r, V1, p, C)
        k_ep = m.k_ep[r]; k_op = m.k_op[r]
        ll += k_ep * _safelog(p_ep)
        ll += k_op * _safelog(1.0 - p_ep)
        s = State(s.n_o, s.n_b,
                  add_i(s.n_n, r, k_ep),
                  add_i(s.n_pe, r, -(k_ep + k_op)))
    end

    return ll
end

"""
    loglik_actions(markets, p)

Sum of per-market log-likelihoods for the period-1 actions.
"""
function loglik_actions(markets::Vector{MarketData}, p::Params)
    states = all_states(p.N_max)
    V1 = compute_terminal_values(states, p)
    ll = 0.0
    for m in markets
        ll += loglik_market(m, p, V1)
    end
    return ll
end

# ---------------------------------------------------------------------------
# Step 2: MLE for (κ, φ) given γ̂
# ---------------------------------------------------------------------------
"""
    mle_kappa_phi(markets, γ̂, p_base; kappa0, phi0, trace)

Maximize the period-1 action log-likelihood over (κ, φ) by NelderMead,
holding γ fixed at `γ̂`. Returns `(κ̂, φ̂, ℓ̂, result)`.
"""
function mle_kappa_phi(markets::Vector{MarketData}, γ̂::NTuple{R,Float64},
                       p_base::Params;
                       kappa0::Float64 = 1.0, phi0::Float64 = 1.0,
                       trace::Bool = true)

    states = all_states(p_base.N_max)
    function neg_ll(θ::Vector{Float64})
        κ = θ[1]; φ = θ[2]
        p = _with_params(p_base; kappa = κ, phi = φ, gamma = γ̂)
        V1 = compute_terminal_values(states, p)
        ll = 0.0
        for m in markets
            ll += loglik_market(m, p, V1)
        end
        return -ll
    end

    opts = Optim.Options(show_trace = trace, iterations = 1000,
                         g_tol = 1e-8, f_abstol = 1e-10)
    res = optimize(neg_ll, [kappa0, phi0], NelderMead(), opts)
    θ̂ = Optim.minimizer(res)
    return (θ̂[1], θ̂[2], -Optim.minimum(res), res)
end

# ---------------------------------------------------------------------------
# BHHH standard errors via finite-difference gradients of per-market ℓ_m
# ---------------------------------------------------------------------------
"""
    bhhh_se(markets, p_hat; step)

Outer-product-of-gradients standard errors for (κ, φ) at `p_hat`. γ is held
fixed. Uses central finite differences on per-market log-likelihoods.
"""
function bhhh_se(markets::Vector{MarketData}, p_hat::Params; step::Float64 = 1e-4)
    states = all_states(p_hat.N_max)

    function ll_per_market(κ::Float64, φ::Float64)
        p = _with_params(p_hat; kappa = κ, phi = φ)
        V1 = compute_terminal_values(states, p)
        return [loglik_market(m, p, V1) for m in markets]
    end

    κ0 = p_hat.kappa; φ0 = p_hat.phi
    lκp = ll_per_market(κ0 + step, φ0)
    lκm = ll_per_market(κ0 - step, φ0)
    lφp = ll_per_market(κ0, φ0 + step)
    lφm = ll_per_market(κ0, φ0 - step)

    M = length(markets)
    g = zeros(M, 2)
    for i in 1:M
        g[i, 1] = (lκp[i] - lκm[i]) / (2 * step)
        g[i, 2] = (lφp[i] - lφm[i]) / (2 * step)
    end
    B = g' * g
    V = inv(B)
    return (se_kappa = sqrt(V[1, 1]), se_phi = sqrt(V[2, 2]), cov = V)
end
