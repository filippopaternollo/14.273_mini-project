"""
Constrained social planner counterfactual for the regional Igami model.

The planner uses the same logit form firms use, but with the social
marginal value substituted for the private one (plug-in design choice
(a) in `notes/planner_counterfactual.md`):

    π_planner = Λ((ΔW_social − cost) / σ),

with  ΔW_social = E[Σ_r W_r | own acts] − E[Σ_r W_r | own does NOT act]
computed by backward induction on the same OLD → BOTH → NEW → PE
sub-stage tree as `solver.jl`.

Two static-production conventions are supported:

  - `competitive_static = false` — Cournot in both periods (dynamic-only
                                   first best; isolates the spillover
                                   internalization channel).
  - `competitive_static = true`  — P = MC in both periods (full first
                                   best; adds the static markup
                                   correction).

Period-1 social welfare at `s_0` is independent of all stage decisions
(production happens at `s_0` in this model — see `simulate.jl:91-96`),
so it cancels in every ΔW_social.  The FP machinery therefore only
needs the period-2 terminal social value
`V1_social[s] = CS(s) + Σ PS_r(s)` (or `CS(s)` alone under
`competitive_static = true`).
"""

using Random

# ---------------------------------------------------------------------------
# Static block: competitive (P = MC) outcome
# ---------------------------------------------------------------------------

"""
    competitive_outcome(s, p) → (q_oo, q_bo, q_bn, q_nn, Q_o, Q_n)

Closed-form P=MC outcome of the global two-market system.  With constant
marginal cost and homogeneous goods, only the lowest-`c_n[r]` *active*
new-gen region produces; old-gen output is split symmetrically across
all active old-tech firms (old- and both-types share the same `c_o`).

Demand inverts to a 2×2 linear system at P_o = c_o, P_n = c_n_min.
Returns per-firm quantities (zeros for inactive slots) and total
market quantities `(Q_o, Q_n)` consistent with those quantities.
"""
function competitive_outcome(s::State, p::Params)
    cn  = c_n_vec(s, p)
    b_  = p.B / p.M
    s_  = p.rho * p.B / p.M

    active_old = ntuple(r -> s.n_o[r] + s.n_b[r] > 0, R)
    active_new = ntuple(r -> s.n_b[r] + s.n_n[r] > 0, R)
    n_old_active = sum(s.n_o) + sum(s.n_b)
    has_old = n_old_active > 0
    has_new = any(active_new)

    cn_min = +Inf; r_star = 0
    if has_new
        for r in 1:R
            if active_new[r] && cn[r] < cn_min
                cn_min = cn[r]
                r_star = r
            end
        end
    end

    Q_o = 0.0; Q_n = 0.0
    if has_old && has_new
        # A − b·Q_o − s·Q_n = c_o
        # A − b·Q_n − s·Q_o = c_n_min
        det = b_*b_ - s_*s_
        Q_o = (b_ * (p.A - p.c_o) - s_ * (p.A - cn_min)) / det
        Q_n = (b_ * (p.A - cn_min) - s_ * (p.A - p.c_o)) / det
    elseif has_old
        Q_o = (p.A - p.c_o) / b_
    elseif has_new
        Q_n = (p.A - cn_min) / b_
    end
    Q_o = max(Q_o, 0.0)
    Q_n = max(Q_n, 0.0)

    q_oo_v = zeros(R); q_bo_v = zeros(R)
    q_bn_v = zeros(R); q_nn_v = zeros(R)

    if has_old && Q_o > 0.0
        per_firm_o = Q_o / n_old_active
        for r in 1:R
            s.n_o[r] > 0 && (q_oo_v[r] = per_firm_o)
            s.n_b[r] > 0 && (q_bo_v[r] = per_firm_o)
        end
    end
    if has_new && r_star > 0 && Q_n > 0.0
        n_n_at_star = s.n_b[r_star] + s.n_n[r_star]
        per_firm_n = Q_n / n_n_at_star
        s.n_b[r_star] > 0 && (q_bn_v[r_star] = per_firm_n)
        s.n_n[r_star] > 0 && (q_nn_v[r_star] = per_firm_n)
    end

    return (Tuple(q_oo_v), Tuple(q_bo_v), Tuple(q_bn_v), Tuple(q_nn_v), Q_o, Q_n)
end

"""
    social_welfare_static(s, p; competitive) → (cs, ps_by_region)

Single-period social welfare components at state `s`.  If
`competitive=false`, Cournot CS + per-region Cournot PS.  If `true`,
competitive (P=MC) CS and zero PS by construction.
"""
function social_welfare_static(s::State, p::Params; competitive::Bool)
    if competitive
        _, _, _, _, Q_o, Q_n = competitive_outcome(s, p)
        cs = consumer_surplus_from_quantities(Q_o, Q_n, p)
        return (cs, ntuple(_ -> 0.0, R))
    else
        return (consumer_surplus(s, p), producer_surplus_by_region(s, p))
    end
end

# ---------------------------------------------------------------------------
# Terminal V1 (period-2 social welfare per state)
# ---------------------------------------------------------------------------

"""
    compute_terminal_planner_values(states, p; competitive_static) → Dict{State,Float64}

Period-2 terminal social welfare per state.  Used as the boundary value
at the bottom of the sub-stage backward induction.
"""
function compute_terminal_planner_values(states::Vector{State}, p::Params;
                                         competitive_static::Bool)
    V1 = Dict{State, Float64}()
    for s in states
        cs, ps_r = social_welfare_static(s, p; competitive = competitive_static)
        V1[s] = cs + sum(ps_r)
    end
    return V1
end

# ---------------------------------------------------------------------------
# Caches (planner)
# ---------------------------------------------------------------------------

"""
    SolveCachesPlanner

Mirrors `SolveCaches` in `solver.jl`.  PE caches are global (depend only
on `(s, r, V1_social, p)`); NEW/BOTH/OLD caches are per-`s_orig` (BOTH
and OLD use `s_orig.n_b[r]` / `s_orig.n_o[r]` as the decider count, so
their fixed points share state across `s_orig`).
"""
mutable struct SolveCachesPlanner
    pe_W::Dict{Tuple{State,Int}, Float64}
    pe_ccp::Dict{Tuple{State,Int}, Float64}
    new_W::Dict{Tuple{State,Int}, Float64}
    new_ccp::Dict{Tuple{State,Int}, Float64}
    both_W::Dict{Tuple{State,Int}, Float64}
    both_ccp::Dict{Tuple{State,Int}, Float64}
    old_W::Dict{Tuple{State,Int}, Float64}
    old_ccp::Dict{Tuple{State,Int}, Tuple{Float64,Float64}}
end

# ---------------------------------------------------------------------------
# PE stage (last mover, global cache)
# ---------------------------------------------------------------------------
function solve_pe_region_planner(s::State, r::Int,
                                  V1::Dict{State,Float64}, p::Params,
                                  C::SolveCachesPlanner;
                                  tol::Float64 = 1e-10, max_iter::Int = 500)
    key = (s, r)
    haskey(C.pe_ccp, key) && return C.pe_ccp[key]
    n = s.n_pe[r]
    if n == 0
        C.pe_ccp[key] = 0.0
        return 0.0
    end
    p_r = 0.5
    for _ in 1:max_iter
        W_act = -p.phi
        W_no  = 0.0
        for v in 0:(n - 1)
            lp = log_binomial_prob(n - 1, v, p_r)
            lp == -Inf && continue
            prob = exp(lp)
            s_act = State(s.n_o, s.n_b,
                          add_i(s.n_n, r, v + 1),
                          set_i(s.n_pe, r, 0))
            s_no  = State(s.n_o, s.n_b,
                          add_i(s.n_n, r, v),
                          set_i(s.n_pe, r, 0))
            W_act += p.beta * prob * W_after_pe_region_planner(s_act, r + 1, V1, p, C)
            W_no  += p.beta * prob * W_after_pe_region_planner(s_no,  r + 1, V1, p, C)
        end
        new_p = logit2(W_act, W_no, p.sigma)
        diff = abs(new_p - p_r)
        p_r = new_p
        diff < tol && break
    end
    C.pe_ccp[key] = p_r
    return p_r
end

function W_after_pe_region_planner(s::State, r::Int,
                                    V1::Dict{State,Float64}, p::Params,
                                    C::SolveCachesPlanner)
    if r > R
        return get(V1, s, 0.0)
    end
    key = (s, r)
    haskey(C.pe_W, key) && return C.pe_W[key]
    n = s.n_pe[r]
    if n == 0
        W = W_after_pe_region_planner(s, r + 1, V1, p, C)
        C.pe_W[key] = W
        return W
    end
    p_r = solve_pe_region_planner(s, r, V1, p, C)
    W = 0.0
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(s.n_o, s.n_b,
                       add_i(s.n_n, r, v),
                       set_i(s.n_pe, r, 0))
        W += prob * W_after_pe_region_planner(s_next, r + 1, V1, p, C)
    end
    C.pe_W[key] = W
    return W
end

# ---------------------------------------------------------------------------
# NEW stage
# ---------------------------------------------------------------------------
function solve_new_region_planner(s::State, r::Int,
                                   V1::Dict{State,Float64}, p::Params,
                                   C::SolveCachesPlanner;
                                   tol::Float64 = 1e-10, max_iter::Int = 500)
    key = (s, r)
    haskey(C.new_ccp, key) && return C.new_ccp[key]
    n = s.n_n[r]
    if n == 0
        C.new_ccp[key] = 0.0
        return 0.0
    end
    p_r = 0.5
    for _ in 1:max_iter
        W_stay = 0.0; W_exit = 0.0
        for v in 0:(n - 1)
            lp = log_binomial_prob(n - 1, v, p_r)
            lp == -Inf && continue
            prob = exp(lp)
            s_stay = State(s.n_o, s.n_b, set_i(s.n_n, r, v + 1), s.n_pe)
            s_exit = State(s.n_o, s.n_b, set_i(s.n_n, r, v),     s.n_pe)
            W_stay += p.beta * prob * W_after_new_region_planner(s_stay, r + 1, V1, p, C)
            W_exit += p.beta * prob * W_after_new_region_planner(s_exit, r + 1, V1, p, C)
        end
        new_p = logit2(W_stay, W_exit, p.sigma)
        diff = abs(new_p - p_r)
        p_r = new_p
        diff < tol && break
    end
    C.new_ccp[key] = p_r
    return p_r
end

function W_after_new_region_planner(s::State, r::Int,
                                     V1::Dict{State,Float64}, p::Params,
                                     C::SolveCachesPlanner)
    if r > R
        return W_after_pe_region_planner(s, 1, V1, p, C)
    end
    key = (s, r)
    haskey(C.new_W, key) && return C.new_W[key]
    n = s.n_n[r]
    if n == 0
        W = W_after_new_region_planner(s, r + 1, V1, p, C)
        C.new_W[key] = W
        return W
    end
    p_r = solve_new_region_planner(s, r, V1, p, C)
    W = 0.0
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(s.n_o, s.n_b, set_i(s.n_n, r, v), s.n_pe)
        W += prob * W_after_new_region_planner(s_next, r + 1, V1, p, C)
    end
    C.new_W[key] = W
    return W
end

# ---------------------------------------------------------------------------
# BOTH stage — deciders are s_orig.n_b[r]
# ---------------------------------------------------------------------------
function solve_both_region_planner(s::State, r::Int, s_orig::State,
                                    V1::Dict{State,Float64}, p::Params,
                                    C::SolveCachesPlanner;
                                    tol::Float64 = 1e-10, max_iter::Int = 500)
    key = (s, r)
    haskey(C.both_ccp, key) && return C.both_ccp[key]
    n = s_orig.n_b[r]
    if n == 0
        C.both_ccp[key] = 0.0
        return 0.0
    end
    p_r = 0.5
    for _ in 1:max_iter
        W_stay = 0.0; W_exit = 0.0
        for v in 0:(n - 1)
            lp = log_binomial_prob(n - 1, v, p_r)
            lp == -Inf && continue
            prob = exp(lp)
            exiters_stay = n - (v + 1)
            exiters_exit = n - v
            s_stay = State(s.n_o, add_i(s.n_b, r, -exiters_stay), s.n_n, s.n_pe)
            s_exit = State(s.n_o, add_i(s.n_b, r, -exiters_exit), s.n_n, s.n_pe)
            W_stay += p.beta * prob * W_after_both_region_planner(s_stay, r + 1, s_orig, V1, p, C)
            W_exit += p.beta * prob * W_after_both_region_planner(s_exit, r + 1, s_orig, V1, p, C)
        end
        new_p = logit2(W_stay, W_exit, p.sigma)
        diff = abs(new_p - p_r)
        p_r = new_p
        diff < tol && break
    end
    C.both_ccp[key] = p_r
    return p_r
end

function W_after_both_region_planner(s::State, r::Int, s_orig::State,
                                      V1::Dict{State,Float64}, p::Params,
                                      C::SolveCachesPlanner)
    if r > R
        return W_after_new_region_planner(s, 1, V1, p, C)
    end
    key = (s, r)
    haskey(C.both_W, key) && return C.both_W[key]
    n = s_orig.n_b[r]
    if n == 0
        W = W_after_both_region_planner(s, r + 1, s_orig, V1, p, C)
        C.both_W[key] = W
        return W
    end
    p_r = solve_both_region_planner(s, r, s_orig, V1, p, C)
    W = 0.0
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        exiters = n - v
        s_next = State(s.n_o, add_i(s.n_b, r, -exiters), s.n_n, s.n_pe)
        W += prob * W_after_both_region_planner(s_next, r + 1, s_orig, V1, p, C)
    end
    C.both_W[key] = W
    return W
end

# ---------------------------------------------------------------------------
# OLD stage — 3-way (stay / innovate / exit), deciders are s_orig.n_o[r]
# ---------------------------------------------------------------------------
function solve_old_region_planner(s::State, r::Int, s_orig::State,
                                   V1::Dict{State,Float64}, p::Params,
                                   C::SolveCachesPlanner;
                                   tol::Float64 = 1e-10, max_iter::Int = 500)
    key = (s, r)
    haskey(C.old_ccp, key) && return C.old_ccp[key]
    n = s_orig.n_o[r]
    if n == 0
        C.old_ccp[key] = (0.0, 0.0)
        return (0.0, 0.0)
    end

    p_s = 1/3; p_i = 1/3
    for _ in 1:max_iter
        W_stay = 0.0; W_innov = 0.0; W_exit = 0.0
        p_e = max(0.0, 1.0 - p_s - p_i)

        for k_so in 0:(n - 1), k_io in 0:(n - 1 - k_so)
            k_eo = n - 1 - k_so - k_io
            lp = log_multinomial_prob(n - 1, k_so, k_io, k_eo, p_s, p_i, p_e)
            lp == -Inf && continue
            prob = exp(lp)
            s_stay = State(set_i(s.n_o, r, k_so + 1),
                           add_i(s.n_b, r, k_io),
                           s.n_n, s.n_pe)
            s_inn  = State(set_i(s.n_o, r, k_so),
                           add_i(s.n_b, r, k_io + 1),
                           s.n_n, s.n_pe)
            s_exit = State(set_i(s.n_o, r, k_so),
                           add_i(s.n_b, r, k_io),
                           s.n_n, s.n_pe)
            W_stay  += p.beta * prob * W_after_old_region_planner(s_stay, r + 1, s_orig, V1, p, C)
            W_innov += p.beta * prob * W_after_old_region_planner(s_inn,  r + 1, s_orig, V1, p, C)
            W_exit  += p.beta * prob * W_after_old_region_planner(s_exit, r + 1, s_orig, V1, p, C)
        end
        # Innovation costs the social resource κ (no subsidy under planner).
        W_innov -= p.kappa

        vmax = max(W_stay, W_innov, W_exit)
        e_s = exp((W_stay  - vmax) / p.sigma)
        e_i = exp((W_innov - vmax) / p.sigma)
        e_e = exp((W_exit  - vmax) / p.sigma)
        denom = e_s + e_i + e_e
        np_s = e_s / denom
        np_i = e_i / denom
        diff = abs(np_s - p_s) + abs(np_i - p_i)
        p_s = np_s; p_i = np_i
        diff < tol && break
    end

    C.old_ccp[key] = (p_s, p_i)
    return (p_s, p_i)
end

function W_after_old_region_planner(s::State, r::Int, s_orig::State,
                                     V1::Dict{State,Float64}, p::Params,
                                     C::SolveCachesPlanner)
    if r > R
        return W_after_both_region_planner(s, 1, s_orig, V1, p, C)
    end
    key = (s, r)
    haskey(C.old_W, key) && return C.old_W[key]
    n = s_orig.n_o[r]
    if n == 0
        W = W_after_old_region_planner(s, r + 1, s_orig, V1, p, C)
        C.old_W[key] = W
        return W
    end
    p_s, p_i = solve_old_region_planner(s, r, s_orig, V1, p, C)
    p_e = max(0.0, 1.0 - p_s - p_i)
    W = 0.0
    for k_so in 0:n, k_io in 0:(n - k_so)
        k_eo = n - k_so - k_io
        lp = log_multinomial_prob(n, k_so, k_io, k_eo, p_s, p_i, p_e)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(set_i(s.n_o, r, k_so),
                       add_i(s.n_b, r, k_io),
                       s.n_n, s.n_pe)
        W += prob * W_after_old_region_planner(s_next, r + 1, s_orig, V1, p, C)
    end
    C.old_W[key] = W
    return W
end

# ---------------------------------------------------------------------------
# Forward traversal: per-region welfare aggregation for one s_0
# ---------------------------------------------------------------------------
mutable struct PlannerAccum
    s_orig::State
    p::Params
    competitive_static::Bool
    mso::Vector{Float64}; mio::Vector{Float64}
    msb::Vector{Float64}; msn::Vector{Float64}; mep::Vector{Float64}
    k_innov::Vector{Float64}; k_enter::Vector{Float64}
    cs_p2::Float64
    ps_p2::Vector{Float64}
end

PlannerAccum(s_0::State, p::Params, competitive_static::Bool) =
    PlannerAccum(s_0, p, competitive_static,
                 zeros(R), zeros(R),
                 zeros(R), zeros(R), zeros(R),
                 zeros(R), zeros(R),
                 0.0, zeros(R))

function _accumulate_terminal!(acc::PlannerAccum, s::State, w::Float64)
    cs, ps_r = social_welfare_static(s, acc.p; competitive = acc.competitive_static)
    acc.cs_p2 += w * cs
    for r in 1:R
        acc.ps_p2[r] += w * ps_r[r]
    end
end

function traverse_pe_planner!(s::State, r::Int, w::Float64, acc::PlannerAccum,
                               V1::Dict{State,Float64}, C::SolveCachesPlanner)
    if r > R
        _accumulate_terminal!(acc, s, w)
        return
    end
    n = s.n_pe[r]
    if n == 0
        return traverse_pe_planner!(s, r + 1, w, acc, V1, C)
    end
    p_r = solve_pe_region_planner(s, r, V1, acc.p, C)
    acc.mep[r]    += w * p_r
    acc.k_enter[r] += w * n * p_r
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(s.n_o, s.n_b,
                       add_i(s.n_n, r, v),
                       set_i(s.n_pe, r, 0))
        traverse_pe_planner!(s_next, r + 1, w * prob, acc, V1, C)
    end
end

function traverse_new_planner!(s::State, r::Int, w::Float64, acc::PlannerAccum,
                                V1::Dict{State,Float64}, C::SolveCachesPlanner)
    if r > R
        return traverse_pe_planner!(s, 1, w, acc, V1, C)
    end
    n = s.n_n[r]
    if n == 0
        return traverse_new_planner!(s, r + 1, w, acc, V1, C)
    end
    p_r = solve_new_region_planner(s, r, V1, acc.p, C)
    acc.msn[r] += w * p_r
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(s.n_o, s.n_b, set_i(s.n_n, r, v), s.n_pe)
        traverse_new_planner!(s_next, r + 1, w * prob, acc, V1, C)
    end
end

function traverse_both_planner!(s::State, r::Int, w::Float64, acc::PlannerAccum,
                                 V1::Dict{State,Float64}, C::SolveCachesPlanner)
    if r > R
        return traverse_new_planner!(s, 1, w, acc, V1, C)
    end
    n = acc.s_orig.n_b[r]
    if n == 0
        return traverse_both_planner!(s, r + 1, w, acc, V1, C)
    end
    p_r = solve_both_region_planner(s, r, acc.s_orig, V1, acc.p, C)
    acc.msb[r] += w * p_r
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        exiters = n - v
        s_next = State(s.n_o, add_i(s.n_b, r, -exiters), s.n_n, s.n_pe)
        traverse_both_planner!(s_next, r + 1, w * prob, acc, V1, C)
    end
end

function traverse_old_planner!(s::State, r::Int, w::Float64, acc::PlannerAccum,
                                V1::Dict{State,Float64}, C::SolveCachesPlanner)
    if r > R
        return traverse_both_planner!(s, 1, w, acc, V1, C)
    end
    n = acc.s_orig.n_o[r]
    if n == 0
        return traverse_old_planner!(s, r + 1, w, acc, V1, C)
    end
    p_s, p_i = solve_old_region_planner(s, r, acc.s_orig, V1, acc.p, C)
    p_e = max(0.0, 1.0 - p_s - p_i)
    acc.mso[r] += w * p_s
    acc.mio[r] += w * p_i
    acc.k_innov[r] += w * n * p_i
    for k_so in 0:n, k_io in 0:(n - k_so)
        k_eo = n - k_so - k_io
        lp = log_multinomial_prob(n, k_so, k_io, k_eo, p_s, p_i, p_e)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(set_i(s.n_o, r, k_so),
                       add_i(s.n_b, r, k_io),
                       s.n_n, s.n_pe)
        traverse_old_planner!(s_next, r + 1, w * prob, acc, V1, C)
    end
end

# ---------------------------------------------------------------------------
# Driver: per-market planner welfare from s_0
# ---------------------------------------------------------------------------

"""
    planner_welfare_at(s_0, p, V1_social; competitive_static, pe_caches…)
        → NamedTuple

Closed-form planner welfare from initial state `s_0`.  Returns the same
NamedTuple shape as `welfare_for_market` (welfare.jl:121) so the run
script can treat Eq / DynOnly / FullFB uniformly.

Period-1 social welfare is evaluated directly at `s_0` (production at
`s_0` is independent of decisions in this model).  Period-2 components
come from the forward traversal under the planner's CCPs.

Subsidy fields are zero (no subsidy under the planner).  `n_old_by_region`
and `n_pe_by_region` are decider counts at `s_0` — same denominators the
equilibrium rate reports use.
"""
function planner_welfare_at(s_0::State, p::Params,
                             V1_social::Dict{State,Float64};
                             competitive_static::Bool,
                             pe_W_cache::Dict{Tuple{State,Int}, Float64} =
                                Dict{Tuple{State,Int}, Float64}(),
                             pe_ccp_cache::Dict{Tuple{State,Int}, Float64} =
                                Dict{Tuple{State,Int}, Float64}())
    C = SolveCachesPlanner(
        pe_W_cache, pe_ccp_cache,
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, Tuple{Float64,Float64}}(),
    )
    acc = PlannerAccum(s_0, p, competitive_static)
    traverse_old_planner!(s_0, 1, 1.0, acc, V1_social, C)

    cs_p1, ps_r1 = social_welfare_static(s_0, p; competitive = competitive_static)
    cs_p2 = acc.cs_p2

    ps_r = ntuple(r -> ps_r1[r] + p.beta * acc.ps_p2[r], R)
    costs_r = ntuple(r -> p.kappa * acc.k_innov[r] + p.phi * acc.k_enter[r], R)
    cs_total_disc = cs_p1 + p.beta * cs_p2
    welfare_r = ntuple(r -> cs_total_disc / R + ps_r[r] - costs_r[r], R)

    return (
        cs_p1                      = cs_p1,
        cs_p2                      = cs_p2,
        ps_by_region               = ps_r,
        costs_by_region            = costs_r,
        subsidy_received_by_region = ntuple(_ -> 0.0, R),
        gov_outlay_total           = 0.0,
        k_innov_by_region          = ntuple(r -> acc.k_innov[r], R),
        k_enter_by_region          = ntuple(r -> acc.k_enter[r], R),
        n_old_by_region            = ntuple(r -> s_0.n_o[r],  R),
        n_pe_by_region             = ntuple(r -> s_0.n_pe[r], R),
        welfare_by_region          = welfare_r,
    )
end

# ---------------------------------------------------------------------------
# Top-level Monte-Carlo driver
# ---------------------------------------------------------------------------

"""
    expected_planner_welfare(p; n_markets, seed, competitive_static) → NamedTuple

Mean planner welfare over `n_markets` draws of `s_0`, using the same CRN
seed scheme as `expected_welfare_mc` (welfare.jl:212): each market `k`
gets `MersenneTwister(seed + k)` and `random_s0(rng, p)`.  Output shape
matches `expected_welfare_mc` so the run script can treat all three
scenarios uniformly.
"""
function expected_planner_welfare(p::Params; n_markets::Int = 5000,
                                    seed::Int = 20260424,
                                    competitive_static::Bool)
    states = all_states(p.N_max)
    V1_social = compute_terminal_planner_values(states, p;
                                                 competitive_static = competitive_static)
    pe_W_cache   = Dict{Tuple{State,Int}, Float64}()
    pe_ccp_cache = Dict{Tuple{State,Int}, Float64}()

    cs_p1 = 0.0; cs_p2 = 0.0
    ps_r       = zeros(Float64, R)
    costs_r    = zeros(Float64, R)
    welfare_r  = zeros(Float64, R)
    k_innov_r  = zeros(Float64, R)
    k_enter_r  = zeros(Float64, R)
    n_old_r    = zeros(Int, R)
    n_pe_r     = zeros(Int, R)

    for k in 1:n_markets
        rng = MersenneTwister(seed + k)
        s_0 = random_s0(rng, p)
        w   = planner_welfare_at(s_0, p, V1_social;
                                  competitive_static = competitive_static,
                                  pe_W_cache = pe_W_cache,
                                  pe_ccp_cache = pe_ccp_cache)
        cs_p1 += w.cs_p1
        cs_p2 += w.cs_p2
        for r in 1:R
            ps_r[r]      += w.ps_by_region[r]
            costs_r[r]   += w.costs_by_region[r]
            welfare_r[r] += w.welfare_by_region[r]
            k_innov_r[r] += w.k_innov_by_region[r]
            k_enter_r[r] += w.k_enter_by_region[r]
            n_old_r[r]   += w.n_old_by_region[r]
            n_pe_r[r]    += w.n_pe_by_region[r]
        end
    end

    K = n_markets
    cs_p1 /= K
    cs_p2 /= K
    ps_avg      = ntuple(r -> ps_r[r]      / K, R)
    costs_avg   = ntuple(r -> costs_r[r]   / K, R)
    welfare_avg = ntuple(r -> welfare_r[r] / K, R)
    innov_rate  = ntuple(r -> n_old_r[r] > 0 ? k_innov_r[r] / n_old_r[r] : 0.0, R)
    enter_rate  = ntuple(r -> n_pe_r[r]  > 0 ? k_enter_r[r] / n_pe_r[r]  : 0.0, R)

    return (
        cs_p1                       = cs_p1,
        cs_p2                       = cs_p2,
        ps_by_region                = ps_avg,
        costs_by_region             = costs_avg,
        subsidy_received_by_region  = ntuple(_ -> 0.0, R),
        gov_outlay_total            = 0.0,
        welfare_by_region           = welfare_avg,
        innov_rate_by_region        = innov_rate,
        enter_rate_by_region        = enter_rate,
        total_welfare               = sum(welfare_avg),
        n_markets                   = K,
        seed                        = seed,
    )
end
