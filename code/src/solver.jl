"""
2-period backward-induction solver for the Igami model with **regional**
agglomeration and **sequential** moves across regions.

Within each stage (old → both → new → pe), regions act in order r=1, 2, 3.
Region r observes the realized outcome of regions 1..r-1 before choosing.
This turns each stage's 3-variable joint fixed point over (p₁,p₂,p₃) into
three **scalar** fixed points, solved in reverse order (r=R → 1) with CCPs
cached by the sub-state.

Cache layout: PE stage depends only on (s, r) and V1 so its caches are
shared across s_orig.  The other three stages depend on flow profits at
s_orig and are rebuilt inside `solve_state`.
"""

# ---------------------------------------------------------------------------
# Small structs & helpers
# ---------------------------------------------------------------------------
struct EV
    old::NTuple{R,Float64}
    both::NTuple{R,Float64}
    new::NTuple{R,Float64}
end

const ZERO_EV = EV(ntuple(_ -> 0.0, R), ntuple(_ -> 0.0, R), ntuple(_ -> 0.0, R))

set_i(t::NTuple{R,Int}, i::Int, v::Int) = ntuple(k -> k == i ? v : t[k], R)
add_i(t::NTuple{R,Int}, i::Int, d::Int) = ntuple(k -> k == i ? t[k] + d : t[k], R)

# ---------------------------------------------------------------------------
# Terminal values (period-1 Cournot)
# ---------------------------------------------------------------------------
function compute_terminal_values(states::Vector{State}, p::Params) :: Dict{State, EV}
    V1 = Dict{State, EV}()
    for s in states
        cn = c_n_vec(s, p)
        pi_o, pi_b, pi_n = cournot_profits_regional(s.n_o, s.n_b, s.n_n, p.c_o, cn, p)
        V1[s] = EV(pi_o, pi_b, pi_n)
    end
    return V1
end

# ---------------------------------------------------------------------------
# Solve context + caches
# ---------------------------------------------------------------------------
struct SolveContext
    s_orig::State
    flow_o::NTuple{R,Float64}
    flow_b::NTuple{R,Float64}
    flow_n::NTuple{R,Float64}
end

mutable struct SolveCaches
    # Global (shared across s_orig)
    pe_ccp::Dict{Tuple{State,Int}, Float64}
    ev_pe::Dict{Tuple{State,Int}, EV}
    # Per-s_orig (rebuilt inside solve_state)
    new_ccp::Dict{Tuple{State,Int}, Float64}
    ev_new::Dict{Tuple{State,Int}, EV}
    both_ccp::Dict{Tuple{State,Int}, Float64}
    ev_both::Dict{Tuple{State,Int}, EV}
    old_ccp::Dict{Tuple{State,Int}, Tuple{Float64,Float64}}
    ev_old::Dict{Tuple{State,Int}, EV}
end

# ---------------------------------------------------------------------------
# PE stage (last mover, global cache)
# ---------------------------------------------------------------------------
function solve_pe_region(s::State, r::Int, V1::Dict{State,EV}, p::Params,
                         C::SolveCaches; tol::Float64=1e-10, max_iter::Int=500)
    key = (s, r)
    haskey(C.pe_ccp, key) && return C.pe_ccp[key]
    n = s.n_pe[r]
    if n == 0
        C.pe_ccp[key] = 0.0
        return 0.0
    end
    p_r = 0.5
    for _ in 1:max_iter
        u_enter = -p.phi
        for v in 0:(n - 1)
            lp = log_binomial_prob(n - 1, v, p_r)
            lp == -Inf && continue
            prob = exp(lp)
            # PE firms are one-shot: after region r resolves, all of its
            # potential entrants are gone (entered or stayed out), matching
            # the simulator's add_i(s.n_pe, r, -length(deciders)).
            s_next = State(s.n_o, s.n_b,
                           add_i(s.n_n, r, v + 1),
                           set_i(s.n_pe, r, 0))
            ev = ev_after_pe_region(s_next, r + 1, V1, p, C)
            u_enter += p.beta * prob * ev.new[r]
        end
        new_p = logit2(u_enter, 0.0, p.sigma)
        diff = abs(new_p - p_r)
        p_r = new_p
        diff < tol && break
    end
    C.pe_ccp[key] = p_r
    return p_r
end

function ev_after_pe_region(s::State, r::Int, V1::Dict{State,EV},
                             p::Params, C::SolveCaches)
    if r > R
        return get(V1, s, ZERO_EV)
    end
    key = (s, r)
    haskey(C.ev_pe, key) && return C.ev_pe[key]

    n = s.n_pe[r]
    if n == 0
        ev = ev_after_pe_region(s, r + 1, V1, p, C)
        C.ev_pe[key] = ev
        return ev
    end

    p_r = solve_pe_region(s, r, V1, p, C)

    old = zeros(R); both = zeros(R); new_ = zeros(R)
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        # All PE firms in region r are gone after the region resolves
        # (v entered, the rest stayed out). Mirrors the simulator.
        s_next = State(s.n_o, s.n_b,
                       add_i(s.n_n, r, v),
                       set_i(s.n_pe, r, 0))
        ev = ev_after_pe_region(s_next, r + 1, V1, p, C)
        for k in 1:R
            old[k]  += prob * ev.old[k]
            both[k] += prob * ev.both[k]
            new_[k] += prob * ev.new[k]
        end
    end
    result = EV(Tuple(old), Tuple(both), Tuple(new_))
    C.ev_pe[key] = result
    return result
end

# ---------------------------------------------------------------------------
# NEW stage
# ---------------------------------------------------------------------------
function solve_new_region(s::State, r::Int, ctx::SolveContext,
                          V1::Dict{State,EV}, p::Params, C::SolveCaches;
                          tol::Float64=1e-10, max_iter::Int=500)
    key = (s, r)
    haskey(C.new_ccp, key) && return C.new_ccp[key]
    n = s.n_n[r]
    if n == 0
        C.new_ccp[key] = 0.0
        return 0.0
    end
    p_r = 0.5
    for _ in 1:max_iter
        u_stay = ctx.flow_n[r]
        for v in 0:(n - 1)            # peer survivors
            lp = log_binomial_prob(n - 1, v, p_r)
            lp == -Inf && continue
            prob = exp(lp)
            # Own stays ⇒ n_n[r] becomes v + 1
            s_next = State(s.n_o, s.n_b, set_i(s.n_n, r, v + 1), s.n_pe)
            ev = ev_after_new_region(s_next, r + 1, ctx, V1, p, C)
            u_stay += p.beta * prob * ev.new[r]
        end
        new_p = logit2(u_stay, 0.0, p.sigma)
        diff = abs(new_p - p_r)
        p_r = new_p
        diff < tol && break
    end
    C.new_ccp[key] = p_r
    return p_r
end

function ev_after_new_region(s::State, r::Int, ctx::SolveContext,
                              V1::Dict{State,EV}, p::Params, C::SolveCaches)
    if r > R
        return ev_after_pe_region(s, 1, V1, p, C)
    end
    key = (s, r)
    haskey(C.ev_new, key) && return C.ev_new[key]

    n = s.n_n[r]
    if n == 0
        ev = ev_after_new_region(s, r + 1, ctx, V1, p, C)
        C.ev_new[key] = ev
        return ev
    end

    p_r = solve_new_region(s, r, ctx, V1, p, C)

    old = zeros(R); both = zeros(R); new_ = zeros(R)
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(s.n_o, s.n_b, set_i(s.n_n, r, v), s.n_pe)
        ev = ev_after_new_region(s_next, r + 1, ctx, V1, p, C)
        for k in 1:R
            old[k]  += prob * ev.old[k]
            both[k] += prob * ev.both[k]
            new_[k] += prob * ev.new[k]
        end
    end
    result = EV(Tuple(old), Tuple(both), Tuple(new_))
    C.ev_new[key] = result
    return result
end

# ---------------------------------------------------------------------------
# BOTH stage — deciders are ctx.s_orig.n_b[r] (locked-in k_io already in s.n_b)
# ---------------------------------------------------------------------------
function solve_both_region(s::State, r::Int, ctx::SolveContext,
                           V1::Dict{State,EV}, p::Params, C::SolveCaches;
                           tol::Float64=1e-10, max_iter::Int=500)
    key = (s, r)
    haskey(C.both_ccp, key) && return C.both_ccp[key]
    n = ctx.s_orig.n_b[r]
    if n == 0
        C.both_ccp[key] = 0.0
        return 0.0
    end
    p_r = 0.5
    for _ in 1:max_iter
        u_stay = ctx.flow_b[r]
        for v in 0:(n - 1)            # peer deciders that stay
            lp = log_binomial_prob(n - 1, v, p_r)
            lp == -Inf && continue
            prob = exp(lp)
            exiters = n - (v + 1)     # own stays, so survivors = v+1
            s_next = State(s.n_o, add_i(s.n_b, r, -exiters), s.n_n, s.n_pe)
            ev = ev_after_both_region(s_next, r + 1, ctx, V1, p, C)
            u_stay += p.beta * prob * ev.both[r]
        end
        new_p = logit2(u_stay, 0.0, p.sigma)
        diff = abs(new_p - p_r)
        p_r = new_p
        diff < tol && break
    end
    C.both_ccp[key] = p_r
    return p_r
end

function ev_after_both_region(s::State, r::Int, ctx::SolveContext,
                               V1::Dict{State,EV}, p::Params, C::SolveCaches)
    if r > R
        return ev_after_new_region(s, 1, ctx, V1, p, C)
    end
    key = (s, r)
    haskey(C.ev_both, key) && return C.ev_both[key]

    n = ctx.s_orig.n_b[r]
    if n == 0
        ev = ev_after_both_region(s, r + 1, ctx, V1, p, C)
        C.ev_both[key] = ev
        return ev
    end

    p_r = solve_both_region(s, r, ctx, V1, p, C)

    old = zeros(R); both = zeros(R); new_ = zeros(R)
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        exiters = n - v
        s_next = State(s.n_o, add_i(s.n_b, r, -exiters), s.n_n, s.n_pe)
        ev = ev_after_both_region(s_next, r + 1, ctx, V1, p, C)
        for k in 1:R
            old[k]  += prob * ev.old[k]
            both[k] += prob * ev.both[k]
            new_[k] += prob * ev.new[k]
        end
    end
    result = EV(Tuple(old), Tuple(both), Tuple(new_))
    C.ev_both[key] = result
    return result
end

# ---------------------------------------------------------------------------
# OLD stage — 3-way choice (stay / innovate / exit), deciders s_orig.n_o[r]
# ---------------------------------------------------------------------------
function solve_old_region(s::State, r::Int, ctx::SolveContext,
                          V1::Dict{State,EV}, p::Params, C::SolveCaches;
                          tol::Float64=1e-10, max_iter::Int=500)
    key = (s, r)
    haskey(C.old_ccp, key) && return C.old_ccp[key]
    n = ctx.s_orig.n_o[r]
    if n == 0
        C.old_ccp[key] = (0.0, 0.0)
        return (0.0, 0.0)
    end

    p_s = 1/3; p_i = 1/3
    for _ in 1:max_iter
        u_stay  = ctx.flow_o[r]
        u_innov = ctx.flow_o[r] - p.kappa
        p_e = max(0.0, 1.0 - p_s - p_i)

        for k_so in 0:(n - 1), k_io in 0:(n - 1 - k_so)
            k_eo = n - 1 - k_so - k_io
            lp = log_multinomial_prob(n - 1, k_so, k_io, k_eo, p_s, p_i, p_e)
            lp == -Inf && continue
            prob = exp(lp)

            # Own stays: region-r n_o becomes k_so + 1, n_b gains k_io
            s_stay = State(set_i(s.n_o, r, k_so + 1),
                           add_i(s.n_b, r, k_io),
                           s.n_n, s.n_pe)
            ev_s = ev_after_old_region(s_stay, r + 1, ctx, V1, p, C)
            u_stay += p.beta * prob * ev_s.old[r]

            # Own innovates: n_o becomes k_so, n_b gains k_io + 1 (own locks in)
            s_inn = State(set_i(s.n_o, r, k_so),
                          add_i(s.n_b, r, k_io + 1),
                          s.n_n, s.n_pe)
            ev_i = ev_after_old_region(s_inn, r + 1, ctx, V1, p, C)
            u_innov += p.beta * prob * ev_i.both[r]
        end

        vmax = max(u_stay, u_innov, 0.0)
        e_s = exp((u_stay  - vmax) / p.sigma)
        e_i = exp((u_innov - vmax) / p.sigma)
        e_e = exp((0.0     - vmax) / p.sigma)
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

function ev_after_old_region(s::State, r::Int, ctx::SolveContext,
                              V1::Dict{State,EV}, p::Params, C::SolveCaches)
    if r > R
        return ev_after_both_region(s, 1, ctx, V1, p, C)
    end
    key = (s, r)
    haskey(C.ev_old, key) && return C.ev_old[key]

    n = ctx.s_orig.n_o[r]
    if n == 0
        ev = ev_after_old_region(s, r + 1, ctx, V1, p, C)
        C.ev_old[key] = ev
        return ev
    end

    p_s, p_i = solve_old_region(s, r, ctx, V1, p, C)
    p_e = max(0.0, 1.0 - p_s - p_i)

    old = zeros(R); both = zeros(R); new_ = zeros(R)
    for k_so in 0:n, k_io in 0:(n - k_so)
        k_eo = n - k_so - k_io
        lp = log_multinomial_prob(n, k_so, k_io, k_eo, p_s, p_i, p_e)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(set_i(s.n_o, r, k_so),
                       add_i(s.n_b, r, k_io),
                       s.n_n, s.n_pe)
        ev = ev_after_old_region(s_next, r + 1, ctx, V1, p, C)
        for k in 1:R
            old[k]  += prob * ev.old[k]
            both[k] += prob * ev.both[k]
            new_[k] += prob * ev.new[k]
        end
    end
    result = EV(Tuple(old), Tuple(both), Tuple(new_))
    C.ev_old[key] = result
    return result
end

# ---------------------------------------------------------------------------
# Forward traversal for marginal CCPs
# ---------------------------------------------------------------------------
function traverse_pe!(s::State, r::Int, w::Float64, ctx::SolveContext,
                      V1::Dict{State,EV}, p::Params, C::SolveCaches,
                      mep::Vector{Float64})
    r > R && return
    n = s.n_pe[r]
    if n == 0
        return traverse_pe!(s, r + 1, w, ctx, V1, p, C, mep)
    end
    p_r = solve_pe_region(s, r, V1, p, C)
    mep[r] += w * p_r
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        # PE firms are one-shot: zero out n_pe[r] after the region resolves.
        s_next = State(s.n_o, s.n_b,
                       add_i(s.n_n, r, v),
                       set_i(s.n_pe, r, 0))
        traverse_pe!(s_next, r + 1, w * prob, ctx, V1, p, C, mep)
    end
end

function traverse_new!(s::State, r::Int, w::Float64, ctx::SolveContext,
                       V1::Dict{State,EV}, p::Params, C::SolveCaches,
                       msn::Vector{Float64}, mep::Vector{Float64})
    if r > R
        return traverse_pe!(s, 1, w, ctx, V1, p, C, mep)
    end
    n = s.n_n[r]
    if n == 0
        return traverse_new!(s, r + 1, w, ctx, V1, p, C, msn, mep)
    end
    p_r = solve_new_region(s, r, ctx, V1, p, C)
    msn[r] += w * p_r
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(s.n_o, s.n_b, set_i(s.n_n, r, v), s.n_pe)
        traverse_new!(s_next, r + 1, w * prob, ctx, V1, p, C, msn, mep)
    end
end

function traverse_both!(s::State, r::Int, w::Float64, ctx::SolveContext,
                         V1::Dict{State,EV}, p::Params, C::SolveCaches,
                         msb::Vector{Float64}, msn::Vector{Float64},
                         mep::Vector{Float64})
    if r > R
        return traverse_new!(s, 1, w, ctx, V1, p, C, msn, mep)
    end
    n = ctx.s_orig.n_b[r]
    if n == 0
        return traverse_both!(s, r + 1, w, ctx, V1, p, C, msb, msn, mep)
    end
    p_r = solve_both_region(s, r, ctx, V1, p, C)
    msb[r] += w * p_r
    for v in 0:n
        lp = log_binomial_prob(n, v, p_r)
        lp == -Inf && continue
        prob = exp(lp)
        exiters = n - v
        s_next = State(s.n_o, add_i(s.n_b, r, -exiters), s.n_n, s.n_pe)
        traverse_both!(s_next, r + 1, w * prob, ctx, V1, p, C, msb, msn, mep)
    end
end

function traverse_old!(s::State, r::Int, w::Float64, ctx::SolveContext,
                        V1::Dict{State,EV}, p::Params, C::SolveCaches,
                        mso::Vector{Float64}, mio::Vector{Float64},
                        msb::Vector{Float64}, msn::Vector{Float64},
                        mep::Vector{Float64})
    if r > R
        return traverse_both!(s, 1, w, ctx, V1, p, C, msb, msn, mep)
    end
    n = ctx.s_orig.n_o[r]
    if n == 0
        return traverse_old!(s, r + 1, w, ctx, V1, p, C, mso, mio, msb, msn, mep)
    end
    p_s, p_i = solve_old_region(s, r, ctx, V1, p, C)
    p_e = max(0.0, 1.0 - p_s - p_i)
    mso[r] += w * p_s
    mio[r] += w * p_i
    for k_so in 0:n, k_io in 0:(n - k_so)
        k_eo = n - k_so - k_io
        lp = log_multinomial_prob(n, k_so, k_io, k_eo, p_s, p_i, p_e)
        lp == -Inf && continue
        prob = exp(lp)
        s_next = State(set_i(s.n_o, r, k_so),
                       add_i(s.n_b, r, k_io),
                       s.n_n, s.n_pe)
        traverse_old!(s_next, r + 1, w * prob, ctx, V1, p, C,
                      mso, mio, msb, msn, mep)
    end
end

# ---------------------------------------------------------------------------
# Top-level: solve a single initial state
# ---------------------------------------------------------------------------
function solve_state(s_orig::State, V1::Dict{State,EV},
                     pe_ccp_cache::Dict{Tuple{State,Int}, Float64},
                     ev_after_pe_cache::Dict{Tuple{State,Int}, EV},
                     p::Params) :: StateCCPs

    cn = c_n_vec(s_orig, p)
    pi_o, pi_b, pi_n = cournot_profits_regional(
        s_orig.n_o, s_orig.n_b, s_orig.n_n, p.c_o, cn, p)

    ctx = SolveContext(s_orig, pi_o, pi_b, pi_n)
    C = SolveCaches(
        pe_ccp_cache, ev_after_pe_cache,
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, EV}(),
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, EV}(),
        Dict{Tuple{State,Int}, Tuple{Float64,Float64}}(),
        Dict{Tuple{State,Int}, EV}(),
    )

    mso = zeros(R); mio = zeros(R)
    msb = zeros(R); msn = zeros(R); mep = zeros(R)

    traverse_old!(s_orig, 1, 1.0, ctx, V1, p, C, mso, mio, msb, msn, mep)

    return StateCCPs(
        (mso[1], mso[2], mso[3]),
        (mio[1], mio[2], mio[3]),
        (msb[1], msb[2], msb[3]),
        (msn[1], msn[2], msn[3]),
        (mep[1], mep[2], mep[3]))
end

# ---------------------------------------------------------------------------
# Convenience top-level solver
# ---------------------------------------------------------------------------
function solve_initial(s0::State, p::Params)
    states = all_states(p.N_max)
    V1 = compute_terminal_values(states, p)
    pe_ccp_cache      = Dict{Tuple{State,Int}, Float64}()
    ev_after_pe_cache = Dict{Tuple{State,Int}, EV}()
    ccps0 = solve_state(s0, V1, pe_ccp_cache, ev_after_pe_cache, p)
    return V1, ccps0
end
