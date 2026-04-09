"""
State space and sequential-move stage solvers for the 2-period Igami model.

State: (n_o, n_b, n_n, n_pe)
  n_o:  old incumbents (have not yet innovated)
  n_b:  "both" incumbents (have already innovated)
  n_n:  new entrants (using new technology)
  n_pe: potential entrants

Within each period moves are sequential: old → both → new → pe.
Each type observes the realized outcomes of earlier movers before acting,
so each type solves a single-agent problem (fixed-point only within the
same type, no cross-type simultaneous interaction).

Key convention on the both-stage:
  When k_io old firms innovate they become "both" at t+1 WITHOUT getting
  a separate exit decision at the both-stage.  Only the original n_b both
  firms (n_b_deciding) participate at Stage 2.  The k_io innovators are
  "locked in" and always survive as both.
"""
struct State
    n_o::Int
    n_b::Int
    n_n::Int
    n_pe::Int
end

Base.hash(s::State, h::UInt) = hash((s.n_o, s.n_b, s.n_n, s.n_pe), h)
Base.:(==)(a::State, b::State) = (a.n_o == b.n_o && a.n_b == b.n_b &&
                                   a.n_n == b.n_n && a.n_pe == b.n_pe)

"""
Enumerate all feasible states with n_o+n_b+n_n+n_pe ≤ N_max.
"""
function all_states(N_max::Int)
    states = State[]
    for n_o in 0:N_max
        for n_b in 0:(N_max - n_o)
            for n_n in 0:(N_max - n_o - n_b)
                for n_pe in 0:(N_max - n_o - n_b - n_n)
                    push!(states, State(n_o, n_b, n_n, n_pe))
                end
            end
        end
    end
    return states
end

"""
Effective new-technology marginal cost with agglomeration.
c_n,t = c_n0 - γ * (n_b + n_n)
Clamped at zero from below.
"""
function c_n_eff(s::State, p::Params)
    return max(0.0, p.c_n0 - p.gamma * (s.n_b + s.n_n))
end

# ---------------------------------------------------------------------------
# CCPs struct — stores old-firm CCPs (at original state) plus marginal
# CCPs for the other types (averaged over equilibrium paths from s).
# Fields kept compatible with run_2period.jl.
# ---------------------------------------------------------------------------
struct StateCCPs
    p_so::Float64   # P(stay    | old, s)
    p_io::Float64   # P(innovate| old, s)
    p_sb::Float64   # marginal P(stay | both),  weighted over reachable s_after_old
    p_sn::Float64   # marginal P(stay | new),   weighted over reachable s_after_both
    p_ep::Float64   # marginal P(enter | pe),   weighted over reachable s_after_new
end

# ---------------------------------------------------------------------------
# Multinomial / Binomial helpers (log-probability for numerical stability)
# ---------------------------------------------------------------------------
using SpecialFunctions: lgamma

function log_multinomial_prob(n::Int, k1::Int, k2::Int, k3::Int,
                              p1::Float64, p2::Float64, p3::Float64)
    k1 + k2 + k3 == n || return -Inf
    (p1 < 0 || p2 < 0 || p3 < 0) && return -Inf
    if p1 == 0.0 && k1 > 0; return -Inf; end
    if p2 == 0.0 && k2 > 0; return -Inf; end
    if p3 == 0.0 && k3 > 0; return -Inf; end
    lp  = lgamma(n + 1) - lgamma(k1 + 1) - lgamma(k2 + 1) - lgamma(k3 + 1)
    lp += k1 * (p1 > 0 ? log(p1) : 0.0)
    lp += k2 * (p2 > 0 ? log(p2) : 0.0)
    lp += k3 * (p3 > 0 ? log(p3) : 0.0)
    return lp
end

function log_binomial_prob(n::Int, k::Int, p::Float64)
    (k < 0 || k > n) && return -Inf
    p == 0.0 && k > 0  && return -Inf
    p == 1.0 && k < n  && return -Inf
    lp = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    lp += k * (p > 0 ? log(p) : 0.0) + (n - k) * (p < 1 ? log(1 - p) : 0.0)
    return lp
end

# ---------------------------------------------------------------------------
# Logit binary helper
# ---------------------------------------------------------------------------
function logit2(u1::Float64, u2::Float64, sigma::Float64) :: Float64
    vmax = max(u1, u2)
    e1 = exp((u1 - vmax) / sigma)
    e2 = exp((u2 - vmax) / sigma)
    return e1 / (e1 + e2)
end

# ---------------------------------------------------------------------------
# Stage 4 (last): PE firms
#
# s = s_after_new = (n_o_p, n_b_p, n_n_p, n_pe)
# Each PE firm decides {enter, quit}.
# ū_pe(enter) = -phi + β * Σ_v Bin(v; n_pe-1, p_ep) * V1[State(n_o_p, n_b_p, n_n_p+1+v, n_pe-1-v)][:new]
# ū_pe(quit)  = 0   (V1[:pe] = 0 always)
# ---------------------------------------------------------------------------
function solve_pe_stage(
        s::State,
        V1::Dict{State, Dict{Symbol, Float64}},
        p::Params;
        tol::Float64  = 1e-10,
        max_iter::Int = 2000) :: Float64

    s.n_pe == 0 && return 0.0

    p_ep = 0.5

    for _ in 1:max_iter
        u_enter = -p.phi
        # ū_pe(quit) = 0, no iteration needed there

        for v in 0:(s.n_pe - 1)
            lp = log_binomial_prob(s.n_pe - 1, v, p_ep)
            lp == -Inf && continue
            s1 = State(s.n_o, s.n_b, s.n_n + 1 + v, s.n_pe - 1 - v)
            haskey(V1, s1) || continue
            u_enter += p.beta * exp(lp) * V1[s1][:new]
        end

        new_p_ep = logit2(u_enter, 0.0, p.sigma)
        abs(new_p_ep - p_ep) < tol && return new_p_ep
        p_ep = new_p_ep
    end

    return p_ep
end

# ---------------------------------------------------------------------------
# Stage 3: New firms
#
# s = s_after_both = (n_o_p, n_b_p, n_n, n_pe)   [n_n = original n_n]
# flow_n = π_n(s_orig)
# Each new firm decides {stay, exit}.
# ū_n(stay) = flow_n + β * Σ_z Bin(z; n_n-1, p_sn) *
#                          Σ_v Bin(v; n_pe, p_ep(State(n_o_p, n_b_p, 1+z, n_pe))) *
#                          V1[State(n_o_p, n_b_p, 1+z+v, n_pe-v)][:new]
# ū_n(exit) = 0
# ---------------------------------------------------------------------------
function solve_new_stage(
        s::State,
        flow_n::Float64,
        pe_ccps::Dict{State, Float64},
        V1::Dict{State, Dict{Symbol, Float64}},
        p::Params;
        tol::Float64  = 1e-10,
        max_iter::Int = 2000) :: Float64

    s.n_n == 0 && return 0.0

    p_sn = 0.5

    for _ in 1:max_iter
        u_stay = flow_n

        for z in 0:(s.n_n - 1)
            lp_z = log_binomial_prob(s.n_n - 1, z, p_sn)
            lp_z == -Inf && continue
            prob_z = exp(lp_z)

            # s_after_new when firm i stays: n_n_p = 1+z
            s_an = State(s.n_o, s.n_b, 1 + z, s.n_pe)
            p_ep = get(pe_ccps, s_an, 0.0)

            for v in 0:s.n_pe
                lp_v = log_binomial_prob(s.n_pe, v, p_ep)
                lp_v == -Inf && continue
                s1 = State(s.n_o, s.n_b, 1 + z + v, s.n_pe - v)
                haskey(V1, s1) || continue
                u_stay += p.beta * prob_z * exp(lp_v) * V1[s1][:new]
            end
        end

        new_p_sn = logit2(u_stay, 0.0, p.sigma)
        abs(new_p_sn - p_sn) < tol && return new_p_sn
        p_sn = new_p_sn
    end

    return p_sn
end

# ---------------------------------------------------------------------------
# Stage 2: Both firms (original n_b only; locked-in innovators are auto-both)
#
# k_so        = surviving old firms (from old stage)
# n_b_deciding = s.n_b = original both firms participating in this stage
# n_b_locked  = k_io  = innovated-old firms (auto-both, no exit decision)
# n_n, n_pe   = original counts (unchanged by old stage)
# flow_b      = π_b(s_orig)
#
# ū_b(stay) = flow_b + β * Σ_w Bin(w; n_b_deciding-1, p_sb) *
#               Σ_z Bin(z; n_n, p_sn(State(k_so, w+1+n_b_locked, n_n, n_pe))) *
#               Σ_v Bin(v; n_pe, p_ep(State(k_so, w+1+n_b_locked, z, n_pe))) *
#               V1[State(k_so, w+1+n_b_locked, z+v, n_pe-v)][:both]
# ū_b(exit)  = 0
# ---------------------------------------------------------------------------
function solve_both_stage(
        k_so::Int,
        n_b_deciding::Int,
        n_b_locked::Int,
        n_n::Int,
        n_pe::Int,
        flow_b::Float64,
        new_ccps::Dict{State, Float64},
        pe_ccps::Dict{State, Float64},
        V1::Dict{State, Dict{Symbol, Float64}},
        p::Params;
        tol::Float64  = 1e-10,
        max_iter::Int = 2000) :: Float64

    n_b_deciding == 0 && return 0.0

    p_sb = 0.5

    for _ in 1:max_iter
        u_stay = flow_b

        for w in 0:(n_b_deciding - 1)
            lp_w = log_binomial_prob(n_b_deciding - 1, w, p_sb)
            lp_w == -Inf && continue
            prob_w = exp(lp_w)

            n_b_after = w + 1 + n_b_locked   # w others + me + locked-in

            s_ab = State(k_so, n_b_after, n_n, n_pe)
            p_sn = get(new_ccps, s_ab, 0.0)

            for z in 0:n_n
                lp_z = log_binomial_prob(n_n, z, p_sn)
                lp_z == -Inf && continue
                prob_z = exp(lp_z)

                s_an = State(k_so, n_b_after, z, n_pe)
                p_ep = get(pe_ccps, s_an, 0.0)

                for v in 0:n_pe
                    lp_v = log_binomial_prob(n_pe, v, p_ep)
                    lp_v == -Inf && continue
                    s1 = State(k_so, n_b_after, z + v, n_pe - v)
                    haskey(V1, s1) || continue
                    u_stay += p.beta * prob_w * prob_z * exp(lp_v) * V1[s1][:both]
                end
            end
        end

        new_p_sb = logit2(u_stay, 0.0, p.sigma)
        abs(new_p_sb - p_sb) < tol && return new_p_sb
        p_sb = new_p_sb
    end

    return p_sb
end

# ---------------------------------------------------------------------------
# Helper: EV for a surviving firm starting from s_after_old
#
# Given that firm i will be of type `firm_type_t1` at t=1, integrate over:
#   - both stage: n_b_deciding deciding firms (p_sb from both_ccps[(k_so,k_io)]),
#                 plus n_b_locked locked-in innovators
#   - new stage:  n_n deciding firms (p_sn from new_ccps[s_after_both])
#   - PE stage:   n_pe firms (p_ep from pe_ccps[s_after_new])
#
# Returns E[V1[firm_type_t1](s1)].
# ---------------------------------------------------------------------------
function ev_from_sao(
        k_so::Int,
        n_b_deciding::Int,
        n_b_locked::Int,
        n_n::Int,
        n_pe::Int,
        firm_type_t1::Symbol,
        both_ccps::Dict{Tuple{Int,Int}, Float64},
        new_ccps::Dict{State, Float64},
        pe_ccps::Dict{State, Float64},
        V1::Dict{State, Dict{Symbol, Float64}}) :: Float64

    p_sb = get(both_ccps, (k_so, n_b_locked), 0.0)

    ev = 0.0

    for w in 0:n_b_deciding
        lp_w = log_binomial_prob(n_b_deciding, w, p_sb)
        lp_w == -Inf && continue
        prob_w = exp(lp_w)

        n_b_after = w + n_b_locked

        s_ab = State(k_so, n_b_after, n_n, n_pe)
        p_sn = get(new_ccps, s_ab, 0.0)

        for z in 0:n_n
            lp_z = log_binomial_prob(n_n, z, p_sn)
            lp_z == -Inf && continue
            prob_z = exp(lp_z)

            s_an = State(k_so, n_b_after, z, n_pe)
            p_ep = get(pe_ccps, s_an, 0.0)

            for v in 0:n_pe
                lp_v = log_binomial_prob(n_pe, v, p_ep)
                lp_v == -Inf && continue
                s1 = State(k_so, n_b_after, z + v, n_pe - v)
                haskey(V1, s1) || continue
                ev += prob_w * prob_z * exp(lp_v) * V1[s1][firm_type_t1]
            end
        end
    end

    return ev
end

# ---------------------------------------------------------------------------
# Stage 1 (first): Old firms
#
# s = original state (n_o, n_b, n_n, n_pe)
# flow_o = π_o(s_orig)
# both_ccps: (k_so, k_io) → p_sb
# new_ccps:  State(k_so, n_b_after, n_n, n_pe) → p_sn
# pe_ccps:   State(k_so, n_b_after, n_n_p, n_pe) → p_ep
#
# Old firm i decides {stay, innovate, exit}:
#   stay:     contributes 1 to n_o at t=1; s_after_old has (k_so'+1, n_b+k_io', ...)
#   innovate: contributes 1 to n_b at t=1 (locked-in); s_after_old has (k_so', n_b+k_io'+1, ...)
#   exit:     u_exit = 0
#
# Returns (p_so, p_io).
# ---------------------------------------------------------------------------
function solve_old_stage(
        s::State,
        flow_o::Float64,
        both_ccps::Dict{Tuple{Int,Int}, Float64},
        new_ccps::Dict{State, Float64},
        pe_ccps::Dict{State, Float64},
        V1::Dict{State, Dict{Symbol, Float64}},
        p::Params;
        tol::Float64  = 1e-10,
        max_iter::Int = 2000) :: Tuple{Float64, Float64}

    s.n_o == 0 && return (0.0, 0.0)

    p_so = 1/3
    p_io = 1/3

    for _ in 1:max_iter
        p_eo = max(0.0, 1.0 - p_so - p_io)

        u_stay   = flow_o
        u_innov  = flow_o - p.kappa
        u_exit   = 0.0

        # Sum over OTHER n_o-1 old firms' actions
        for k_so_o in 0:(s.n_o - 1)
            for k_io_o in 0:(s.n_o - 1 - k_so_o)
                k_eo_o = s.n_o - 1 - k_so_o - k_io_o

                lp = log_multinomial_prob(s.n_o - 1, k_so_o, k_io_o, k_eo_o,
                                          p_so, p_io, p_eo)
                lp == -Inf && continue
                prob = exp(lp)

                # --- firm i stays: n_o survivors = k_so_o + 1, k_io = k_io_o ---
                ev_s = ev_from_sao(k_so_o + 1, s.n_b, k_io_o,
                                   s.n_n, s.n_pe, :old,
                                   both_ccps, new_ccps, pe_ccps, V1)
                u_stay += p.beta * prob * ev_s

                # --- firm i innovates: n_o survivors = k_so_o, k_io = k_io_o + 1 ---
                ev_i = ev_from_sao(k_so_o, s.n_b, k_io_o + 1,
                                   s.n_n, s.n_pe, :both,
                                   both_ccps, new_ccps, pe_ccps, V1)
                u_innov += p.beta * prob * ev_i
            end
        end

        # Logit update (multinomial over 3 actions)
        vmax  = max(u_stay, u_innov, u_exit)
        e_s   = exp((u_stay  - vmax) / p.sigma)
        e_i   = exp((u_innov - vmax) / p.sigma)
        e_e   = exp((u_exit  - vmax) / p.sigma)
        denom = e_s + e_i + e_e

        new_p_so = e_s / denom
        new_p_io = e_i / denom

        diff = abs(new_p_so - p_so) + abs(new_p_io - p_io)
        p_so = new_p_so
        p_io = new_p_io
        diff < tol && break
    end

    return p_so, p_io
end
