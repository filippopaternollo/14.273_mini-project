"""
State space and sequential-move stage solvers for the 2-period Igami model
with *regional* agglomeration.  `R = 3` regions are baked in via the
`R` constant from `parameters.jl`.

State (all per region, NTuple{3,Int}):
  n_o  : old incumbents in region r (not yet innovated)
  n_b  : "both" incumbents in region r (already innovated)
  n_n  : new entrants in region r (new technology)
  n_pe : potential entrants in region r

Within each period moves are sequential: old → both → new → pe.  Within a
stage, firms in the same (type, region) slot solve a single-agent problem
taking CCPs of other regions and later stages as given; a joint fixed point
across regions is iterated to convergence.
"""

struct State
    n_o::NTuple{R,Int}
    n_b::NTuple{R,Int}
    n_n::NTuple{R,Int}
    n_pe::NTuple{R,Int}
end

Base.hash(s::State, h::UInt) = hash((s.n_o, s.n_b, s.n_n, s.n_pe), h)
Base.:(==)(a::State, b::State) =
    a.n_o == b.n_o && a.n_b == b.n_b && a.n_n == b.n_n && a.n_pe == b.n_pe

total_firms(s::State) = sum(s.n_o) + sum(s.n_b) + sum(s.n_n) + sum(s.n_pe)

"""
    all_states(N_max)

Enumerate every state whose total firm count is ≤ `N_max`.  For R=3 and
N_max=6 this yields ~18k states.
"""
function all_states(N_max::Int)
    states = State[]
    for no1 in 0:N_max, no2 in 0:(N_max - no1), no3 in 0:(N_max - no1 - no2)
        t1 = no1 + no2 + no3
        for nb1 in 0:(N_max - t1), nb2 in 0:(N_max - t1 - nb1),
            nb3 in 0:(N_max - t1 - nb1 - nb2)
            t2 = t1 + nb1 + nb2 + nb3
            for nn1 in 0:(N_max - t2), nn2 in 0:(N_max - t2 - nn1),
                nn3 in 0:(N_max - t2 - nn1 - nn2)
                t3 = t2 + nn1 + nn2 + nn3
                for np1 in 0:(N_max - t3), np2 in 0:(N_max - t3 - np1),
                    np3 in 0:(N_max - t3 - np1 - np2)
                    push!(states, State(
                        (no1, no2, no3),
                        (nb1, nb2, nb3),
                        (nn1, nn2, nn3),
                        (np1, np2, np3)))
                end
            end
        end
    end
    return states
end

"""
    c_n_eff(s, r, p)

Regional effective new-technology marginal cost.  Spillovers are local:
`c_{n,r} = c_n0 - γ_r · (n_b^r + n_n^r)`, clamped at 0.
"""
c_n_eff(s::State, r::Int, p::Params) =
    max(0.0, p.c_n0 - p.gamma[r] * (s.n_b[r] + s.n_n[r]))

c_n_vec(s::State, p::Params) = ntuple(r -> c_n_eff(s, r, p), R)

# ---------------------------------------------------------------------------
# CCPs struct — per-region action probabilities
# ---------------------------------------------------------------------------
struct StateCCPs
    p_so::NTuple{R,Float64}   # P(stay    | old, r)
    p_io::NTuple{R,Float64}   # P(innov   | old, r)
    p_sb::NTuple{R,Float64}   # marginal P(stay | both, r)
    p_sn::NTuple{R,Float64}   # marginal P(stay | new, r)
    p_ep::NTuple{R,Float64}   # marginal P(enter | pe, r)
end

# ---------------------------------------------------------------------------
# Probability helpers (log-space for numerical stability)
# ---------------------------------------------------------------------------
using SpecialFunctions: lgamma

function log_binomial_prob(n::Int, k::Int, p::Float64)
    (k < 0 || k > n) && return -Inf
    p == 0.0 && k > 0 && return -Inf
    p == 1.0 && k < n && return -Inf
    lp = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    lp += k * (p > 0 ? log(p) : 0.0) + (n - k) * (p < 1 ? log(1 - p) : 0.0)
    return lp
end

function log_multinomial_prob(n::Int, k1::Int, k2::Int, k3::Int,
                              p1::Float64, p2::Float64, p3::Float64)
    k1 + k2 + k3 == n || return -Inf
    (p1 < 0 || p2 < 0 || p3 < 0) && return -Inf
    p1 == 0.0 && k1 > 0 && return -Inf
    p2 == 0.0 && k2 > 0 && return -Inf
    p3 == 0.0 && k3 > 0 && return -Inf
    lp  = lgamma(n + 1) - lgamma(k1 + 1) - lgamma(k2 + 1) - lgamma(k3 + 1)
    lp += k1 * (p1 > 0 ? log(p1) : 0.0)
    lp += k2 * (p2 > 0 ? log(p2) : 0.0)
    lp += k3 * (p3 > 0 ? log(p3) : 0.0)
    return lp
end

function logit2(u1::Float64, u2::Float64, sigma::Float64)
    vmax = max(u1, u2)
    e1 = exp((u1 - vmax) / sigma)
    e2 = exp((u2 - vmax) / sigma)
    return e1 / (e1 + e2)
end
