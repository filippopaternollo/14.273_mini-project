"""
State space and transition distribution for the 2-period Igami model.

State: (n_o, n_b, n_n, n_pe)
  n_o:  old incumbents (have not innovated)
  n_b:  "both" incumbents (have adopted new technology)
  n_n:  new entrants (using new technology)
  n_pe: potential entrants

Active firms = n_o + n_b + n_n; total with pe = n_o + n_b + n_n + n_pe.
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
# CCPs struct: one probability per action per firm type, for a given state.
# Actions:
#   old  → stay (p_so), innovate (p_io), exit = 1-p_so-p_io
#   both → stay (p_sb), exit = 1-p_sb
#   new  → stay (p_sn), exit = 1-p_sn
#   pe   → enter (p_ep), quit = 1-p_ep
# ---------------------------------------------------------------------------
struct StateCCPs
    p_so::Float64   # P(stay    | old)
    p_io::Float64   # P(innovate| old)
    p_sb::Float64   # P(stay    | both)
    p_sn::Float64   # P(stay    | new)
    p_ep::Float64   # P(enter   | pe)
end

function uniform_ccps()
    return StateCCPs(1/3, 1/3, 0.5, 0.5, 0.5)
end

# ---------------------------------------------------------------------------
# Multinomial / Binomial helpers (log-probability for numerical stability)
# ---------------------------------------------------------------------------
using SpecialFunctions: lgamma

function log_multinomial_prob(n::Int, k1::Int, k2::Int, k3::Int,
                              p1::Float64, p2::Float64, p3::Float64)
    k1 + k2 + k3 == n || return -Inf
    (p1 < 0 || p2 < 0 || p3 < 0) && return -Inf
    # Handle degenerate cases
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
# Expected continuation value
#
# Compute E[ V1[s1][next_type] | s0, action_i, type_i, ccps ]
#
# Given:
#   - firm i is of `type_i` and takes `action_i`
#   - all OTHER firms follow `ccps`
#
# Transition:
#   Other old firms (n_o-1 of them, if type_i==:old, else n_o):
#       each independently: stay (p_so), innovate (p_io), exit (1-p_so-p_io)
#   Both firms (n_b, minus 1 if type_i==:both):
#       each independently: stay (p_sb), exit (1-p_sb)
#   New firms (n_n, minus 1 if type_i==:new):
#       each independently: stay (p_sn), exit (1-p_sn)
#   PE firms (n_pe, minus 1 if type_i==:pe):
#       each independently: enter (p_ep), quit (1-p_ep)
# ---------------------------------------------------------------------------
function expected_continuation(
        action_i::Symbol,
        type_i::Symbol,
        s::State,
        V1::Dict{State, Dict{Symbol, Float64}},
        ccps::StateCCPs,
        p::Params)

    # Determine firm i's contribution to next-period state and its next type
    delta_o  = (type_i == :old  && action_i == :stay)     ? 1 : 0
    delta_b  = (type_i == :old  && action_i == :innovate) ? 1 :
               (type_i == :both && action_i == :stay)     ? 1 : 0
    delta_n  = (type_i == :new  && action_i == :stay)     ? 1 :
               (type_i == :pe   && action_i == :enter)    ? 1 : 0
    delta_pe = (type_i == :pe   && action_i == :quit)     ? 1 : 0

    next_type_i = if type_i == :old
        action_i == :stay     ? :old  :
        action_i == :innovate ? :both : nothing
    elseif type_i == :both
        action_i == :stay     ? :both : nothing
    elseif type_i == :new
        action_i == :stay     ? :new  : nothing
    else  # :pe
        action_i == :enter    ? :new  : :pe
    end

    # If firm i exits, continuation value is 0 (out of market)
    next_type_i === nothing && return 0.0

    # Number of OTHER firms of each type
    n_o_others  = s.n_o  - (type_i == :old  ? 1 : 0)
    n_b_others  = s.n_b  - (type_i == :both ? 1 : 0)
    n_n_others  = s.n_n  - (type_i == :new  ? 1 : 0)
    n_pe_others = s.n_pe - (type_i == :pe   ? 1 : 0)

    p_so = ccps.p_so; p_io = ccps.p_io
    p_eo = max(0.0, 1.0 - p_so - p_io)
    p_sb = ccps.p_sb
    p_sn = ccps.p_sn
    p_ep = ccps.p_ep

    total = 0.0

    # Iterate over actions of other old firms (multinomial)
    for x_stay in 0:n_o_others
        for x_innov in 0:(n_o_others - x_stay)
            x_exit = n_o_others - x_stay - x_innov
            lp_o = log_multinomial_prob(n_o_others, x_stay, x_innov, x_exit,
                                        p_so, p_io, p_eo)
            lp_o == -Inf && continue
            prob_o = exp(lp_o)

            # Other both firms
            for w_stay in 0:n_b_others
                lp_b = log_binomial_prob(n_b_others, w_stay, p_sb)
                lp_b == -Inf && continue
                prob_b = exp(lp_b)

                # Other new firms
                for z_stay in 0:n_n_others
                    lp_n = log_binomial_prob(n_n_others, z_stay, p_sn)
                    lp_n == -Inf && continue
                    prob_n = exp(lp_n)

                    # PE firms
                    for v_enter in 0:n_pe_others
                        lp_pe = log_binomial_prob(n_pe_others, v_enter, p_ep)
                        lp_pe == -Inf && continue
                        prob_pe = exp(lp_pe)

                        joint_prob = prob_o * prob_b * prob_n * prob_pe

                        s1 = State(
                            delta_o  + x_stay,
                            delta_b  + x_innov + w_stay,
                            delta_n  + z_stay  + v_enter,
                            delta_pe + (n_pe_others - v_enter)
                        )

                        # Look up V1 (state may have more firms than N_max if
                        # calibration allows; skip if not in dict)
                        if haskey(V1, s1)
                            total += joint_prob * V1[s1][next_type_i]
                        end
                    end
                end
            end
        end
    end

    return total
end
