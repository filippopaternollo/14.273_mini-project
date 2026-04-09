"""
2-period backward induction solver for the Igami model with agglomeration.

Solution method (Igami 2017):
With private i.i.d. EVT1 cost shocks, each firm's optimization is equivalent
to a single-agent problem taking others' CCPs as given. This yields:

    CCP(a | type, s) = exp( V̄_a(type, s) / σ ) / Σ_{a'} exp( V̄_{a'}(type, s) / σ )

The within-period equilibrium — where CCPs of firms of the same type affect
each other through the transition distribution — is solved as a fixed-point
iteration on the CCP vector for each state.

Two-period structure:
  t=1 (terminal): V1(type, s1) = π_type(s1)   [Cournot profit, no choices]
  t=0:            solve for CCPs via backward induction
"""

# ---------------------------------------------------------------------------
# Terminal period values (t = 1)
# ---------------------------------------------------------------------------
"""
V1[s][type]: Cournot profit earned by a firm of `type` in state `s` at t=1.
Agglomeration is in effect: c_n depends on n_b+n_n via c_n_eff(s, p).
"""
function compute_terminal_values(
        states::Vector{State},
        p::Params) :: Dict{State, Dict{Symbol, Float64}}

    V1 = Dict{State, Dict{Symbol, Float64}}()

    for s in states
        c_n  = c_n_eff(s, p)
        pi_o, pi_b, pi_n = cournot_profits(s.n_o, s.n_b, s.n_n, p.c_o, c_n, p)

        V1[s] = Dict{Symbol, Float64}(
            :old  => pi_o,
            :both => pi_b,
            :new  => pi_n,
            :pe   => 0.0   # potential entrants don't produce
        )
    end

    return V1
end

# ---------------------------------------------------------------------------
# Flow payoff from action (cost component paid at t=0)
# ---------------------------------------------------------------------------
function action_cost(type::Symbol, action::Symbol, p::Params) :: Float64
    type == :old && action == :innovate && return -p.kappa
    type == :pe  && action == :enter    && return -p.phi
    return 0.0
end

# ---------------------------------------------------------------------------
# Systematic utility of each action for each firm type
# ---------------------------------------------------------------------------
"""
Compute the systematic utility V̄_a for each action available to `type` at s,
given `V1` (terminal values) and `ccps` (others' CCPs at t=0).

Flow profit is earned this period (at type's current Cournot profit).
Action cost is subtracted.
Continuation value is E[V1(next_type, s') | s, a_i, CCPs_{-i}].
"""
function systematic_utilities(
        type::Symbol,
        s::State,
        V1::Dict{State, Dict{Symbol, Float64}},
        ccps::StateCCPs,
        p::Params) :: Dict{Symbol, Float64}

    c_n  = c_n_eff(s, p)
    pi_o, pi_b, pi_n = cournot_profits(s.n_o, s.n_b, s.n_n, p.c_o, c_n, p)
    flow = type == :old ? pi_o : type == :both ? pi_b : type == :new ? pi_n : 0.0

    utils = Dict{Symbol, Float64}()

    if type == :old
        for a in (:stay, :innovate, :exit)
            a == :exit && (utils[:exit] = 0.0; continue)
            ev = expected_continuation(a, :old, s, V1, ccps, p)
            utils[a] = flow + action_cost(:old, a, p) + p.beta * ev
        end

    elseif type == :both
        for a in (:stay, :exit)
            a == :exit && (utils[:exit] = 0.0; continue)
            ev = expected_continuation(a, :both, s, V1, ccps, p)
            utils[a] = flow + p.beta * ev
        end

    elseif type == :new
        for a in (:stay, :exit)
            a == :exit && (utils[:exit] = 0.0; continue)
            ev = expected_continuation(a, :new, s, V1, ccps, p)
            utils[a] = flow + p.beta * ev
        end

    else  # :pe
        for a in (:enter, :quit)
            # pe earns no flow profit until entry
            ev = expected_continuation(a, :pe, s, V1, ccps, p)
            utils[a] = action_cost(:pe, a, p) + p.beta * ev
        end
    end

    return utils
end

# ---------------------------------------------------------------------------
# Logit CCP update from systematic utilities
# ---------------------------------------------------------------------------
function logit_ccps(utils::Dict{Symbol, Float64}, sigma::Float64) :: Dict{Symbol, Float64}
    vals = collect(values(utils))
    vmax = maximum(vals)   # subtract max for numerical stability
    denom = sum(exp((v - vmax) / sigma) for v in vals)
    return Dict(a => exp((u - vmax) / sigma) / denom for (a, u) in utils)
end

# ---------------------------------------------------------------------------
# Within-period fixed-point: solve for symmetric equilibrium CCPs at state s
#
# Each firm type's CCP satisfies the logit fixed-point given the CCPs of
# the same type (through the transition distribution).
# We iterate until convergence.
# ---------------------------------------------------------------------------
"""
Solve for equilibrium CCPs at t=0 for state `s`, given `V1`.

Returns a `StateCCPs` at convergence.
"""
function solve_state_ccps(
        s::State,
        V1::Dict{State, Dict{Symbol, Float64}},
        p::Params;
        tol::Float64     = 1e-10,
        max_iter::Int    = 2000) :: StateCCPs

    ccps = uniform_ccps()

    for _ in 1:max_iter
        # Compute updated CCPs for each type given current ccps
        u_o  = systematic_utilities(:old,  s, V1, ccps, p)
        u_b  = systematic_utilities(:both, s, V1, ccps, p)
        u_n  = systematic_utilities(:new,  s, V1, ccps, p)
        u_pe = systematic_utilities(:pe,   s, V1, ccps, p)

        c_o  = logit_ccps(u_o,  p.sigma)
        c_b  = logit_ccps(u_b,  p.sigma)
        c_n  = logit_ccps(u_n,  p.sigma)
        c_pe = logit_ccps(u_pe, p.sigma)

        new_ccps = StateCCPs(
            get(c_o,  :stay,    1/3),
            get(c_o,  :innovate,1/3),
            get(c_b,  :stay,    0.5),
            get(c_n,  :stay,    0.5),
            get(c_pe, :enter,   0.5)
        )

        # Check convergence
        diff = abs(new_ccps.p_so - ccps.p_so) +
               abs(new_ccps.p_io - ccps.p_io) +
               abs(new_ccps.p_sb - ccps.p_sb) +
               abs(new_ccps.p_sn - ccps.p_sn) +
               abs(new_ccps.p_ep - ccps.p_ep)

        ccps = new_ccps
        diff < tol && break
    end

    return ccps
end

# ---------------------------------------------------------------------------
# Main solver: full 2-period backward induction
# ---------------------------------------------------------------------------
"""
Solve the 2-period model with agglomeration.

Returns:
  - V1   : Dict{State, Dict{Symbol,Float64}}  terminal period values
  - ccps : Dict{State, StateCCPs}             equilibrium CCPs at t=0
"""
function solve_2period(p::Params) :: Tuple{
        Dict{State, Dict{Symbol, Float64}},
        Dict{State, StateCCPs}}

    states = all_states(p.N_max)

    # Step 1: terminal values
    V1 = compute_terminal_values(states, p)

    # Step 2: backward induction — solve CCPs at t=0 for each state
    ccps = Dict{State, StateCCPs}()
    for s in states
        ccps[s] = solve_state_ccps(s, V1, p)
    end

    return V1, ccps
end
