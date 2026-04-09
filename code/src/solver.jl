"""
2-period backward induction solver for the Igami model with sequential moves.

Solution method (Igami 2017):
Within each period, firms move sequentially: old → both → new → pe.
Each type observes the realized outcomes of all earlier movers, so each
type solves a single-agent problem given beliefs over same-type and
later-type CCPs.  With i.i.d. EVT1 private cost shocks the solution
yields closed-form logit CCPs at each stage.

Solving order within a period (backward):
  Stage 4 (PE):   solved first — no earlier-mover uncertainty remains
  Stage 3 (new):  solved using pre-computed PE CCPs
  Stage 2 (both): solved using pre-computed new + PE CCPs
  Stage 1 (old):  solved last using all lower-stage CCPs

Two-period structure:
  t=1 (terminal): V1(type, s1) = π_type(s1)   [Cournot profit, no choices]
  t=0:            solve CCPs via backward induction + sequential-move logic
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
# Sequential within-period solver for a single state s
# ---------------------------------------------------------------------------
"""
Solve the sequential-move game at t=0 for original state `s`, given `V1`.

Procedure:
  1. Compute flow profits at s (all types use π(s_orig)).
  2. Enumerate all intermediate states reachable after old+both+new moves.
  3. Solve stages in reverse order: PE → new → both → old.
  4. Compute old-firm CCPs (p_so, p_io) via fixed-point.
  5. Compute marginal CCPs for other types by integrating over the
     equilibrium transition from s.

Returns a `StateCCPs`.
"""
function solve_state_sequential(
        s::State,
        V1::Dict{State, Dict{Symbol, Float64}},
        p::Params) :: StateCCPs

    c_n   = c_n_eff(s, p)
    pi_o, pi_b, pi_n = cournot_profits(s.n_o, s.n_b, s.n_n, p.c_o, c_n, p)

    # ------------------------------------------------------------------
    # Stage 4: PE CCPs
    # Indexed by s_after_new = State(k_so, n_b_after_both, z_sn, n_pe)
    # where k_so ∈ [0,n_o], n_b_after_both ∈ [0, n_b+n_o-k_so],
    #       z_sn ∈ [0, n_n],  n_pe = s.n_pe (unchanged).
    # ------------------------------------------------------------------
    pe_ccps = Dict{State, Float64}()

    for k_so in 0:s.n_o
        for k_io in 0:(s.n_o - k_so)
            for w_sb in 0:s.n_b                # ONLY deciding both can exit
                n_b_ab = w_sb + k_io           # after both stage
                for z_sn in 0:s.n_n
                    sn = State(k_so, n_b_ab, z_sn, s.n_pe)
                    haskey(pe_ccps, sn) && continue
                    pe_ccps[sn] = solve_pe_stage(sn, V1, p)
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # Stage 3: New firm CCPs
    # Indexed by s_after_both = State(k_so, n_b_ab, n_n, n_pe)
    # ------------------------------------------------------------------
    new_ccps = Dict{State, Float64}()

    for k_so in 0:s.n_o
        for k_io in 0:(s.n_o - k_so)
            for w_sb in 0:s.n_b
                n_b_ab = w_sb + k_io
                sab = State(k_so, n_b_ab, s.n_n, s.n_pe)
                haskey(new_ccps, sab) && continue
                new_ccps[sab] = solve_new_stage(sab, pi_n, pe_ccps, V1, p)
            end
        end
    end

    # ------------------------------------------------------------------
    # Stage 2: Both firm CCPs
    # Keyed by (k_so, k_io) — same n_b_deciding = s.n_b is implicit.
    # ------------------------------------------------------------------
    both_ccps = Dict{Tuple{Int,Int}, Float64}()

    for k_so in 0:s.n_o
        for k_io in 0:(s.n_o - k_so)
            key = (k_so, k_io)
            haskey(both_ccps, key) && continue
            both_ccps[key] = solve_both_stage(
                k_so, s.n_b, k_io, s.n_n, s.n_pe,
                pi_b, new_ccps, pe_ccps, V1, p)
        end
    end

    # ------------------------------------------------------------------
    # Stage 1: Old firm CCPs
    # ------------------------------------------------------------------
    p_so, p_io = solve_old_stage(s, pi_o, both_ccps, new_ccps, pe_ccps, V1, p)
    p_eo = max(0.0, 1.0 - p_so - p_io)

    # ------------------------------------------------------------------
    # Marginal CCPs for both / new / PE — average over equilibrium paths
    # ------------------------------------------------------------------
    marginal_p_sb = 0.0
    marginal_p_sn = 0.0
    marginal_p_ep = 0.0

    for k_so in 0:s.n_o
        for k_io in 0:(s.n_o - k_so)
            k_eo = s.n_o - k_so - k_io
            lp_old = log_multinomial_prob(s.n_o, k_so, k_io, k_eo, p_so, p_io, p_eo)
            lp_old == -Inf && continue
            prob_old = exp(lp_old)

            p_sb = get(both_ccps, (k_so, k_io), 0.0)
            marginal_p_sb += prob_old * p_sb

            for w_sb in 0:s.n_b
                lp_b = log_binomial_prob(s.n_b, w_sb, p_sb)
                lp_b == -Inf && continue
                prob_b = exp(lp_b)

                n_b_ab = w_sb + k_io
                sab    = State(k_so, n_b_ab, s.n_n, s.n_pe)
                p_sn   = get(new_ccps, sab, 0.0)
                marginal_p_sn += prob_old * prob_b * p_sn

                for z_sn in 0:s.n_n
                    lp_n = log_binomial_prob(s.n_n, z_sn, p_sn)
                    lp_n == -Inf && continue
                    prob_n = exp(lp_n)

                    sn  = State(k_so, n_b_ab, z_sn, s.n_pe)
                    p_ep = get(pe_ccps, sn, 0.0)
                    marginal_p_ep += prob_old * prob_b * prob_n * p_ep
                end
            end
        end
    end

    return StateCCPs(p_so, p_io, marginal_p_sb, marginal_p_sn, marginal_p_ep)
end

# ---------------------------------------------------------------------------
# Main solver: full 2-period backward induction
# ---------------------------------------------------------------------------
"""
Solve the 2-period model with agglomeration and sequential moves.

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

    # Step 2: backward induction — solve sequential CCPs at t=0 for each state
    ccps = Dict{State, StateCCPs}()
    for s in states
        ccps[s] = solve_state_sequential(s, V1, p)
    end

    return V1, ccps
end
