"""
Firm-level simulation of the 2-period regional model.

For each "market" (an independent Monte-Carlo replication of the 2-period
game) we walk the same sequential structure the solver uses
(old → both → new → pe, regions r = 1..R within each stage) and draw each
firm's realized action from the CCP returned by the stage solver at the
current sub-state. Period-1 and period-2 Cournot quantities and profits
are computed from `cournot_quantities_regional` / `cournot_profits_regional`.

Public entry points:
  - `simulate_market(s0, p, rng, V1, pe_ccp_cache, ev_after_pe_cache, market_id)`
  - `random_s0(rng, p)`  — draw a random initial state inside the state space
"""

using Random
using DataFrames

# ---------------------------------------------------------------------------
# Random initial state
# ---------------------------------------------------------------------------
"""
    random_s0(rng, p; max_per_slot = 2)

Draw each region's `(n_o, n_b, n_n, n_pe)` i.i.d. from `0..max_per_slot`,
rejecting the draw if the market is empty or if the implied total firm
count (active + potential entrants) exceeds `p.N_max`. The upper bound
matches `all_states(p.N_max)`, which enumerates every state with total
firm count `≤ N_max`, so the returned `s0` is guaranteed to live in
the enumerated state space.
"""
function random_s0(rng::AbstractRNG, p::Params; max_per_slot::Int = 2)
    while true
        n_o  = ntuple(_ -> rand(rng, 0:max_per_slot), R)
        n_b  = ntuple(_ -> rand(rng, 0:max_per_slot), R)
        n_n  = ntuple(_ -> rand(rng, 0:max_per_slot), R)
        n_pe = ntuple(_ -> rand(rng, 0:max_per_slot), R)
        total = sum(n_o) + sum(n_b) + sum(n_n) + sum(n_pe)
        total == 0 && continue
        total > p.N_max && continue
        return State(n_o, n_b, n_n, n_pe)
    end
end

# ---------------------------------------------------------------------------
# Firm bookkeeping
# ---------------------------------------------------------------------------
mutable struct FirmRecord
    id::Int
    region::Int
    type0::Symbol           # :old, :both, :new, :pe
    action::Symbol          # :stay, :innovate, :exit, :enter, :stay_out
    type1::Symbol           # :old, :both, :new, :gone
end

action_string(a::Symbol) =
    a === :stay      ? "stay"      :
    a === :innovate  ? "innovate"  :
    a === :exit      ? "exit"      :
    a === :enter     ? "enter"     :
    a === :stay_out  ? "stay_out"  : ""

type_string(t::Symbol) =
    t === :old  ? "old"  :
    t === :both ? "both" :
    t === :new  ? "new"  :
    t === :pe   ? "pe"   : ""

# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------
function draw_categorical(rng::AbstractRNG, probs::NTuple{3,Float64})
    u = rand(rng)
    c = 0.0
    @inbounds for k in 1:3
        c += probs[k]
        u <= c && return k
    end
    return 3
end

# ---------------------------------------------------------------------------
# Simulate a single market
# ---------------------------------------------------------------------------
function simulate_market(s0::State, p::Params, rng::AbstractRNG,
                         V1::Dict{State,EV},
                         pe_ccp_cache::Dict{Tuple{State,Int}, Float64},
                         ev_after_pe_cache::Dict{Tuple{State,Int}, EV};
                         market_id::Int = 1)

    # ── Period-1 (decision period) Cournot at s0 ────────────────────────
    cn0 = c_n_vec(s0, p)
    q_oo0, q_bo0, q_bn0, q_nn0 =
        cournot_quantities_regional(s0.n_o, s0.n_b, s0.n_n, p.c_o, cn0, p)
    pi_o0, pi_b0, pi_n0 =
        cournot_profits_regional(s0.n_o, s0.n_b, s0.n_n, p.c_o, cn0, p)

    # ── Build firm roster at period 1 ────────────────────────────────────
    firms = FirmRecord[]
    next_id = 1
    for r in 1:R
        for _ in 1:s0.n_o[r];  push!(firms, FirmRecord(next_id, r, :old,  :stay, :gone)); next_id += 1; end
        for _ in 1:s0.n_b[r];  push!(firms, FirmRecord(next_id, r, :both, :stay, :gone)); next_id += 1; end
        for _ in 1:s0.n_n[r];  push!(firms, FirmRecord(next_id, r, :new,  :stay, :gone)); next_id += 1; end
        for _ in 1:s0.n_pe[r]; push!(firms, FirmRecord(next_id, r, :pe,   :stay_out, :gone)); next_id += 1; end
    end

    # ── Solve-context and caches (reuse PE caches across markets) ────────
    ctx = SolveContext(s0, pi_o0, pi_b0, pi_n0)
    C = SolveCaches(
        pe_ccp_cache, ev_after_pe_cache,
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, EV}(),
        Dict{Tuple{State,Int}, Float64}(),
        Dict{Tuple{State,Int}, EV}(),
        Dict{Tuple{State,Int}, Tuple{Float64,Float64}}(),
        Dict{Tuple{State,Int}, EV}(),
    )

    # ── Walk the stages, drawing actions ─────────────────────────────────
    s = s0
    firms_by_slot(r::Int, t::Symbol) =
        [f for f in firms if f.region == r && f.type0 === t]

    # OLD stage
    for r in 1:R
        deciders = firms_by_slot(r, :old)
        isempty(deciders) && continue
        p_s, p_i = solve_old_region(s, r, ctx, V1, p, C)
        p_e = max(0.0, 1.0 - p_s - p_i)
        k_so = 0; k_io = 0; k_eo = 0
        for f in deciders
            k = draw_categorical(rng, (p_s, p_i, p_e))
            if k == 1
                f.action = :stay;     f.type1 = :old;  k_so += 1
            elseif k == 2
                f.action = :innovate; f.type1 = :both; k_io += 1
            else
                f.action = :exit;     f.type1 = :gone; k_eo += 1
            end
        end
        s = State(set_i(s.n_o, r, k_so),
                  add_i(s.n_b, r, k_io),
                  s.n_n, s.n_pe)
    end

    # BOTH stage (original both firms only)
    for r in 1:R
        deciders = firms_by_slot(r, :both)
        isempty(deciders) && continue
        p_stay_b = solve_both_region(s, r, ctx, V1, p, C)
        survivors = 0
        for f in deciders
            if rand(rng) < p_stay_b
                f.action = :stay; f.type1 = :both
                survivors += 1
            else
                f.action = :exit; f.type1 = :gone
            end
        end
        exiters = length(deciders) - survivors
        s = State(s.n_o, add_i(s.n_b, r, -exiters), s.n_n, s.n_pe)
    end

    # NEW stage (original new firms only)
    for r in 1:R
        deciders = firms_by_slot(r, :new)
        isempty(deciders) && continue
        p_stay_n = solve_new_region(s, r, ctx, V1, p, C)
        survivors = 0
        for f in deciders
            if rand(rng) < p_stay_n
                f.action = :stay; f.type1 = :new
                survivors += 1
            else
                f.action = :exit; f.type1 = :gone
            end
        end
        s = State(s.n_o, s.n_b, set_i(s.n_n, r, survivors), s.n_pe)
    end

    # PE stage
    for r in 1:R
        deciders = firms_by_slot(r, :pe)
        isempty(deciders) && continue
        p_enter = solve_pe_region(s, r, V1, p, C)
        entered = 0
        for f in deciders
            if rand(rng) < p_enter
                f.action = :enter; f.type1 = :new
                entered += 1
            else
                f.action = :stay_out; f.type1 = :gone
            end
        end
        s = State(s.n_o, s.n_b,
                  add_i(s.n_n, r, entered),
                  add_i(s.n_pe, r, -length(deciders)))
    end

    s1 = s

    # ── Period-2 Cournot at s1 ──────────────────────────────────────────
    cn1 = c_n_vec(s1, p)
    q_oo1, q_bo1, q_bn1, q_nn1 =
        cournot_quantities_regional(s1.n_o, s1.n_b, s1.n_n, p.c_o, cn1, p)
    pi_o1, pi_b1, pi_n1 =
        cournot_profits_regional(s1.n_o, s1.n_b, s1.n_n, p.c_o, cn1, p)

    # ── Emit rows ───────────────────────────────────────────────────────
    rows = NamedTuple[]

    for f in firms
        r = f.region
        q_old_p1, q_new_p1, profit_p1 =
            if f.type0 === :old
                (q_oo0[r], 0.0, pi_o0[r])
            elseif f.type0 === :both
                (q_bo0[r], q_bn0[r], pi_b0[r])
            elseif f.type0 === :new
                (0.0, q_nn0[r], pi_n0[r])
            else  # :pe
                (0.0, 0.0, 0.0)
            end
        push!(rows, (
            market_id = market_id,
            period    = 1,
            firm_id   = f.id,
            region    = r,
            firm_type = type_string(f.type0),
            q_old     = q_old_p1,
            q_new     = q_new_p1,
            profit    = profit_p1,
            action    = action_string(f.action),
        ))
    end

    for f in firms
        f.type1 === :gone && continue
        r = f.region
        q_old_p2, q_new_p2, profit_p2 =
            if f.type1 === :old
                (q_oo1[r], 0.0, pi_o1[r])
            elseif f.type1 === :both
                (q_bo1[r], q_bn1[r], pi_b1[r])
            else  # :new
                (0.0, q_nn1[r], pi_n1[r])
            end
        push!(rows, (
            market_id = market_id,
            period    = 2,
            firm_id   = f.id,
            region    = r,
            firm_type = type_string(f.type1),
            q_old     = q_old_p2,
            q_new     = q_new_p2,
            profit    = profit_p2,
            action    = "",
        ))
    end

    return rows
end

