"""
Welfare accounting for the 2-period regional model.

The two-market linear inverse demand
  P_o = A − b·Q_o − s_·Q_n,  P_n = A − b·Q_n − s_·Q_o,
  b = B/M,  s_ = ρ·B/M,
is rationalised by the quasi-linear utility
  U(Q_o, Q_n) = A·(Q_o + Q_n) − (b/2)·(Q_o² + Q_n²) − s_·Q_o·Q_n,
giving consumer surplus
  CS = (b/2)·(Q_o² + Q_n²) + s_·Q_o·Q_n.

Cournot is global, so CS is a single global object.  For per-region
reporting we allocate it equally across the `R = 3` regions
(equal-population assumption — see `notes/merger_counterfactual.md`).
Producer surplus is local: each firm's per-period Cournot profit is
charged to its own region.  Innovation/entry costs are paid by the firm's
region in period 1.

Per-region two-period discounted welfare:

  W_r = (CS(s_0) + β·CS(s_1)) / R                           ← global CS, equal split
      + PS_r(s_0) + β·PS_r(s_1)                             ← own-region producer surplus
      − κ · k_innov_r − φ · k_enter_r                       ← own-region costs paid

By construction Σ_r W_r = CS_total + Σ_r PS_r − Σ_r costs_r.  This
identity is verified numerically by `code/scripts/run_merger.jl`
(`check_sum_identity`); residuals are at machine precision.

The `expected_welfare_mc(p; ...)` driver runs `n_markets` independent
simulations through `simulate_market`, drawing each market's `s_0` from
`random_s0(rng, p)` (matching the DGP in `simulate_data.jl`).  Using the
same `seed` across calibrations gives common random numbers, so the
welfare *difference* between two calibrations is computed with much
smaller variance than either level.
"""

using Random

# ---------------------------------------------------------------------------
# Surplus helpers (per-state, deterministic given a state)
# ---------------------------------------------------------------------------

"""
    consumer_surplus_from_quantities(Q_o, Q_n, p) → Float64

Quasi-linear CS at total market quantities `(Q_o, Q_n)`.
"""
function consumer_surplus_from_quantities(Q_o::Float64, Q_n::Float64, p::Params)
    b_ = p.B / p.M
    s_ = p.rho * p.B / p.M
    return 0.5 * b_ * (Q_o^2 + Q_n^2) + s_ * Q_o * Q_n
end

"""
    consumer_surplus(s, p) → Float64

CS at the Cournot equilibrium of state `s`.  Solves the regional Cournot
system once and aggregates to total `(Q_o, Q_n)`.
"""
function consumer_surplus(s::State, p::Params)
    cn = c_n_vec(s, p)
    q_oo, q_bo, q_bn, q_nn =
        cournot_quantities_regional(s.n_o, s.n_b, s.n_n, p.c_o, cn, p)
    Q_o = 0.0
    Q_n = 0.0
    for r in 1:R
        Q_o += s.n_o[r] * q_oo[r] + s.n_b[r] * q_bo[r]
        Q_n += s.n_b[r] * q_bn[r] + s.n_n[r] * q_nn[r]
    end
    return consumer_surplus_from_quantities(Q_o, Q_n, p)
end

"""
    producer_surplus_by_region(s, p) → NTuple{R,Float64}

PS_r = π_o[r]·n_o[r] + π_b[r]·n_b[r] + π_n[r]·n_n[r], using
`cournot_profits_regional`.
"""
function producer_surplus_by_region(s::State, p::Params)
    cn = c_n_vec(s, p)
    pi_o, pi_b, pi_n =
        cournot_profits_regional(s.n_o, s.n_b, s.n_n, p.c_o, cn, p)
    return ntuple(r -> pi_o[r] * s.n_o[r] + pi_b[r] * s.n_b[r] + pi_n[r] * s.n_n[r],
                  R)
end

# ---------------------------------------------------------------------------
# Per-market welfare from simulated firm rows
# ---------------------------------------------------------------------------

"""
    welfare_for_market(rows, p) → NamedTuple

Aggregate one market's worth of firm rows (the `Vector{NamedTuple}`
returned by `simulate_market`) into welfare components.  Reads quantities
and profits the simulator already computed; aggregates totals (for CS),
per-region sums (for PS), and counts of `innovate` / `enter` actions
(for costs paid).  Returns

  cs_p1, cs_p2                      Float64           CS in each period
  ps_by_region                      NTuple{R,Float64} period-1 + β·period-2 PS, per region
  costs_by_region                   NTuple{R,Float64} κ·k_innov_r + φ·k_enter_r (full social
                                                      resource cost; subsidy does not enter)
  subsidy_received_by_region        NTuple{R,Float64} τ_r·k_innov_r — firm-side receipts of the
                                                      per-region innovation subsidy.  Under
                                                      sovereign-funding accounting this is a
                                                      transfer that cancels inside W_r and is
                                                      reported for descriptive purposes only.
  gov_outlay_total                  Float64           Σ_r subsidy_received_by_region — gross
                                                      government outlay summed across regions
                                                      (same units as costs_by_region).
  k_innov_by_region                 NTuple{R,Int}
  k_enter_by_region                 NTuple{R,Int}
  n_old_by_region, n_pe_by_region   NTuple{R,Int}     period-1 deciders (rate denominators)
  welfare_by_region                 NTuple{R,Float64}

Period-2 PS is discounted by `p.beta` so `ps_by_region[r] +
costs_by_region[r]` participate in the welfare sum without further
discounting outside this function.
"""
function welfare_for_market(rows, p::Params)
    Q_o1 = 0.0;  Q_n1 = 0.0
    Q_o2 = 0.0;  Q_n2 = 0.0
    ps_r       = zeros(Float64, R)
    k_innov_r  = zeros(Int, R)
    k_enter_r  = zeros(Int, R)
    n_old_r    = zeros(Int, R)
    n_pe_r     = zeros(Int, R)

    for row in rows
        r = row.region
        if row.period == 1
            Q_o1     += row.q_old
            Q_n1     += row.q_new
            ps_r[r]  += row.profit
            if row.firm_type == "old"
                n_old_r[r] += 1
                row.action == "innovate" && (k_innov_r[r] += 1)
            elseif row.firm_type == "pe"
                n_pe_r[r] += 1
                row.action == "enter" && (k_enter_r[r] += 1)
            end
        else  # period 2 — discount profits here
            Q_o2     += row.q_old
            Q_n2     += row.q_new
            ps_r[r]  += p.beta * row.profit
        end
    end

    cs_p1 = consumer_surplus_from_quantities(Q_o1, Q_n1, p)
    cs_p2 = consumer_surplus_from_quantities(Q_o2, Q_n2, p)
    cs_total_disc = cs_p1 + p.beta * cs_p2

    # Resource cost is the full κ × k_innov (and φ × k_enter); the subsidy is
    # a transfer that does NOT reduce social resource use.
    costs_r = ntuple(r -> p.kappa * k_innov_r[r] + p.phi * k_enter_r[r], R)

    # Each region is treated as a sovereign country: its own government funds
    # its own subsidy from its own taxpayers. The transfer τ_r·k_innov_r flows
    # from region r's taxpayers to region r's firms and cancels exactly within
    # region r's welfare — it never crosses borders. Regions with no subsidy
    # see no transfer term at all. We still track the (gross) outlay and
    # firm-side receipts so the reporting layer can show them.
    subsidy_received_r = ntuple(r -> p.subsidy[r] * k_innov_r[r], R)
    gov_outlay_total   = sum(subsidy_received_r)

    welfare_r = ntuple(r -> cs_total_disc / R + ps_r[r] - costs_r[r], R)

    return (
        cs_p1                      = cs_p1,
        cs_p2                      = cs_p2,
        ps_by_region               = ntuple(r -> ps_r[r],      R),
        costs_by_region            = costs_r,
        subsidy_received_by_region = subsidy_received_r,
        gov_outlay_total           = gov_outlay_total,
        k_innov_by_region          = ntuple(r -> k_innov_r[r], R),
        k_enter_by_region          = ntuple(r -> k_enter_r[r], R),
        n_old_by_region            = ntuple(r -> n_old_r[r],   R),
        n_pe_by_region             = ntuple(r -> n_pe_r[r],    R),
        welfare_by_region          = welfare_r,
    )
end

# ---------------------------------------------------------------------------
# Monte-Carlo welfare driver
# ---------------------------------------------------------------------------

"""
    expected_welfare_mc(p; n_markets = 5000, seed = 20260424) → NamedTuple

Run `n_markets` simulations with `MersenneTwister(seed + k)` per market.
Each market draws its own `s_0` via `random_s0(rng, p)` — the same DGP
`simulate_data.jl` uses — and the same RNG drives EVT1 shocks inside
`simulate_market`.  Pass the same `seed` across two calibrations to get
common random numbers (identical s_0 and shock paths under both), so the
welfare *difference* is essentially noise-free.

Returns means across markets:

  cs_p1, cs_p2                                      Float64
  ps_by_region, costs_by_region, welfare_by_region  NTuple{R,Float64}
  subsidy_received_by_region                        NTuple{R,Float64}   mean τ_r·k_innov_r
                                                                        (firm-side receipts;
                                                                        transfer, cancels in W_r)
  gov_outlay_total                                  Float64             mean Σ_r τ_r·k_innov_r
                                                                        (gross outlay per market)
  innov_rate_by_region                              NTuple{R,Float64}   pooled #innov / #old
  enter_rate_by_region                              NTuple{R,Float64}   pooled #enter / #pe
  total_welfare                                     Float64             Σ_r welfare_by_region
  n_markets, seed                                   Int                 reproducibility
"""
function expected_welfare_mc(p::Params; n_markets::Int = 5000,
                              seed::Int = 20260424)
    states = all_states(p.N_max)
    V1 = compute_terminal_values(states, p)
    pe_ccp = Dict{Tuple{State,Int}, Float64}()
    ev_pe  = Dict{Tuple{State,Int}, EV}()

    cs_p1 = 0.0;  cs_p2 = 0.0
    ps_r            = zeros(Float64, R)
    costs_r         = zeros(Float64, R)
    welfare_r       = zeros(Float64, R)
    subsidy_recv_r  = zeros(Float64, R)
    gov_outlay_tot  = 0.0
    k_innov_r       = zeros(Int, R)
    k_enter_r       = zeros(Int, R)
    n_old_r         = zeros(Int, R)
    n_pe_r          = zeros(Int, R)

    for k in 1:n_markets
        rng = MersenneTwister(seed + k)
        s0  = random_s0(rng, p)
        rows = simulate_market(s0, p, rng, V1, pe_ccp, ev_pe; market_id = k)
        w = welfare_for_market(rows, p)
        cs_p1 += w.cs_p1
        cs_p2 += w.cs_p2
        gov_outlay_tot += w.gov_outlay_total
        for r in 1:R
            ps_r[r]           += w.ps_by_region[r]
            costs_r[r]        += w.costs_by_region[r]
            welfare_r[r]      += w.welfare_by_region[r]
            subsidy_recv_r[r] += w.subsidy_received_by_region[r]
            k_innov_r[r]      += w.k_innov_by_region[r]
            k_enter_r[r]      += w.k_enter_by_region[r]
            n_old_r[r]        += w.n_old_by_region[r]
            n_pe_r[r]         += w.n_pe_by_region[r]
        end
    end

    K = n_markets
    cs_p1 /= K
    cs_p2 /= K
    ps_avg           = ntuple(r -> ps_r[r] / K,            R)
    costs_avg        = ntuple(r -> costs_r[r] / K,         R)
    welfare_avg      = ntuple(r -> welfare_r[r] / K,       R)
    subsidy_recv_avg = ntuple(r -> subsidy_recv_r[r] / K,  R)
    innov_rate       = ntuple(r -> n_old_r[r] > 0 ? k_innov_r[r] / n_old_r[r] : 0.0, R)
    enter_rate       = ntuple(r -> n_pe_r[r]  > 0 ? k_enter_r[r] / n_pe_r[r]  : 0.0, R)

    return (
        cs_p1                       = cs_p1,
        cs_p2                       = cs_p2,
        ps_by_region                = ps_avg,
        costs_by_region             = costs_avg,
        subsidy_received_by_region  = subsidy_recv_avg,
        gov_outlay_total            = gov_outlay_tot / K,
        welfare_by_region           = welfare_avg,
        innov_rate_by_region        = innov_rate,
        enter_rate_by_region        = enter_rate,
        total_welfare               = sum(welfare_avg),
        n_markets                   = K,
        seed                        = seed,
    )
end
