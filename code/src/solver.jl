"""
2-period backward-induction solver for the Igami model with **regional**
agglomeration and sequential moves.

Notation (within-period move order): old → both → new → pe.

Within each stage, firms of the same (type, region) slot solve a single-agent
problem.  Regions are coupled through (i) the global Cournot flow profit and
(ii) the joint continuation V1 after all stages resolve.  A Jacobi-style fixed
point is iterated across regions at each stage.

Design: caches split by whether they depend on `s_orig`.
  - Global caches (shared across all initial states):
      `pe_ccp_cache`       :: Dict{State, NTuple{R,Float64}}
      `ev_after_pe_cache`  :: Dict{State, EV}
  - Per-`s_orig` caches (rebuilt inside `solve_state`):
      `new_ccp_cache`     , `ev_after_new_cache`
      `both_ccp_cache`    , `ev_after_both_cache`
"""

# ---------------------------------------------------------------------------
# Small structs
# ---------------------------------------------------------------------------
struct EV
    old::NTuple{R,Float64}
    both::NTuple{R,Float64}
    new::NTuple{R,Float64}
end

const ZERO_EV = EV(ntuple(_ -> 0.0, R), ntuple(_ -> 0.0, R), ntuple(_ -> 0.0, R))

# ---------------------------------------------------------------------------
# Tuple helpers
# ---------------------------------------------------------------------------
set_i(t::NTuple{R,Int}, i::Int, v::Int) = ntuple(k -> k == i ? v : t[k], R)
add_i(t::NTuple{R,Int}, i::Int, d::Int) = ntuple(k -> k == i ? t[k] + d : t[k], R)

add_tuples(a::NTuple{R,Int}, b::NTuple{R,Int}) = ntuple(k -> a[k] + b[k], R)
sub_tuples(a::NTuple{R,Int}, b::NTuple{R,Int}) = ntuple(k -> a[k] - b[k], R)

# ---------------------------------------------------------------------------
# Terminal values V1[s][(type, r)] — per-firm Cournot profit at s
# ---------------------------------------------------------------------------
"""
Compute V1 for every state.  PE value is 0.
Return type: `Dict{State, EV}`.
"""
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
# Stage 4 (PE) — joint 3-region fixed point at state s_an
# ---------------------------------------------------------------------------
"""
Return equilibrium `(p_ep_1, p_ep_2, p_ep_3)` at `s_an`.  Depends only on
`s_an` and `V1`, so the result is safely cacheable across `s_orig`.
"""
function solve_pe_fp(s_an::State, V1::Dict{State, EV}, p::Params;
                     tol::Float64 = 1e-10, max_iter::Int = 2000)
    # Shortcut: no PE anywhere
    any(s_an.n_pe[r] > 0 for r in 1:R) || return ntuple(_ -> 0.0, R)

    p_ep = ntuple(r -> s_an.n_pe[r] > 0 ? 0.5 : 0.0, R)

    for _ in 1:max_iter
        new_p_ep_vec = zeros(R)

        for r_own in 1:R
            if s_an.n_pe[r_own] == 0
                new_p_ep_vec[r_own] = 0.0
                continue
            end

            # n_others: PE count excluding own in own region
            n1 = s_an.n_pe[1] - (r_own == 1 ? 1 : 0)
            n2 = s_an.n_pe[2] - (r_own == 2 ? 1 : 0)
            n3 = s_an.n_pe[3] - (r_own == 3 ? 1 : 0)

            u_enter = -p.phi

            for v1 in 0:n1, v2 in 0:n2, v3 in 0:n3
                lp = log_binomial_prob(n1, v1, p_ep[1]) +
                     log_binomial_prob(n2, v2, p_ep[2]) +
                     log_binomial_prob(n3, v3, p_ep[3])
                lp == -Inf && continue
                prob = exp(lp)

                # Own PE enters region r_own: add 1 to own region, pull from own n_pe
                add_n = (v1 + (r_own == 1 ? 1 : 0),
                         v2 + (r_own == 2 ? 1 : 0),
                         v3 + (r_own == 3 ? 1 : 0))
                s1 = State(
                    s_an.n_o, s_an.n_b,
                    add_tuples(s_an.n_n, add_n),
                    sub_tuples(s_an.n_pe, add_n))
                haskey(V1, s1) || continue
                u_enter += p.beta * prob * V1[s1].new[r_own]
            end

            new_p_ep_vec[r_own] = logit2(u_enter, 0.0, p.sigma)
        end

        new_p_ep = (new_p_ep_vec[1], new_p_ep_vec[2], new_p_ep_vec[3])
        diff = abs(new_p_ep[1] - p_ep[1]) + abs(new_p_ep[2] - p_ep[2]) +
               abs(new_p_ep[3] - p_ep[3])
        p_ep = new_p_ep
        diff < tol && return p_ep
    end
    return p_ep
end

function get_pe_ccps!(cache::Dict{State, NTuple{R,Float64}},
                     s_an::State, V1, p::Params)
    haskey(cache, s_an) && return cache[s_an]
    val = solve_pe_fp(s_an, V1, p)
    cache[s_an] = val
    return val
end

# ---------------------------------------------------------------------------
# EV after PE — continuation from s_an for a non-PE observer
# ---------------------------------------------------------------------------
"""
E[V1[(T, r)](s1) | s_an, PE-CCPs] integrated over the joint PE-entry
distribution at s_an.  Observer's action does not contribute (only stage-4
actors are PE firms, which aren't asking for a continuation here).
"""
function compute_ev_after_pe(s_an::State, V1::Dict{State, EV}, p::Params,
                              pe_ccp_cache::Dict{State, NTuple{R,Float64}})
    p_ep = get_pe_ccps!(pe_ccp_cache, s_an, V1, p)

    old = zeros(R); both = zeros(R); new_ = zeros(R)

    n1, n2, n3 = s_an.n_pe
    for v1 in 0:n1, v2 in 0:n2, v3 in 0:n3
        lp = log_binomial_prob(n1, v1, p_ep[1]) +
             log_binomial_prob(n2, v2, p_ep[2]) +
             log_binomial_prob(n3, v3, p_ep[3])
        lp == -Inf && continue
        prob = exp(lp)

        add_n = (v1, v2, v3)
        s1 = State(
            s_an.n_o, s_an.n_b,
            add_tuples(s_an.n_n, add_n),
            sub_tuples(s_an.n_pe, add_n))
        haskey(V1, s1) || continue

        v1_ev = V1[s1]
        for r in 1:R
            old[r]  += prob * v1_ev.old[r]
            both[r] += prob * v1_ev.both[r]
            new_[r] += prob * v1_ev.new[r]
        end
    end

    return EV((old[1], old[2], old[3]),
              (both[1], both[2], both[3]),
              (new_[1], new_[2], new_[3]))
end

function get_ev_after_pe!(ev_cache::Dict{State, EV},
                          pe_ccp_cache::Dict{State, NTuple{R,Float64}},
                          s_an::State, V1, p::Params)
    haskey(ev_cache, s_an) && return ev_cache[s_an]
    ev = compute_ev_after_pe(s_an, V1, p, pe_ccp_cache)
    ev_cache[s_an] = ev
    return ev
end

# ---------------------------------------------------------------------------
# Stage 3 (new) — joint 3-region fixed point at state s_ab
# ---------------------------------------------------------------------------
"""
Solve for `(p_sn_1, p_sn_2, p_sn_3)` at `s_ab`, given flow profit `flow_n`
(per-region NTuple evaluated at s_orig) and the global EV-after-PE cache.
"""
function solve_new_fp(s_ab::State, flow_n::NTuple{R,Float64},
                      ev_after_pe_cache::Dict{State, EV},
                      pe_ccp_cache::Dict{State, NTuple{R,Float64}},
                      V1::Dict{State, EV}, p::Params;
                      tol::Float64 = 1e-10, max_iter::Int = 2000)

    any(s_ab.n_n[r] > 0 for r in 1:R) || return ntuple(_ -> 0.0, R)

    p_sn = ntuple(r -> s_ab.n_n[r] > 0 ? 0.5 : 0.0, R)

    for _ in 1:max_iter
        new_p_sn_vec = zeros(R)

        for r_own in 1:R
            if s_ab.n_n[r_own] == 0
                new_p_sn_vec[r_own] = 0.0
                continue
            end

            n1 = s_ab.n_n[1] - (r_own == 1 ? 1 : 0)
            n2 = s_ab.n_n[2] - (r_own == 2 ? 1 : 0)
            n3 = s_ab.n_n[3] - (r_own == 3 ? 1 : 0)

            u_stay = flow_n[r_own]

            for z1 in 0:n1, z2 in 0:n2, z3 in 0:n3
                lp = log_binomial_prob(n1, z1, p_sn[1]) +
                     log_binomial_prob(n2, z2, p_sn[2]) +
                     log_binomial_prob(n3, z3, p_sn[3])
                lp == -Inf && continue
                prob = exp(lp)

                # Own stays ⇒ own region gains 1 survivor
                n_n_after = (z1 + (r_own == 1 ? 1 : 0),
                             z2 + (r_own == 2 ? 1 : 0),
                             z3 + (r_own == 3 ? 1 : 0))
                s_an = State(s_ab.n_o, s_ab.n_b, n_n_after, s_ab.n_pe)
                ev = get_ev_after_pe!(ev_after_pe_cache, pe_ccp_cache, s_an, V1, p)
                u_stay += p.beta * prob * ev.new[r_own]
            end

            new_p_sn_vec[r_own] = logit2(u_stay, 0.0, p.sigma)
        end

        new_p_sn = (new_p_sn_vec[1], new_p_sn_vec[2], new_p_sn_vec[3])
        diff = abs(new_p_sn[1] - p_sn[1]) + abs(new_p_sn[2] - p_sn[2]) +
               abs(new_p_sn[3] - p_sn[3])
        p_sn = new_p_sn
        diff < tol && return p_sn
    end
    return p_sn
end

"""
E[V1[(T,r)](s1)] integrated through stages 3+4 starting from s_ab.  The
observer here is NOT a stage-3 decider.
"""
function compute_ev_after_new(s_ab::State, p_sn::NTuple{R,Float64},
                              ev_after_pe_cache::Dict{State, EV},
                              pe_ccp_cache::Dict{State, NTuple{R,Float64}},
                              V1::Dict{State, EV}, p::Params)
    old = zeros(R); both = zeros(R); new_ = zeros(R)

    n1, n2, n3 = s_ab.n_n
    for z1 in 0:n1, z2 in 0:n2, z3 in 0:n3
        lp = log_binomial_prob(n1, z1, p_sn[1]) +
             log_binomial_prob(n2, z2, p_sn[2]) +
             log_binomial_prob(n3, z3, p_sn[3])
        lp == -Inf && continue
        prob = exp(lp)

        s_an = State(s_ab.n_o, s_ab.n_b, (z1, z2, z3), s_ab.n_pe)
        ev = get_ev_after_pe!(ev_after_pe_cache, pe_ccp_cache, s_an, V1, p)

        for r in 1:R
            old[r]  += prob * ev.old[r]
            both[r] += prob * ev.both[r]
            new_[r] += prob * ev.new[r]
        end
    end

    return EV((old[1], old[2], old[3]),
              (both[1], both[2], both[3]),
              (new_[1], new_[2], new_[3]))
end

# ---------------------------------------------------------------------------
# Stage 2 (both) — joint fixed point at (k_so, k_io)
#
# Sub-state is captured by (k_so::NTuple{3,Int}, k_io::NTuple{3,Int}).
# Deciding-both count per region is s_orig.n_b[r]; locked-in is k_io[r].
# ---------------------------------------------------------------------------
"""
Solve for `(p_sb_1, p_sb_2, p_sb_3)` given k_so, k_io, and the stage-3 CCP
map.  `flow_b` is π_b(s_orig) per region.
"""
function solve_both_fp(k_so::NTuple{R,Int}, k_io::NTuple{R,Int},
                       n_b_deciding::NTuple{R,Int},
                       n_n::NTuple{R,Int}, n_pe::NTuple{R,Int},
                       flow_b::NTuple{R,Float64},
                       new_ccp_cache::Dict{State, NTuple{R,Float64}},
                       ev_after_new_cache::Dict{State, EV},
                       ev_after_pe_cache::Dict{State, EV},
                       pe_ccp_cache::Dict{State, NTuple{R,Float64}},
                       flow_n::NTuple{R,Float64},
                       V1::Dict{State, EV}, p::Params;
                       tol::Float64 = 1e-10, max_iter::Int = 2000)

    any(n_b_deciding[r] > 0 for r in 1:R) || return ntuple(_ -> 0.0, R)

    p_sb = ntuple(r -> n_b_deciding[r] > 0 ? 0.5 : 0.0, R)

    for _ in 1:max_iter
        new_p_sb_vec = zeros(R)

        for r_own in 1:R
            if n_b_deciding[r_own] == 0
                new_p_sb_vec[r_own] = 0.0
                continue
            end

            n1 = n_b_deciding[1] - (r_own == 1 ? 1 : 0)
            n2 = n_b_deciding[2] - (r_own == 2 ? 1 : 0)
            n3 = n_b_deciding[3] - (r_own == 3 ? 1 : 0)

            u_stay = flow_b[r_own]

            for w1 in 0:n1, w2 in 0:n2, w3 in 0:n3
                lp = log_binomial_prob(n1, w1, p_sb[1]) +
                     log_binomial_prob(n2, w2, p_sb[2]) +
                     log_binomial_prob(n3, w3, p_sb[3])
                lp == -Inf && continue
                prob = exp(lp)

                n_b_after = (w1 + k_io[1] + (r_own == 1 ? 1 : 0),
                             w2 + k_io[2] + (r_own == 2 ? 1 : 0),
                             w3 + k_io[3] + (r_own == 3 ? 1 : 0))
                s_ab = State(k_so, n_b_after, n_n, n_pe)

                # Need EV_after_new[s_ab].  Compute via stage-3 solve + integration.
                if !haskey(ev_after_new_cache, s_ab)
                    p_sn = get!(new_ccp_cache, s_ab) do
                        solve_new_fp(s_ab, flow_n, ev_after_pe_cache,
                                     pe_ccp_cache, V1, p)
                    end
                    ev_after_new_cache[s_ab] = compute_ev_after_new(
                        s_ab, p_sn, ev_after_pe_cache, pe_ccp_cache, V1, p)
                end
                ev = ev_after_new_cache[s_ab]

                u_stay += p.beta * prob * ev.both[r_own]
            end

            new_p_sb_vec[r_own] = logit2(u_stay, 0.0, p.sigma)
        end

        new_p_sb = (new_p_sb_vec[1], new_p_sb_vec[2], new_p_sb_vec[3])
        diff = abs(new_p_sb[1] - p_sb[1]) + abs(new_p_sb[2] - p_sb[2]) +
               abs(new_p_sb[3] - p_sb[3])
        p_sb = new_p_sb
        diff < tol && return p_sb
    end
    return p_sb
end

"""
E[V1[(T,r)](s1)] integrated through stages 2+3+4 starting from (k_so, k_io)
with fixed n_b_deciding, n_n, n_pe.  Observer is not a stage-2 decider.
"""
function compute_ev_after_both(k_so::NTuple{R,Int}, k_io::NTuple{R,Int},
                               n_b_deciding::NTuple{R,Int},
                               n_n::NTuple{R,Int}, n_pe::NTuple{R,Int},
                               p_sb::NTuple{R,Float64},
                               flow_n::NTuple{R,Float64},
                               new_ccp_cache::Dict{State, NTuple{R,Float64}},
                               ev_after_new_cache::Dict{State, EV},
                               ev_after_pe_cache::Dict{State, EV},
                               pe_ccp_cache::Dict{State, NTuple{R,Float64}},
                               V1::Dict{State, EV}, p::Params)

    old = zeros(R); both = zeros(R); new_ = zeros(R)

    n1, n2, n3 = n_b_deciding
    for w1 in 0:n1, w2 in 0:n2, w3 in 0:n3
        lp = log_binomial_prob(n1, w1, p_sb[1]) +
             log_binomial_prob(n2, w2, p_sb[2]) +
             log_binomial_prob(n3, w3, p_sb[3])
        lp == -Inf && continue
        prob = exp(lp)

        n_b_after = (w1 + k_io[1], w2 + k_io[2], w3 + k_io[3])
        s_ab = State(k_so, n_b_after, n_n, n_pe)

        if !haskey(ev_after_new_cache, s_ab)
            p_sn = get!(new_ccp_cache, s_ab) do
                solve_new_fp(s_ab, flow_n, ev_after_pe_cache,
                             pe_ccp_cache, V1, p)
            end
            ev_after_new_cache[s_ab] = compute_ev_after_new(
                s_ab, p_sn, ev_after_pe_cache, pe_ccp_cache, V1, p)
        end
        ev = ev_after_new_cache[s_ab]

        for r in 1:R
            old[r]  += prob * ev.old[r]
            both[r] += prob * ev.both[r]
            new_[r] += prob * ev.new[r]
        end
    end

    return EV((old[1], old[2], old[3]),
              (both[1], both[2], both[3]),
              (new_[1], new_[2], new_[3]))
end

# ---------------------------------------------------------------------------
# Stage 1 (old) — joint fixed point over (p_so_r, p_io_r) for r=1..3
# ---------------------------------------------------------------------------
"""
Solve stage 1 at `s_orig`.  Returns `(p_so, p_io)` as NTuples.
"""
function solve_old_fp(s_orig::State,
                      flow_o::NTuple{R,Float64},
                      flow_b::NTuple{R,Float64},
                      flow_n::NTuple{R,Float64},
                      ev_after_both_cache::Dict{Tuple{NTuple{R,Int},NTuple{R,Int}}, EV},
                      both_ccp_cache::Dict{Tuple{NTuple{R,Int},NTuple{R,Int}}, NTuple{R,Float64}},
                      new_ccp_cache::Dict{State, NTuple{R,Float64}},
                      ev_after_new_cache::Dict{State, EV},
                      ev_after_pe_cache::Dict{State, EV},
                      pe_ccp_cache::Dict{State, NTuple{R,Float64}},
                      V1::Dict{State, EV}, p::Params;
                      tol::Float64 = 1e-10, max_iter::Int = 2000)

    any(s_orig.n_o[r] > 0 for r in 1:R) ||
        return (ntuple(_ -> 0.0, R), ntuple(_ -> 0.0, R))

    p_so = ntuple(r -> s_orig.n_o[r] > 0 ? 1/3 : 0.0, R)
    p_io = ntuple(r -> s_orig.n_o[r] > 0 ? 1/3 : 0.0, R)

    for _ in 1:max_iter
        new_p_so_vec = zeros(R)
        new_p_io_vec = zeros(R)

        for r_own in 1:R
            if s_orig.n_o[r_own] == 0
                new_p_so_vec[r_own] = 0.0
                new_p_io_vec[r_own] = 0.0
                continue
            end

            # Other old firms per region (own region has n_o[r_own]-1 others)
            n_other = ntuple(r -> s_orig.n_o[r] - (r == r_own ? 1 : 0), R)
            p_eo = ntuple(r -> max(0.0, 1.0 - p_so[r] - p_io[r]), R)

            u_stay  = flow_o[r_own]
            u_innov = flow_o[r_own] - p.kappa

            # Nested sum over (k_so_r, k_io_r) for each region r from its others
            for kso1 in 0:n_other[1], kio1 in 0:(n_other[1] - kso1)
                keo1 = n_other[1] - kso1 - kio1
                lp1 = log_multinomial_prob(n_other[1], kso1, kio1, keo1,
                                            p_so[1], p_io[1], p_eo[1])
                lp1 == -Inf && continue

                for kso2 in 0:n_other[2], kio2 in 0:(n_other[2] - kso2)
                    keo2 = n_other[2] - kso2 - kio2
                    lp2 = log_multinomial_prob(n_other[2], kso2, kio2, keo2,
                                                p_so[2], p_io[2], p_eo[2])
                    lp2 == -Inf && continue

                    for kso3 in 0:n_other[3], kio3 in 0:(n_other[3] - kso3)
                        keo3 = n_other[3] - kso3 - kio3
                        lp3 = log_multinomial_prob(n_other[3], kso3, kio3, keo3,
                                                    p_so[3], p_io[3], p_eo[3])
                        lp3 == -Inf && continue

                        prob = exp(lp1 + lp2 + lp3)
                        kso_others = (kso1, kso2, kso3)
                        kio_others = (kio1, kio2, kio3)

                        # --- own stays: own adds 1 to k_so in own region ---
                        k_so_s = ntuple(r -> kso_others[r] + (r == r_own ? 1 : 0), R)
                        k_io_s = kio_others

                        ev_s = get_or_compute_ev_after_both!(
                            k_so_s, k_io_s, s_orig.n_b, s_orig.n_n, s_orig.n_pe,
                            flow_b, flow_n,
                            both_ccp_cache, ev_after_both_cache,
                            new_ccp_cache, ev_after_new_cache,
                            ev_after_pe_cache, pe_ccp_cache, V1, p)
                        u_stay += p.beta * prob * ev_s.old[r_own]

                        # --- own innovates: firm moves into the "both" slot
                        # (locked-in at k_io[r_own]+1). Continuation is V1[:both, r_own].
                        k_so_i = kso_others
                        k_io_i = ntuple(r -> kio_others[r] + (r == r_own ? 1 : 0), R)

                        ev_i = get_or_compute_ev_after_both!(
                            k_so_i, k_io_i, s_orig.n_b, s_orig.n_n, s_orig.n_pe,
                            flow_b, flow_n,
                            both_ccp_cache, ev_after_both_cache,
                            new_ccp_cache, ev_after_new_cache,
                            ev_after_pe_cache, pe_ccp_cache, V1, p)
                        u_innov += p.beta * prob * ev_i.both[r_own]
                    end
                end
            end

            # Logit update over 3 actions
            vmax = max(u_stay, u_innov, 0.0)
            e_s = exp((u_stay  - vmax) / p.sigma)
            e_i = exp((u_innov - vmax) / p.sigma)
            e_e = exp((0.0     - vmax) / p.sigma)
            denom = e_s + e_i + e_e

            new_p_so_vec[r_own] = e_s / denom
            new_p_io_vec[r_own] = e_i / denom
        end

        new_p_so = (new_p_so_vec[1], new_p_so_vec[2], new_p_so_vec[3])
        new_p_io = (new_p_io_vec[1], new_p_io_vec[2], new_p_io_vec[3])
        diff = 0.0
        for r in 1:R
            diff += abs(new_p_so[r] - p_so[r]) + abs(new_p_io[r] - p_io[r])
        end
        p_so = new_p_so
        p_io = new_p_io
        diff < tol && return (p_so, p_io)
    end

    return (p_so, p_io)
end

"""
Cached wrapper: compute `EV_after_both` for a (k_so, k_io) sub-state if not
already in the cache.  Triggers stage-2 fixed-point solve on cache miss.
"""
function get_or_compute_ev_after_both!(
        k_so::NTuple{R,Int}, k_io::NTuple{R,Int},
        n_b_deciding::NTuple{R,Int}, n_n::NTuple{R,Int}, n_pe::NTuple{R,Int},
        flow_b::NTuple{R,Float64}, flow_n::NTuple{R,Float64},
        both_ccp_cache::Dict{Tuple{NTuple{R,Int},NTuple{R,Int}}, NTuple{R,Float64}},
        ev_after_both_cache::Dict{Tuple{NTuple{R,Int},NTuple{R,Int}}, EV},
        new_ccp_cache::Dict{State, NTuple{R,Float64}},
        ev_after_new_cache::Dict{State, EV},
        ev_after_pe_cache::Dict{State, EV},
        pe_ccp_cache::Dict{State, NTuple{R,Float64}},
        V1::Dict{State, EV}, p::Params)

    key = (k_so, k_io)
    haskey(ev_after_both_cache, key) && return ev_after_both_cache[key]

    p_sb = get!(both_ccp_cache, key) do
        solve_both_fp(k_so, k_io, n_b_deciding, n_n, n_pe, flow_b,
                      new_ccp_cache, ev_after_new_cache,
                      ev_after_pe_cache, pe_ccp_cache, flow_n, V1, p)
    end

    ev = compute_ev_after_both(k_so, k_io, n_b_deciding, n_n, n_pe, p_sb,
                                flow_n, new_ccp_cache, ev_after_new_cache,
                                ev_after_pe_cache, pe_ccp_cache, V1, p)
    ev_after_both_cache[key] = ev
    return ev
end

# ---------------------------------------------------------------------------
# Top-level: solve one initial state
# ---------------------------------------------------------------------------
"""
Solve the model at a single `s_orig` given pre-built `V1` and the global
PE caches.  Per-state caches (new / both) are built internally.
"""
function solve_state(s_orig::State,
                     V1::Dict{State, EV},
                     ev_after_pe_cache::Dict{State, EV},
                     pe_ccp_cache::Dict{State, NTuple{R,Float64}},
                     p::Params) :: StateCCPs

    cn = c_n_vec(s_orig, p)
    pi_o, pi_b, pi_n = cournot_profits_regional(
        s_orig.n_o, s_orig.n_b, s_orig.n_n, p.c_o, cn, p)

    new_ccp_cache       = Dict{State, NTuple{R,Float64}}()
    ev_after_new_cache  = Dict{State, EV}()
    both_ccp_cache      = Dict{Tuple{NTuple{R,Int},NTuple{R,Int}}, NTuple{R,Float64}}()
    ev_after_both_cache = Dict{Tuple{NTuple{R,Int},NTuple{R,Int}}, EV}()

    p_so, p_io = solve_old_fp(
        s_orig, pi_o, pi_b, pi_n,
        ev_after_both_cache, both_ccp_cache,
        new_ccp_cache, ev_after_new_cache,
        ev_after_pe_cache, pe_ccp_cache,
        V1, p)

    # ------------------------------------------------------------------
    # Marginal CCPs for both / new / pe — average over the equilibrium
    # path from s_orig.
    # ------------------------------------------------------------------
    p_eo = ntuple(r -> max(0.0, 1.0 - p_so[r] - p_io[r]), R)

    marg_sb = zeros(R)
    marg_sn = zeros(R)
    marg_ep = zeros(R)

    # Enumerate stage-1 outcomes (k_so, k_io) independently across regions
    for kso1 in 0:s_orig.n_o[1], kio1 in 0:(s_orig.n_o[1] - kso1)
        keo1 = s_orig.n_o[1] - kso1 - kio1
        lp1 = log_multinomial_prob(s_orig.n_o[1], kso1, kio1, keo1,
                                    p_so[1], p_io[1], p_eo[1])
        lp1 == -Inf && continue

        for kso2 in 0:s_orig.n_o[2], kio2 in 0:(s_orig.n_o[2] - kso2)
            keo2 = s_orig.n_o[2] - kso2 - kio2
            lp2 = log_multinomial_prob(s_orig.n_o[2], kso2, kio2, keo2,
                                        p_so[2], p_io[2], p_eo[2])
            lp2 == -Inf && continue

            for kso3 in 0:s_orig.n_o[3], kio3 in 0:(s_orig.n_o[3] - kso3)
                keo3 = s_orig.n_o[3] - kso3 - kio3
                lp3 = log_multinomial_prob(s_orig.n_o[3], kso3, kio3, keo3,
                                            p_so[3], p_io[3], p_eo[3])
                lp3 == -Inf && continue

                w_old = exp(lp1 + lp2 + lp3)
                k_so = (kso1, kso2, kso3)
                k_io = (kio1, kio2, kio3)

                p_sb = get(both_ccp_cache, (k_so, k_io), ntuple(_ -> 0.0, R))
                for r in 1:R
                    marg_sb[r] += w_old * p_sb[r]
                end

                # Enumerate stage-2 outcomes (w_sb per region)
                for w1 in 0:s_orig.n_b[1], w2 in 0:s_orig.n_b[2], w3 in 0:s_orig.n_b[3]
                    lpb = log_binomial_prob(s_orig.n_b[1], w1, p_sb[1]) +
                          log_binomial_prob(s_orig.n_b[2], w2, p_sb[2]) +
                          log_binomial_prob(s_orig.n_b[3], w3, p_sb[3])
                    lpb == -Inf && continue
                    w_bot = exp(lpb)

                    n_b_ab = (w1 + kio1, w2 + kio2, w3 + kio3)
                    s_ab = State(k_so, n_b_ab, s_orig.n_n, s_orig.n_pe)
                    p_sn = get(new_ccp_cache, s_ab, ntuple(_ -> 0.0, R))
                    for r in 1:R
                        marg_sn[r] += w_old * w_bot * p_sn[r]
                    end

                    # Stage-3 outcomes
                    for z1 in 0:s_orig.n_n[1], z2 in 0:s_orig.n_n[2],
                        z3 in 0:s_orig.n_n[3]
                        lpn = log_binomial_prob(s_orig.n_n[1], z1, p_sn[1]) +
                              log_binomial_prob(s_orig.n_n[2], z2, p_sn[2]) +
                              log_binomial_prob(s_orig.n_n[3], z3, p_sn[3])
                        lpn == -Inf && continue
                        w_new = exp(lpn)

                        s_an = State(k_so, n_b_ab, (z1, z2, z3), s_orig.n_pe)
                        p_ep = get_pe_ccps!(pe_ccp_cache, s_an, V1, p)
                        for r in 1:R
                            marg_ep[r] += w_old * w_bot * w_new * p_ep[r]
                        end
                    end
                end
            end
        end
    end

    return StateCCPs(p_so, p_io,
                     (marg_sb[1], marg_sb[2], marg_sb[3]),
                     (marg_sn[1], marg_sn[2], marg_sn[3]),
                     (marg_ep[1], marg_ep[2], marg_ep[3]))
end

# ---------------------------------------------------------------------------
# Convenience top-level solvers
# ---------------------------------------------------------------------------
"""
Solve the 2-period regional model.  Returns `(V1, ccps)` where `ccps` is a
`Dict{State, StateCCPs}` populated for every state in the N_max-bounded
state space.

For large N_max the enumeration over all states is expensive; use
`solve_initial(s0, p)` to solve a single state of interest.
"""
function solve_2period(p::Params)
    states = all_states(p.N_max)
    V1 = compute_terminal_values(states, p)

    pe_ccp_cache      = Dict{State, NTuple{R,Float64}}()
    ev_after_pe_cache = Dict{State, EV}()

    ccps = Dict{State, StateCCPs}()
    for s in states
        ccps[s] = solve_state(s, V1, ev_after_pe_cache, pe_ccp_cache, p)
    end
    return V1, ccps
end

"""
Solve the regional model at a single initial state `s0`.  Much faster than
`solve_2period` when you only care about CCPs at one (or a few) states.
Returns `(V1, ccps_at_s0)`.
"""
function solve_initial(s0::State, p::Params)
    states = all_states(p.N_max)
    V1 = compute_terminal_values(states, p)
    pe_ccp_cache      = Dict{State, NTuple{R,Float64}}()
    ev_after_pe_cache = Dict{State, EV}()
    ccps0 = solve_state(s0, V1, ev_after_pe_cache, pe_ccp_cache, p)
    return V1, ccps0
end
