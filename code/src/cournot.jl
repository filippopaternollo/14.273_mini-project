using LinearAlgebra

# Backwards-compatible single-region Cournot helper, used only in tests.  The
# regional version below is what the solver calls.

"""
Two-market Cournot equilibrium with consumer substitution between generations.

Demand system (Igami 2017):
  P_o = A - B*(Q_o/M) - S*(Q_n/M)   (old-gen market)
  P_n = A - B*(Q_n/M) - S*(Q_o/M)   (new-gen market)
where S = rho*B ∈ [0, B) is the cross-market slope.

Three groups:
  n_o  old-tech firms:  produce old-gen only (quantity q_oo)
  n_b  "both" firms:    produce old-gen AND new-gen (quantities q_bo, q_bn)
  n_n  new-tech firms:  produce new-gen only (quantity q_nn)

"Both" firms are multi-product and internalize cannibalization: their new-gen
production reduces demand for their own old-gen product (and vice versa).

FOCs in symmetric equilibrium (b = B/M, s = rho*B/M):
  FOC_old:    A - b*(n_o+1)*q_oo - b*n_b*q_bo     - s*n_b*q_bn     - s*n_n*q_nn     = c_o
  FOC_both_o: A - b*n_o*q_oo     - b*(n_b+1)*q_bo  - s*(n_b+1)*q_bn - s*n_n*q_nn     = c_o
  FOC_both_n: A - s*n_o*q_oo     - s*(n_b+1)*q_bo  - b*(n_b+1)*q_bn - b*n_n*q_nn     = c_n
  FOC_new:    A - s*n_o*q_oo     - s*n_b*q_bo      - b*n_b*q_bn     - b*(n_n+1)*q_nn = c_n

Corner handling: each product (q_oo, q_bo, q_bn, q_nn) is independently constrained
to be non-negative. If an interior solution gives q ≤ 0 for a product, that product
is dropped from the market and the system is re-solved. In particular, a "both" firm
may end up selling only new-gen (q_bo = 0) if old-gen is not profitable — this is the
strong cannibalization case where the incumbent abandons old-gen production.

Profit formulas (derived from FOCs, where b = B/M, s = rho*B/M):
  pi_o = b * q_oo^2
  pi_b = b*(q_bo^2 + q_bn^2) + 2s*q_bo*q_bn   ← cross term encodes cannibalization
  pi_n = b * q_nn^2

Returns (pi_o, pi_b, pi_n): per-firm profits for each group.
"""
function cournot_profits(n_o::Int, n_b::Int, n_n::Int,
                         c_o::Float64, c_n::Float64, p::Params)
    A, B, M = p.A, p.B, p.M
    b = B / M
    s = p.rho * B / M

    if n_o == 0 && n_b == 0 && n_n == 0
        return (0.0, 0.0, 0.0)
    end

    # Each of the 4 product quantities has its own active flag
    a_oo = n_o > 0   # old firms in old-gen market
    a_bo = n_b > 0   # both firms in old-gen market
    a_bn = n_b > 0   # both firms in new-gen market
    a_nn = n_n > 0   # new firms in new-gen market

    for _ in 1:5
        q_oo, q_bo, q_bn, q_nn =
            _build_and_solve(a_oo, a_bo, a_bn, a_nn, n_o, n_b, n_n, c_o, c_n, A, b, s)

        changed = false
        if a_oo && q_oo <= 0.0; a_oo = false; changed = true; end
        if a_bo && q_bo <= 0.0; a_bo = false; changed = true; end
        if a_bn && q_bn <= 0.0; a_bn = false; changed = true; end
        if a_nn && q_nn <= 0.0; a_nn = false; changed = true; end

        if !changed
            pi_o = b * q_oo^2
            pi_b = b * (q_bo^2 + q_bn^2) + 2 * s * q_bo * q_bn
            pi_n = b * q_nn^2
            return (pi_o, pi_b, pi_n)
        end
    end

    return (0.0, 0.0, 0.0)
end

"""
Build and solve the Cournot linear system for the active products.

Active flags: a_oo (old firms sell old-gen), a_bo (both firms sell old-gen),
a_bn (both firms sell new-gen), a_nn (new firms sell new-gen). Each flag can
be set independently, allowing corner solutions where a "both" firm sells only
one generation (e.g., a_bo=false, a_bn=true when old-gen is not profitable).

One FOC per active product. Returns (q_oo, q_bo, q_bn, q_nn) with 0.0 for inactive.
"""
function _build_and_solve(a_oo::Bool, a_bo::Bool, a_bn::Bool, a_nn::Bool,
                           n_o::Int, n_b::Int, n_n::Int,
                           c_o::Float64, c_n::Float64,
                           A::Float64, b::Float64, s::Float64)
    vars = Symbol[]
    a_oo && push!(vars, :q_oo)
    a_bo && push!(vars, :q_bo)
    a_bn && push!(vars, :q_bn)
    a_nn && push!(vars, :q_nn)

    k = length(vars)
    k == 0 && return (0.0, 0.0, 0.0, 0.0)

    idx = Dict(v => i for (i, v) in enumerate(vars))
    Mat = zeros(k, k)
    rhs = zeros(k)

    for (i, eq) in enumerate(vars)
        if eq == :q_oo          # FOC for old-type firm (old-gen market)
            rhs[i] = A - c_o
            a_oo && (Mat[i, idx[:q_oo]] += b * (n_o + 1))
            a_bo && (Mat[i, idx[:q_bo]] += b * n_b)
            a_bn && (Mat[i, idx[:q_bn]] += s * n_b)
            a_nn && (Mat[i, idx[:q_nn]] += s * n_n)

        elseif eq == :q_bo      # FOC for both-type firm, old-gen market
            rhs[i] = A - c_o
            a_oo && (Mat[i, idx[:q_oo]] += b * n_o)
            a_bo && (Mat[i, idx[:q_bo]] += b * (n_b + 1))
            a_bn && (Mat[i, idx[:q_bn]] += s * (n_b + 1))
            a_nn && (Mat[i, idx[:q_nn]] += s * n_n)

        elseif eq == :q_bn      # FOC for both-type firm, new-gen market
            rhs[i] = A - c_n
            a_oo && (Mat[i, idx[:q_oo]] += s * n_o)
            a_bo && (Mat[i, idx[:q_bo]] += s * (n_b + 1))
            a_bn && (Mat[i, idx[:q_bn]] += b * (n_b + 1))
            a_nn && (Mat[i, idx[:q_nn]] += b * n_n)

        elseif eq == :q_nn      # FOC for new-type firm (new-gen market)
            rhs[i] = A - c_n
            a_oo && (Mat[i, idx[:q_oo]] += s * n_o)
            a_bo && (Mat[i, idx[:q_bo]] += s * n_b)
            a_bn && (Mat[i, idx[:q_bn]] += b * n_b)
            a_nn && (Mat[i, idx[:q_nn]] += b * (n_n + 1))
        end
    end

    q_vec = Mat \ rhs

    q_oo = a_oo ? q_vec[idx[:q_oo]] : 0.0
    q_bo = a_bo ? q_vec[idx[:q_bo]] : 0.0
    q_bn = a_bn ? q_vec[idx[:q_bn]] : 0.0
    q_nn = a_nn ? q_vec[idx[:q_nn]] : 0.0

    return (q_oo, q_bo, q_bn, q_nn)
end

# ===========================================================================
#  Regional Cournot: global competition, per-region new-tech costs
# ===========================================================================
# Two-market Cournot with R regional cost types on the new-gen side; all
# firms compete in a single global market pair. Per-firm profits satisfy
#   pi_o_r = b · q_oo_r², pi_n_r = b · q_nn_r²,
#   pi_b_r = b · (q_bo_r² + q_bn_r²) + 2s · q_bo_r · q_bn_r.
"""
    cournot_quantities_regional(n_o, n_b, n_n, c_o, c_n, p)

Same solution routine as `cournot_profits_regional`, but returns the
four `NTuple{R,Float64}` of equilibrium quantities
`(q_oo, q_bo, q_bn, q_nn)` (per-firm, per-region, per-product). A slot
with no firm or a dropped corner returns 0.0.
"""
function cournot_quantities_regional(n_o::NTuple{R,Int}, n_b::NTuple{R,Int},
                                      n_n::NTuple{R,Int}, c_o::Float64,
                                      c_n::NTuple{R,Float64}, p::Params)
    q_oo, q_bo, q_bn, q_nn = _solve_regional_quantities(n_o, n_b, n_n, c_o, c_n, p)
    return (ntuple(r -> q_oo[r], R),
            ntuple(r -> q_bo[r], R),
            ntuple(r -> q_bn[r], R),
            ntuple(r -> q_nn[r], R))
end

function cournot_profits_regional(n_o::NTuple{R,Int}, n_b::NTuple{R,Int},
                                   n_n::NTuple{R,Int}, c_o::Float64,
                                   c_n::NTuple{R,Float64}, p::Params)
    B, M = p.B, p.M
    b_ = B / M
    s_ = p.rho * B / M
    q_oo, q_bo, q_bn, q_nn = _solve_regional_quantities(n_o, n_b, n_n, c_o, c_n, p)
    pi_o = ntuple(r -> b_ * q_oo[r]^2, R)
    pi_b = ntuple(r -> b_ * (q_bo[r]^2 + q_bn[r]^2) + 2 * s_ * q_bo[r] * q_bn[r], R)
    pi_n = ntuple(r -> b_ * q_nn[r]^2, R)
    return (pi_o, pi_b, pi_n)
end

"""
    _solve_regional_quantities(n_o, n_b, n_n, c_o, c_n, p) -> (q_oo, q_bo, q_bn, q_nn)

Internal helper: solves the regional asymmetric Cournot system with
iterative corner drops and returns the four length-R quantity vectors
(as `Vector{Float64}`, slot 0 where no firm exists or the corner is
inactive).
"""
function _solve_regional_quantities(n_o::NTuple{R,Int}, n_b::NTuple{R,Int},
                                     n_n::NTuple{R,Int}, c_o::Float64,
                                     c_n::NTuple{R,Float64}, p::Params)
    A, B, M = p.A, p.B, p.M
    b_ = B / M
    s_ = p.rho * B / M

    zero_v() = zeros(R)

    if sum(n_o) + sum(n_b) + sum(n_n) == 0
        return (zero_v(), zero_v(), zero_v(), zero_v())
    end

    act_oo = [n_o[r] > 0 for r in 1:R]
    act_bo = [n_b[r] > 0 for r in 1:R]
    act_bn = [n_b[r] > 0 for r in 1:R]
    act_nn = [n_n[r] > 0 for r in 1:R]

    q_oo = zeros(R); q_bo = zeros(R); q_bn = zeros(R); q_nn = zeros(R)

    for _ in 1:(4 * R + 2)
        fill!(q_oo, 0.0); fill!(q_bo, 0.0)
        fill!(q_bn, 0.0); fill!(q_nn, 0.0)

        vars = Tuple{Symbol,Int}[]
        for r in 1:R
            act_oo[r] && push!(vars, (:oo, r))
            act_bo[r] && push!(vars, (:bo, r))
            act_bn[r] && push!(vars, (:bn, r))
            act_nn[r] && push!(vars, (:nn, r))
        end
        k = length(vars)
        k == 0 && return (zero_v(), zero_v(), zero_v(), zero_v())

        idx = Dict{Tuple{Symbol,Int}, Int}()
        for (i, v) in enumerate(vars); idx[v] = i; end

        Mat = zeros(k, k)
        rhs = zeros(k)

        for (i, (kind, r)) in enumerate(vars)
            if kind == :oo
                rhs[i] = A - c_o
                Mat[i, idx[(:oo, r)]] += b_ * (n_o[r] + 1)
                for r2 in 1:R
                    r2 == r && continue
                    act_oo[r2] && (Mat[i, idx[(:oo, r2)]] += b_ * n_o[r2])
                end
                for r2 in 1:R
                    act_bo[r2] && (Mat[i, idx[(:bo, r2)]] += b_ * n_b[r2])
                end
                for r2 in 1:R
                    act_bn[r2] && (Mat[i, idx[(:bn, r2)]] += s_ * n_b[r2])
                end
                for r2 in 1:R
                    act_nn[r2] && (Mat[i, idx[(:nn, r2)]] += s_ * n_n[r2])
                end

            elseif kind == :bo
                rhs[i] = A - c_o
                Mat[i, idx[(:bo, r)]] += b_ * (n_b[r] + 1)
                for r2 in 1:R
                    r2 == r && continue
                    act_bo[r2] && (Mat[i, idx[(:bo, r2)]] += b_ * n_b[r2])
                end
                for r2 in 1:R
                    act_oo[r2] && (Mat[i, idx[(:oo, r2)]] += b_ * n_o[r2])
                end
                # The same-slot cross term uses (n_b[r]+1) because the firm's
                # own new-gen choice co-moves with its own old-gen choice.
                if act_bn[r]
                    Mat[i, idx[(:bn, r)]] += s_ * (n_b[r] + 1)
                end
                for r2 in 1:R
                    r2 == r && continue
                    act_bn[r2] && (Mat[i, idx[(:bn, r2)]] += s_ * n_b[r2])
                end
                for r2 in 1:R
                    act_nn[r2] && (Mat[i, idx[(:nn, r2)]] += s_ * n_n[r2])
                end

            elseif kind == :bn
                rhs[i] = A - c_n[r]
                Mat[i, idx[(:bn, r)]] += b_ * (n_b[r] + 1)
                for r2 in 1:R
                    r2 == r && continue
                    act_bn[r2] && (Mat[i, idx[(:bn, r2)]] += b_ * n_b[r2])
                end
                for r2 in 1:R
                    act_nn[r2] && (Mat[i, idx[(:nn, r2)]] += b_ * n_n[r2])
                end
                if act_bo[r]
                    Mat[i, idx[(:bo, r)]] += s_ * (n_b[r] + 1)
                end
                for r2 in 1:R
                    r2 == r && continue
                    act_bo[r2] && (Mat[i, idx[(:bo, r2)]] += s_ * n_b[r2])
                end
                for r2 in 1:R
                    act_oo[r2] && (Mat[i, idx[(:oo, r2)]] += s_ * n_o[r2])
                end

            elseif kind == :nn
                rhs[i] = A - c_n[r]
                Mat[i, idx[(:nn, r)]] += b_ * (n_n[r] + 1)
                for r2 in 1:R
                    r2 == r && continue
                    act_nn[r2] && (Mat[i, idx[(:nn, r2)]] += b_ * n_n[r2])
                end
                for r2 in 1:R
                    act_bn[r2] && (Mat[i, idx[(:bn, r2)]] += b_ * n_b[r2])
                end
                for r2 in 1:R
                    act_oo[r2] && (Mat[i, idx[(:oo, r2)]] += s_ * n_o[r2])
                end
                for r2 in 1:R
                    act_bo[r2] && (Mat[i, idx[(:bo, r2)]] += s_ * n_b[r2])
                end
            end
        end

        q_vec = Mat \ rhs

        for (i, (kind, r)) in enumerate(vars)
            if     kind == :oo; q_oo[r] = q_vec[i]
            elseif kind == :bo; q_bo[r] = q_vec[i]
            elseif kind == :bn; q_bn[r] = q_vec[i]
            elseif kind == :nn; q_nn[r] = q_vec[i]
            end
        end

        changed = false
        for r in 1:R
            if act_oo[r] && q_oo[r] <= 0.0; act_oo[r] = false; changed = true; end
            if act_bo[r] && q_bo[r] <= 0.0; act_bo[r] = false; changed = true; end
            if act_bn[r] && q_bn[r] <= 0.0; act_bn[r] = false; changed = true; end
            if act_nn[r] && q_nn[r] <= 0.0; act_nn[r] = false; changed = true; end
        end

        if !changed
            return (q_oo, q_bo, q_bn, q_nn)
        end
    end

    return (zero_v(), zero_v(), zero_v(), zero_v())
end
