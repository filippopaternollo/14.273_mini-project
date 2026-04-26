using LinearAlgebra

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
