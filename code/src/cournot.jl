"""
Static Cournot equilibrium for two symmetric groups of firms.

Inverse demand: P = A - B*(Q/M)
Group 1: n_o firms at marginal cost c_o  (old technology)
Group 2: n_in firms at marginal cost c_n (new technology)

FOC for old firm i:
    A - B*(n_o+1)*q_o/M - B*n_in*q_n/M = c_o
    → (n_o+1)*q_o + n_in*q_n = M*(A-c_o)/B     ... (1)

FOC for new firm j:
    A - B*n_o*q_o/M - B*(n_in+1)*q_n/M = c_n
    → n_o*q_o + (n_in+1)*q_n = M*(A-c_n)/B     ... (2)

System determinant: (n_o+1)*(n_in+1) - n_o*n_in = n_o + n_in + 1

Returns (pi_o, pi_n): per-firm profits for each group.
If a group would produce negative quantity, it is excluded and the
single-group symmetric equilibrium is used for the remaining group.
"""
function cournot_profits(n_o::Int, n_in::Int, c_o::Float64, c_n::Float64, p::Params)
    A, B, M = p.A, p.B, p.M

    if n_o == 0 && n_in == 0
        return (0.0, 0.0)
    end

    alpha_o = M * (A - c_o) / B
    alpha_n = M * (A - c_n) / B

    if n_o == 0
        q_n = max(0.0, alpha_n / (n_in + 1))
        return (0.0, (B / M) * q_n^2)
    end

    if n_in == 0
        q_o = max(0.0, alpha_o / (n_o + 1))
        return ((B / M) * q_o^2, 0.0)
    end

    # General 2x2 case
    denom = n_o + n_in + 1
    q_o = ((n_in + 1) * alpha_o - n_in * alpha_n) / denom
    q_n = ((n_o + 1) * alpha_n - n_o * alpha_o) / denom

    # Handle corner: one group is priced out
    if q_o <= 0.0 && q_n <= 0.0
        return (0.0, 0.0)
    elseif q_o <= 0.0
        q_n_only = max(0.0, alpha_n / (n_in + 1))
        return (0.0, (B / M) * q_n_only^2)
    elseif q_n <= 0.0
        q_o_only = max(0.0, alpha_o / (n_o + 1))
        return ((B / M) * q_o_only^2, 0.0)
    end

    return ((B / M) * q_o^2, (B / M) * q_n^2)
end
