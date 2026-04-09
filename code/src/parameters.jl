"""
Model parameters for the 2-period Igami (2017) extension with agglomeration.

Calibration is illustrative (not estimated from data).
"""
struct Params
    A::Float64      # demand intercept: P_g = A - B*(Q_g/M) - rho*B*(Q_{-g}/M)
    B::Float64      # own-market demand slope
    M::Float64      # market size
    c_o::Float64    # old-tech marginal cost
    c_n0::Float64   # new-tech baseline marginal cost (before agglomeration)
    beta::Float64   # discount factor
    kappa::Float64  # innovation cost (old → both)
    phi::Float64    # entry cost (pe → new)
    sigma::Float64  # scale of EVT1 private cost shocks
    gamma::Float64  # agglomeration parameter (γ ≥ 0)
    rho::Float64    # cross-market substitution: S = rho*B; rho ∈ [0,1)
    N_max::Int      # max total active firms (bounds state space)
end

"""
Default plausible calibration. Pass gamma= or rho= to vary agglomeration/substitution.
rho=0.5: moderate substitution between old-gen and new-gen products.
"""
function default_params(; gamma::Float64 = 0.05, rho::Float64 = 0.5)
    return Params(
        3.0,    # A
        1.0,    # B
        1.0,    # M
        1.5,    # c_o
        0.5,    # c_n0
        0.9,    # beta
        0.3,    # kappa
        0.2,    # phi
        1.0,    # sigma
        gamma,  # gamma
        rho,    # rho
        8       # N_max
    )
end
