"""
Model parameters for the 2-period Igami (2017) extension with *regional*
agglomeration.

Calibration is illustrative (not estimated from data).  The model has a fixed
number of regions `R = 3`.  Agglomeration spillovers are region-specific:
region `r`'s new-tech cost falls in the number of innovators located in `r`.
"""
const R = 3

struct Params
    A::Float64                      # demand intercept: P_g = A - B*(Q_g/M) - rho*B*(Q_{-g}/M)
    B::Float64                      # own-market demand slope
    M::Float64                      # market size
    c_o::Float64                    # old-tech marginal cost (common across regions)
    c_n0::Float64                   # new-tech baseline marginal cost
    beta::Float64                   # discount factor
    kappa::Float64                  # innovation cost (old → both)
    phi::Float64                    # entry cost (pe → new)
    sigma::Float64                  # scale of EVT1 private cost shocks
    gamma::NTuple{R,Float64}        # per-region agglomeration parameters (γ_r ≥ 0)
    rho::Float64                    # cross-market substitution: S = rho*B; rho ∈ [0,1)
    N_max::Int                      # max total firms incl. potential entrants (bounds state space)
end

"""
    default_params(; gamma = 0.05, rho = 0.5, N_max = 6)

Default plausible calibration.  `gamma` may be a scalar (applied to every
region) or an `NTuple{3,Float64}` for region-specific values.  `N_max` defaults
to 6 on this branch because the regional state space has 4·R = 12 bins and
blows up quickly.
"""
function default_params(; gamma = 0.05,
                          rho::Float64 = 0.5,
                          N_max::Int = 6)
    γ = gamma isa Number ? ntuple(_ -> Float64(gamma), R) :
                           NTuple{R,Float64}(gamma)
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
        γ,      # gamma per region
        rho,
        N_max
    )
end
