"""
Model parameters for the 2-period Igami (2017) extension with *regional*
agglomeration.

Calibration is illustrative (not estimated from data).  The model has a fixed
number of regions `R = 3`.  Agglomeration spillovers are bloc-specific:
region `r`'s new-tech cost falls in the number of innovators located in `r`'s
*spillover bloc* (singleton blocs, the default, recover purely local spillovers).

`blocs` partitions the regions into *spillover pools*: regions sharing a
bloc id pool their innovator count when the new-tech cost is computed
(see `c_n_eff`).  The default `(1, 2, 3)` puts each region in its own bloc
and recovers the original local-spillover model bit-for-bit.  Counterfactuals
that "merge" two regions set their bloc ids equal — e.g. `(1, 1, 2)`
represents an alliance between regions 1 and 2.
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
    blocs::NTuple{R,Int}            # spillover-pool ids; default (1,2,3) → singletons
    subsidy::NTuple{R,Float64}      # per-region innovation subsidy τ_r ≥ 0; firm pays κ − τ_r
end

"""
    default_params(; gamma = 0.15, sigma = 0.5, rho = 0.5, N_max = 6,
                     blocs = (1, 2, 3), kappa = 0.3, phi = 0.2)

Default plausible calibration.  `gamma` may be a scalar (applied to every
region) or an `NTuple{3,Float64}` for region-specific values.  `N_max` defaults
to 6 on this branch because the regional state space has 4·R = 12 bins and
blows up quickly.  `blocs` defaults to singleton pools; pass e.g. `(1, 1, 2)`
to merge regions 1 and 2 into a single spillover pool.  `kappa`, `phi`, and
`sigma` are exposed so counterfactual scripts can plug in estimated values
without constructing a `Params` from scratch.

The calibration `(γ, σ) = (0.15, 0.5)` is chosen to put innovation in the
*responsive* region of the logit (γ below the c_n floor at γ ≳ 0.30, σ low
enough that spillover changes shift CCPs by visible amounts).  At this
calibration the EU–US alliance counterfactual produces innovation-rate
shifts of order 6 % and welfare gains of a few percent in the allied
regions.  The earlier calibration `(0.05, 1.0)` damped both effects to
under 1 %.
"""
function default_params(; gamma = 0.15,
                          sigma::Float64 = 0.5,
                          rho::Float64 = 0.5,
                          N_max::Int = 6,
                          blocs::NTuple{R,Int} = (1, 2, 3),
                          kappa::Float64 = 0.3,
                          phi::Float64 = 0.2,
                          subsidy = (0.0, 0.0, 0.0))
    γ = gamma isa Number ? ntuple(_ -> Float64(gamma), R) :
                           NTuple{R,Float64}(gamma)
    τ = subsidy isa Number ? ntuple(_ -> Float64(subsidy), R) :
                             NTuple{R,Float64}(subsidy)
    return Params(
        3.0,    # A
        1.0,    # B
        1.0,    # M
        1.5,    # c_o
        0.5,    # c_n0
        0.9,    # beta
        kappa,
        phi,
        sigma,
        γ,      # gamma per region
        rho,
        N_max,
        blocs,
        τ       # per-region innovation subsidy
    )
end
