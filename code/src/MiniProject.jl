module MiniProject

include("parameters.jl")
include("cournot.jl")
include("state_space.jl")
include("solver.jl")
include("simulate.jl")
include("estimate.jl")
include("welfare.jl")
include("planner.jl")

export R
export Params, default_params
export State, StateCCPs, EV, all_states
export c_n_eff, c_n_vec
export cournot_profits_regional, cournot_quantities_regional
export compute_terminal_values, solve_initial, solve_state
export simulate_market, random_s0
export MarketData, recover_markets, nls_gamma, loglik_actions,
       loglik_market, mle_kappa_phi, bhhh_se
export consumer_surplus, consumer_surplus_from_quantities,
       producer_surplus_by_region, welfare_for_market, expected_welfare_mc
export competitive_outcome, social_welfare_static,
       planner_welfare_at, expected_planner_welfare

end
