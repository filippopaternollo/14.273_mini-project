module MiniProject

include("parameters.jl")
include("cournot.jl")
include("state_space.jl")
include("solver.jl")
include("simulate.jl")
include("estimate.jl")

export R
export Params, default_params
export State, StateCCPs, EV, all_states, total_firms
export c_n_eff, c_n_vec
export cournot_profits, cournot_profits_regional, cournot_quantities_regional
export compute_terminal_values, solve_2period, solve_initial, solve_state
export simulate_market
export MarketData, recover_markets, nls_gamma, loglik_actions,
       loglik_market, mle_kappa_phi, bhhh_se

end
