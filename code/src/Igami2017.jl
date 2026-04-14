module Igami2017

include("parameters.jl")
include("cournot.jl")
include("state_space.jl")
include("solver.jl")

export R
export Params, default_params
export State, StateCCPs, EV, all_states, total_firms
export c_n_eff, c_n_vec
export cournot_profits, cournot_profits_regional
export compute_terminal_values, solve_2period, solve_initial, solve_state

end
