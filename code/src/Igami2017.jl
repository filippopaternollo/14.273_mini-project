module Igami2017

include("parameters.jl")
include("cournot.jl")
include("state_space.jl")
include("solver.jl")

export Params, default_params
export State, StateCCPs, all_states, c_n_eff, uniform_ccps, expected_continuation
export cournot_profits
export compute_terminal_values, solve_2period, systematic_utilities

end
