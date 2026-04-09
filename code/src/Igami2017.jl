module Igami2017

include("parameters.jl")
include("cournot.jl")
include("state_space.jl")
include("solver.jl")

export Params, default_params
export State, StateCCPs, all_states, c_n_eff
export cournot_profits
export compute_terminal_values, solve_2period
export solve_pe_stage, solve_new_stage, solve_both_stage, solve_old_stage

end
