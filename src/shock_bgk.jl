"""
Run only the BGK normal-shock simulation and save a JLD2 result.

The initial data, active-flux transport, physical/velocity grids, and boundary
conditions are identical to `shock_boltzmann.jl`; only the local collision
backend is changed.  Pass an optional first command-line argument to change the
output filename.
"""

include(joinpath(@__DIR__, "shock.jl"))
using JLD2

const DEFAULT_BGK_SHOCK_DATA = joinpath(
    @__DIR__, "..", "data", "shock_bgk.jld2",
)

"""Solve the BGK case and write its portable result dictionary."""
function run_bgk_shock(; output_path=DEFAULT_BGK_SHOCK_DATA, kwargs...)
    result = solve_normal_shock_active_flux(;
        collision_model=:bgk,
        kwargs...,
    )
    data = normal_shock_result_data(result)
    mkpath(dirname(output_path))
    JLD2.jldsave(output_path, true; result=data)
    print_normal_shock_diagnostics(result)
    println("  saved JLD2 result: ", abspath(output_path))
    println("  file size (MiB): ", round(filesize(output_path) / 2^20; digits=3))
    return data
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_path = isempty(ARGS) ? DEFAULT_BGK_SHOCK_DATA : abspath(ARGS[1])
    run_bgk_shock(; output_path)
end
