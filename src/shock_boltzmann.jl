"""
Run only the full-Boltzmann normal-shock simulation and save a JLD2 result.

This is the expensive part of Case 3.  The plotting program never includes
the solver or rebuilds the fast-spectral kernel; it reads the portable arrays
written here.  Pass an optional first command-line argument to change the
output filename.
"""

include(joinpath(@__DIR__, "shock.jl"))
using JLD2

const DEFAULT_BOLTZMANN_SHOCK_DATA = joinpath(
    @__DIR__, "..", "data", "shock_boltzmann.jld2",
)

"""Solve the full Boltzmann case and write its portable result dictionary."""
function run_boltzmann_shock(;
    output_path=DEFAULT_BOLTZMANN_SHOCK_DATA,
    kwargs...,
)
    result = solve_normal_shock_active_flux(;
        collision_model=:fsm,
        kwargs...,
    )
    data = normal_shock_result_data(result)
    mkpath(dirname(output_path))

    # Compression is worthwhile because the cell and interface distributions
    # contain large smooth/near-zero velocity-space regions.
    JLD2.jldsave(output_path, true; result=data)
    print_normal_shock_diagnostics(result)
    println("  saved JLD2 result: ", abspath(output_path))
    println("  file size (MiB): ", round(filesize(output_path) / 2^20; digits=3))
    return data
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_path = isempty(ARGS) ? DEFAULT_BOLTZMANN_SHOCK_DATA : abspath(ARGS[1])
    run_boltzmann_shock(; output_path)
end
