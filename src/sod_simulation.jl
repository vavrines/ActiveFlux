"""
Run the active-flux full-Boltzmann Sod simulations and save JLD2 results.

The default program executes the three reference Knudsen numbers `1e-4`,
`1e-2`, and `1`, writing one file immediately after each calculation. Pass an
optional first command-line argument to change the output directory. The
plotting stack is not loaded by this program. Launch Julia with `--threads=4`
to use four thread-private FSM collision workers.
"""

include(joinpath(@__DIR__, "sod.jl"))
using JLD2

const SOD_REFERENCE_KNUDSEN_NUMBERS = (1e-4, 1e-2, 1.0)
const DEFAULT_SOD_DATA_DIRECTORY = joinpath(@__DIR__, "..", "data")

"""Return a stable filename for one of the three documented Knudsen values."""
function sod_data_filename(knudsen)
    isapprox(knudsen, 1e-4; rtol=0, atol=eps()) && return "sod_kn1e-4.jld2"
    isapprox(knudsen, 1e-2; rtol=0, atol=eps()) && return "sod_kn1e-2.jld2"
    isapprox(knudsen, 1.0; rtol=0, atol=eps()) && return "sod_kn1.jld2"
    tag = replace(string(knudsen), "." => "p", "-" => "m", "+" => "")
    return "sod_kn$(tag).jld2"
end

"""Solve one full-Boltzmann Sod case and write its compressed result."""
function run_sod_simulation(;
    knudsen,
    output_path=joinpath(
        DEFAULT_SOD_DATA_DIRECTORY, sod_data_filename(knudsen),
    ),
    kwargs...,
)
    result = solve_sod_active_flux(; knudsen, kwargs...)
    data = sod_result_data(result)
    mkpath(dirname(output_path))
    JLD2.jldsave(output_path, true; result=data)

    print_sod_diagnostics(result)
    println("  saved JLD2 result: ", abspath(output_path))
    println("  file size (MiB): ", round(filesize(output_path) / 2^20; digits=3))
    return data
end

"""Run and save all requested Knudsen-number cases sequentially."""
function run_sod_reference_cases(;
    output_directory=DEFAULT_SOD_DATA_DIRECTORY,
    knudsen_numbers=SOD_REFERENCE_KNUDSEN_NUMBERS,
    kwargs...,
)
    mkpath(output_directory)
    results = Dict{Float64,Any}()
    for knudsen in knudsen_numbers
        println("Starting full-Boltzmann Sod case at Kn=$knudsen")
        output_path = joinpath(output_directory, sod_data_filename(knudsen))
        results[Float64(knudsen)] = run_sod_simulation(;
            knudsen,
            output_path,
            kwargs...,
        )
    end
    return results
end

"""
Run the same small-Knudsen problem with all three transport limiter modes.

This controlled experiment is intentionally separate from the three-Knudsen
reference sweep. Its default coarse velocity grid makes the limiter comparison
affordable; it must not be used as the production FSM resolution for Figure 4.
Each portable result records Euler errors and the minimum distribution both
before and after the collision projection.
"""
function run_sod_limiter_comparison(;
    output_directory=joinpath(DEFAULT_SOD_DATA_DIRECTORY, "limiter_comparison"),
    limiter_modes=SOD_LIMITER_MODES,
    knudsen=1e-4,
    nx=50,
    nu=16,
    nv=12,
    nw=12,
    kwargs...,
)
    mkpath(output_directory)
    results = Dict{Symbol,Any}()
    for limiter in limiter_modes
        limiter = sod_limiter_mode(limiter)
        println("Starting Sod limiter comparison with limiter=$limiter")
        output_path = joinpath(output_directory, "sod_limiter_$(limiter).jld2")
        results[limiter] = run_sod_simulation(;
            knudsen,
            output_path,
            nx,
            nu,
            nv,
            nw,
            limiter,
            kwargs...,
        )
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_directory = isempty(ARGS) ? DEFAULT_SOD_DATA_DIRECTORY : abspath(ARGS[1])
    run_sod_reference_cases(; output_directory)
end
