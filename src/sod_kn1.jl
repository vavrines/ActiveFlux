"""Run only the production full-Boltzmann Sod case at Kn=1."""

include(joinpath(@__DIR__, "sod_simulation.jl"))

function main_sod_kn1(args=ARGS; kwargs...)
    output_directory = isempty(args) ? DEFAULT_SOD_DATA_DIRECTORY :
                       abspath(first(args))
    output_path = joinpath(output_directory, sod_data_filename(1.0))
    return run_sod_simulation(; knudsen=1.0, output_path, kwargs...)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_sod_kn1()
