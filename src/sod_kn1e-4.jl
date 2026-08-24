"""Run only the production full-Boltzmann Sod case at Kn=1e-4."""

include(joinpath(@__DIR__, "sod_simulation.jl"))

function main_sod_kn1e_4(args=ARGS; kwargs...)
    output_directory = isempty(args) ? DEFAULT_SOD_DATA_DIRECTORY :
                       abspath(first(args))
    output_path = joinpath(output_directory, sod_data_filename(1e-4))
    return run_sod_simulation(; knudsen=1e-4, output_path, kwargs...)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_sod_kn1e_4()
