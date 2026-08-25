"""
Plot the full-Boltzmann Sod comparison from three saved JLD2 results.

This program performs no kinetic simulation. Its optional command-line
arguments are, in order,

    Kn1e-4.jld2  Kn1e-2.jld2  Kn1.jld2  output-figure.pdf
"""

using JLD2

# A direct script run writes a PDF and does not need an interactive GR window.
if (abspath(PROGRAM_FILE) == @__FILE__) && !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "100"
end
using Plots

const DEFAULT_SOD_DATA_PATHS = (
    joinpath(@__DIR__, "..", "data", "sod_kn1e-4.jld2"),
    joinpath(@__DIR__, "..", "data", "sod_kn1e-2.jld2"),
    joinpath(@__DIR__, "..", "data", "sod_kn1.jld2"),
)
const DEFAULT_SOD_FIGURE = joinpath(
    @__DIR__, "..", "tex", "figures", "sod-shock-tube.pdf",
)
const DEFAULT_SOD_CONTINUUM_FIGURE = joinpath(
    @__DIR__, "..", "tex", "figures", "sod-continuum-check.png",
)

"""Load and minimally validate one portable Sod result."""
function load_sod_data(path, expected_knudsen)
    isfile(path) || error(
        "missing Sod result: $path\nRun sod_simulation.jl first.",
    )
    data = JLD2.load(path, "result")
    get(data, "format_version", 0) == 4 ||
        error("unsupported Sod-result format in $path")
    get(data, "case", "") == "sod" ||
        error("expected Sod data in $path")
    get(data, "collision_model", "") == "fsm" ||
        error("expected full-Boltzmann FSM data in $path")
    parameters = data["parameters"]
    get(parameters, "fsm_modes", nothing) == 5 || error(
        "Figure 4 requires a five-mode FSM result in $path; " *
        "rerun the simulation with nm=5",
    )
    get(parameters, "transport_limiter", "") == "macroscopic_local" ||
        error("Figure 4 requires macroscopic-local limiter data in $path")
    knudsen = parameters["knudsen"]
    isapprox(knudsen, expected_knudsen; rtol=1e-12) || error(
        "expected Kn=$expected_knudsen in $path, found $knudsen",
    )
    return data
end

"""Build the three-Knudsen comparison solely from saved profile arrays."""
function plot_saved_sod(data_sets)
    first_data = first(data_sets)
    first_profile = first_data["profile"]
    x = first_profile["x"]
    reference_parameters = first_data["parameters"]
    reference_grid = first_data["grid"]
    for data in data_sets
        data["profile"]["x"] == x || error("Sod physical grids differ")
        parameters = data["parameters"]
        for key in (
            "nx", "nu", "nv", "nw", "x0", "x1", "gamma", "cfl",
            "fsm_modes", "alpha", "omega", "transport_limiter",
            "sensor_smooth", "sensor_nonsmooth",
        )
            parameters[key] == reference_parameters[key] || error(
                "Sod parameter '$key' differs across saved results",
            )
        end
        for key in ("u", "v", "w")
            data["grid"][key] == reference_grid[key] || error(
                "Sod velocity grid '$key' differs across saved results",
            )
        end
    end

    fields = ("density", "velocity", "pressure", "heat_flux")
    ylabels = ("density", "velocity", "pressure", "heat flux")
    panels = [plot(; ylabel=label) for label in ylabels]
    labels = ("Kn=1e-4", "Kn=1e-2", "Kn=1")
    line_styles = (:solid, :dash, :dot)
    for (data, label, line_style) in zip(data_sets, labels, line_styles)
        profile = data["profile"]
        for (panel, field) in zip(panels, fields)
            plot!(panel, x, profile[field]; label, lw=1.8, line=line_style)
        end
    end
    for (panel, exact_field) in zip(
        panels[1:3],
        ("exact_density", "exact_velocity", "exact_pressure"),
    )
        plot!(
            panel, x, first_profile[exact_field];
            label="Euler exact", color=:black, line=:dashdot, lw=1.4,
        )
    end
    plot!(panels[3]; xlabel="x")
    plot!(panels[4]; xlabel="x")

    gamma = first(data_sets)["parameters"]["gamma"]
    time = first(data_sets)["time"]
    return plot(
        panels...;
        layout=(2, 2),
        size=(920, 700),
        plot_title="Full-Boltzmann Sod shock tube: " *
                   "t=$(round(time; digits=4)), γ=$(round(gamma; digits=4))",
    )
end

"""Plot the saved small-Knudsen result against the exact Euler solution."""
function plot_sod_continuum_check(;
    data_path=first(DEFAULT_SOD_DATA_PATHS),
    output_path=DEFAULT_SOD_CONTINUUM_FIGURE,
)
    data = load_sod_data(data_path, 1e-4)
    profile = data["profile"]
    x = profile["x"]
    panels = Any[]
    for (field, label) in (
        ("density", "density"),
        ("velocity", "velocity"),
        ("pressure", "pressure"),
    )
        panel = plot(
            x,
            profile[field];
            label="Active Flux, Kn=1e-4",
            lw=2,
            ylabel=label,
        )
        plot!(
            panel,
            x,
            profile["exact_" * field];
            label="Euler exact",
            color=:black,
            line=:dashdot,
            lw=1.5,
        )
        push!(panels, panel)
    end
    plot!(last(panels); xlabel="x")
    parameters = data["parameters"]
    nx = parameters["nx"]
    nu = parameters["nu"]
    nv = parameters["nv"]
    nw = parameters["nw"]
    figure = plot(
        panels...;
        layout=(3, 1),
        size=(800, 900),
        plot_title="Sod continuum check: nx=$nx, velocity=$nu×$nv×$nw",
    )
    mkpath(dirname(output_path))
    savefig(figure, output_path)
    println("Saved Sod continuum-limit check: ", abspath(output_path))
    return (; figure, data)
end

"""Load three simulations, build the figure, and write it to disk."""
function plot_saved_sod(;
    data_paths=DEFAULT_SOD_DATA_PATHS,
    output_path=DEFAULT_SOD_FIGURE,
)
    expected_knudsen = (1e-4, 1e-2, 1.0)
    length(data_paths) == 3 || error("three Sod data paths are required")
    data_sets = [
        load_sod_data(path, knudsen)
        for (path, knudsen) in zip(data_paths, expected_knudsen)
    ]
    figure = plot_saved_sod(data_sets)
    mkpath(dirname(output_path))
    savefig(figure, output_path)

    println("Saved full-Boltzmann Sod Knudsen comparison")
    for path in data_paths
        println("  data: ", abspath(path))
    end
    println("  figure: ", abspath(output_path))
    return (; figure, data_sets)
end

if abspath(PROGRAM_FILE) == @__FILE__
    data_paths = length(ARGS) >= 3 ? Tuple(abspath.(ARGS[1:3])) :
                 DEFAULT_SOD_DATA_PATHS
    output_path = length(ARGS) >= 4 ? abspath(ARGS[4]) : DEFAULT_SOD_FIGURE
    plot_saved_sod(; data_paths, output_path)
end
