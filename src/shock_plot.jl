"""
Plot the normal-shock comparison from two previously saved JLD2 files.

This program performs no kinetic simulation.  It can therefore be rerun
quickly while adjusting labels, layout, or derived diagnostics.  Its optional
command-line arguments are, in order,

    Boltzmann-data.jld2  BGK-data.jld2  output-figure.pdf
"""

using JLD2
using LinearAlgebra

# A direct script run writes a PDF and does not need an interactive GR window.
if (abspath(PROGRAM_FILE) == @__FILE__) && !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "100"
end
using Plots

const DEFAULT_BOLTZMANN_SHOCK_DATA = joinpath(
    @__DIR__, "..", "data", "shock_boltzmann.jld2",
)
const DEFAULT_BGK_SHOCK_DATA = joinpath(
    @__DIR__, "..", "data", "shock_bgk.jld2",
)
const DEFAULT_SHOCK_FIGURE = joinpath(
    @__DIR__, "..", "tex", "figures", "normal-shock-comparison.pdf",
)

"""Load and minimally validate one portable normal-shock result."""
function load_normal_shock_data(path, expected_model)
    isfile(path) || error(
        "missing $expected_model result: $path\n" *
        "Run the corresponding simulation program first.",
    )
    data = JLD2.load(path, "result")
    get(data, "format_version", 0) == 1 ||
        error("unsupported shock-result format in $path")
    data["collision_model"] == expected_model || error(
        "expected $expected_model data in $path, found " *
        string(data["collision_model"]),
    )
    if expected_model == "fsm"
        get(data["parameters"], "fsm_modes", nothing) == 5 || error(
            "the Boltzmann comparison requires a five-mode FSM result in " *
            "$path; rerun shock_boltzmann.jl with nm=5",
        )
    end
    return data
end

"""Compute relative profile differences after checking that the grids agree."""
function saved_shock_differences(boltzmann, bgk)
    bp = boltzmann["profile"]
    gp = bgk["profile"]
    bp["x"] == gp["x"] || error("Boltzmann and BGK physical grids differ")

    relative_l2(field) = norm(bp[field] - gp[field]) /
                         max(norm(bp[field]), eps())
    return Dict(
        "density" => relative_l2("density"),
        "velocity" => relative_l2("velocity"),
        "temperature" => relative_l2("temperature"),
        "heat_flux" => relative_l2("heat_flux"),
    )
end

"""Build the four-panel FSM/BGK comparison solely from saved arrays."""
function plot_saved_normal_shock(boltzmann, bgk)
    bp = boltzmann["profile"]
    gp = bgk["profile"]
    left = boltzmann["reservoir"]["primitive_left"]
    right = boltzmann["reservoir"]["primitive_right"]
    temperature_left = 1 / (2left[5])
    temperature_right = 1 / (2right[5])

    p_density = plot(
        bp["x"], bp["density"];
        label="Boltzmann FSM", lw=2, ylabel="density",
    )
    plot!(p_density, gp["x"], gp["density"]; label="BGK", lw=2, line=:dash)
    hline!(p_density, [left[1], right[1]]; label=false, color=:gray, line=:dot)

    p_velocity = plot(
        bp["x"], bp["velocity"];
        label="Boltzmann FSM", lw=2, ylabel="velocity",
    )
    plot!(p_velocity, gp["x"], gp["velocity"]; label="BGK", lw=2, line=:dash)
    hline!(p_velocity, [left[2], right[2]]; label=false, color=:gray, line=:dot)

    p_temperature = plot(
        bp["x"], bp["temperature"];
        label="Boltzmann FSM", lw=2, xlabel="x", ylabel="temperature",
    )
    plot!(
        p_temperature, gp["x"], gp["temperature"];
        label="BGK", lw=2, line=:dash,
    )
    hline!(
        p_temperature, [temperature_left, temperature_right];
        label=false, color=:gray, line=:dot,
    )

    p_heat = plot(
        bp["x"], bp["heat_flux"];
        label="Boltzmann FSM", lw=2, xlabel="x", ylabel="heat flux",
    )
    plot!(p_heat, gp["x"], gp["heat_flux"]; label="BGK", lw=2, line=:dash)

    mach = boltzmann["parameters"]["mach"]
    time = boltzmann["time"]
    return plot(
        p_density,
        p_velocity,
        p_temperature,
        p_heat;
        layout=(2, 2),
        size=(920, 700),
        plot_title="Preliminary normal shock: Ma=$mach, " *
                   "t=$(round(time; digits=3))",
    )
end

"""Load both simulations, report their differences, and write the figure."""
function plot_saved_normal_shock(;
    boltzmann_path=DEFAULT_BOLTZMANN_SHOCK_DATA,
    bgk_path=DEFAULT_BGK_SHOCK_DATA,
    output_path=DEFAULT_SHOCK_FIGURE,
)
    boltzmann = load_normal_shock_data(boltzmann_path, "fsm")
    bgk = load_normal_shock_data(bgk_path, "bgk")
    differences = saved_shock_differences(boltzmann, bgk)
    figure = plot_saved_normal_shock(boltzmann, bgk)
    mkpath(dirname(output_path))
    savefig(figure, output_path)

    println("Saved normal-shock comparison")
    println("  Boltzmann data: ", abspath(boltzmann_path))
    println("  BGK data: ", abspath(bgk_path))
    println("  relative profile differences: ", differences)
    println("  figure: ", abspath(output_path))
    return (; figure, boltzmann, bgk, differences)
end

if abspath(PROGRAM_FILE) == @__FILE__
    boltzmann_path = length(ARGS) >= 1 ? abspath(ARGS[1]) :
                       DEFAULT_BOLTZMANN_SHOCK_DATA
    bgk_path = length(ARGS) >= 2 ? abspath(ARGS[2]) : DEFAULT_BGK_SHOCK_DATA
    output_path = length(ARGS) >= 3 ? abspath(ARGS[3]) : DEFAULT_SHOCK_FIGURE
    plot_saved_normal_shock(; boltzmann_path, bgk_path, output_path)
end
