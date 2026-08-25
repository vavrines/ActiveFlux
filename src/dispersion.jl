"""
Fourier analysis of the one-dimensional active-flux transport operator.

The script evaluates the two eigenbranches associated with the cell average
and the shared interface point value. It compares the semi-discrete symbol
with the fully discrete SSPRK33 amplification factors and writes the figure
used in the manuscript to `tex/figures/transport-spectrum.pdf`.
"""

using LinearAlgebra

# A direct script run writes a PDF and does not need an interactive GR window.
if (abspath(PROGRAM_FILE) == @__FILE__) && !haskey(ENV, "GKSwstype")
    ENV["GKSwstype"] = "100"
end
using Plots

const DEFAULT_DISPERSION_FIGURE = joinpath(
    @__DIR__, "..", "tex", "figures", "transport-spectrum.pdf",
)

"""Return the physical and internal eigenvalues of the dimensionless symbol."""
function active_flux_modes(theta)
    shift = cis(-theta)
    symbol = ComplexF64[
        0.0 1.0-shift
        -6.0 4.0+2.0shift
    ]
    modes = eigvals(symbol)

    # At zero wavenumber the physical mode is exactly zero. For positive
    # wavenumber it is the right-propagating branch with positive imaginary
    # part. This directional criterion also remains unambiguous at theta=pi,
    # where the two eigenvalues happen to have the same magnitude.
    physical_index = iszero(theta) ? argmin(abs.(modes)) : argmax(imag.(modes))
    internal_index = 3 - physical_index
    return modes[physical_index], modes[internal_index]
end

"""Stability polynomial of the three-stage, third-order SSP Runge--Kutta method."""
ssprk33(z) = 1 + z + z^2 / 2 + z^3 / 6

"""Compute semi-discrete and SSPRK33 spectral diagnostics on one theta grid."""
function spectral_diagnostics(thetas, cfl_values)
    physical = ComplexF64[]
    internal = ComplexF64[]
    for theta in thetas
        physical_mode, internal_mode = active_flux_modes(theta)
        push!(physical, physical_mode)
        push!(internal, internal_mode)
    end

    phase_semidiscrete = ones(length(thetas))
    phase_semidiscrete[2:end] .= imag.(physical[2:end]) ./ thetas[2:end]
    damping_semidiscrete = real.(physical)

    fully_discrete = map(cfl_values) do cfl
        physical_factor = ssprk33.(-cfl .* physical)
        internal_factor = ssprk33.(-cfl .* internal)

        phase_ratio = ones(length(thetas))
        phase_ratio[2:end] .=
            -angle.(physical_factor[2:end]) ./ (cfl .* thetas[2:end])
        damping_rate = -log.(abs.(physical_factor)) ./ cfl
        internal_amplitude = abs.(internal_factor)
        (; cfl, phase_ratio, damping_rate, internal_amplitude)
    end
    return (; phase_semidiscrete, damping_semidiscrete, fully_discrete)
end

"""Maximum amplification over both eigenbranches and all resolved wavenumbers."""
function spectral_radius(cfl, thetas)
    radius = 0.0
    for theta in thetas
        physical, internal = active_flux_modes(theta)
        radius = max(
            radius,
            abs(ssprk33(-cfl * physical)),
            abs(ssprk33(-cfl * internal)),
        )
    end
    return radius
end

"""Locate the SSPRK33 transport CFL limit by bisection of the spectral radius."""
function stable_cfl_limit(thetas; lower=0.0, upper=0.5, iterations=60)
    spectral_radius(lower, thetas) <= 1 + 1e-12 ||
        error("lower CFL bracket is unstable")
    spectral_radius(upper, thetas) > 1 + 1e-12 ||
        error("upper CFL bracket is stable")
    for _ in 1:iterations
        midpoint = (lower + upper) / 2
        if spectral_radius(midpoint, thetas) <= 1 + 1e-12
            lower = midpoint
        else
            upper = midpoint
        end
    end
    return lower
end

"""Build and save the three-panel transport spectrum figure."""
function plot_transport_spectrum(;
    output_path=DEFAULT_DISPERSION_FIGURE,
    cfl_values=(0.1, 0.25, 0.4),
    ntheta=1201,
)
    thetas = collect(range(0.0, pi; length=ntheta))
    diagnostics = spectral_diagnostics(thetas, cfl_values)
    normalized_wavenumber = thetas ./ pi
    colors = (:royalblue3, :darkorange2, :seagreen4)

    common = (
        xlabel="normalized wavenumber",
        xlims=(0, 1),
        grid=:on,
        gridalpha=0.18,
        framestyle=:box,
        legendfontsize=8,
        guidefontsize=9,
        tickfontsize=8,
        titlefontsize=10,
    )

    phase_panel = plot(
        normalized_wavenumber,
        diagnostics.phase_semidiscrete;
        label="semi-discrete",
        color=:black,
        line=:dash,
        lw=1.8,
        ylabel="phase-speed ratio",
        title="(a) Dispersion",
        ylims=(0.98, 1.075),
        common...,
    )
    hline!(phase_panel, [1.0]; label=false, color=:gray55, line=:dot, lw=1.0)

    damping_panel = plot(
        normalized_wavenumber,
        diagnostics.damping_semidiscrete;
        label="semi-discrete",
        color=:black,
        line=:dash,
        lw=1.8,
        ylabel="normalized damping",
        title="(b) Dissipation",
        ylims=(0, 1.7),
        common...,
    )

    internal_panel = plot(;
        ylabel="internal-mode amplitude",
        title="(c) Internal branch",
        ylims=(0, 1.02),
        common...,
    )

    for (curve, color) in zip(diagnostics.fully_discrete, colors)
        label = "SSPRK33, c=$(curve.cfl)"
        plot!(
            phase_panel,
            normalized_wavenumber,
            curve.phase_ratio;
            label,
            color,
            lw=1.6,
        )
        plot!(
            damping_panel,
            normalized_wavenumber,
            curve.damping_rate;
            label,
            color,
            lw=1.6,
        )
        plot!(
            internal_panel,
            normalized_wavenumber,
            curve.internal_amplitude;
            label="c=$(curve.cfl)",
            color,
            lw=1.6,
        )
    end

    figure = plot(
        phase_panel,
        damping_panel,
        internal_panel;
        layout=(1, 3),
        size=(1080, 345),
        margin=3Plots.mm,
    )
    mkpath(dirname(output_path))
    savefig(figure, output_path)

    # A finer theta grid makes the reported stability threshold independent
    # of the plotting resolution to the shown four significant digits.
    stability_thetas = collect(range(0.0, pi; length=20001))
    cfl_limit = stable_cfl_limit(stability_thetas)
    println("Saved transport spectrum: ", abspath(output_path))
    println("SSPRK33 active-flux transport CFL limit: ", cfl_limit)
    return (; figure, cfl_limit, diagnostics)
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_path = isempty(ARGS) ? DEFAULT_DISPERSION_FIGURE : abspath(ARGS[1])
    plot_transport_spectrum(; output_path)
end
