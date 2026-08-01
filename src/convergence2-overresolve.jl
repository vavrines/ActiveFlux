"""
Convergence test 2: temporally over-resolved active-flux BGK solver.

The full solver combines third-order active-flux transport with second-order
Strang splitting. A fixed-CFL refinement has `dt=O(dx)`, so the splitting error
is `O(dx^2)` and masks the transport order. Here each grid instead uses the
balanced refinement

    dt = C * dx^(3/2) / umax.

The splitting error then becomes `O(dx^3)`, compatible with the `O(dx^3)`
transport error. Its molecular CFL decreases only as `sqrt(dx)`, avoiding the
long small-CFL pre-asymptotic regime produced by `dt=O(dx^2)`. Since no exact
BGK solution is available, a fine-grid BGK solution is restricted
conservatively onto each comparison grid.
"""

include(joinpath(@__DIR__, "bgk_advection_convergence.jl"))

"""Return the refined time step `C*dx^power/umax` for one solver setup."""
function overresolved_timestep(ks, coefficient, power)
    dx = ks.ps.dx[1]
    umax = maximum(abs, ks.vs.u)
    return coefficient * dx^power / umax
end

"""Solve BGK with a time step proportional to `dx^time_power`."""
function solve_overresolved_bgk(;
    nx,
    nu,
    final_time,
    knudsen,
    time_coefficient,
    time_power,
)
    # Constructing this lightweight setup first gives the exact velocity bound
    # used by the solver's midpoint velocity grid.
    setup = create_solver(; nx, nu, knudsen, max_time=final_time)
    target_dt = overresolved_timestep(setup, time_coefficient, time_power)

    # Use a uniform step that divides the requested final time exactly. This
    # avoids a mesh-dependent short final Strang step in convergence studies.
    planned_steps = ceil(Int, final_time / target_dt)
    dt = final_time / planned_steps

    result = solve_bgk_active_flux(;
        nx,
        nu,
        knudsen,
        max_time=final_time,
        fixed_dt=dt,
    )
    return (;
        result,
        dt,
        target_dt,
        maximum_cfl=maximum(abs, result.ks.vs.u) * dt / result.ks.ps.dx[1],
    )
end

"""
Measure full-BGK convergence with negligible leading Strang error.

All grids use the same discrete velocity space. The default fine reference is
twice as fine as the finest measured grid (and four times finer than the next
grid), while the short final time keeps the temporally refined study affordable.
"""
function overresolved_bgk_convergence_test(;
    grids=(20, 40, 80, 160, 320),
    reference_nx=640,
    nu=24,
    final_time=0.005,
    knudsen=1e-2,
    time_coefficient=1.0,
    time_power=1.5,
)
    all(reference_nx % nx == 0 for nx in grids) ||
        throw(ArgumentError("all grids must divide reference_nx=$reference_nx"))
    issorted(collect(grids)) || throw(ArgumentError("grids must be ordered coarse to fine"))

    println(
        "Computing over-resolved BGK reference: nx=$reference_nx, nu=$nu, " *
        "dt=C*dx^$time_power/umax",
    )
    reference = solve_overresolved_bgk(;
        nx=reference_nx,
        nu,
        final_time,
        knudsen,
        time_coefficient,
        time_power,
    )
    reference_f = distribution_matrix(reference.result)
    reference_w = conservative_matrix(reference.result)

    rows = NamedTuple[]
    previous_f_error = nothing
    previous_density_error = nothing

    for nx in grids
        println("Computing over-resolved comparison: nx=$nx, nu=$nu")
        solved = solve_overresolved_bgk(;
            nx,
            nu,
            final_time,
            knudsen,
            time_coefficient,
            time_power,
        )
        result = solved.result

        numerical_f = distribution_matrix(result)
        numerical_w = conservative_matrix(result)
        restricted_f = restrict_cell_averages(reference_f, nx)
        restricted_w = restrict_cell_averages(reference_w, nx)

        f_error = relative_distribution_error(result.ks, numerical_f, restricted_f)
        density_error = relative_scalar_error(
            result.ks.ps.dx[1],
            numerical_w[:, 1],
            restricted_w[:, 1],
        )
        f_order = isnothing(previous_f_error) ?
                  NaN : log2(previous_f_error / f_error)
        density_order = isnothing(previous_density_error) ?
                        NaN : log2(previous_density_error / density_error)

        push!(rows, (;
            nx,
            dx=result.ks.ps.dx[1],
            dt=solved.dt,
            maximum_cfl=solved.maximum_cfl,
            steps=result.steps,
            f_error,
            f_order,
            density_error,
            density_order,
            minimum_distribution=result.stats.minimum_distribution,
            conservation_drift=maximum(abs, result.stats.relative_drift),
        ))
        previous_f_error = f_error
        previous_density_error = density_error
    end

    return (;
        rows,
        reference,
        grids,
        reference_nx,
        nu,
        final_time,
        knudsen,
        time_coefficient,
        time_power,
    )
end

function print_overresolved_convergence(report)
    println()
    println("Temporally over-resolved active-flux BGK convergence")
    println("Each row is (nx, dt, max_cfl, f_error, f_order, density_error, density_order)")
    for row in report.rows
        println((
            nx=row.nx,
            dt=row.dt,
            maximum_cfl=row.maximum_cfl,
            f_error=row.f_error,
            f_order=row.f_order,
            density_error=row.density_error,
            density_order=row.density_order,
        ))
    end
    return nothing
end

"""Require transport-dominated third-order convergence and valid solutions."""
function validate_overresolved_convergence(
    report;
    minimum_order=2.8,
    maximum_order_difference=0.15,
    conservation_tolerance=1e-7,
)
    # The small-CFL hierarchy is visibly pre-asymptotic on its coarse grids.
    # Require the finest observed rate to reach the third-order regime.
    final_row = last(report.rows)
    final_row.f_order >= minimum_order ||
        error("over-resolved distribution convergence fell below order $minimum_order")
    final_row.density_order >= minimum_order ||
        error("over-resolved density convergence fell below order $minimum_order")
    abs(final_row.f_order - final_row.density_order) <= maximum_order_difference ||
        error("distribution and density asymptotic orders are not compatible")
    all(row.minimum_distribution > 0 for row in report.rows) ||
        error("an over-resolved BGK distribution became non-positive")
    all(row.conservation_drift < conservation_tolerance for row in report.rows) ||
        error("conservation drift exceeded $conservation_tolerance")
    report.reference.result.stats.minimum_distribution > 0 ||
        error("the fine-grid reference distribution became non-positive")
    maximum(abs, report.reference.result.stats.relative_drift) < conservation_tolerance ||
        error("reference conservation drift exceeded $conservation_tolerance")
    return true
end

function plot_overresolved_convergence(report)
    dx = [row.dx for row in report.rows]
    f_error = [row.f_error for row in report.rows]
    density_error = [row.density_error for row in report.rows]

    figure = plot(
        dx,
        f_error;
        marker=:circle,
        xscale=:log10,
        yscale=:log10,
        xlabel="Δx",
        ylabel="relative L² error",
        label="distribution f",
        lw=1.5,
    )
    plot!(figure, dx, density_error; marker=:square, label="density", lw=1.5)
    third_order = f_error[1] .* (dx ./ dx[1]) .^ 3
    plot!(figure, dx, third_order; label="O(Δx³)", line=:dash, color=:black)
    return figure
end

if abspath(PROGRAM_FILE) == @__FILE__
    report = overresolved_bgk_convergence_test()
    validate_overresolved_convergence(report)
    print_overresolved_convergence(report)
    display(plot_overresolved_convergence(report))
end
