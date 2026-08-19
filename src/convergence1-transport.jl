"""
Convergence test 1: exact test of the active-flux transport operator.

This script disables BGK relaxation and solves

    ∂ₜf + u ∂ₓf = 0

for every point `u` on the discrete velocity grid. On the periodic domain the
exact solution is `f(x,u,t) = f₀(x-u*t,u)`, so the transport discretization can
be tested independently of Strang splitting and BGK collision errors.
"""

include(joinpath(@__DIR__, "advection.jl"))

"""Evaluate the initial Maxwellian at one physical point and one velocity."""
function initial_distribution(x, velocity)
    ρ, U, λ = initial_primitive(x)
    return ρ * sqrt(λ / π) * exp(-λ * (velocity - U)^2)
end

"""Map a coordinate periodically onto the physical domain."""
function periodic_coordinate(x, x0, x1)
    return x0 + mod(x - x0, x1 - x0)
end

"""
Compute exact physical-cell averages of the free-transport solution.

The same four-point Gauss-Legendre rule used for numerical initialization is
applied to the smooth exact solution. Its quadrature error is much higher order
than the third-order transport error measured here.
"""
function exact_transport_cell_averages(ks, time)
    nx, nu = ks.ps.nx, ks.vs.nu
    exact = zeros(nx, nu)

    for i in 1:nx
        xcenter = ks.ps.x[i]
        dx = ks.ps.dx[i]
        for q in eachindex(GL4_NODES)
            xq = xcenter + 0.5 * dx * GL4_NODES[q]
            quadrature_weight = 0.5 * GL4_WEIGHTS[q]
            for j in 1:nu
                characteristic_foot = periodic_coordinate(
                    xq - ks.vs.u[j] * time,
                    ks.ps.x0,
                    ks.ps.x1,
                )
                exact[i, j] += quadrature_weight *
                               initial_distribution(characteristic_foot, ks.vs.u[j])
            end
        end
    end
    return exact
end

"""Relative L2 error of `f`, including physical and velocity quadrature."""
function transport_distribution_error(ks, numerical, exact)
    dx = ks.ps.dx[1]
    numerator = 0.0
    denominator = 0.0
    for i in axes(numerical, 1), j in axes(numerical, 2)
        weighted_volume = dx * ks.vs.weights[j]
        numerator += weighted_volume * (numerical[i, j] - exact[i, j])^2
        denominator += weighted_volume * exact[i, j]^2
    end
    return sqrt(numerator / denominator)
end

"""Relative physical-space L2 error for a cell-average scalar field."""
function transport_scalar_error(dx, numerical, exact)
    numerator = dx * sum(abs2, numerical - exact)
    denominator = dx * sum(abs2, exact)
    return sqrt(numerator / denominator)
end

"""Integrate only the active-flux transport operator to `final_time`."""
function solve_transport_active_flux(;
    nx=100,
    nu=32,
    cfl=0.5,
    final_time=0.1,
)
    ks = create_solver(; nx, nu, cfl, max_time=final_time)
    ctr, face = initialize_solution(ks)

    f_old = OffsetArray(zeros(nx + 3, nu), 0:nx+2, 1:nu)
    time = 0.0
    steps = 0
    while time < final_time
        dt = active_flux_timestep(ks, time)
        transport_step!(ks, ctr, face, dt, f_old)
        time += dt
        steps += 1
    end

    exact_f = exact_transport_cell_averages(ks, time)
    numerical_f = zeros(nx, nu)
    for i in 1:nx
        numerical_f[i, :] .= ctr[i].f
    end

    # Compute exact density consistently from the discrete velocity quadrature.
    exact_density = exact_f * ks.vs.weights
    numerical_density = [ctr[i].w[1] for i in 1:nx]
    f_error = transport_distribution_error(ks, numerical_f, exact_f)
    density_error = transport_scalar_error(
        ks.ps.dx[1],
        numerical_density,
        exact_density,
    )
    periodic_mismatch = maximum(abs, face[1].f - face[nx+1].f)

    return (;
        ks,
        ctr,
        face,
        time,
        steps,
        numerical_f,
        exact_f,
        f_error,
        density_error,
        minimum_distribution=minimum(numerical_f),
        periodic_mismatch,
    )
end

"""Run exact transport comparisons on successively refined physical grids."""
function transport_convergence_test(;
    grids=(20, 40, 80, 160, 320),
    nu=32,
    final_time=0.1,
    cfl=0.5,
)
    issorted(collect(grids)) || throw(ArgumentError("grids must be ordered coarse to fine"))

    rows = NamedTuple[]
    previous_f_error = nothing
    previous_density_error = nothing

    for nx in grids
        println("Computing exact transport comparison: nx=$nx, nu=$nu")
        result = solve_transport_active_flux(; nx, nu, cfl, final_time)
        f_order = isnothing(previous_f_error) ?
                  NaN : log2(previous_f_error / result.f_error)
        density_order = isnothing(previous_density_error) ?
                        NaN : log2(previous_density_error / result.density_error)

        push!(rows, (;
            nx,
            dx=result.ks.ps.dx[1],
            f_error=result.f_error,
            f_order,
            density_error=result.density_error,
            density_order,
            minimum_distribution=result.minimum_distribution,
            periodic_mismatch=result.periodic_mismatch,
        ))
        previous_f_error = result.f_error
        previous_density_error = result.density_error
    end
    return (; rows, grids, nu, final_time, cfl)
end

function print_transport_convergence(report)
    println()
    println("Exact active-flux transport convergence")
    println("Each row is (nx, f_error, f_order, density_error, density_order)")
    for row in report.rows
        println((
            nx=row.nx,
            f_error=row.f_error,
            f_order=row.f_order,
            density_error=row.density_error,
            density_order=row.density_order,
        ))
    end
    return nothing
end

"""Require asymptotic third-order transport convergence and valid solutions."""
function validate_transport_convergence(report; minimum_order=2.7)
    # The coarsest pair can be pre-asymptotic; enforce the threshold on all
    # subsequent refinements.
    measured_rows = report.rows[3:end]
    all(row.f_order >= minimum_order for row in measured_rows) ||
        error("distribution transport convergence fell below order $minimum_order")
    all(row.density_order >= minimum_order for row in measured_rows) ||
        error("density transport convergence fell below order $minimum_order")
    all(row.minimum_distribution > 0 for row in report.rows) ||
        error("the transport solution became non-positive")
    all(row.periodic_mismatch == 0 for row in report.rows) ||
        error("periodic interface values do not match")
    return true
end

function plot_transport_convergence(report)
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
    report = transport_convergence_test()
    validate_transport_convergence(report)
    print_transport_convergence(report)
    display(plot_transport_convergence(report))
end
