"""
Sod shock tube for the one-dimensional active-flux BGK solver.

The kinetic model is the pure `1d1f1v` equation with `K=0` and `γ=3`.  The
standard Sod states are initially separated at `x=0.5`:

    (ρ, U, p)_L = (1.0,   0.0, 1.0),
    (ρ, U, p)_R = (0.125, 0.0, 0.1).

Transport uses the active-flux point values and cell averages from
`advection.jl`.  Because a shock problem is nonsmooth, a kinetic jump sensor
blends the high-order active-flux evolution locally with a positive first-order
upwind evolution.  BGK relaxation remains exact in time and is evaluated from
a positivity-limited quadratic reconstruction inside each physical cell.
"""

include(joinpath(@__DIR__, "advection.jl"))

"""Build the KitBase data for the `1d1f1v`, `K=0`, `γ=3` Sod problem."""
function create_sod_solver(;
    nx=200,
    nu=72,
    knudsen=1e-4,
    cfl=0.45,
    max_time=0.12,
)
    set = Setup(;
        case="sod",
        space="1d1f1v",
        collision="bgk",
        interpOrder=2,
        boundary=["fix", "fix"],
        cfl,
        maxTime=max_time,
    )
    ps = PSpace1D(0.0, 1.0, nx, 1)
    vs = VSpace1D(-5.0, 5.0, nu)
    gas = Gas(; Kn=knudsen, K=0.0, γ=3.0)

    # KitBase primitive variables are [ρ,U,λ], where p=ρ/(2λ).  Therefore the
    # canonical Sod pressures p_L=1 and p_R=0.1 correspond to λ_L=0.5 and
    # λ_R=0.625 for the densities below.
    primL = [1.0, 0.0, 0.5]
    primR = [0.125, 0.0, 0.625]
    wL = prim_conserve(primL, gas.γ)
    wR = prim_conserve(primR, gas.γ)
    midpoint = 0.5 * (ps.x0 + ps.x1)
    parameters = (; midpoint, primL, primR, wL, wR, u=vs.u, γ=gas.γ)

    fw = (x, p) -> x <= p.midpoint ? p.wL : p.wR
    ff = (x, p) -> maxwellian(p.u, x <= p.midpoint ? p.primL : p.primR)
    bc = (x, p) -> x <= p.midpoint ? p.primL : p.primR
    ib = IB1F(fw, ff, bc, parameters)
    return SolverSet(set, ps, vs, gas, ib)
end

"""Set one cell or interface state from a velocity distribution."""
function set_distribution!(state, distribution, ks)
    state.f .= distribution
    update_macroscopic!(state, ks)
    return nothing
end

"""Return the left and right reservoir Maxwellians attached to the Sod data."""
function sod_reservoir_distributions(ks)
    left = maxwellian(ks.vs.u, ks.ib.p.primL)
    right = maxwellian(ks.vs.u, ks.ib.p.primR)
    return left, right
end

"""
Apply fixed kinetic reservoirs to ghosts and incoming boundary characteristics.

At the left boundary positive molecular velocities enter from the left state;
at the right boundary negative velocities enter from the right state.  The
opposite half ranges are outgoing and retain the value evolved from the
interior.  This is the kinetic analogue of a non-reflecting shock-tube boundary.
"""
function apply_sod_boundary!(ks, ctr, face, left_distribution, right_distribution)
    nx = ks.ps.nx

    set_distribution!(ctr[0], left_distribution, ks)
    set_distribution!(ctr[nx+1], right_distribution, ks)
    set_distribution!(face[0], left_distribution, ks)
    set_distribution!(face[nx+2], right_distribution, ks)

    for j in eachindex(ks.vs.u)
        if ks.vs.u[j] >= 0
            face[1].f[j] = left_distribution[j]
            face[1].favg[j] = left_distribution[j]
        else
            face[nx+1].f[j] = right_distribution[j]
            face[nx+1].favg[j] = right_distribution[j]
        end
    end
    update_macroscopic!(face[1], ks)
    update_macroscopic!(face[nx+1], ks)
    return nothing
end

"""Initialize exact cell averages and shared interface values for the jump."""
function initialize_sod_solution(ks)
    nx, nu = ks.ps.nx, ks.vs.nu
    ctr = OffsetArray{AFControlVolume1F}(undef, 0:nx+1)
    face = OffsetArray{AFInterface1F}(undef, 0:nx+2)

    for i in 0:nx+1
        ctr[i] = AFControlVolume1F(zeros(3), zeros(3), zeros(nu))
    end
    for i in 0:nx+2
        face[i] = AFInterface1F(zeros(3), zeros(3), zeros(nu), zeros(nu))
    end

    left_distribution, right_distribution = sod_reservoir_distributions(ks)
    discontinuity = 0.5 * (ks.ps.x0 + ks.ps.x1)

    # Integrate the piecewise-constant initial distribution exactly.  This also
    # works if a user chooses a mesh for which x=0.5 cuts through a cell.
    for i in 1:nx
        dx = ks.ps.dx[i]
        cell_left = ks.ps.x[i] - 0.5dx
        left_fraction = clamp((discontinuity - cell_left) / dx, 0.0, 1.0)
        @. ctr[i].f = left_fraction * left_distribution +
                         (1 - left_fraction) * right_distribution
        update_macroscopic!(ctr[i], ks)
    end

    for i in 1:nx+1
        xface = ks.ps.x0 + (i - 1) * ks.ps.dx[1]
        if xface < discontinuity
            face[i].f .= left_distribution
        elseif xface > discontinuity
            face[i].f .= right_distribution
        else
            # At the initial material interface, use the kinetic upwind trace:
            # right-moving particles originate on the left and vice versa.
            for j in 1:nu
                face[i].f[j] = ks.vs.u[j] >= 0 ?
                               left_distribution[j] : right_distribution[j]
            end
        end
        face[i].favg .= face[i].f
        update_macroscopic!(face[i], ks)
    end

    apply_sod_boundary!(
        ks,
        ctr,
        face,
        left_distribution,
        right_distribution,
    )
    return ctr, face, left_distribution, right_distribution
end

"""
Evaluate the cell quadratic at all Gauss points and enforce nonnegativity.

For each molecular velocity, one common scaling factor contracts the four
point values toward their positive cell average.  The contraction preserves
the Gauss-integrated cell average because the original quadratic has that
average.  It is inactive wherever the active-flux reconstruction is positive.
"""
function limited_collision_points!(point_values, ks, ctr, face, i)
    for q in eachindex(GL4_NODES)
        ξ = 0.5 * (GL4_NODES[q] + 1)
        left_basis = 3ξ^2 - 4ξ + 1
        average_basis = 6ξ - 6ξ^2
        right_basis = 3ξ^2 - 2ξ
        @views @. point_values[q, :] = left_basis * face[i].f +
                                             average_basis * ctr[i].f +
                                             right_basis * face[i+1].f
    end

    for j in axes(point_values, 2)
        minimum_point = minimum(@view point_values[:, j])
        cell_average = ctr[i].f[j]
        if minimum_point < 0
            # A tiny safety factor prevents roundoff from landing just below 0.
            θ = min(1.0, 0.999999999999 * cell_average /
                         (cell_average - minimum_point))
            for q in axes(point_values, 1)
                point_values[q, j] = cell_average +
                                     θ * (point_values[q, j] - cell_average)
            end
        end
    end
    return nothing
end

"""Perform a positivity-preserving, high-order BGK collision substep."""
function sod_collision_step!(
    ks,
    ctr,
    face,
    dt,
    left_distribution,
    right_distribution,
)
    point_values = zeros(length(GL4_NODES), ks.vs.nu)
    updated_average = zeros(ks.vs.nu)

    for i in 1:ks.ps.nx
        limited_collision_points!(point_values, ks, ctr, face, i)
        fill!(updated_average, 0)
        for q in eachindex(GL4_NODES)
            point_distribution = @view point_values[q, :]
            relax_distribution!(point_distribution, ks, dt)
            @. updated_average += 0.5 * GL4_WEIGHTS[q] * point_distribution
        end
        ctr[i].f .= updated_average
        update_macroscopic!(ctr[i], ks)
    end

    # Unlike the periodic wave, the shock tube has nx+1 distinct physical
    # interfaces; each is relaxed before incoming reservoir data are restored.
    for i in 1:ks.ps.nx+1
        relax_bgk!(face[i], ks, dt)
    end
    apply_sod_boundary!(
        ks,
        ctr,
        face,
        left_distribution,
        right_distribution,
    )
    return nothing
end

"""Smoothly turn off antidiffusion across a kinetic discontinuity."""
function sod_jump_blend(left_value, right_value; smooth=0.02, nonsmooth=0.20)
    jump = abs(right_value - left_value) /
           (abs(right_value) + abs(left_value) + eps(Float64))
    return clamp((nonsmooth - jump) / (nonsmooth - smooth), 0.0, 1.0)
end

"""
Advance kinetic transport with shock-limited active flux.

The high-order point value and time average are computed with the same
characteristic formulas as the smooth solver.  At a detected jump they are
blended with the first-order upwind value.  A final conservative positivity
correction scales only the remaining antidiffusive flux for each molecular
velocity; it is normally inactive after jump sensing.
"""
function sod_transport_step!(
    ks,
    ctr,
    face,
    dt,
    f_old,
    left_distribution,
    right_distribution,
)
    nx, nu = ks.ps.nx, ks.vs.nu
    dx = ks.ps.dx[1]
    high_point = zeros(nx + 1, nu)
    high_average = zeros(nx + 1, nu)
    low_value = zeros(nx + 1, nu)
    face_blend = ones(nx + 1, nu)

    for i in 0:nx+2
        f_old[i, :] .= face[i].f
    end

    for i in 1:nx+1, j in 1:nu
        velocity = ks.vs.u[j]
        c = abs(velocity) * dt / dx
        c <= 1 + 10eps(c) || error("active-flux CFL condition violated: c=$c")
        ηnear, ηfar = 4 - 3c, 3c - 2
        ϕnear, ϕfar = 2 - c, c - 1

        if velocity >= 0
            fleft, fcell, fright = f_old[i-1, j], ctr[i-1].f[j], f_old[i, j]
            high_point[i, j] = fright - c * ηfar * (fcell - fleft) -
                               c * ηnear * (fright - fcell)
            high_average[i, j] = fright - c * ϕfar * (fcell - fleft) -
                                 c * ϕnear * (fright - fcell)
            low_value[i, j] = ctr[i-1].f[j]
        else
            fleft, fcell, fright = f_old[i, j], ctr[i].f[j], f_old[i+1, j]
            high_point[i, j] = fleft - c * ηnear * (fleft - fcell) -
                               c * ηfar * (fcell - fright)
            high_average[i, j] = fleft - c * ϕnear * (fleft - fcell) -
                                 c * ϕfar * (fcell - fright)
            low_value[i, j] = ctr[i].f[j]
        end

        face_blend[i, j] = sod_jump_blend(ctr[i-1].f[j], ctr[i].f[j])
    end

    # Blend only faces identified as nonsmooth.  Smooth portions retain the
    # original active-flux point evolution and time-averaged kinetic flux.
    for i in 1:nx+1, j in 1:nu
        θ = face_blend[i, j]
        face[i].f[j] = low_value[i, j] + θ * (high_point[i, j] - low_value[i, j])
        face[i].favg[j] = low_value[i, j] +
                          θ * (high_average[i, j] - low_value[i, j])
    end

    # The first-order candidate is positive for molecular CFL <= 1.  If an
    # antidiffusive active-flux correction would make any cell negative, scale
    # that correction conservatively for this velocity on every face.
    for j in 1:nu
        velocity_scale = dt / dx * ks.vs.u[j]
        correction_scale = 1.0
        for i in 1:nx
            low_candidate = ctr[i].f[j] + velocity_scale *
                            (low_value[i, j] - low_value[i+1, j])
            limited_candidate = ctr[i].f[j] + velocity_scale *
                                (face[i].favg[j] - face[i+1].favg[j])
            if limited_candidate < 0
                correction_scale = min(
                    correction_scale,
                    0.999999999999 * low_candidate /
                    (low_candidate - limited_candidate),
                )
            end
        end

        if correction_scale < 1
            for i in 1:nx+1
                face[i].f[j] = low_value[i, j] + correction_scale *
                               (face[i].f[j] - low_value[i, j])
                face[i].favg[j] = low_value[i, j] + correction_scale *
                                  (face[i].favg[j] - low_value[i, j])
            end
        end

        # Point values are additional active-flux degrees of freedom rather
        # than conservative quantities.  Projecting a remaining negative point
        # value to zero therefore does not alter the finite-volume balance, and
        # guarantees physical moments in the following BGK half-step.
        for i in 1:nx+1
            face[i].f[j] = max(face[i].f[j], 0.0)
        end
    end

    for i in 1:nx+1
        update_macroscopic!(face[i], ks)
    end
    for i in 1:nx
        for j in 1:nu
            scale = dt / dx * ks.vs.u[j]
            ctr[i].f[j] += scale * (face[i].favg[j] - face[i+1].favg[j])
        end
        update_macroscopic!(ctr[i], ks)
    end

    apply_sod_boundary!(
        ks,
        ctr,
        face,
        left_distribution,
        right_distribution,
    )
    return nothing
end

"""Solve the active-flux BGK Sod shock tube to the requested final time."""
function solve_sod_active_flux(; kwargs...)
    ks = create_sod_solver(; kwargs...)
    ctr, face, left_distribution, right_distribution = initialize_sod_solution(ks)
    f_old = OffsetArray(zeros(ks.ps.nx + 3, ks.vs.nu), 0:ks.ps.nx+2, 1:ks.vs.nu)

    time = 0.0
    steps = 0
    while time < ks.set.maxTime
        dt = active_flux_timestep(ks, time)
        sod_collision_step!(
            ks, ctr, face, 0.5dt, left_distribution, right_distribution,
        )
        sod_transport_step!(
            ks, ctr, face, dt, f_old, left_distribution, right_distribution,
        )
        sod_collision_step!(
            ks, ctr, face, 0.5dt, left_distribution, right_distribution,
        )
        time += dt
        steps += 1
    end

    minimum_distribution = minimum(minimum(ctr[i].f) for i in 1:ks.ps.nx)
    return (; ks, ctr, face, time, steps, minimum_distribution)
end

"""Plot density, velocity, and pressure against the inviscid Sod solution."""
function plot_sod_solution(result)
    ks, ctr = result.ks, result.ctr
    x = collect(ks.ps.x[1:ks.ps.nx])
    density = [ctr[i].prim[1] for i in 1:ks.ps.nx]
    velocity = [ctr[i].prim[2] for i in 1:ks.ps.nx]
    pressure = [0.5 * ctr[i].prim[1] / ctr[i].prim[end] for i in 1:ks.ps.nx]

    exact = KitBase.sample_riemann_solution(
        (1.0, 0.0, 1.0),
        (0.125, 0.0, 0.1),
        x .- 0.5,
        result.time,
        ks.gas.γ,
    )
    exact_density = first.(exact)
    exact_velocity = getindex.(exact, 2)
    exact_pressure = last.(exact)

    p1 = plot(x, density; label="active flux BGK", ylabel="density", lw=1.6)
    plot!(p1, x, exact_density; label="Euler exact", color=:black, line=:dash)
    p2 = plot(x, velocity; label="active flux BGK", ylabel="velocity", lw=1.6)
    plot!(p2, x, exact_velocity; label="Euler exact", color=:black, line=:dash)
    p3 = plot(
        x,
        pressure;
        label="active flux BGK",
        xlabel="x",
        ylabel="pressure",
        lw=1.6,
    )
    plot!(p3, x, exact_pressure; label="Euler exact", color=:black, line=:dash)
    return plot(
        p1,
        p2,
        p3;
        layout=(3, 1),
        size=(760, 820),
        plot_title="Sod shock tube: t=$(round(result.time; digits=4)), γ=$(ks.gas.γ)",
    )
end

function print_sod_diagnostics(result)
    println("Active-flux BGK Sod integration completed")
    println("  model: 1d1f1v, K=0, γ=3")
    println("  final time: ", result.time)
    println("  time steps: ", result.steps)
    println("  minimum cell-average distribution: ", result.minimum_distribution)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = solve_sod_active_flux()
    print_sod_diagnostics(result)
    display(plot_sod_solution(result))
end
