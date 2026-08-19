"""
One-dimensional periodic BGK advection case solved with active flux.

The kinetic equation is

    ∂ₜf + u ∂ₓf = (M[f] - f) / τ,

on a periodic physical domain. Both physical space and molecular velocity are
one-dimensional (1d1f1v). The active flux method advects the single
distribution f(x,u,t) for every discrete molecular velocity. The production
driver packs the active-flux degrees of freedom into an OrdinaryDiffEq
ODEProblem, while the characteristic--Strang functions are retained for the
two spatial-convergence scripts.
"""

using KitBase
using KitBase.OffsetArrays
using OrdinaryDiffEq: ODEProblem, SSPRK33, SplitODEProblem, Tsit5, solve
using OrdinaryDiffEqSDIRK: KenCarp4
using Plots

# Four-point Gauss-Legendre rule used to initialize genuine cell averages of
# the distribution function. The weights sum to two on the interval [-1, 1].
const GL4_NODES = (
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526,
)
const GL4_WEIGHTS = (
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538,
)

"""
Cell-average degrees of freedom for the one-distribution BGK system.

`w` and `prim` follow the KitBase conventions for conservative and primitive
variables. `f` is a cell average in physical space but remains a point value
on the discrete velocity grid.
"""
mutable struct AFControlVolume1F{TW,TP,TF}
    w::TW
    prim::TP
    f::TF
end

"""
Shared point values at a physical-space cell interface.

Besides the current point value `f`, each interface stores `favg`, its time
average over the current transport step. Multiplication by the molecular
velocity turns this average into the kinetic numerical flux.
"""
mutable struct AFInterface1F{TW,TP,TF,TFA}
    w::TW
    prim::TP
    f::TF
    favg::TFA
end

"""
Return the smooth periodic primitive state used by the wave test.

KitBase orders the one-dimensional primitive variables as `[ρ, U, λ]`, with
pressure `p=ρ/(2λ)`. Choosing `λ=ρ` therefore produces a density wave in a
uniform velocity field while keeping the initial pressure equal to `1/2`.
"""
function initial_primitive(x)
    ρ = 1.0 + 0.1 * sin(2π * x)
    U = 1.0
    λ = ρ
    return [ρ, U, λ]
end

"""Build KitBase solver data consistent with its pure-1D gas examples."""
function create_solver(; nx=100, nu=80, knudsen=1e-2, cfl=0.5, max_time=1.0)
    set = Setup(;
        case="bgk_advection",
        space="1d1f1v",
        collision="bgk",
        interpOrder=2,
        boundary=["period", "period"],
        cfl=cfl,
        maxTime=max_time,
    )
    ps = PSpace1D(0.0, 1.0, nx, 1)
    vs = VSpace1D(-6.0, 6.0, nu)

    # With one molecular-velocity dimension and no reduced internal-energy
    # distribution, K=0 and γ=3 describe the pure 1D monatomic model.
    gas = Gas(; Kn=knudsen, K=0.0, γ=3.0)

    # These functions have the same roles as `fw`, `ff`, and `bc` in the
    # Kinetic.jl examples. `IB1F` keeps the initial/boundary data attached to
    # the SolverSet even though periodic boundaries are filled directly below.
    fw = function (x, p)
        return prim_conserve(initial_primitive(x), p.γ)
    end
    ff = function (x, p)
        return maxwellian(p.u, initial_primitive(x))
    end
    bc = (x, p) -> initial_primitive(x)
    ib = IB1F(fw, ff, bc, (u=vs.u, γ=gas.γ))

    return SolverSet(set, ps, vs, gas, ib)
end

"""Recompute macroscopic moments stored in one cell or interface state."""
function update_macroscopic!(state, ks)
    state.w .= moments_conserve(
        state.f,
        ks.vs.u,
        ks.vs.weights,
        KitBase.VDF{1,1},
    )
    state.prim .= conserve_prim(state.w, ks.gas.γ)
    return nothing
end

"""Copy all solution fields between two cell states."""
function copy_state!(destination::AFControlVolume1F, source::AFControlVolume1F)
    destination.w .= source.w
    destination.prim .= source.prim
    destination.f .= source.f
    return nothing
end

"""Copy all solution fields between two interface states."""
function copy_state!(destination::AFInterface1F, source::AFInterface1F)
    destination.w .= source.w
    destination.prim .= source.prim
    destination.f .= source.f
    destination.favg .= source.favg
    return nothing
end

"""Fill periodic cell ghosts and interface duplicates/ghosts."""
function apply_periodic_boundary!(ks, ctr, face)
    nx = ks.ps.nx

    # Physical cell ghosts: cell 0 is a copy of cell nx, and cell nx+1 is a
    # copy of cell 1.
    copy_state!(ctr[0], ctr[nx])
    copy_state!(ctr[nx+1], ctr[1])

    # face[1] and face[nx+1] represent the same periodic interface. face[0]
    # and face[nx+2] provide the next layer needed by the quadratic upwind
    # reconstruction on either side of the domain.
    copy_state!(face[nx+1], face[1])
    copy_state!(face[0], face[nx])
    copy_state!(face[nx+2], face[2])
    return nothing
end

"""Initialize active-flux cell averages and shared interface point values."""
function initialize_solution(ks)
    nx, nu = ks.ps.nx, ks.vs.nu
    ps = ks.ps

    # One cell ghost is required on each side. Interfaces require one duplicate
    # periodic face plus one additional ghost face on each side.
    ctr = OffsetArray{AFControlVolume1F}(undef, 0:nx+1)
    face = OffsetArray{AFInterface1F}(undef, 0:nx+2)

    for i in 0:nx+1
        ctr[i] = AFControlVolume1F(zeros(3), zeros(3), zeros(nu))
    end
    for i in 0:nx+2
        face[i] = AFInterface1F(zeros(3), zeros(3), zeros(nu), zeros(nu))
    end

    # A cell degree of freedom must be the physical-space average of the
    # distribution, not its value at the cell center. Map the four-point
    # Gauss-Legendre rule from [-1,1] to every physical cell.
    for i in 1:nx
        xcenter = ps.x[i]
        dx = ps.dx[i]
        for q in eachindex(GL4_NODES)
            xq = xcenter + 0.5 * dx * GL4_NODES[q]
            fq = ks.ib.ff(xq, ks.ib.p)
            ctr[i].f .+= 0.5 * GL4_WEIGHTS[q] .* fq
        end
        update_macroscopic!(ctr[i], ks)
    end

    # There are nx distinct periodic interfaces. The extra physical index
    # nx+1 is filled as an exact copy of interface 1.
    for i in 1:nx
        xface = ps.x0 + (i - 1) * ps.dx[1]
        face[i].f .= ks.ib.ff(xface, ks.ib.p)
        update_macroscopic!(face[i], ks)
    end

    apply_periodic_boundary!(ks, ctr, face)
    return ctr, face
end

"""
Construct a Maxwellian whose *discrete* mass, momentum, and energy equal `w`.

The analytic Maxwellian has the requested continuous moments, but integration
on a finite, truncated velocity grid leaves a small moment defect. That defect
would make a nominally collision-invariant quantity drift during relaxation.
We remove it with the weighted projection

    M_d(u_j) = M(u_j) * (1 + α₀ + α₁*u_j + α₂*u_j²/2),

where the three coefficients are chosen so the velocity quadrature of `M_d`
matches `w`. The correction is tiny for the present `[-6,6]` grid, retains
positivity for the smooth test, and makes the semi-discrete BGK collision step
conservative to roundoff.
"""
function discrete_conservative_maxwellian(ks, w, prim)
    equilibrium = maxwellian(ks.vs.u, prim)
    equilibrium_moments = moments_conserve(
        equilibrium,
        ks.vs.u,
        ks.vs.weights,
        KitBase.VDF{1,1},
    )
    defect = w - equilibrium_moments

    # The Gram matrix maps the polynomial multiplier coefficients to their
    # induced changes in the three discrete collision-invariant moments.
    gram = zeros(eltype(equilibrium), 3, 3)
    for j in eachindex(equilibrium)
        basis = (one(eltype(equilibrium)), ks.vs.u[j], 0.5 * ks.vs.u[j]^2)
        weighted_equilibrium = ks.vs.weights[j] * equilibrium[j]
        for row in 1:3, column in 1:3
            gram[row, column] += weighted_equilibrium * basis[row] * basis[column]
        end
    end
    correction = gram \ defect

    for j in eachindex(equilibrium)
        velocity = ks.vs.u[j]
        multiplier = 1 + correction[1] + correction[2] * velocity +
                     0.5 * correction[3] * velocity^2
        equilibrium[j] *= multiplier
    end
    return equilibrium
end

"""Apply exact, discretely conservative BGK relaxation to one distribution."""
function relax_distribution!(f, ks, dt)
    w = moments_conserve(f, ks.vs.u, ks.vs.weights, KitBase.VDF{1,1})
    prim = conserve_prim(w, ks.gas.γ)
    τ = vhs_collision_time(prim, ks.gas.μᵣ, ks.gas.ω)
    equilibrium = discrete_conservative_maxwellian(ks, w, prim)
    decay = exp(-dt / τ)

    @. f = equilibrium + decay * (f - equilibrium)
    return nothing
end

"""
Apply an exact pointwise BGK relaxation for a time interval `dt`.

During a collision-only step the conservative moments, Maxwellian, and
collision time are constant. The exact pointwise solution is therefore

    f(t+dt) = M + exp(-dt/τ) * (f(t)-M).
"""
function relax_bgk!(state, ks, dt)
    relax_distribution!(state.f, ks, dt)

    # Refresh the redundant macroscopic fields after evolving the distribution.
    update_macroscopic!(state, ks)
    return nothing
end

"""
Apply one spatially high-order collision substep.

The BGK operator is nonlinear, so relaxing a cell average as `M[fbar]` is only
second-order consistent with the physical-space average of `M[f]`. Instead,
reconstruct the active-flux quadratic at four Gauss points, relax each
pointwise distribution exactly, and integrate the results back to the updated
cell average. Shared interface point values are relaxed directly afterward.
"""
function collision_step!(ks, ctr, face, dt)
    point_distribution = zeros(ks.vs.nu)
    updated_average = zeros(ks.vs.nu)

    for i in 1:ks.ps.nx
        fill!(updated_average, 0)

        for q in eachindex(GL4_NODES)
            # Convert the Gauss node from [-1,1] to the cell coordinate ξ in
            # [0,1], then evaluate the quadratic determined by the left point
            # value, cell average, and right point value.
            ξ = 0.5 * (GL4_NODES[q] + 1)
            left_basis = 3ξ^2 - 4ξ + 1
            average_basis = 6ξ - 6ξ^2
            right_basis = 3ξ^2 - 2ξ

            @. point_distribution = left_basis * face[i].f +
                                    average_basis * ctr[i].f +
                                    right_basis * face[i+1].f
            relax_distribution!(point_distribution, ks, dt)
            @. updated_average += 0.5 * GL4_WEIGHTS[q] * point_distribution
        end

        ctr[i].f .= updated_average
        update_macroscopic!(ctr[i], ks)
    end

    # Cell reconstruction above must use old-time interface data, so interface
    # relaxation is deliberately performed only after all cells are complete.
    for i in 1:ks.ps.nx
        relax_bgk!(face[i], ks, dt)
    end
    apply_periodic_boundary!(ks, ctr, face)
    return nothing
end

"""
Advance the collisionless transport equation with active flux.

For every molecular velocity `u[j]`, the kinetic transport equation is a
scalar advection equation with its own signed CFL number. Positive velocities
use the cell to the left of an interface; negative velocities use the cell to
the right. The interface point value is obtained by evaluating the continuous
quadratic reconstruction at the characteristic foot. `favg` is the exact time
average of that quadratic, equivalently its Simpson average.
"""
function transport_step!(ks, ctr, face, dt, f_old)
    nx, nu = ks.ps.nx, ks.vs.nu
    dx = ks.ps.dx[1]

    # All interface updates must use the same old-time data, so save the face
    # point values before overwriting them.
    for i in 0:nx+2
        f_old[i, :] .= face[i].f
    end

    for i in 1:nx
        for j in 1:nu
            velocity = ks.vs.u[j]
            c = abs(velocity) * dt / dx
            c <= 1 + 10eps(c) || error("active-flux CFL condition violated: c=$c")

            ηnear = 4 - 3c
            ηfar = 3c - 2
            ϕnear = 2 - c
            ϕfar = c - 1

            if velocity >= 0
                # The characteristic enters from the cell on the left. Its
                # reconstruction uses face i-1, cell i-1, and face i.
                fleft, fcell, fright = f_old[i-1, j], ctr[i-1].f[j], f_old[i, j]

                face[i].f[j] = fright - c * ηfar * (fcell - fleft) -
                               c * ηnear * (fright - fcell)
                face[i].favg[j] = fright - c * ϕfar * (fcell - fleft) -
                                  c * ϕnear * (fright - fcell)
            else
                # For negative velocity, trace into the cell on the right. The
                # formulas are the mirror image of the positive-speed update.
                fleft, fcell, fright = f_old[i, j], ctr[i].f[j], f_old[i+1, j]

                face[i].f[j] = fleft - c * ηnear * (fleft - fcell) -
                               c * ηfar * (fcell - fright)
                face[i].favg[j] = fleft - c * ϕnear * (fleft - fcell) -
                                  c * ϕfar * (fcell - fright)
            end
        end
        update_macroscopic!(face[i], ks)
    end

    # Close the periodic interface before updating the cell averages so cell nx
    # sees exactly the same right flux that cell 1 sees as its left flux.
    copy_state!(face[nx+1], face[1])

    # Conservative finite-volume update. The kinetic flux is u*f, while the
    # active-flux face array contains the time-averaged distribution f.
    for i in 1:nx
        for j in 1:nu
            scale = dt / dx * ks.vs.u[j]
            ctr[i].f[j] += scale * (face[i].favg[j] - face[i+1].favg[j])
        end
        update_macroscopic!(ctr[i], ks)
    end

    apply_periodic_boundary!(ks, ctr, face)
    return nothing
end

"""Transport-limited time step; exact BGK relaxation adds no stiffness limit."""
function active_flux_timestep(ks, time)
    dx = minimum(ks.ps.dx[1:ks.ps.nx])
    vmax = maximum(abs, ks.vs.u)
    dt = ks.set.cfl * dx / vmax
    return min(dt, ks.set.maxTime - time)
end

"""Integrate global discrete mass, momentum, and energy."""
function global_conserved(ks, ctr)
    total = zeros(3)
    for i in 1:ks.ps.nx
        total .+= ks.ps.dx[i] .* ctr[i].w
    end
    return total
end

"""Collect conservation, periodicity, and positivity diagnostics."""
function diagnostics(ks, ctr, face, initial_total)
    final_total = global_conserved(ks, ctr)
    scale = max.(abs.(initial_total), eps())
    relative_drift = (final_total .- initial_total) ./ scale
    minimum_distribution = minimum(minimum(ctr[i].f) for i in 1:ks.ps.nx)
    periodic_mismatch = maximum(abs, face[1].f - face[ks.ps.nx+1].f)

    return (;
        initial_total,
        final_total,
        relative_drift,
        minimum_distribution,
        periodic_mismatch,
    )
end

"""
Solve the periodic BGK wave problem.

The Strang sequence is collision(dt/2), transport(dt), collision(dt/2).
Transport is third-order active flux for each discrete velocity; the coupled
BGK algorithm is second-order in time because of operator splitting.
"""
function solve_bgk_active_flux(; fixed_dt=nothing, kwargs...)
    ks = create_solver(; kwargs...)
    ctr, face = initialize_solution(ks)
    initial_total = global_conserved(ks, ctr)

    if !isnothing(fixed_dt)
        fixed_dt > 0 || throw(ArgumentError("fixed_dt must be positive"))
        maximum(abs, ks.vs.u) * fixed_dt / ks.ps.dx[1] <= 1 ||
            throw(ArgumentError("fixed_dt violates the active-flux CFL condition"))
    end

    nx, nu = ks.ps.nx, ks.vs.nu
    f_old = OffsetArray(zeros(nx + 3, nu), 0:nx+2, 1:nu)

    time = 0.0
    steps = 0
    while time < ks.set.maxTime
        dt = if isnothing(fixed_dt)
            active_flux_timestep(ks, time)
        else
            min(fixed_dt, ks.set.maxTime - time)
        end
        collision_step!(ks, ctr, face, 0.5dt)
        transport_step!(ks, ctr, face, dt, f_old)
        collision_step!(ks, ctr, face, 0.5dt)
        time += dt
        steps += 1
    end

    stats = diagnostics(ks, ctr, face, initial_total)
    return (; ks, ctr, face, time, steps, stats)
end

# ---------------------------------------------------------------------------
# Semi-discrete OrdinaryDiffEq formulation used by this advection case.
# ---------------------------------------------------------------------------

"""Parameters passed to every in-place active-flux right-hand side."""
struct AdvectionRHSParameters{KS}
    ks::KS
end

"""Packed-state column containing the interface at the left of cell i."""
@inline advection_interface_column(parameters::AdvectionRHSParameters, i) =
    parameters.ks.ps.nx + i

"""
Pack the active-flux degrees of freedom into an OrdinaryDiffEq state array.

Rows enumerate molecular velocities. Columns 1:nx store physical-space cell
averages and columns nx+1:2nx store the nx distinct periodic interface point
values. The duplicate and ghost interfaces remain implementation details of
the plotting and diagnostic containers and therefore are not integrated.
"""
function pack_advection_state(ks, ctr, face)
    state = zeros(eltype(ctr[1].f), ks.vs.nu, 2 * ks.ps.nx)
    for i in 1:ks.ps.nx
        @views state[:, i] .= ctr[i].f
        @views state[:, ks.ps.nx+i] .= face[i].f
    end
    return state
end

"""Copy a packed ODE state back to active-flux cells and interfaces."""
function unpack_advection_state!(state, ks, ctr, face)
    nx = ks.ps.nx
    size(state) == (ks.vs.nu, 2 * nx) ||
        throw(DimensionMismatch("expected a $(ks.vs.nu) by $(2 * nx) state"))

    for i in 1:nx
        @views ctr[i].f .= state[:, i]
        update_macroscopic!(ctr[i], ks)

        @views face[i].f .= state[:, nx+i]
        face[i].favg .= face[i].f
        update_macroscopic!(face[i], ks)
    end
    apply_periodic_boundary!(ks, ctr, face)
    return nothing
end

"""
Evaluate the semi-discrete periodic active-flux transport operator.

For a positive velocity, an interface uses the derivative at the right end of
the quadratic in the upwind cell,

    p'(1)/dx = (2*f_left - 6*f_average + 4*f_right)/dx.

For a negative velocity it uses the derivative at the left end of the cell on
the right,

    p'(0)/dx = (-4*f_left + 6*f_average - 2*f_right)/dx.

The cell-average equation is the conservative difference of the shared point
values. This function contains no Runge--Kutta coefficient or step size, so it
can be reused by any compatible OrdinaryDiffEq algorithm.
"""
function advection_transport_rhs!(derivative, state, parameters, time)
    ks = parameters.ks
    nx, nu = ks.ps.nx, ks.vs.nu
    dx = ks.ps.dx[1]
    size(state) == (nu, 2 * nx) ||
        throw(DimensionMismatch("invalid advection state size $(size(state))"))
    fill!(derivative, zero(eltype(derivative)))

    # Conservative evolution of the cell-average degrees of freedom.
    for i in 1:nx
        right_interface = mod1(i + 1, nx)
        left_column = advection_interface_column(parameters, i)
        right_column = advection_interface_column(parameters, right_interface)
        for j in 1:nu
            derivative[j, i] = -ks.vs.u[j] / dx *
                               (state[j, right_column] - state[j, left_column])
        end
    end

    # Upwind evolution of every shared interface point value.
    for i in 1:nx
        point_column = advection_interface_column(parameters, i)
        left_cell = mod1(i - 1, nx)
        right_cell = i
        far_left_column = advection_interface_column(parameters, left_cell)
        far_right_column =
            advection_interface_column(parameters, mod1(right_cell + 1, nx))

        for j in 1:nu
            velocity = ks.vs.u[j]
            spatial_derivative = if velocity >= 0
                (2 * state[j, far_left_column] -
                 6 * state[j, left_cell] + 4 * state[j, point_column]) / dx
            else
                (-4 * state[j, point_column] +
                 6 * state[j, right_cell] - 2 * state[j, far_right_column]) / dx
            end
            derivative[j, point_column] = -velocity * spatial_derivative
        end
    end
    return nothing
end

"""Evaluate the discretely conservative BGK source of one distribution."""
function conservative_bgk_rhs!(collision, distribution, ks)
    moments = moments_conserve(
        distribution,
        ks.vs.u,
        ks.vs.weights,
        KitBase.VDF{1,1},
    )
    primitive = conserve_prim(moments, ks.gas.γ)
    collision_time = vhs_collision_time(primitive, ks.gas.μᵣ, ks.gas.ω)
    equilibrium = discrete_conservative_maxwellian(ks, moments, primitive)
    @. collision = (equilibrium - distribution) / collision_time
    return nothing
end

"""
Evaluate the finite-Knudsen BGK collision part of the packed state.

For a cell-average degree of freedom, the quadratic active-flux
reconstruction is evaluated at four Gauss points, the nonlinear BGK source is
formed at each point, and those sources are integrated back to a cell average.
Interface degrees of freedom are collided pointwise. Thus the RHS preserves
the high-order physical-space treatment of collision used by the original
Strang implementation.
"""
function advection_collision_rhs!(derivative, state, parameters, time)
    ks = parameters.ks
    nx, nu = ks.ps.nx, ks.vs.nu
    fill!(derivative, zero(eltype(derivative)))
    point_distribution = similar(state, eltype(state), nu)
    point_collision = similar(state, eltype(state), nu)

    for i in 1:nx
        left_column = advection_interface_column(parameters, i)
        right_column =
            advection_interface_column(parameters, mod1(i + 1, nx))
        average_collision = @view derivative[:, i]

        for q in eachindex(GL4_NODES)
            ξ = 0.5 * (GL4_NODES[q] + 1)
            left_basis = 3ξ^2 - 4ξ + 1
            average_basis = 6ξ - 6ξ^2
            right_basis = 3ξ^2 - 2ξ
            @views @. point_distribution =
                left_basis * state[:, left_column] +
                average_basis * state[:, i] +
                right_basis * state[:, right_column]
            conservative_bgk_rhs!(point_collision, point_distribution, ks)
            @. average_collision += 0.5 * GL4_WEIGHTS[q] * point_collision
        end
    end

    for i in 1:nx
        column = advection_interface_column(parameters, i)
        conservative_bgk_rhs!(
            @view(derivative[:, column]),
            @view(state[:, column]),
            ks,
        )
    end
    return nothing
end

"""
Evaluate the equilibrium-projection BGK collision used by the IMEX option.

Every active-flux degree of freedom is relaxed toward a discrete Maxwellian
with the same moments. This gives the exact discrete equilibrium null space
needed by the formal Euler-limit AP argument. It is kept separate from the
Gauss-quadrature source because the latter is the finite-Knudsen high-order
choice, whereas a uniformly accurate AP collision reconstruction remains a
future step.
"""
function advection_equilibrium_collision_rhs!(
    derivative,
    state,
    parameters,
    time,
)
    ks = parameters.ks
    fill!(derivative, zero(eltype(derivative)))
    for column in axes(state, 2)
        conservative_bgk_rhs!(
            @view(derivative[:, column]),
            @view(state[:, column]),
            ks,
        )
    end
    return nothing
end

"""Add active-flux transport and quadrature BGK collision in place."""
function advection_bgk_rhs!(derivative, state, parameters, time)
    advection_transport_rhs!(derivative, state, parameters, time)
    collision = similar(derivative)
    advection_collision_rhs!(collision, state, parameters, time)
    derivative .+= collision
    return nothing
end

"""Add transport and equilibrium-projection collision in place."""
function advection_ap_bgk_rhs!(derivative, state, parameters, time)
    advection_transport_rhs!(derivative, state, parameters, time)
    collision = similar(derivative)
    advection_equilibrium_collision_rhs!(collision, state, parameters, time)
    derivative .+= collision
    return nothing
end

"""
Build the OrdinaryDiffEq problem for the periodic advection case.

The default returns one ODEProblem for transport plus the high-order
quadrature collision. With split=true, collision is the implicit component
and transport is the explicit component of a SplitODEProblem. The split mode
defaults to equilibrium projection because its discrete equilibrium manifold
is the one used by the formal AP analysis in the manuscript.
"""
function advection_ode_problem(;
    split=false,
    collision_discretization=nothing,
    kwargs...,
)
    selected_collision = if isnothing(collision_discretization)
        split ? :equilibrium_projection : :quadrature
    else
        collision_discretization
    end
    selected_collision in (:quadrature, :equilibrium_projection) ||
        throw(ArgumentError(
            "collision_discretization must be :quadrature or " *
            ":equilibrium_projection",
        ))

    ks = create_solver(; kwargs...)
    ctr, face = initialize_solution(ks)
    initial_state = pack_advection_state(ks, ctr, face)
    parameters = AdvectionRHSParameters(ks)
    time_span = (0.0, ks.set.maxTime)

    problem = if split
        collision_rhs! = selected_collision === :quadrature ?
                         advection_collision_rhs! :
                         advection_equilibrium_collision_rhs!
        SplitODEProblem(
            collision_rhs!,
            advection_transport_rhs!,
            initial_state,
            time_span,
            parameters,
        )
    else
        combined_rhs! = selected_collision === :quadrature ?
                        advection_bgk_rhs! : advection_ap_bgk_rhs!
        ODEProblem(combined_rhs!, initial_state, time_span, parameters)
    end
    return (; problem, ks, ctr, face, parameters, selected_collision)
end

"""
Solve the periodic BGK advection case through OrdinaryDiffEq.

Tsit5 is the default for the finite-Knudsen unsplit problem. Setting split=true
selects the stiffly accurate KenCarp4 IMEX method unless another algorithm is
provided. In both cases dtmax enforces the molecular transport CFL; explicit
integration may impose a smaller collision time step. The returned solution
object exposes accepted/rejected steps and can be inspected with the standard
SciML tooling.
"""
function solve_advection_active_flux(;
    split=false,
    algorithm=nothing,
    collision_discretization=nothing,
    reltol=1e-7,
    abstol=1e-9,
    kwargs...,
)
    setup = advection_ode_problem(;
        split,
        collision_discretization,
        kwargs...,
    )
    ks = setup.ks
    initial_total = global_conserved(ks, setup.ctr)
    transport_dt = ks.set.cfl * minimum(ks.ps.dx[1:ks.ps.nx]) /
                   maximum(abs, ks.vs.u)
    selected_algorithm = if isnothing(algorithm)
        split ? KenCarp4() : Tsit5()
    else
        algorithm
    end

    solution = solve(
        setup.problem,
        selected_algorithm;
        dt=transport_dt,
        dtmax=transport_dt,
        reltol,
        abstol,
        save_everystep=false,
    )
    unpack_advection_state!(last(solution.u), ks, setup.ctr, setup.face)
    stats = diagnostics(ks, setup.ctr, setup.face, initial_total)
    return (;
        ks,
        ctr=setup.ctr,
        face=setup.face,
        time=last(solution.t),
        steps=solution.destats.naccept,
        stats,
        solution,
        algorithm=selected_algorithm,
        formulation=:ordinarydiffeq,
        split,
        collision_discretization=setup.selected_collision,
    )
end

"""Plot the initial and final density, velocity, and pressure."""
function plot_solution(result)
    ks, ctr = result.ks, result.ctr
    x = collect(ks.ps.x[1:ks.ps.nx])
    density = [ctr[i].prim[1] for i in 1:ks.ps.nx]
    velocity = [ctr[i].prim[2] for i in 1:ks.ps.nx]
    pressure = [0.5 * ctr[i].prim[1] / ctr[i].prim[end] for i in 1:ks.ps.nx]
    initial_density = first.(initial_primitive.(x))

    p1 = plot(x, initial_density; label="initial", ylabel="density", lw=1.5)
    plot!(p1, x, density; label="active flux BGK", lw=1.5)
    p2 = plot(x, velocity; label=false, ylabel="velocity", lw=1.5)
    p3 = plot(x, pressure; label=false, xlabel="x", ylabel="pressure", lw=1.5)
    return plot(p1, p2, p3; layout=(3, 1), size=(700, 750))
end

function print_diagnostics(result)
    println("Active-flux BGK integration completed")
    println("  model: 1d1f1v, K=0, γ=3")
    println("  final time: ", result.time)
    println("  time steps: ", result.steps)
    println("  relative [mass, momentum, energy] drift: ", result.stats.relative_drift)
    println("  minimum distribution value: ", result.stats.minimum_distribution)
    println("  periodic interface mismatch: ", result.stats.periodic_mismatch)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = solve_advection_active_flux()
    print_diagnostics(result)
    display(plot_solution(result))
end
