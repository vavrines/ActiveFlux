"""
Normal-shock structure for the one-dimensional active-flux kinetic solver.

The physical domain is one-dimensional, but the monatomic distribution uses
all three molecular-velocity components (`1d1f3v`, `K=0`, `gamma=5/3`).  The
same active-flux transport operator is combined with either

  * the full Boltzmann collision integral evaluated by KitBase's fast spectral
    method (FSM), or
  * the BGK relaxation model.

Both models use the same Rankine--Hugoniot reservoirs, physical and velocity
grids, initial data, kinetic boundary conditions, Strang splitting, and
OrdinaryDiffEq explicit-Euler transport integration.  Consequently their plotted
difference isolates the collision model rather than a change of discretization.

The default calculation is deliberately a short, coarse preliminary run.  A
converged steady shock needs a longer final time and grid-refinement studies;
the keywords of `solve_normal_shock_active_flux` expose those parameters.  The
executable programs `shock_boltzmann.jl` and `shock_bgk.jl` run the two
collision models independently, while `shock_plot.jl` compares their saved
JLD2 results without repeating either simulation.
"""

using KitBase
using KitBase.OffsetArrays
using LinearAlgebra
using OrdinaryDiffEq: Euler, ODEProblem, solve

# Four Gauss--Legendre points are enough to integrate the quadratic active-flux
# reconstruction exactly.  They are also used to construct genuine initial
# cell averages and to average the nonlinear collision operator in x.
const SHOCK_GL4_NODES = (
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526,
)
const SHOCK_GL4_WEIGHTS = (
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538,
)

"""Cell-average active-flux degree of freedom for a `1d1f3v` gas."""
mutable struct ShockControlVolume{TW,TP,TF}
    w::TW
    prim::TP
    f::TF
end

"""Shared physical-interface point value of the three-velocity distribution."""
mutable struct ShockInterface{TW,TP,TF,TFA}
    w::TW
    prim::TP
    f::TF
    favg::TFA
end

"""
Reusable velocity-space storage for BGK and fast-spectral collision steps.

The five columns of `invariants` contain

    1, u, v, w, (u^2+v^2+w^2)/2.

Writing velocity moments as a small dense matrix product both documents the
conservative projection and avoids millions of scalar allocations during the
many local collision solves.
"""
mutable struct ShockCollisionWorkspace{T,TA,TM,TV}
    shape::NTuple{3,Int}
    invariants::TM
    weights::TV
    weighted_values::TV
    weighted_basis::TM
    multiplier::TV
    base::TA
    equilibrium::TA
    q::TA
    q_equilibrium::TA
    stage::TA
    trial::TA
    target::Vector{T}
    current::Vector{T}
    defect::Vector{T}
    coefficients::Vector{T}
    gram::Matrix{T}
end

"""Parameters and scratch arrays of the semi-discrete transport operator."""
struct NormalShockTransportParameters{KS,TV,TM,T}
    ks::KS
    velocity_x::TV
    left_distribution::TV
    right_distribution::TV
    low_face::TM
    limited_face::TM
    step_size::T
end

"""Return the upstream and downstream monatomic Rankine--Hugoniot states."""
function normal_shock_states(mach, gamma)
    primitive_left_1v, primitive_right_1v = KitBase.ib_rh(mach, gamma)
    primitive_left = [
        primitive_left_1v[1], primitive_left_1v[2], 0.0, 0.0,
        primitive_left_1v[3],
    ]
    primitive_right = [
        primitive_right_1v[1], primitive_right_1v[2], 0.0, 0.0,
        primitive_right_1v[3],
    ]
    return primitive_left, primitive_right
end

"""
Build one normal-shock solver set.

`nu` is intentionally larger than `nv` and `nw`: the shock has a nonzero
streamwise velocity, so the shifted distribution needs more resolution in u.
The `(32,24,24)` default is much cheaper than a production FSM grid while
already making the discrete Maxwellian mass error small enough for a useful
short-time comparison.
"""
function create_normal_shock_solver(;
    collision_model=:fsm,
    nx=16,
    nu=32,
    nv=24,
    nw=24,
    xlimit=25.0,
    velocity_limit=10.0,
    mach=2.0,
    knudsen=1.0,
    cfl=0.35,
    max_time=0.30,
    nm=3,
    alpha=1.0,
    omega=0.72,
)
    collision_model in (:fsm, :bgk) ||
        throw(ArgumentError("collision_model must be :fsm or :bgk"))

    set = Setup(;
        case="normal_shock",
        space="1d1f3v",
        collision=String(collision_model),
        interpOrder=2,
        boundary=["fix", "fix"],
        cfl,
        maxTime=max_time,
    )
    ps = PSpace1D(-xlimit, xlimit, nx, 1)
    vs = VSpace3D(
        -velocity_limit, velocity_limit, nu,
        -velocity_limit, velocity_limit, nv,
        -velocity_limit, velocity_limit, nw,
    )

    gamma = 5 / 3
    reference_viscosity = ref_vhs_vis(knudsen, alpha, 0.5)
    spectral_kernel = collision_model === :fsm ?
                      fsm_kernel(vs, reference_viscosity, nm, alpha) : nothing
    gas = Gas(;
        Kn=knudsen,
        Ma=mach,
        K=0.0,
        γ=gamma,
        ω=omega,
        αᵣ=alpha,
        ωᵣ=0.5,
        μᵣ=reference_viscosity,
        fsm=spectral_kernel,
    )

    primitive_left, primitive_right = normal_shock_states(mach, gamma)
    conservative_left = prim_conserve(primitive_left, gamma)
    conservative_right = prim_conserve(primitive_right, gamma)
    midpoint = 0.5 * (ps.x0 + ps.x1)
    parameters = (;
        midpoint,
        primitive_left,
        primitive_right,
        conservative_left,
        conservative_right,
        u=vs.u,
        v=vs.v,
        velocity_w=vs.w,
    )

    fw = (x, p) -> x <= p.midpoint ?
                      p.conservative_left : p.conservative_right
    ff = (x, p) -> maxwellian(
        p.u,
        p.v,
        p.velocity_w,
        x <= p.midpoint ? p.primitive_left : p.primitive_right,
    )
    bc = (x, p) -> x <= p.midpoint ?
                      p.primitive_left : p.primitive_right
    ib = IB1F(fw, ff, bc, parameters)
    return SolverSet(set, ps, vs, gas, ib)
end

"""Allocate the invariant matrix and all collision temporaries once."""
function create_shock_collision_workspace(ks)
    shape = size(ks.vs.u)
    nvelocity = length(ks.vs.u)
    T = eltype(ks.vs.u)
    invariants = Matrix{T}(undef, nvelocity, 5)

    for linear_index in eachindex(ks.vs.u)
        u = ks.vs.u[linear_index]
        v = ks.vs.v[linear_index]
        w = ks.vs.w[linear_index]
        invariants[linear_index, 1] = 1
        invariants[linear_index, 2] = u
        invariants[linear_index, 3] = v
        invariants[linear_index, 4] = w
        invariants[linear_index, 5] = 0.5 * (u^2 + v^2 + w^2)
    end

    array() = zeros(T, shape)
    return ShockCollisionWorkspace(
        shape,
        invariants,
        collect(vec(ks.vs.weights)),
        zeros(T, nvelocity),
        zeros(T, nvelocity, 5),
        zeros(T, nvelocity),
        array(),
        array(),
        array(),
        array(),
        array(),
        array(),
        zeros(T, 5),
        zeros(T, 5),
        zeros(T, 5),
        zeros(T, 5),
        zeros(T, 5, 5),
    )
end

"""Compute the five discrete collision-invariant moments without allocation."""
function shock_moments!(moments, distribution, workspace)
    workspace.weighted_values .= workspace.weights .* vec(distribution)
    mul!(moments, transpose(workspace.invariants), workspace.weighted_values)
    return moments
end

"""Form the weighted Gram matrix of the five collision invariants."""
function shock_gram!(gram, distribution, workspace)
    workspace.weighted_values .= workspace.weights .* vec(distribution)
    for column in 1:5
        @views @. workspace.weighted_basis[:, column] =
            workspace.weighted_values * workspace.invariants[:, column]
    end
    mul!(gram, transpose(workspace.invariants), workspace.weighted_basis)
    return gram
end

"""
Project a distribution to nonnegative values while preserving target moments.

Clipping the small Fourier oscillations created by an FSM update would destroy
mass, momentum, and energy.  Instead, after clipping, this routine finds the
positive exponential correction

    f_new = max(f_trial,0) exp(alpha dot psi)

whose five discrete moments equal `target`.  Newton's Jacobian is exactly the
invariant Gram matrix.  For the small spectral undershoots encountered here,
three or four iterations normally reach roundoff.
"""
function positive_moment_projection!(
    output,
    trial,
    target,
    workspace;
    tolerance=5e-13,
    maximum_iterations=12,
)
    workspace.base .= max.(trial, zero(eltype(trial)))
    fill!(workspace.coefficients, 0)
    target_scale = max.(abs.(target), one(eltype(target)))

    for iteration in 1:maximum_iterations
        mul!(
            workspace.multiplier,
            workspace.invariants,
            workspace.coefficients,
        )
        for velocity_index in eachindex(output)
            exponent = clamp(
                workspace.multiplier[velocity_index], -50, 50,
            )
            output[velocity_index] = workspace.base[velocity_index] * exp(exponent)
        end
        shock_moments!(workspace.current, output, workspace)
        @. workspace.defect = target - workspace.current
        residual = maximum(abs.(workspace.defect) ./ target_scale)
        residual <= tolerance && return residual

        shock_gram!(workspace.gram, output, workspace)
        increment = workspace.gram \ workspace.defect

        # A damped Newton increment protects unusual coarse-grid states from
        # overflowing the exponential while leaving ordinary corrections full.
        damping = min(1.0, 1.0 / max(maximum(abs, increment), 1.0))
        @. workspace.coefficients += damping * increment
    end

    shock_moments!(workspace.current, output, workspace)
    residual = maximum(abs.(target .- workspace.current) ./ target_scale)
    residual <= 1e-9 || error(
        "positive moment projection did not converge; residual=$residual",
    )
    return residual
end

"""Construct a positive Maxwellian with exactly the requested discrete moments."""
function discrete_maxwellian_3v!(equilibrium, target, primitive, ks, workspace)
    maxwellian!(
        equilibrium,
        ks.vs.u,
        ks.vs.v,
        ks.vs.w,
        primitive[1],
        primitive[2],
        primitive[3],
        primitive[4],
        primitive[5],
    )
    positive_moment_projection!(equilibrium, equilibrium, target, workspace)
    return equilibrium
end

"""Recompute conservative and primitive variables stored beside a distribution."""
function update_shock_macroscopic!(state, ks, workspace)
    shock_moments!(state.w, state.f, workspace)
    state.prim .= conserve_prim(state.w, ks.gas.γ)
    return nothing
end

"""Set one state from a supplied distribution and refresh its moments."""
function set_shock_distribution!(state, distribution, ks, workspace)
    state.f .= distribution
    hasproperty(state, :favg) && (state.favg .= distribution)
    update_shock_macroscopic!(state, ks, workspace)
    return nothing
end

"""Construct discretely conservative upstream and downstream reservoirs."""
function normal_shock_reservoirs(ks, workspace)
    left = similar(ks.vs.u)
    right = similar(ks.vs.u)
    discrete_maxwellian_3v!(
        left,
        ks.ib.p.conservative_left,
        ks.ib.p.primitive_left,
        ks,
        workspace,
    )
    discrete_maxwellian_3v!(
        right,
        ks.ib.p.conservative_right,
        ks.ib.p.primitive_right,
        ks,
        workspace,
    )
    return left, right
end

"""
Apply fixed kinetic reservoirs to ghosts and incoming half characteristics.

At x=x_left, molecules with u>0 enter from the upstream Maxwellian.  At
x=x_right, molecules with u<0 enter from downstream.  The complementary half
ranges leave the domain and keep the state evolved from the interior.
"""
function apply_normal_shock_boundary!(
    ks,
    ctr,
    face,
    left_distribution,
    right_distribution,
    workspace,
)
    nx = ks.ps.nx
    set_shock_distribution!(ctr[0], left_distribution, ks, workspace)
    set_shock_distribution!(ctr[nx+1], right_distribution, ks, workspace)
    set_shock_distribution!(face[0], left_distribution, ks, workspace)
    set_shock_distribution!(face[nx+2], right_distribution, ks, workspace)

    for velocity_index in eachindex(ks.vs.u)
        if ks.vs.u[velocity_index] >= 0
            face[1].f[velocity_index] = left_distribution[velocity_index]
        else
            face[nx+1].f[velocity_index] = right_distribution[velocity_index]
        end
    end
    face[1].favg .= face[1].f
    face[nx+1].favg .= face[nx+1].f
    update_shock_macroscopic!(face[1], ks, workspace)
    update_shock_macroscopic!(face[nx+1], ks, workspace)
    return nothing
end

"""Smooth transition used only to start the finite-time shock calculation."""
@inline shock_blending_fraction(x, width) = 0.5 * (1 + tanh(x / width))

"""
Initialize active-flux cell averages and interface point values.

The initial distribution is a smooth convex mixture of the two reservoir
Maxwellians.  This is not asserted to be a steady kinetic shock; it merely
reduces startup ringing while the short calculation begins to form one.
"""
function initialize_normal_shock_solution(ks, workspace; initial_width=2.0)
    nx = ks.ps.nx
    shape = size(ks.vs.u)
    ctr = OffsetArray{ShockControlVolume}(undef, 0:nx+1)
    face = OffsetArray{ShockInterface}(undef, 0:nx+2)

    for i in 0:nx+1
        ctr[i] = ShockControlVolume(zeros(5), zeros(5), zeros(shape))
    end
    for i in 0:nx+2
        face[i] = ShockInterface(
            zeros(5), zeros(5), zeros(shape), zeros(shape),
        )
    end

    left_distribution, right_distribution =
        normal_shock_reservoirs(ks, workspace)
    temporary = similar(ks.vs.u)

    # Integrate the smooth x profile, rather than sampling it at cell centers.
    for i in 1:nx
        fill!(ctr[i].f, 0)
        for q in eachindex(SHOCK_GL4_NODES)
            x = ks.ps.x[i] + 0.5 * ks.ps.dx[i] * SHOCK_GL4_NODES[q]
            blend = shock_blending_fraction(x, initial_width)
            @. temporary = (1 - blend) * left_distribution +
                           blend * right_distribution
            @. ctr[i].f += 0.5 * SHOCK_GL4_WEIGHTS[q] * temporary
        end
        update_shock_macroscopic!(ctr[i], ks, workspace)
    end

    for i in 1:nx+1
        xface = ks.ps.x0 + (i - 1) * ks.ps.dx[1]
        blend = shock_blending_fraction(xface, initial_width)
        @. face[i].f = (1 - blend) * left_distribution +
                       blend * right_distribution
        face[i].favg .= face[i].f
        update_shock_macroscopic!(face[i], ks, workspace)
    end

    apply_normal_shock_boundary!(
        ks, ctr, face, left_distribution, right_distribution, workspace,
    )
    return ctr, face, left_distribution, right_distribution
end

"""Pack cell averages and interface values into an OrdinaryDiffEq state."""
function pack_normal_shock_state(ks, ctr, face)
    nvelocity = length(ks.vs.u)
    nx = ks.ps.nx
    state = zeros(eltype(ctr[1].f), nvelocity, 2nx + 1)
    for i in 1:nx
        @views state[:, i] .= vec(ctr[i].f)
    end
    for i in 1:nx+1
        @views state[:, nx+i] .= vec(face[i].f)
    end
    return state
end

"""Packed-state column occupied by physical interface i."""
@inline normal_shock_interface_column(parameters, i) = parameters.ks.ps.nx + i

"""Return a cell value or reservoir adjacent to physical interface i."""
@inline function normal_shock_adjacent_value(state, parameters, i, j, side)
    nx = parameters.ks.ps.nx
    if side === :left
        return i == 1 ? parameters.left_distribution[j] : state[j, i-1]
    end
    return i == nx + 1 ? parameters.right_distribution[j] : state[j, i]
end

"""Continuous switch from active flux to first-order upwinding at a jump."""
@inline function normal_shock_jump_blend(left, right; smooth=0.02, nonsmooth=0.20)
    jump = abs(right - left) / (abs(right) + abs(left) + eps(Float64))
    return clamp((nonsmooth - jump) / (nonsmooth - smooth), 0.0, 1.0)
end

"""
Evaluate the shock-limited semi-discrete active-flux transport RHS.

For every velocity node this is the scalar equation f_t+u f_x=0.  The
cell-average flux uses the evolved shared point value in smooth regions and is
blended toward positive kinetic upwinding near a jump.  A conservative flux
correction limiter protects the forward-Euler update used by this inexpensive
preliminary calculation.  Interface point values use the upwind
derivative of the neighboring quadratic active-flux reconstruction.
"""
function normal_shock_transport_rhs!(derivative, state, parameters, time)
    ks = parameters.ks
    nx = ks.ps.nx
    nvelocity = length(parameters.velocity_x)
    dx = ks.ps.dx[1]
    dt = parameters.step_size
    size(state) == (nvelocity, 2nx + 1) ||
        throw(DimensionMismatch("invalid packed normal-shock state"))
    fill!(derivative, zero(eltype(derivative)))

    low_face = parameters.low_face
    limited_face = parameters.limited_face

    for i in 1:nx+1, j in 1:nvelocity
        velocity = parameters.velocity_x[j]
        left = normal_shock_adjacent_value(state, parameters, i, j, :left)
        right = normal_shock_adjacent_value(state, parameters, i, j, :right)
        upwind = velocity >= 0 ? left : right
        point = state[j, normal_shock_interface_column(parameters, i)]
        blend = normal_shock_jump_blend(left, right)
        incoming = (i == 1 && velocity >= 0) ||
                   (i == nx + 1 && velocity < 0)
        low_face[j, i] = upwind
        limited_face[j, i] = incoming ? upwind :
                              upwind + blend * (point - upwind)
    end

    # One scale per velocity component retains the conservative difference of
    # two neighboring face fluxes while preventing a negative cell average.
    for j in 1:nvelocity
        velocity = parameters.velocity_x[j]
        correction_scale = 1.0
        for i in 1:nx
            low_rhs = -velocity / dx *
                      (low_face[j, i+1] - low_face[j, i])
            high_rhs = -velocity / dx *
                       (limited_face[j, i+1] - limited_face[j, i])
            low_candidate = state[j, i] + dt * low_rhs
            high_candidate = state[j, i] + dt * high_rhs
            if high_candidate < 0
                denominator = low_candidate - high_candidate
                denominator > 0 || continue
                correction_scale = min(
                    correction_scale,
                    0.999999999999 * max(low_candidate, 0) / denominator,
                )
            end
        end
        if correction_scale < 1
            for i in 1:nx+1
                limited_face[j, i] = low_face[j, i] + correction_scale *
                                     (limited_face[j, i] - low_face[j, i])
            end
        end
    end

    for i in 1:nx, j in 1:nvelocity
        derivative[j, i] = -parameters.velocity_x[j] / dx *
                           (limited_face[j, i+1] - limited_face[j, i])
    end

    for i in 1:nx+1, j in 1:nvelocity
        velocity = parameters.velocity_x[j]
        point_column = normal_shock_interface_column(parameters, i)
        incoming = (i == 1 && velocity >= 0) ||
                   (i == nx + 1 && velocity < 0)
        if incoming
            derivative[j, point_column] = 0
            continue
        end

        left = normal_shock_adjacent_value(state, parameters, i, j, :left)
        right = normal_shock_adjacent_value(state, parameters, i, j, :right)
        upwind = velocity >= 0 ? left : right
        blend = normal_shock_jump_blend(left, right)
        high_derivative = if velocity >= 0
            far_left = state[
                j, normal_shock_interface_column(parameters, i-1),
            ]
            -velocity / dx *
            (2far_left - 6state[j, i-1] + 4state[j, point_column])
        else
            far_right = state[
                j, normal_shock_interface_column(parameters, i+1),
            ]
            -velocity / dx *
            (-4state[j, point_column] + 6state[j, i] - 2far_right)
        end
        low_derivative = -abs(velocity) / dx *
                         (state[j, point_column] - upwind)
        point_derivative = low_derivative + blend *
                           (high_derivative - low_derivative)
        if state[j, point_column] + dt * point_derivative < 0
            point_derivative = -0.999999999999 *
                               state[j, point_column] / dt
        end
        derivative[j, point_column] = point_derivative
    end
    return nothing
end

"""Copy an OrdinaryDiffEq state back into active-flux storage."""
function unpack_normal_shock_state!(
    state,
    ks,
    ctr,
    face,
    left_distribution,
    right_distribution,
    workspace,
)
    nx = ks.ps.nx
    shape = size(ks.vs.u)
    for i in 1:nx
        ctr[i].f .= reshape(@view(state[:, i]), shape)
        update_shock_macroscopic!(ctr[i], ks, workspace)
    end
    for i in 1:nx+1
        face[i].f .= reshape(@view(state[:, nx+i]), shape)
        face[i].favg .= face[i].f
        update_shock_macroscopic!(face[i], ks, workspace)
    end
    apply_normal_shock_boundary!(
        ks, ctr, face, left_distribution, right_distribution, workspace,
    )
    return nothing
end

"""Advance one collisionless active-flux interval with explicit Euler."""
function normal_shock_transport_rk_step!(
    ks,
    ctr,
    face,
    dt,
    left_distribution,
    right_distribution,
    workspace;
    algorithm=Euler(),
)
    initial_state = pack_normal_shock_state(ks, ctr, face)
    nvelocity = length(ks.vs.u)
    parameters = NormalShockTransportParameters(
        ks,
        collect(vec(ks.vs.u)),
        collect(vec(left_distribution)),
        collect(vec(right_distribution)),
        zeros(eltype(initial_state), nvelocity, ks.ps.nx + 1),
        zeros(eltype(initial_state), nvelocity, ks.ps.nx + 1),
        dt,
    )
    problem = ODEProblem(
        normal_shock_transport_rhs!,
        initial_state,
        (zero(dt), dt),
        parameters,
    )
    solution = solve(
        problem,
        algorithm;
        adaptive=false,
        dt,
        save_everystep=false,
    )
    unpack_normal_shock_state!(
        last(solution.u),
        ks,
        ctr,
        face,
        left_distribution,
        right_distribution,
        workspace,
    )
    return solution
end

"""Remove the five discrete moments of a fast-spectral collision array."""
function project_fsm_collision!(collision, distribution, workspace)
    shock_moments!(workspace.defect, collision, workspace)
    shock_gram!(workspace.gram, distribution, workspace)
    workspace.coefficients .= workspace.gram \ workspace.defect
    mul!(
        workspace.multiplier,
        workspace.invariants,
        workspace.coefficients,
    )
    for velocity_index in eachindex(collision)
        collision[velocity_index] -= distribution[velocity_index] *
                                     workspace.multiplier[velocity_index]
    end
    shock_moments!(workspace.current, collision, workspace)
    return maximum(abs, workspace.current)
end

"""
Evaluate a conservative, equilibrium-corrected full Boltzmann FSM RHS.

On a finite Fourier grid, `Q(M,M)` is small but not exactly zero.  Subtracting
the spectral residual of the Maxwellian with the same discrete moments makes
the collision step well balanced without changing the continuous operator.
The subsequent five-moment projection removes the remaining FFT quadrature
defect to roundoff.
"""
function fsm_collision_rhs!(collision, distribution, ks, workspace)
    shock_moments!(workspace.target, distribution, workspace)
    primitive = conserve_prim(workspace.target, ks.gas.γ)
    discrete_maxwellian_3v!(
        workspace.equilibrium,
        workspace.target,
        primitive,
        ks,
        workspace,
    )
    boltzmann_fft!(collision, distribution, ks.gas.fsm)
    boltzmann_fft!(
        workspace.q_equilibrium,
        workspace.equilibrium,
        ks.gas.fsm,
    )
    @. collision -= workspace.q_equilibrium
    return project_fsm_collision!(collision, distribution, workspace)
end

"""Exact, discretely conservative BGK relaxation of one local distribution."""
function bgk_collision_update!(distribution, ks, dt, workspace)
    shock_moments!(workspace.target, distribution, workspace)
    primitive = conserve_prim(workspace.target, ks.gas.γ)
    discrete_maxwellian_3v!(
        workspace.equilibrium,
        workspace.target,
        primitive,
        ks,
        workspace,
    )
    collision_time = vhs_collision_time(
        primitive,
        ks.gas.μᵣ,
        ks.gas.ω,
    )
    decay = exp(-dt / collision_time)
    @. distribution = workspace.equilibrium +
                      decay * (distribution - workspace.equilibrium)
    return 0.0
end

"""
Second-order positive and conservative update of the full Boltzmann collision.

An explicit midpoint step is adequate for the present `Kn=1` preliminary
calculation.  After its midpoint and final stages, the exponential moment
projection removes small Fourier undershoots while preserving the five moments
of the incoming distribution.  `collision_substeps` can be increased for a
longer or more collisional calculation.
"""
function fsm_collision_update!(
    distribution,
    ks,
    dt,
    workspace;
    collision_substeps=1,
)
    substep = dt / collision_substeps
    largest_projection_residual = 0.0

    for substep_index in 1:collision_substeps
        shock_moments!(workspace.target, distribution, workspace)
        workspace.current .= workspace.target

        largest_projection_residual = max(
            largest_projection_residual,
            fsm_collision_rhs!(workspace.q, distribution, ks, workspace),
        )
        @. workspace.stage = distribution + 0.5substep * workspace.q
        positive_moment_projection!(
            workspace.stage,
            workspace.stage,
            workspace.current,
            workspace,
        )

        largest_projection_residual = max(
            largest_projection_residual,
            fsm_collision_rhs!(workspace.q, workspace.stage, ks, workspace),
        )
        @. workspace.trial = distribution + substep * workspace.q
        positive_moment_projection!(
            distribution,
            workspace.trial,
            workspace.current,
            workspace,
        )
    end
    return largest_projection_residual
end

"""Dispatch one local collision update to BGK or full Boltzmann FSM."""
function local_shock_collision_update!(
    distribution,
    ks,
    dt,
    workspace,
    collision_model;
    collision_substeps=1,
)
    if collision_model === :bgk
        return bgk_collision_update!(distribution, ks, dt, workspace)
    elseif collision_model === :fsm
        return fsm_collision_update!(
            distribution,
            ks,
            dt,
            workspace;
            collision_substeps,
        )
    end
    throw(ArgumentError("unknown collision model $collision_model"))
end

"""Reconstruct four nonnegative distributions inside cell i."""
function limited_shock_collision_points!(point_values, ks, ctr, face, i)
    nvelocity = length(ks.vs.u)
    for q in eachindex(SHOCK_GL4_NODES)
        xi = 0.5 * (SHOCK_GL4_NODES[q] + 1)
        left_basis = 3xi^2 - 4xi + 1
        average_basis = 6xi - 6xi^2
        right_basis = 3xi^2 - 2xi
        @views point_values[q, :] .=
            left_basis .* vec(face[i].f) .+
            average_basis .* vec(ctr[i].f) .+
            right_basis .* vec(face[i+1].f)
    end

    for j in 1:nvelocity
        minimum_point = minimum(@view point_values[:, j])
        average = ctr[i].f[j]
        if minimum_point < 0
            theta = min(
                1.0,
                0.999999999999 * average / (average - minimum_point),
            )
            for q in axes(point_values, 1)
                point_values[q, j] = average +
                                     theta * (point_values[q, j] - average)
            end
        end
    end
    return nothing
end

"""Apply one x-high-order collision half step to cells and interfaces."""
function normal_shock_collision_step!(
    ks,
    ctr,
    face,
    dt,
    left_distribution,
    right_distribution,
    workspace,
    collision_model;
    collision_substeps=1,
)
    nvelocity = length(ks.vs.u)
    point_values = zeros(eltype(ks.vs.u), 4, nvelocity)
    point_distribution = similar(ks.vs.u)
    updated_average = similar(ks.vs.u)
    maximum_collision_residual = 0.0

    # Collision is nonlinear in f.  Therefore Q must be evaluated on the
    # reconstructed point distributions before Gauss averaging back to fbar.
    for i in 1:ks.ps.nx
        limited_shock_collision_points!(point_values, ks, ctr, face, i)
        fill!(updated_average, 0)
        for q in eachindex(SHOCK_GL4_NODES)
            point_distribution .= reshape(
                @view(point_values[q, :]), size(ks.vs.u),
            )
            residual = local_shock_collision_update!(
                point_distribution,
                ks,
                dt,
                workspace,
                collision_model;
                collision_substeps,
            )
            maximum_collision_residual = max(
                maximum_collision_residual, residual,
            )
            @. updated_average +=
                0.5 * SHOCK_GL4_WEIGHTS[q] * point_distribution
        end
        ctr[i].f .= updated_average
        update_shock_macroscopic!(ctr[i], ks, workspace)
    end

    for i in 1:ks.ps.nx+1
        residual = local_shock_collision_update!(
            face[i].f,
            ks,
            dt,
            workspace,
            collision_model;
            collision_substeps,
        )
        maximum_collision_residual = max(maximum_collision_residual, residual)
        face[i].favg .= face[i].f
        update_shock_macroscopic!(face[i], ks, workspace)
    end

    apply_normal_shock_boundary!(
        ks, ctr, face, left_distribution, right_distribution, workspace,
    )
    return maximum_collision_residual
end

"""Transport CFL step; the collision part is handled in separate half steps."""
function normal_shock_timestep(ks, time)
    dt = ks.set.cfl * minimum(ks.ps.dx[1:ks.ps.nx]) /
         maximum(abs, ks.vs.u)
    return min(dt, ks.set.maxTime - time)
end

"""
Solve the finite-time active-flux normal shock with one collision backend.

The Strang sequence is collision(dt/2), transport(dt), collision(dt/2).
Transport is the same `ODEProblem` for FSM and BGK, while the local collision
dispatch above supplies the only model-dependent operation.
"""
function solve_normal_shock_active_flux(;
    collision_model=:fsm,
    algorithm=Euler(),
    collision_substeps=1,
    initial_width=2.0,
    kwargs...,
)
    ks = create_normal_shock_solver(; collision_model, kwargs...)
    workspace = create_shock_collision_workspace(ks)
    ctr, face, left_distribution, right_distribution =
        initialize_normal_shock_solution(ks, workspace; initial_width)

    time = 0.0
    steps = 0
    maximum_collision_residual = 0.0
    elapsed = @elapsed while time < ks.set.maxTime - eps(ks.set.maxTime)
        dt = normal_shock_timestep(ks, time)
        maximum_collision_residual = max(
            maximum_collision_residual,
            normal_shock_collision_step!(
                ks,
                ctr,
                face,
                0.5dt,
                left_distribution,
                right_distribution,
                workspace,
                collision_model;
                collision_substeps,
            ),
        )
        normal_shock_transport_rk_step!(
            ks,
            ctr,
            face,
            dt,
            left_distribution,
            right_distribution,
            workspace;
            algorithm,
        )
        maximum_collision_residual = max(
            maximum_collision_residual,
            normal_shock_collision_step!(
                ks,
                ctr,
                face,
                0.5dt,
                left_distribution,
                right_distribution,
                workspace,
                collision_model;
                collision_substeps,
            ),
        )
        time += dt
        steps += 1
    end

    minimum_distribution = minimum(
        minimum(ctr[i].f) for i in 1:ks.ps.nx
    )
    return (;
        ks,
        ctr,
        face,
        time,
        steps,
        elapsed,
        collision_model,
        collision_substeps,
        maximum_collision_residual,
        minimum_distribution,
        algorithm,
        formulation=:ordinarydiffeq_strang_1d1f3v,
    )
end

"""Extract density, streamwise velocity, temperature, stress, and heat flux."""
function normal_shock_profile(result)
    ks, ctr = result.ks, result.ctr
    nx = ks.ps.nx
    x = collect(ks.ps.x[1:nx])
    density = [ctr[i].prim[1] for i in 1:nx]
    velocity = [ctr[i].prim[2] for i in 1:nx]
    temperature = [1 / (2ctr[i].prim[5]) for i in 1:nx]
    normal_stress = zeros(nx)
    heat_flux = zeros(nx)

    for i in 1:nx
        U = velocity[i]
        for velocity_index in eachindex(ks.vs.u)
            cx = ks.vs.u[velocity_index] - U
            cy = ks.vs.v[velocity_index]
            cz = ks.vs.w[velocity_index]
            weighted_f = ks.vs.weights[velocity_index] *
                         ctr[i].f[velocity_index]
            normal_stress[i] += cx^2 * weighted_f
            heat_flux[i] += 0.5cx * (cx^2 + cy^2 + cz^2) * weighted_f
        end
    end
    return (; x, density, velocity, temperature, normal_stress, heat_flux)
end

"""
Convert an in-memory solution to portable arrays and scalar metadata.

The returned dictionary deliberately excludes the `SolverSet`, function
closures, OrdinaryDiffEq solution object, and precomputed FSM convolution
kernel.  Those objects are either Julia-session-specific or unnecessarily
large.  Cell and interface distributions, macroscopic variables, grids,
reservoir states, diagnostics, and plotting profiles are retained, so the JLD2
file is sufficient for post-processing without repeating either simulation.
"""
function normal_shock_result_data(result)
    ks, ctr, face = result.ks, result.ctr, result.face
    nx = ks.ps.nx
    velocity_shape = size(ks.vs.u)
    T = eltype(ks.vs.u)

    cell_distribution = Array{T}(undef, velocity_shape..., nx)
    interface_distribution = Array{T}(undef, velocity_shape..., nx + 1)
    conservative = Matrix{T}(undef, 5, nx)
    primitive = Matrix{T}(undef, 5, nx)
    for i in 1:nx
        @views cell_distribution[:, :, :, i] .= ctr[i].f
        @views conservative[:, i] .= ctr[i].w
        @views primitive[:, i] .= ctr[i].prim
    end
    for i in 1:nx+1
        @views interface_distribution[:, :, :, i] .= face[i].f
    end

    profile = normal_shock_profile(result)
    return Dict{String,Any}(
        "format_version" => 1,
        "collision_model" => String(result.collision_model),
        "time" => result.time,
        "steps" => result.steps,
        "elapsed_seconds" => result.elapsed,
        "minimum_distribution" => result.minimum_distribution,
        "maximum_collision_residual" => result.maximum_collision_residual,
        "algorithm" => String(nameof(typeof(result.algorithm))),
        "formulation" => String(result.formulation),
        "parameters" => Dict{String,Any}(
            "mach" => ks.gas.Ma,
            "knudsen" => ks.gas.Kn,
            "gamma" => ks.gas.γ,
            "internal_degrees" => ks.gas.K,
            "cfl" => ks.set.cfl,
            "x0" => ks.ps.x0,
            "x1" => ks.ps.x1,
            "nx" => nx,
            "nu" => ks.vs.nu,
            "nv" => ks.vs.nv,
            "nw" => ks.vs.nw,
        ),
        "grid" => Dict{String,Any}(
            "x" => collect(ks.ps.x[1:nx]),
            "dx" => collect(ks.ps.dx[1:nx]),
            "u" => collect(ks.vs.u[:, 1, 1]),
            "v" => collect(ks.vs.v[1, :, 1]),
            "w" => collect(ks.vs.w[1, 1, :]),
        ),
        "reservoir" => Dict{String,Any}(
            "primitive_left" => copy(ks.ib.p.primitive_left),
            "primitive_right" => copy(ks.ib.p.primitive_right),
            "conservative_left" => copy(ks.ib.p.conservative_left),
            "conservative_right" => copy(ks.ib.p.conservative_right),
        ),
        "state" => Dict{String,Any}(
            "cell_distribution" => cell_distribution,
            "interface_distribution" => interface_distribution,
            "conservative" => conservative,
            "primitive" => primitive,
        ),
        "profile" => Dict{String,Any}(
            "x" => profile.x,
            "density" => profile.density,
            "velocity" => profile.velocity,
            "temperature" => profile.temperature,
            "normal_stress" => profile.normal_stress,
            "heat_flux" => profile.heat_flux,
        ),
    )
end

"""Print diagnostics for one independently executed collision model."""
function print_normal_shock_diagnostics(result)
    println("Active-flux normal-shock simulation completed")
    println("  collision model: ", uppercase(String(result.collision_model)))
    println("  transport integrator: ", nameof(typeof(result.algorithm)))
    println("  final time: ", result.time)
    println("  Strang steps: ", result.steps)
    println("  elapsed seconds: ", round(result.elapsed; digits=3))
    println("  minimum cell-average f: ", result.minimum_distribution)
    if result.collision_model === :fsm
        println(
            "  maximum projected collision-moment residual: ",
            result.maximum_collision_residual,
        )
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("shock.jl provides the shared normal-shock solver.")
    println("Run shock_boltzmann.jl, shock_bgk.jl, and shock_plot.jl in order.")
end
