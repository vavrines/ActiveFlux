"""
Sod shock tube for the one-dimensional active-flux Boltzmann solver.

The physical mesh is one-dimensional, while the monatomic distribution uses
all three molecular-velocity components (`1d1f3v`, `K=0`, `gamma=5/3`). The
standard Sod states are initially separated at `x=0.5`:

    (ρ, U, p)_L = (1.0,   0.0, 1.0),
    (ρ, U, p)_R = (0.125, 0.0, 0.1).

Transport uses a case-specific semi-discrete active-flux right-hand side and
is advanced by an OrdinaryDiffEq strong-stability-preserving Runge--Kutta
method. Because a shock problem is nonsmooth, the default limiter forms one
density/pressure jump sensor per physical face and blends the high-order
interface flux and point-value derivative locally with a positive first-order
upwind operator. A face-local flux-corrected-transport coefficient preserves
cell-average positivity without reducing the correction at unrelated faces.
Unlimited and legacy per-velocity modes remain available for controlled
comparisons. The complete Boltzmann collision integral is
evaluated with KitBase's fast spectral method (FSM), followed by a conservative
five-moment projection and a positive exponential correction. A two-stage
successive BGK penalty gives the homogeneous collision map a strong
asymptotic-preserving limit without replacing the full Boltzmann operator.

This file contains the complete, case-specific numerical core and does not
include any other numerical example. Run `sod_simulation.jl` to
save a portable JLD2 solution and `sod_plot.jl` to create the comparison figure
from that saved result without repeating the kinetic calculation.
"""

using KitBase
using KitBase.OffsetArrays
using LinearAlgebra
using OrdinaryDiffEq: ODEProblem, SSPRK33, solve
using Base.Threads: @threads, maxthreadid, nthreads, threadid

# Four-point Gauss--Legendre quadrature is used both for the positivity-limited
# collision reconstruction and for averaging its relaxed values back to the
# cell degree of freedom. These constants are local to the Sod case so the
# program does not rely on the periodic-advection implementation.
const SOD_GL4_NODES = (
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526,
)
const SOD_GL4_WEIGHTS = (
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538,
)

"""Cell-average active-flux degree of freedom for the Sod distribution."""
mutable struct SodControlVolume{TW,TP,TF}
    w::TW
    prim::TP
    f::TF
end

"""Shared physical-interface point value of the Sod distribution."""
mutable struct SodInterface{TW,TP,TF,TFA}
    w::TW
    prim::TP
    f::TF
    favg::TFA
end

"""Reusable three-velocity storage for the Sod FSM collision operator."""
mutable struct SodCollisionWorkspace{T,TA,TM,TV}
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
    trial::TA
    target::Vector{T}
    current::Vector{T}
    defect::Vector{T}
    coefficients::Vector{T}
    gram::Matrix{T}
end

"""Thread-private collision reconstruction arrays and FSM workspace."""
mutable struct SodCollisionThreadCache{TW,TP,TA}
    workspace::TW
    point_values::TP
    point_distribution::TA
    updated_average::TA
end

"""All thread-local collision data plus a reduction buffer."""
struct SodCollisionThreadPool{TC,TR}
    caches::TC
    residuals::TR
end

"""Allocate invariant matrices and collision temporaries once per simulation."""
function create_sod_collision_workspace(ks)
    shape = size(ks.vs.u)
    nvelocity = length(ks.vs.u)
    T = eltype(ks.vs.u)
    invariants = Matrix{T}(undef, nvelocity, 5)
    for index in eachindex(ks.vs.u)
        u = ks.vs.u[index]
        v = ks.vs.v[index]
        w = ks.vs.w[index]
        invariants[index, 1] = 1
        invariants[index, 2] = u
        invariants[index, 3] = v
        invariants[index, 4] = w
        invariants[index, 5] = 0.5 * (u^2 + v^2 + w^2)
    end

    array() = zeros(T, shape)
    return SodCollisionWorkspace(
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
        zeros(T, 5),
        zeros(T, 5),
        zeros(T, 5),
        zeros(T, 5),
        zeros(T, 5, 5),
    )
end

"""
Allocate one independent collision cache for every Julia thread.

The FSM routines and the conservative moment projection mutate their
workspace. Sharing one instance inside `@threads` would therefore introduce a
race even though different physical cells are being updated. Thread one reuses
the solver's primary workspace; all other threads receive a complete private
copy of the mutable arrays while sharing only the read-only spectral kernel in
`ks`.
"""
function create_sod_collision_thread_pool(ks, primary_workspace)
    make_cache(workspace) = SodCollisionThreadCache(
        workspace,
        zeros(
            eltype(ks.vs.u),
            length(SOD_GL4_NODES),
            length(ks.vs.u),
        ),
        similar(ks.vs.u),
        similar(ks.vs.u),
    )
    first_cache = make_cache(primary_workspace)
    caches = Vector{typeof(first_cache)}(undef, maxthreadid())
    caches[1] = first_cache
    for tid in 2:length(caches)
        workspace = create_sod_collision_workspace(ks)
        caches[tid] = make_cache(workspace)
    end
    return SodCollisionThreadPool(
        caches,
        zeros(eltype(ks.vs.u), length(caches)),
    )
end

"""Compute mass, three momentum components, and energy without allocation."""
function sod_moments!(moments, distribution, workspace)
    workspace.weighted_values .= workspace.weights .* vec(distribution)
    mul!(moments, transpose(workspace.invariants), workspace.weighted_values)
    return moments
end

"""Form the weighted Gram matrix of the five collision invariants."""
function sod_gram!(gram, distribution, workspace)
    workspace.weighted_values .= workspace.weights .* vec(distribution)
    for column in 1:5
        @views @. workspace.weighted_basis[:, column] =
            workspace.weighted_values * workspace.invariants[:, column]
    end
    mul!(gram, transpose(workspace.invariants), workspace.weighted_basis)
    return gram
end

"""
Project a trial distribution to nonnegative values at fixed five moments.

Clipping an FSM update directly would violate conservation. Instead, the
clipped base is multiplied by `exp(alpha dot psi)` and Newton's method chooses
the five coefficients so that the incoming mass, momentum, and energy are
recovered to roundoff.
"""
function sod_positive_moment_projection!(
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
        for index in eachindex(output)
            exponent = clamp(workspace.multiplier[index], -50, 50)
            output[index] = workspace.base[index] * exp(exponent)
        end
        sod_moments!(workspace.current, output, workspace)
        @. workspace.defect = target - workspace.current
        residual = maximum(abs.(workspace.defect) ./ target_scale)
        residual <= tolerance && return residual

        sod_gram!(workspace.gram, output, workspace)
        increment = workspace.gram \ workspace.defect
        damping = min(1.0, 1.0 / max(maximum(abs, increment), 1.0))
        @. workspace.coefficients += damping * increment
    end

    sod_moments!(workspace.current, output, workspace)
    residual = maximum(abs.(target .- workspace.current) ./ target_scale)
    residual <= 1e-9 || error(
        "positive moment projection did not converge; residual=$residual",
    )
    return residual
end

"""Construct a positive Maxwellian with exactly the target discrete moments."""
function sod_discrete_maxwellian!(equilibrium, target, primitive, ks, workspace)
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
    sod_positive_moment_projection!(equilibrium, equilibrium, target, workspace)
    return equilibrium
end

"""Recompute the five conservative and primitive variables of one Sod state."""
function update_sod_macroscopic!(state, ks, workspace)
    sod_moments!(state.w, state.f, workspace)
    state.prim .= conserve_prim(state.w, ks.gas.γ)
    return nothing
end

"""Evaluate the conservative, equilibrium-corrected Boltzmann FSM RHS."""
function sod_fsm_collision_rhs!(collision, distribution, ks, workspace)
    sod_moments!(workspace.target, distribution, workspace)
    primitive = conserve_prim(workspace.target, ks.gas.γ)
    sod_discrete_maxwellian!(
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

    # Remove the remaining FFT/quadrature defect in all five invariants.
    sod_moments!(workspace.defect, collision, workspace)
    sod_gram!(workspace.gram, distribution, workspace)
    workspace.coefficients .= workspace.gram \ workspace.defect
    mul!(
        workspace.multiplier,
        workspace.invariants,
        workspace.coefficients,
    )
    for index in eachindex(collision)
        collision[index] -= distribution[index] * workspace.multiplier[index]
    end
    sod_moments!(workspace.current, collision, workspace)
    return maximum(abs, workspace.current)
end

"""
Advance one homogeneous full-Boltzmann interval by successive penalization.

Let `M` be the discrete Maxwellian with the same five moments as `f`, let
`lambda = dt/tau`, and split the implicit BGK penalty into fractions `alpha`
and `1-alpha`. Eliminating the two implicit relaxation stages gives

    f_new - M = ((1 + lambda) * (f - M) + dt * Q(f,f)) /
                ((1 + alpha * lambda) *
                 (1 + (1-alpha) * lambda)).

The numerator contains the complete conservative FSM operator. For a resolved
collision interval (`lambda << 1`) the formula is `f + dt*Q + O(dt^2)`. For a
stiff interval its denominator is quadratic in `lambda`, so an arbitrary
non-equilibrium input is driven to `M` as `Kn -> 0`. This is the two-stage
successive-penalty construction of Yan and Jin; the former single denominator
only had the weaker, well-prepared AP property and left the Sod solutions close
to free transport. Positivity is restored at the unchanged five moments.
"""
function sod_successive_penalty_fsm_update!(
    distribution,
    ks,
    dt,
    workspace;
    penalty_alpha=0.5,
)
    0 < penalty_alpha < 1 || throw(ArgumentError(
        "penalty_alpha must lie strictly between zero and one",
    ))
    sod_moments!(workspace.target, distribution, workspace)
    primitive = conserve_prim(workspace.target, ks.gas.γ)
    collision_time = vhs_collision_time(primitive, ks.gas.μᵣ, ks.gas.ω)
    sod_discrete_maxwellian!(
        workspace.equilibrium,
        workspace.target,
        primitive,
        ks,
        workspace,
    )
    residual = sod_fsm_collision_rhs!(workspace.q, distribution, ks, workspace)
    lambda = dt / max(collision_time, eps(collision_time))
    denominator = (1 + penalty_alpha * lambda) *
                  (1 + (1 - penalty_alpha) * lambda)
    @. workspace.trial = workspace.equilibrium +
        ((1 + lambda) * (distribution - workspace.equilibrium) +
         dt * workspace.q) / denominator
    sod_positive_moment_projection!(
        distribution,
        workspace.trial,
        workspace.target,
        workspace,
    )
    return residual
end

"""Return the transport-CFL time step clipped at the requested final time."""
function sod_active_flux_timestep(ks, time)
    dx = minimum(ks.ps.dx[1:ks.ps.nx])
    maximum_velocity = maximum(abs, ks.vs.u)
    dt = ks.set.cfl * dx / maximum_velocity
    return min(dt, ks.set.maxTime - time)
end

"""Build the KitBase data for the `1d1f3v` full-Boltzmann Sod problem."""
function create_sod_solver(;
    nx=100,
    nu=64,
    nv=28,
    nw=28,
    knudsen=1e-4,
    velocity_limit=6.0,
    cfl=0.45,
    max_time=0.12,
    nm=5,
    alpha=1.0,
    omega=0.72,
)
    nm == 5 || throw(ArgumentError(
        "the production full-Boltzmann solver requires nm=5; received nm=$nm",
    ))
    set = Setup(;
        case="sod",
        space="1d1f3v",
        collision="fsm",
        interpOrder=2,
        boundary=["fix", "fix"],
        cfl,
        maxTime=max_time,
    )
    ps = PSpace1D(0.0, 1.0, nx, 1)
    vs = VSpace3D(
        -velocity_limit, velocity_limit, nu,
        -velocity_limit, velocity_limit, nv,
        -velocity_limit, velocity_limit, nw,
    )
    reference_viscosity = ref_vhs_vis(knudsen, alpha, 0.5)
    spectral_kernel = fsm_kernel(vs, reference_viscosity, nm, alpha)
    gas = Gas(;
        Kn=knudsen,
        K=0.0,
        γ=5 / 3,
        ω=omega,
        αᵣ=alpha,
        ωᵣ=0.5,
        μᵣ=reference_viscosity,
        fsm=spectral_kernel,
    )

    # KitBase 1d1f3v primitive variables are [rho,U,V,W,lambda], with
    # p=rho/(2lambda). Therefore the
    # canonical Sod pressures p_L=1 and p_R=0.1 correspond to λ_L=0.5 and
    # λ_R=0.625 for the densities below.
    primL = [1.0, 0.0, 0.0, 0.0, 0.5]
    primR = [0.125, 0.0, 0.0, 0.0, 0.625]
    wL = prim_conserve(primL, gas.γ)
    wR = prim_conserve(primR, gas.γ)
    midpoint = 0.5 * (ps.x0 + ps.x1)
    parameters = (;
        midpoint,
        primL,
        primR,
        wL,
        wR,
        u=vs.u,
        v=vs.v,
        velocity_w=vs.w,
        γ=gas.γ,
    )

    fw = (x, p) -> x <= p.midpoint ? p.wL : p.wR
    ff = (x, p) -> maxwellian(
        p.u,
        p.v,
        p.velocity_w,
        x <= p.midpoint ? p.primL : p.primR,
    )
    bc = (x, p) -> x <= p.midpoint ? p.primL : p.primR
    ib = IB1F(fw, ff, bc, parameters)
    return SolverSet(set, ps, vs, gas, ib)
end

"""Set one cell or interface state from a velocity distribution."""
function set_distribution!(state, distribution, ks, workspace)
    state.f .= distribution
    hasproperty(state, :favg) && (state.favg .= distribution)
    update_sod_macroscopic!(state, ks, workspace)
    return nothing
end

"""Return the left and right reservoir Maxwellians attached to the Sod data."""
function sod_reservoir_distributions(ks, workspace)
    left = similar(ks.vs.u)
    right = similar(ks.vs.u)
    sod_discrete_maxwellian!(
        left, ks.ib.p.wL, ks.ib.p.primL, ks, workspace,
    )
    sod_discrete_maxwellian!(
        right, ks.ib.p.wR, ks.ib.p.primR, ks, workspace,
    )
    return left, right
end

"""
Apply fixed kinetic reservoirs to ghosts and incoming boundary characteristics.

At the left boundary positive molecular velocities enter from the left state;
at the right boundary negative velocities enter from the right state.  The
opposite half ranges are outgoing and retain the value evolved from the
interior.  This is the kinetic analogue of a non-reflecting shock-tube boundary.
"""
function apply_sod_boundary!(
    ks,
    ctr,
    face,
    left_distribution,
    right_distribution,
    workspace,
)
    nx = ks.ps.nx

    set_distribution!(ctr[0], left_distribution, ks, workspace)
    set_distribution!(ctr[nx+1], right_distribution, ks, workspace)
    set_distribution!(face[0], left_distribution, ks, workspace)
    set_distribution!(face[nx+2], right_distribution, ks, workspace)

    for j in eachindex(ks.vs.u)
        if ks.vs.u[j] >= 0
            face[1].f[j] = left_distribution[j]
            face[1].favg[j] = left_distribution[j]
        else
            face[nx+1].f[j] = right_distribution[j]
            face[nx+1].favg[j] = right_distribution[j]
        end
    end
    face[1].favg .= face[1].f
    face[nx+1].favg .= face[nx+1].f
    update_sod_macroscopic!(face[1], ks, workspace)
    update_sod_macroscopic!(face[nx+1], ks, workspace)
    return nothing
end

"""Initialize exact cell averages and shared interface values for the jump."""
function initialize_sod_solution(ks, workspace)
    nx = ks.ps.nx
    shape = size(ks.vs.u)
    ctr = OffsetArray{SodControlVolume}(undef, 0:nx+1)
    face = OffsetArray{SodInterface}(undef, 0:nx+2)

    for i in 0:nx+1
        ctr[i] = SodControlVolume(zeros(5), zeros(5), zeros(shape))
    end
    for i in 0:nx+2
        face[i] = SodInterface(
            zeros(5), zeros(5), zeros(shape), zeros(shape),
        )
    end

    left_distribution, right_distribution =
        sod_reservoir_distributions(ks, workspace)
    discontinuity = 0.5 * (ks.ps.x0 + ks.ps.x1)

    # Integrate the piecewise-constant initial distribution exactly.  This also
    # works if a user chooses a mesh for which x=0.5 cuts through a cell.
    for i in 1:nx
        dx = ks.ps.dx[i]
        cell_left = ks.ps.x[i] - 0.5dx
        left_fraction = clamp((discontinuity - cell_left) / dx, 0.0, 1.0)
        @. ctr[i].f = left_fraction * left_distribution +
                         (1 - left_fraction) * right_distribution
        update_sod_macroscopic!(ctr[i], ks, workspace)
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
            for j in eachindex(ks.vs.u)
                face[i].f[j] = ks.vs.u[j] >= 0 ?
                               left_distribution[j] : right_distribution[j]
            end
        end
        face[i].favg .= face[i].f
        update_sod_macroscopic!(face[i], ks, workspace)
    end

    apply_sod_boundary!(
        ks,
        ctr,
        face,
        left_distribution,
        right_distribution,
        workspace,
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
    for q in eachindex(SOD_GL4_NODES)
        ξ = 0.5 * (SOD_GL4_NODES[q] + 1)
        left_basis = 3ξ^2 - 4ξ + 1
        average_basis = 6ξ - 6ξ^2
        right_basis = 3ξ^2 - 2ξ
        @views point_values[q, :] .=
            left_basis .* vec(face[i].f) .+
            average_basis .* vec(ctr[i].f) .+
            right_basis .* vec(face[i+1].f)
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

"""Collide one reconstructed physical cell using thread-private storage."""
function sod_collision_cell!(cache, ks, ctr, face, i, dt, penalty_alpha)
    point_values = cache.point_values
    point_distribution = cache.point_distribution
    updated_average = cache.updated_average
    workspace = cache.workspace
    maximum_residual = 0.0

    limited_collision_points!(point_values, ks, ctr, face, i)
    fill!(updated_average, 0)
    for q in eachindex(SOD_GL4_NODES)
        point_distribution .= reshape(
            @view(point_values[q, :]), size(ks.vs.u),
        )
        residual = sod_successive_penalty_fsm_update!(
            point_distribution,
            ks,
            dt,
            workspace;
            penalty_alpha,
        )
        maximum_residual = max(maximum_residual, residual)
        @. updated_average += 0.5 * SOD_GL4_WEIGHTS[q] * point_distribution
    end
    ctr[i].f .= updated_average
    update_sod_macroscopic!(ctr[i], ks, workspace)
    return maximum_residual
end

"""Collide one active-flux interface using thread-private storage."""
function sod_collision_interface!(cache, ks, face, i, dt, penalty_alpha)
    workspace = cache.workspace
    residual = sod_successive_penalty_fsm_update!(
        face[i].f,
        ks,
        dt,
        workspace;
        penalty_alpha,
    )
    face[i].favg .= face[i].f
    update_sod_macroscopic!(face[i], ks, workspace)
    return residual
end

"""
Perform one positive conservative full-Boltzmann collision step in parallel.

Physical cells are independent during a homogeneous collision interval, as are
the active-flux interface values. Two separate static `@threads` loops update
these sets. Each worker uses its own mutable FSM and projection workspace; the
only shared objects during either loop are read-only solver data and disjoint
cell/interface states.
"""
function sod_collision_step!(
    ks,
    ctr,
    face,
    dt,
    left_distribution,
    right_distribution,
    workspace,
    penalty_alpha,
    thread_pool=nothing,
)
    pool = isnothing(thread_pool) ?
           create_sod_collision_thread_pool(ks, workspace) : thread_pool
    fill!(pool.residuals, 0)

    @threads :static for i in 1:ks.ps.nx
        tid = threadid()
        residual = sod_collision_cell!(
            pool.caches[tid], ks, ctr, face, i, dt, penalty_alpha,
        )
        pool.residuals[tid] = max(pool.residuals[tid], residual)
    end
    maximum_collision_residual = maximum(pool.residuals)

    fill!(pool.residuals, 0)
    # Unlike the periodic wave, the shock tube has nx+1 distinct physical
    # interfaces; each is relaxed before incoming reservoir data are restored.
    @threads :static for i in 1:ks.ps.nx+1
        tid = threadid()
        residual = sod_collision_interface!(
            pool.caches[tid], ks, face, i, dt, penalty_alpha,
        )
        pool.residuals[tid] = max(pool.residuals[tid], residual)
    end
    maximum_collision_residual = max(
        maximum_collision_residual,
        maximum(pool.residuals),
    )

    apply_sod_boundary!(
        ks,
        ctr,
        face,
        left_distribution,
        right_distribution,
        workspace,
    )
    return maximum_collision_residual
end

"""Legacy per-velocity jump sensor retained for controlled comparisons."""
function sod_jump_blend(left_value, right_value; smooth=0.02, nonsmooth=0.20)
    jump = abs(right_value - left_value) /
           (abs(right_value) + abs(left_value) + eps(Float64))
    return clamp((nonsmooth - jump) / (nonsmooth - smooth), 0.0, 1.0)
end

const SOD_LIMITER_MODES = (:none, :legacy, :macroscopic_local)

"""Validate and return one of the supported Sod transport limiter modes."""
function sod_limiter_mode(mode)
    mode in SOD_LIMITER_MODES || throw(ArgumentError(
        "limiter must be one of $(SOD_LIMITER_MODES), received $mode",
    ))
    return mode
end

"""
Return one shock sensor shared by every molecular velocity at an interface.

Using density and pressure avoids the false positives produced when a relative
jump is formed from exponentially small Maxwellian-tail populations. The wider
transition interval keeps the third-order active-flux correction in smooth
rarefactions while still suppressing it at a strong pressure jump.
"""
function sod_macroscopic_jump_blend(
    density_left,
    pressure_left,
    density_right,
    pressure_right;
    smooth=0.08,
    nonsmooth=0.50,
)
    density_jump = abs(density_right - density_left) /
                   (abs(density_right) + abs(density_left) + eps(Float64))
    pressure_jump = abs(pressure_right - pressure_left) /
                    (abs(pressure_right) + abs(pressure_left) + eps(Float64))
    jump = max(density_jump, pressure_jump)
    return clamp((nonsmooth - jump) / (nonsmooth - smooth), 0.0, 1.0)
end

"""Parameters of the case-specific semi-discrete Sod transport operator."""
struct SodTransportParameters{KS,TV,TL,TR,TM,TMM,TS,TPR,T}
    ks::KS
    velocity_x::TV
    left_distribution::TL
    right_distribution::TR
    low_face::TM
    limited_face::TM
    moment_matrix::TMM
    macro_moments::TMM
    density::TS
    pressure::TS
    face_blend::TS
    positivity_ratio::TPR
    limiter::Symbol
    sensor_smooth::T
    sensor_nonsmooth::T
    step_size::T
end

"""Recover density and pressure from one discrete distribution column."""
function sod_density_pressure(distribution, moment_matrix, gamma)
    density = dot(@view(moment_matrix[1, :]), distribution)
    momentum_x = dot(@view(moment_matrix[2, :]), distribution)
    momentum_y = dot(@view(moment_matrix[3, :]), distribution)
    momentum_z = dot(@view(moment_matrix[4, :]), distribution)
    energy = dot(@view(moment_matrix[5, :]), distribution)
    safe_density = max(density, eps(eltype(moment_matrix)))
    pressure = (gamma - 1) * (
        energy - 0.5 * (
            momentum_x^2 + momentum_y^2 + momentum_z^2
        ) / safe_density
    )
    return safe_density, max(pressure, eps(eltype(moment_matrix)))
end

"""Packed-state column containing physical interface i, 1 <= i <= nx+1."""
@inline sod_interface_column(parameters::SodTransportParameters, i) =
    parameters.ks.ps.nx + i

"""
Pack nx cell averages and nx+1 distinct shock-tube interface values.

Unlike the periodic advection case, the left and right boundary interfaces are
different degrees of freedom, so the packed state has 2nx+1 columns.
"""
function pack_sod_state(ks, ctr, face)
    nvelocity = length(ks.vs.u)
    state = zeros(eltype(ctr[1].f), nvelocity, 2 * ks.ps.nx + 1)
    @threads :static for i in 1:ks.ps.nx
        @views state[:, i] .= vec(ctr[i].f)
    end
    @threads :static for i in 1:(ks.ps.nx+1)
        @views state[:, ks.ps.nx+i] .= vec(face[i].f)
    end
    return state
end

"""Copy the final Runge--Kutta state back and restore kinetic reservoirs."""
function unpack_sod_state!(
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
    size(state) == (length(ks.vs.u), 2 * nx + 1) ||
        throw(DimensionMismatch("invalid packed Sod state size $(size(state))"))

    @threads :static for i in 1:nx
        ctr[i].f .= reshape(@view(state[:, i]), shape)
    end
    @threads :static for i in 1:(nx+1)
        face[i].f .= reshape(@view(state[:, nx+i]), shape)
        face[i].favg .= face[i].f
    end

    # Moment evaluation uses the primary mutable workspace and is therefore
    # kept outside the parallel copy loops. Its cost is small relative to the
    # phase-space copies and transport RHS.
    for i in 1:nx
        update_sod_macroscopic!(ctr[i], ks, workspace)
    end
    for i in 1:nx+1
        update_sod_macroscopic!(face[i], ks, workspace)
    end
    apply_sod_boundary!(
        ks,
        ctr,
        face,
        left_distribution,
        right_distribution,
        workspace,
    )
    return nothing
end

"""Return the adjacent cell/reservoir value on one side of interface i."""
@inline function sod_adjacent_value(state, parameters, i, j, side)
    nx = parameters.ks.ps.nx
    if side === :left
        return i == 1 ? parameters.left_distribution[j] : state[j, i-1]
    end
    return i == nx + 1 ? parameters.right_distribution[j] : state[j, i]
end

"""
Evaluate the shock-limited semi-discrete active-flux transport operator.

The cell-average flux starts from positive first-order kinetic upwinding and
adds the sensor-limited active-flux interface correction. The positivity
correction ensures that one forward-Euler step of length `step_size` keeps all
cell averages nonnegative, which is the property needed by the convex stages
of SSPRK33.

The default `:macroscopic_local` mode uses one density/pressure sensor per
physical face and a face-local flux-corrected-transport positivity bound.
`:none` provides an unlimited comparison, while `:legacy` reproduces the
former per-velocity sensor and domain-global positivity coefficient.
Interface point values use the upwind derivative of the quadratic
reconstruction in smooth regions. At a detected jump this derivative is
blended with a relaxation toward the upwind cell average. Incoming boundary
half ranges are fixed by assigning them zero time derivative.
"""
function sod_transport_rhs!(derivative, state, parameters, time)
    ks = parameters.ks
    nx = ks.ps.nx
    nvelocity = length(parameters.velocity_x)
    dx = ks.ps.dx[1]
    dt = parameters.step_size
    size(state) == (nvelocity, 2 * nx + 1) ||
        throw(DimensionMismatch("invalid packed Sod state size $(size(state))"))
    fill!(derivative, zero(eltype(derivative)))

    low_face = parameters.low_face
    limited_face = parameters.limited_face
    limiter = parameters.limiter

    # The recommended sensor is computed once per physical interface from
    # macroscopic cell averages and is then shared by every velocity. This is
    # both cheaper and substantially less dissipative than measuring relative
    # jumps in the tiny populations of the Maxwellian tails.
    if limiter === :macroscopic_local
        parameters.density[1], parameters.pressure[1] = sod_density_pressure(
            parameters.left_distribution,
            parameters.moment_matrix,
            ks.gas.γ,
        )
        mul!(
            parameters.macro_moments,
            parameters.moment_matrix,
            @view(state[:, 1:nx]),
        )
        for i in 1:nx
            density = parameters.macro_moments[1, i]
            momentum_x = parameters.macro_moments[2, i]
            momentum_y = parameters.macro_moments[3, i]
            momentum_z = parameters.macro_moments[4, i]
            energy = parameters.macro_moments[5, i]
            safe_density = max(density, eps(eltype(state)))
            pressure = (ks.gas.γ - 1) * (
                energy - 0.5 * (
                    momentum_x^2 + momentum_y^2 + momentum_z^2
                ) / safe_density
            )
            parameters.density[i+1] = safe_density
            parameters.pressure[i+1] = max(pressure, eps(eltype(state)))
        end
        parameters.density[nx+2], parameters.pressure[nx+2] =
            sod_density_pressure(
                parameters.right_distribution,
                parameters.moment_matrix,
                ks.gas.γ,
            )
        for i in 1:nx+1
            parameters.face_blend[i] = sod_macroscopic_jump_blend(
                parameters.density[i],
                parameters.pressure[i],
                parameters.density[i+1],
                parameters.pressure[i+1],
                smooth=parameters.sensor_smooth,
                nonsmooth=parameters.sensor_nonsmooth,
            )
        end
    elseif limiter === :none
        fill!(parameters.face_blend, one(eltype(parameters.face_blend)))
    end

    # Each physical interface owns a contiguous velocity-space column in the
    # packed arrays, so threading over interfaces gives disjoint writes and
    # cache-friendly inner loops over molecular velocity.
    @threads :static for i in 1:(nx+1)
        for j in 1:nvelocity
            velocity = parameters.velocity_x[j]
            left_value = sod_adjacent_value(state, parameters, i, j, :left)
            right_value = sod_adjacent_value(state, parameters, i, j, :right)
            upwind_value = velocity >= 0 ? left_value : right_value
            point_value = state[j, sod_interface_column(parameters, i)]
            blend = limiter === :legacy ?
                    sod_jump_blend(left_value, right_value) :
                    parameters.face_blend[i]

            # Incoming boundary data are exact reservoirs and carry no
            # antidiffusive correction.
            incoming = (i == 1 && velocity >= 0) ||
                       (i == nx + 1 && velocity < 0)
            low_face[j, i] = upwind_value
            limited_face[j, i] = incoming ? upwind_value :
                upwind_value + blend * (point_value - upwind_value)
        end
    end

    if limiter === :legacy
        # Original comparison method: one troubled cell reduces the active-
        # flux correction at every physical face for that molecular velocity.
        @threads :static for j in 1:nvelocity
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
                        0.999999999999 * low_candidate / denominator,
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
    elseif limiter === :macroscopic_local
        # Local flux-corrected-transport bound. For each cell, compare the
        # available positive low-order state with only the negative parts of
        # the two adjacent antidiffusive fluxes. Each face then receives the
        # bound of the cell from which that correction removes population.
        # Unlike the legacy scalar, this coefficient never propagates a shock
        # restriction to unrelated faces elsewhere in the domain.
        safety = 0.999999999999
        @threads :static for j in 1:nvelocity
            # The positivity ratios are overwritten for every velocity. A
            # thread-private row prevents races while avoiding an
            # nvelocity-by-nx scratch array.
            ratio = @view parameters.positivity_ratio[threadid(), :]
            velocity = parameters.velocity_x[j]
            scale = dt * velocity / dx
            for i in 1:nx
                low_rhs = -velocity / dx *
                          (low_face[j, i+1] - low_face[j, i])
                low_candidate = max(state[j, i] + dt * low_rhs, 0.0)
                left_contribution = scale *
                    (limited_face[j, i] - low_face[j, i])
                right_contribution = -scale *
                    (limited_face[j, i+1] - low_face[j, i+1])
                loss = -min(left_contribution, 0.0) -
                       min(right_contribution, 0.0)
                ratio[i] = loss > 0 ?
                           min(1.0, safety * low_candidate / loss) : 1.0
            end

            for i in 1:nx+1
                correction = limited_face[j, i] - low_face[j, i]
                alpha = 1.0
                # This face is the right boundary of cell i-1.
                if i > 1 && -scale * correction < 0
                    alpha = min(alpha, ratio[i-1])
                end
                # This face is the left boundary of cell i.
                if i <= nx && scale * correction < 0
                    alpha = min(alpha, ratio[i])
                end
                limited_face[j, i] = low_face[j, i] + alpha * correction
            end
        end
    end

    # Conservative cell-average RHS using the limited kinetic face states.
    @threads :static for i in 1:nx
        for j in 1:nvelocity
            derivative[j, i] = -parameters.velocity_x[j] / dx *
                               (limited_face[j, i+1] - limited_face[j, i])
        end
    end

    # Semi-discrete evolution of the additional interface point values.
    @threads :static for i in 1:(nx+1)
        for j in 1:nvelocity
            velocity = parameters.velocity_x[j]
            point_column = sod_interface_column(parameters, i)
            incoming = (i == 1 && velocity >= 0) ||
                       (i == nx + 1 && velocity < 0)
            if incoming
                derivative[j, point_column] = 0
                continue
            end

            left_value = sod_adjacent_value(state, parameters, i, j, :left)
            right_value = sod_adjacent_value(state, parameters, i, j, :right)
            upwind_value = velocity >= 0 ? left_value : right_value
            blend = limiter === :legacy ?
                    sod_jump_blend(left_value, right_value) :
                    parameters.face_blend[i]
            high_derivative = if velocity >= 0
                far_left = state[j, sod_interface_column(parameters, i-1)]
                -velocity / dx *
                (2 * far_left - 6 * state[j, i-1] + 4 * state[j, point_column])
            else
                far_right = state[j, sod_interface_column(parameters, i+1)]
                -velocity / dx *
                (-4 * state[j, point_column] + 6 * state[j, i] - 2 * far_right)
            end
            low_derivative = -abs(velocity) / dx *
                             (state[j, point_column] - upwind_value)
            point_derivative = low_derivative + blend *
                               (high_derivative - low_derivative)

            # Point values are nonconservative degrees of freedom. Scaling
            # their derivative does not alter the finite-volume balance and
            # prevents a negative interface state from entering collision.
            if limiter !== :none &&
               state[j, point_column] + dt * point_derivative < 0
                point_derivative =
                    -0.999999999999 * state[j, point_column] / dt
            end
            derivative[j, point_column] = point_derivative
        end
    end
    return nothing
end

"""Advance one transport interval with an OrdinaryDiffEq Runge--Kutta solve."""
function sod_transport_rk_step!(
    ks,
    ctr,
    face,
    dt,
    left_distribution,
    right_distribution,
    workspace;
    algorithm=SSPRK33(),
    limiter=:macroscopic_local,
    sensor_smooth=0.08,
    sensor_nonsmooth=0.50,
)
    limiter = sod_limiter_mode(limiter)
    0 <= sensor_smooth < sensor_nonsmooth <= 1 || throw(ArgumentError(
        "sensor thresholds must satisfy 0 <= smooth < nonsmooth <= 1",
    ))
    initial_state = pack_sod_state(ks, ctr, face)
    moment_matrix = permutedims(
        workspace.invariants .* reshape(workspace.weights, :, 1),
    )
    parameters = SodTransportParameters(
        ks,
        collect(vec(ks.vs.u)),
        collect(vec(left_distribution)),
        collect(vec(right_distribution)),
        zeros(eltype(initial_state), length(ks.vs.u), ks.ps.nx + 1),
        zeros(eltype(initial_state), length(ks.vs.u), ks.ps.nx + 1),
        moment_matrix,
        zeros(eltype(initial_state), 5, ks.ps.nx),
        zeros(eltype(initial_state), ks.ps.nx + 2),
        zeros(eltype(initial_state), ks.ps.nx + 2),
        ones(eltype(initial_state), ks.ps.nx + 1),
        ones(eltype(initial_state), maxthreadid(), ks.ps.nx),
        limiter,
        sensor_smooth,
        sensor_nonsmooth,
        dt,
    )
    problem = ODEProblem(
        sod_transport_rhs!,
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
    unpack_sod_state!(
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

"""
Measure the final departure from the local discrete Maxwellian.

The weighted relative L2 norm is sensitive to non-equilibrium stress and heat
flux even when the five conserved moments agree. The relative entropy is
integrated over physical cells and divided by total mass. These diagnostics
are essential in a Knudsen sweep: a small-Kn solution must approach its local
Maxwellian rather than merely remain stable.
"""
function sod_nonequilibrium_diagnostics(ks, ctr, workspace)
    distribution_norm2 = zero(eltype(ks.vs.u))
    departure_norm2 = zero(eltype(ks.vs.u))
    relative_entropy = zero(eltype(ks.vs.u))
    total_mass = zero(eltype(ks.vs.u))

    for i in 1:ks.ps.nx
        distribution = ctr[i].f
        sod_moments!(workspace.target, distribution, workspace)
        primitive = conserve_prim(workspace.target, ks.gas.γ)
        sod_discrete_maxwellian!(
            workspace.equilibrium,
            workspace.target,
            primitive,
            ks,
            workspace,
        )
        dx = ks.ps.dx[i]
        total_mass += dx * workspace.target[1]
        for index in eachindex(distribution)
            weight = dx * ks.vs.weights[index]
            value = distribution[index]
            equilibrium = workspace.equilibrium[index]
            distribution_norm2 += weight * value^2
            departure_norm2 += weight * (value - equilibrium)^2
            if value > 0
                relative_entropy += weight * value * log(value / equilibrium)
            end
        end
    end

    return (;
        relative_l2=sqrt(departure_norm2 / distribution_norm2),
        relative_entropy_per_mass=relative_entropy / total_mass,
    )
end

"""
Solve one full-Boltzmann Sod shock tube through OrdinaryDiffEq transport.

Conservative two-stage successive-penalty FSM half steps surround the
case-specific SSPRK33 transport solve. The same algorithm is used from
continuum-like to rarefied reference Knudsen numbers.
"""
function solve_sod_active_flux(;
    algorithm=SSPRK33(),
    penalty_alpha=0.5,
    limiter=:macroscopic_local,
    sensor_smooth=0.08,
    sensor_nonsmooth=0.50,
    progress_interval=0,
    kwargs...,
)
    0 < penalty_alpha < 1 || throw(ArgumentError(
        "penalty_alpha must lie strictly between zero and one",
    ))
    limiter = sod_limiter_mode(limiter)
    0 <= sensor_smooth < sensor_nonsmooth <= 1 || throw(ArgumentError(
        "sensor thresholds must satisfy 0 <= smooth < nonsmooth <= 1",
    ))
    progress_interval >= 0 || throw(ArgumentError(
        "progress_interval must be nonnegative",
    ))
    ks = create_sod_solver(; kwargs...)
    workspace = create_sod_collision_workspace(ks)
    collision_thread_pool = create_sod_collision_thread_pool(ks, workspace)
    ctr, face, left_distribution, right_distribution =
        initialize_sod_solution(ks, workspace)

    time = 0.0
    steps = 0
    maximum_collision_residual = 0.0
    minimum_transport_cell_distribution = Inf
    minimum_transport_interface_distribution = Inf
    elapsed = @elapsed begin
        while time < ks.set.maxTime
            dt = sod_active_flux_timestep(ks, time)
            residual = sod_collision_step!(
                ks,
                ctr,
                face,
                0.5dt,
                left_distribution,
                right_distribution,
                workspace,
                penalty_alpha,
                collision_thread_pool,
            )
            maximum_collision_residual = max(
                maximum_collision_residual, residual,
            )
            sod_transport_rk_step!(
                ks,
                ctr,
                face,
                dt,
                left_distribution,
                right_distribution,
                workspace;
                algorithm,
                limiter,
                sensor_smooth,
                sensor_nonsmooth,
            )
            minimum_transport_cell_distribution = min(
                minimum_transport_cell_distribution,
                minimum(minimum(ctr[i].f) for i in 1:ks.ps.nx),
            )
            minimum_transport_interface_distribution = min(
                minimum_transport_interface_distribution,
                minimum(minimum(face[i].f) for i in 1:ks.ps.nx+1),
            )
            residual = sod_collision_step!(
                ks,
                ctr,
                face,
                0.5dt,
                left_distribution,
                right_distribution,
                workspace,
                penalty_alpha,
                collision_thread_pool,
            )
            maximum_collision_residual = max(
                maximum_collision_residual, residual,
            )
            time += dt
            steps += 1
            if progress_interval > 0 &&
               (steps % progress_interval == 0 || time >= ks.set.maxTime)
                println(
                    "  progress: step=$steps, t=",
                    round(time; digits=8),
                    "/",
                    ks.set.maxTime,
                )
                flush(stdout)
            end
        end
    end

    minimum_distribution = minimum(minimum(ctr[i].f) for i in 1:ks.ps.nx)
    nonequilibrium = sod_nonequilibrium_diagnostics(ks, ctr, workspace)
    return (;
        ks,
        ctr,
        face,
        workspace,
        time,
        steps,
        elapsed,
        minimum_distribution,
        maximum_collision_residual,
        collision_threads=nthreads(),
        minimum_transport_cell_distribution,
        minimum_transport_interface_distribution,
        nonequilibrium,
        penalty_alpha,
        limiter,
        sensor_smooth,
        sensor_nonsmooth,
        algorithm,
        formulation=:ordinarydiffeq_strang_successive_penalty_fsm,
    )
end

"""Return the numerical and exact macroscopic profiles used for comparison."""
function sod_profile(result)
    ks, ctr = result.ks, result.ctr
    x = collect(ks.ps.x[1:ks.ps.nx])
    density = [ctr[i].prim[1] for i in 1:ks.ps.nx]
    velocity = [ctr[i].prim[2] for i in 1:ks.ps.nx]
    pressure = [0.5 * ctr[i].prim[1] / ctr[i].prim[end] for i in 1:ks.ps.nx]
    temperature = [0.5 / ctr[i].prim[end] for i in 1:ks.ps.nx]
    heat_flux = zeros(eltype(ks.vs.u), ks.ps.nx)
    for i in 1:ks.ps.nx
        bulk_velocity = velocity[i]
        for index in eachindex(ks.vs.u)
            cx = ks.vs.u[index] - bulk_velocity
            cy = ks.vs.v[index]
            cz = ks.vs.w[index]
            heat_flux[i] += 0.5 * cx * (cx^2 + cy^2 + cz^2) *
                            ks.vs.weights[index] * ctr[i].f[index]
        end
    end

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
    return (;
        x,
        density,
        velocity,
        pressure,
        temperature,
        heat_flux,
        exact_density,
        exact_velocity,
        exact_pressure,
    )
end

"""Return cell-weighted L1 and L2 errors against the exact Euler solution."""
function sod_euler_error_metrics(result)
    profile = sod_profile(result)
    dx = result.ks.ps.dx[1]
    metrics = Dict{String,Float64}()
    for field in (:density, :velocity, :pressure)
        numerical = getproperty(profile, field)
        exact = getproperty(profile, Symbol("exact_" * String(field)))
        error = numerical .- exact
        metrics[String(field) * "_l1"] = sum(abs.(error)) * dx
        metrics[String(field) * "_l2"] = sqrt(sum(abs2, error) * dx)
    end
    return metrics
end

"""
Convert an in-memory Sod solution into portable arrays and scalar metadata.

The dictionary deliberately excludes the KitBase solver set, function
closures, and OrdinaryDiffEq solution objects. It retains the cell and
interface distributions, conservative and primitive variables, grids,
reservoir states, diagnostics, and both numerical and exact plotting profiles.
Consequently `sod_plot.jl` can visualize the result without loading the kinetic
solver or repeating the simulation.
"""
function sod_result_data(result)
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

    profile = sod_profile(result)
    euler_errors = sod_euler_error_metrics(result)
    return Dict{String,Any}(
        "format_version" => 4,
        "case" => "sod",
        "collision_model" => "fsm",
        "time" => result.time,
        "steps" => result.steps,
        "elapsed_seconds" => result.elapsed,
        "minimum_distribution" => result.minimum_distribution,
        "maximum_collision_residual" => result.maximum_collision_residual,
        "collision_threads" => result.collision_threads,
        "minimum_transport_cell_distribution" =>
            result.minimum_transport_cell_distribution,
        "minimum_transport_interface_distribution" =>
            result.minimum_transport_interface_distribution,
        "nonequilibrium_relative_l2" => result.nonequilibrium.relative_l2,
        "relative_entropy_per_mass" =>
            result.nonequilibrium.relative_entropy_per_mass,
        "euler_errors" => euler_errors,
        "algorithm" => String(nameof(typeof(result.algorithm))),
        "formulation" => String(result.formulation),
        "parameters" => Dict{String,Any}(
            "knudsen" => ks.gas.Kn,
            "gamma" => ks.gas.γ,
            "internal_degrees" => ks.gas.K,
            "cfl" => ks.set.cfl,
            "midpoint" => ks.ib.p.midpoint,
            "x0" => ks.ps.x0,
            "x1" => ks.ps.x1,
            "nx" => nx,
            "nu" => ks.vs.nu,
            "nv" => ks.vs.nv,
            "nw" => ks.vs.nw,
            "fsm_modes" => ks.gas.fsm.nm,
            "alpha" => ks.gas.αᵣ,
            "omega" => ks.gas.ω,
            "penalty_alpha" => result.penalty_alpha,
            "transport_limiter" => String(result.limiter),
            "sensor_smooth" => result.sensor_smooth,
            "sensor_nonsmooth" => result.sensor_nonsmooth,
            "collision_integrator" => "two-stage successive BGK penalty",
        ),
        "grid" => Dict{String,Any}(
            "x" => collect(ks.ps.x[1:nx]),
            "dx" => collect(ks.ps.dx[1:nx]),
            "u" => collect(ks.vs.u[:, 1, 1]),
            "v" => collect(ks.vs.v[1, :, 1]),
            "w" => collect(ks.vs.w[1, 1, :]),
        ),
        "reservoir" => Dict{String,Any}(
            "primitive_left" => copy(ks.ib.p.primL),
            "primitive_right" => copy(ks.ib.p.primR),
            "conservative_left" => copy(ks.ib.p.wL),
            "conservative_right" => copy(ks.ib.p.wR),
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
            "pressure" => profile.pressure,
            "temperature" => profile.temperature,
            "heat_flux" => profile.heat_flux,
            "exact_density" => profile.exact_density,
            "exact_velocity" => profile.exact_velocity,
            "exact_pressure" => profile.exact_pressure,
        ),
    )
end

function print_sod_diagnostics(result)
    println("Active-flux full-Boltzmann Sod integration completed")
    println("  model: 1d1f3v FSM, K=0, γ=5/3")
    println("  Knudsen number: ", result.ks.gas.Kn)
    println("  Julia transport/collision threads: ", result.collision_threads)
    println("  transport limiter: ", result.limiter)
    println(
        "  minimum post-transport cell distribution: ",
        result.minimum_transport_cell_distribution,
    )
    println(
        "  minimum post-transport interface distribution: ",
        result.minimum_transport_interface_distribution,
    )
    println("  final time: ", result.time)
    println("  time steps: ", result.steps)
    println("  elapsed seconds: ", round(result.elapsed; digits=3))
    println("  minimum cell-average distribution: ", result.minimum_distribution)
    println(
        "  maximum projected collision-moment residual: ",
        result.maximum_collision_residual,
    )
    println(
        "  final relative L2 departure from Maxwellian: ",
        result.nonequilibrium.relative_l2,
    )
    println(
        "  final relative entropy per unit mass: ",
        result.nonequilibrium.relative_entropy_per_mass,
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("sod.jl provides the shared Sod shock-tube solver.")
    println("Run sod_simulation.jl and then sod_plot.jl.")
end
