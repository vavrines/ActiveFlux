"""
One-dimensional active flux method for advection equation with convergence and conservation checks.

∂ₜu + ∂ₓu = 0 with periodic boundary conditions
"""

using KitBase
using KitBase.OffsetArrays

function solve_advection(nx; exact_average=false, c=0.5, tend=1.0)
    # Construct a uniform finite-volume mesh on [0, 1]. KitBase adds one ghost
    # cell on either side, so the physical cell indices are 1:nx and the ghost
    # cell indices are 0 and nx+1.
    ps = PSpace1D(0, 1, nx, 1)
    dx = ps.dx[1]

    # Active Flux stores two kinds of degrees of freedom:
    #
    #   uc[i]   = average of u over physical cell i,
    #   uf[i]   = point value of u at a cell interface.
    #
    # The expression sin(2π*x_i) is only the value at the cell center. The
    # exact average of sin(2π*x) over a cell of width dx is
    #
    #   sin(2π*x_i) * sin(π*dx)/(π*dx)
    # = sin(2π*x_i) * sinc(dx),
    #
    # because Julia uses the normalized definition sinc(x)=sin(π*x)/(π*x).
    # Keeping both initialization choices lets this script demonstrate why the
    # distinction between a point value and a cell average matters.
    uc = sin.(2π .* ps.x)
    if exact_average
        uc .*= sinc.(ps.dx)
    end

    # Store interface values on two extra interface ghost locations. For this
    # mesh, uf[i] is the point value at the left boundary of cell i. Therefore
    # cell i is described by the triplet
    #
    #   left interface uf[i], cell average uc[i], right interface uf[i+1].
    #
    # The additional entries uf[0] and uf[nx+2] simplify periodic indexing.
    uf = OffsetArray{Float64}(undef, 0:ps.nx+2)
    for i = 0:ps.nx+1
        uf[i] = sin(2π * (ps.x[i] - ps.dx[i] / 2))
    end
    uf[end] = sin(2π * (ps.x[end] + ps.dx[end] / 2))

    # ufb contains the interface state averaged over one time step. Since the
    # physical flux for u_t + u_x = 0 is f(u)=u, this is also the time-averaged
    # numerical flux used in the conservative finite-volume update.
    ufb = zeros(ps.nx + 1)

    # uf is overwritten with the interface values at the new time level. Keep
    # uf0 as the complete set of interface values at the old time level so all
    # interface updates within one step use the same data.
    uf0 = copy(uf)

    # Coefficients obtained by evaluating and integrating the continuous
    # quadratic reconstruction in the upwind cell. With normalized coordinate
    # ξ in [0,1], the reconstruction in a cell is
    #
    #   p(ξ) = uL*(3ξ²-4ξ+1)
    #        + ubar*(6ξ-6ξ²)
    #        + uR*(3ξ²-2ξ),
    #
    # where uL and uR are its interface point values and ubar is its average.
    ηn = 4 - 3c
    ηf = 3c - 2
    ϕn = 2 - c
    ϕf = c - 1

    # For unit advection speed, c=dt/dx is the CFL number. The characteristic
    # starting from an interface must remain in the adjacent upwind cell, which
    # gives the standard Active Flux condition 0 < c <= 1.
    dt = c * dx

    # This convergence experiment chooses tend and c so tend/dt is integral.
    # round(Int, ...) avoids sensitivity to floating-point representation.
    nt = round(Int, tend / dt)

    # The conservative update should preserve this discrete total mass to
    # roundoff accuracy under periodic boundary conditions.
    mass0 = sum(uc[1:ps.nx]) * dx

    for _ = 1:nt
        # Update every physical interface. The wave speed is +1, so information
        # reaching interface i travels from the cell immediately to its left,
        # whose data are
        #
        #   uL   = uf0[i-1],
        #   ubar = uc[i-1],
        #   uR   = uf0[i].
        #
        # The characteristic foot after one time step has local coordinate
        # ξ=1-c. Evaluating p there gives the new point value uf[i].
        #
        # Integrating p(1-s) for 0 <= s <= c gives ufb[i]. For a quadratic this
        # integral is exact and is equivalent to applying Simpson's rule at the
        # old, half-step, and new interface states.
        for i = 1:ps.nx+1
            uf[i] = uf0[i] - c * ηf * (uc[i-1] - uf0[i-1]) -
                    c * ηn * (uf0[i] - uc[i-1])
            ufb[i] = uf0[i] - c * ϕf * (uc[i-1] - uf0[i-1]) -
                     c * ϕn * (uf0[i] - uc[i-1])
        end

        # Apply the conservative finite-volume balance
        #
        #   ubar_i^(n+1) = ubar_i^n
        #                  - (dt/dx)*(F_{i+1/2} - F_{i-1/2}).
        #
        # Here ufb[i] is the left flux and ufb[i+1] is the right flux of cell i.
        # Neighboring cells use the same interface flux, so their contributions
        # cancel exactly when summed over the periodic domain.
        for i = 1:ps.nx
            uc[i] += c * (ufb[i] - ufb[i+1])
        end

        # Fill periodic ghost cells and ghost interfaces. The two physical
        # representations of the periodic interface, uf[1] and uf[nx+1], are
        # evolved independently from periodic copies of the same data and
        # should consequently remain equal to roundoff accuracy.
        uc[0] = uc[ps.nx]
        uc[ps.nx+1] = uc[1]
        uf[0] = uf[ps.nx]
        uf[ps.nx+2] = uf[2]

        # Make the completed interface solution the old solution for the next
        # time step. Cell averages are updated in place and need no second copy.
        uf0 .= uf
    end

    # At tend=1, unit-speed advection on the unit periodic domain has completed
    # one full revolution, so the exact solution equals the initial sine wave.
    # Construct both interpretations to quantify the initialization issue.
    exact_point = sin.(2π .* ps.x[1:ps.nx])
    exact_cell_average = exact_point .* sinc(dx)

    # Report an L2-like RMS error, conservation error, and periodic-boundary
    # consistency. The point error is returned for optional experimentation;
    # the convergence table below deliberately uses the cell-average error.
    return (
        cell_average_error=sqrt(
            sum(abs2, uc[1:ps.nx] - exact_cell_average) / nx,
        ),
        point_error=sqrt(sum(abs2, uc[1:ps.nx] - exact_point) / nx),
        mass_drift=sum(uc[1:ps.nx]) * dx - mass0,
        periodic_mismatch=abs(uf[1] - uf[ps.nx+1]),
    )
end

function run_convergence_check(; exact_average)
    # Refine by factors of two. If e_h is the error on spacing h, the observed
    # order is log2(e_h/e_{h/2}); a third-order method approaches a rate of 3.
    description = exact_average ?
                  "exact cell-average initialization" :
                  "point-sample initialization from advection.jl"
    println(description)

    previous_error = nothing
    for nx in (25, 50, 100, 200, 400)
        result = solve_advection(nx; exact_average)
        rate = isnothing(previous_error) ?
               NaN : log2(previous_error / result.cell_average_error)

        println((
            nx=nx,
            cell_average_error=result.cell_average_error,
            rate=rate,
            mass_drift=result.mass_drift,
            periodic_mismatch=result.periodic_mismatch,
        ))
        previous_error = result.cell_average_error
    end
end

# First expose the order reduction caused by treating center values as cell
# averages, then repeat with consistent finite-volume initial data. The second
# table should approach third order while mass drift and periodic mismatch stay
# near machine precision.
run_convergence_check(; exact_average=false)
println()
run_convergence_check(; exact_average=true)
