# ActiveFlux

ActiveFlux is a research project for developing the **active flux method** to
kinetic equations. Its main targets are the Boltzmann equation and the
single-relaxation-time Bhatnagar-Gross-Krook (BGK) model. The project is
intended to bring the solution algorithm, reproducible numerical implementations,
and the accompanying mathematical derivations together in one repository.

The active flux method evolves both cell averages and point values at cell
interfaces. This additional interface information gives a compact,
high-order representation of the solution and provides a natural starting
point for constructing numerical schemes for kinetic equations. The work in this repository explores
how that formulation can be combined with particle transport and collisional
relaxation in velocity space.

## Project goals

- Formulate active flux discretizations for the Boltzmann and BGK equations.
- Develop Julia solvers for transport, collision, and relaxation processes.
- Validate the schemes on standard kinetic benchmarks and study their
  accuracy, stability, and asymptotic behavior.
- Provide reproducible examples and numerical results.
- Maintain LaTeX sources for the derivation and documentation of the method.

## Current status

The repository currently contains the one-dimensional scalar transport test,
the smooth periodic `1d1f1v` BGK advection case, the full-Boltzmann Sod shock
tube, and a preliminary `1d1f3v` normal-shock calculation. Each reported case
has targeted source entry points. The advection driver packs the cell averages
and shared interface values into an `ODEProblem` and advances the complete
semi-discrete BGK right-hand side with `OrdinaryDiffEq.jl`. The Sod solver uses
a case-specific shock-limited `SSPRK33` transport solve and a conservative
fast-spectral Boltzmann collision update; separate simulation and plotting
programs exchange three portable JLD2 results. The normal-shock solver uses the
same `ODEProblem` pattern but selects explicit Euler to reduce the cost of the
preliminary FSM/BGK comparison.

Both shock-tube cases use the KitBase fast spectral method in three-dimensional
molecular velocity and apply a discrete five-moment projection. The Sod case
uses a two-stage successive BGK penalty for the full-Boltzmann collision map.
Unlike a single penalized Euler denominator, its quadratic stiff denominator
drives arbitrary non-equilibrium data toward the local Maxwellian as the
Knudsen number vanishes. This supplies a strong AP homogeneous collision step
for reference Knudsen numbers from `1e-4` to `1`, although uniform accuracy of
the complete transport--collision discretization still requires a separate
study. The normal-shock
case compares FSM and BGK collision backends while keeping transport, grids,
initial data, and kinetic boundaries identical. Its default run is
intentionally short and coarse; a steady, grid-refined shock study remains
future work. The two-dimensional solver and a systematic uniform-in-Knudsen
AP study are also not yet complete.

## Getting started

Install [Julia](https://julialang.org/) and instantiate the project from the
repository root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Run the smooth periodic BGK advection case with:

```shell
julia --project=. src/advection.jl
```

This case constructs an `ODEProblem` from the active-flux transport and
finite-Knudsen quadrature collision operators. The default `Tsit5` integrator
can be replaced through the `algorithm` keyword. An AP-oriented split problem
and `KenCarp4` driver remain available through
`solve_advection_active_flux(split=true)`.

Run the three active-flux full-Boltzmann Sod simulations with four Julia
threads and then plot their saved results with:

```shell
julia --threads=4 --project=. src/sod_simulation.jl
julia --project=. src/sod_plot.jl
```

The shock-tube solver uses `1d1f3v`, `K=0`, `γ=5/3`, and kinetic
inflow/outflow boundaries. Its current coarse-run defaults are `nx=50`
physical cells
(101 globally independent active-flux degrees of freedom, or 150 cell-local
average/endpoint entries) and a `28×28×28` molecular-velocity grid. The
complete Boltzmann collision integral is
evaluated by the fast spectral method at four reconstructed physical
quadrature points per cell and at every active-flux interface point. A
five-invariant projection removes the velocity-quadrature defect. The
two-stage successive BGK penalty evaluates the full FSM operator once per
collision call and uses two analytically eliminated implicit relaxation
stages. It reduces to explicit full-Boltzmann evolution when the collision
time is resolved and projects strongly toward the local Maxwellian in the
stiff limit. The same split collision update is used at `Kn=1e-4`, `1e-2`,
and `1`. The default `:macroscopic_local` transport limiter uses one density
and pressure shock sensor shared by all molecular velocities, then applies a
face-local flux-corrected-transport positivity bound. This avoids the former
behavior in which tiny Maxwellian-tail populations triggered the shock sensor
and one troubled cell reduced the active-flux correction across the complete
physical domain. The solver also retains `limiter=:none` and
`limiter=:legacy` for controlled comparisons; `:none` is diagnostic only
because unlimited interface values can become negative before collision.

The first command writes the full states and numerical/exact profiles to
`data/sod_kn1e-4.jld2`, `data/sod_kn1e-2.jld2`, and
`data/sod_kn1.jld2`. The second command reads only those files and writes the
density, velocity, pressure, and heat-flux comparison to
`tex/figures/sod-shock-tube.pdf`. An optional argument to the simulation
program selects the output directory. The plotting program optionally accepts
the three input paths followed by its output path. The solver implementation
in `src/sod.jl` is self-contained at the case level: it defines its own
active-flux states, FSM collision workspace, quadrature data, boundaries, and
time-step routine and does not include `advection.jl` or another example. Each
saved result also records the velocity-weighted relative departure from the
local Maxwellian and the relative entropy per unit mass, so a Knudsen sweep
checks asymptotic relaxation rather than stability alone. It additionally
stores Euler-profile errors and the minimum cell/interface distribution
immediately after transport, before the positive collision projection can hide
an unlimited undershoot.

The reconstructed cell collisions and active-flux interface collisions use
`Threads.@threads`. Each worker owns independent reconstruction arrays, FSM
temporaries, and conservative-projection storage; the spectral kernel is
read-only and shared. Launching Julia with `--threads=4` is therefore required
to obtain the intended four-thread execution. To compute only the preliminary
small-Knudsen result, use:

```shell
julia --threads=4 --project=. -e 'include("src/sod_simulation.jl"); run_sod_simulation(knudsen=1e-4)'
julia --project=. -e 'include("src/sod_plot.jl"); plot_sod_continuum_check()'
```

The second command writes the density, velocity, and pressure comparison with
the exact Euler Riemann solution to
`tex/figures/sod-continuum-check.png` without repeating the simulation.

The affordable limiter comparison uses the same `nx=50` physical mesh but a
coarse velocity grid solely to isolate the transport treatment:

```shell
julia --threads=4 --project=. -e 'include("src/sod_simulation.jl"); run_sod_limiter_comparison()'
```

Its three JLD2 files are written under `data/limiter_comparison/`. These data
must not replace the `28×28×28` results used for Figure 4.

Case 3 is split into two independent simulations and a lightweight plotting
step so that the expensive full-Boltzmann result can be reused:

```shell
julia --project=. src/shock_boltzmann.jl
julia --project=. src/shock_bgk.jl
julia --project=. src/shock_plot.jl
```

The first two commands save complete portable results to
`data/shock_boltzmann.jld2` and `data/shock_bgk.jld2`. The third command reads
only those files and writes the density, velocity, temperature, and heat-flux
comparison to `tex/figures/normal-shock-comparison.pdf`; it never starts a
kinetic simulation. Optional command-line arguments select the simulation
output path, or (for the plotting program) the two input paths and figure path.
The shared solver in `src/shock.jl` uses `1d1f3v`, `K=0`, and `gamma=5/3`.
Its defaults are meant for a quick implementation check; longer final times
and refined physical and velocity grids are needed for a steady resolved
study.

The two convergence programs deliberately retain the exact characteristic
active-flux update and Strang splitting because they isolate spatial accuracy:

```shell
julia --project=. src/convergence1-transport.jl
julia --project=. src/convergence2-overresolve.jl
```

They are not used as the time-integration drivers for the advection or Sod
figures.

## License

This project is distributed under the terms in [LICENSE](LICENSE).
