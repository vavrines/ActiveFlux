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

This repository is at an early stage of development.
Future revisions will add solver modules, benchmark cases, tests, and a
directory containing the LaTeX sources.

## Getting started

Install [Julia](https://julialang.org/) and instantiate the project from the
repository root:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Run the current example with:

```sh
julia --project=. example/advection.jl
```

The example solves the constant-coefficient advection equation on a periodic
one-dimensional domain and plots the numerical solution alongside its initial
state.

## License

This project is distributed under the terms in [LICENSE](LICENSE).
