# CFD Solver — Development Notes

## Project Overview

2D incompressible Navier-Stokes finite volume solver based on the SIMPLE
algorithm. Core solver in `navier_stokes.py` with lid-driven cavity flow
benchmark included.

## Repository

**URL: https://github.com/DKword17/cfd-solver**

## Modules

| Module | Contents |
|--------|----------|
| `solver/` | FVM discretisation, linear solvers, SIMPLE/SIMPLEC/PISO, Rhie-Chow |
| `turbulence/` | Spalart-Allmaras, k-ε, k-ω SST, LES Smagorinsky, wall functions |
| `mesh/` | Structured 2D mesh generation, AMR, coordinate transforms, quality metrics, CGNS/PLOT3D I/O |
| `boundary/` | BC framework, turbulent inlet (SEM), FSI coupling, conjugate HT, porous jump |
| `kernel/` | CUDA Poisson solver, Rhie-Chow GPU kernel, TVD limiters |
| `verification/` | MMS, benchmark suite (Ghia, Poiseuille, backward step, cylinder), grid convergence (GCI) |

## Verification Criteria

| Module | Acceptance |
|--------|-----------|
| Solver | Re=100 cavity vortex centre within 1% of Ghia et al. |
| Turbulence | k-ω SST Cf error < 5% on flat plate |
| Mesh | GCI < 5% over 5 refinement levels |
| Boundary | Cylinder Re=40 separation angle error < 2° |
| GPU | Poisson solver speedup > 40× (256² grid) |
| Verification | MMS spatial order ≥ 1.8 (2nd-order upwind) |
