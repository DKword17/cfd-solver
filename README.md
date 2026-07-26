# CFD Solver

**Finite Volume Navier-Stokes Solver — SIMPLE Algorithm on Collocated Grids**

Implements the SIMPLE algorithm for 2D incompressible laminar and turbulent flows.
Core solver includes:
- Finite volume discretisation (2nd order central / QUICK / MUSCL)
- Rhie-Chow interpolation for collocated grids
- Boundary conditions (wall, inlet, outlet, symmetry, periodic)
- Lid-driven cavity flow benchmark (Ghia et al. 1982)

Contents:
- `navier_stokes.py` — Core SIMPLE solver + cavity flow benchmark
- `docs/TEAM_BRIEFING.md` — Team structure and task assignments

Quick Start:
    $ python navier_stokes.py
