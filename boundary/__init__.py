"""
boundary/__init__.py
=====================

Boundary conditions module for the CFD solver.

This package provides a comprehensive suite of boundary condition
implementations for incompressible Navier-Stokes simulations. It includes
generic boundary condition types (wall, inlet, outlet, symmetry, periodic,
opening), turbulent inflow generation techniques, fluid-structure interaction
coupling, conjugate heat transfer interface treatment, and porous media jump
conditions.

Modules:
    conditions:          Generic boundary condition framework with Wall, Inlet,
                         Outlet, SymmetryPlane, Periodic, and Opening types.
    turbulent_inlet:     Turbulent inflow generation methods including the
                         Synthetic Eddy Method, vortex method, digital filter
                         method, and prescribed Reynolds stress specification.
    fsi_coupling:        Fluid-Structure Interaction coupling with interface
                         mapping, load transfer, and displacement transfer.
    conjugate_ht:        Conjugate heat transfer treatment at fluid-solid
                         interfaces ensuring heat flux continuity and
                         temperature coupling.
    porous_jump:         Porous media jump conditions implementing the
                         Darcy-Forchheimer law for pressure drop across
                         porous interfaces.

Author:     Ananya Patel
            Indian Institute of Technology Bombay
Date:       2026-07-27
"""

from boundary.conditions import (
    Wall,
    Inlet,
    Outlet,
    SymmetryPlane,
    Periodic,
    Opening,
    apply_boundary_conditions,
)

from boundary.turbulent_inlet import (
    SyntheticEddyMethod,
    VortexMethod2D,
    DigitalFilterMethod,
    prescribed_reynolds_stresses,
)

from boundary.fsi_coupling import (
    FSICoupling,
    InterfaceMapping,
    LoadTransfer,
    DisplacementTransfer,
)

from boundary.conjugate_ht import (
    ConjugateHeatTransfer,
    FluidSolidInterface,
)

from boundary.porous_jump import (
    PorousJump,
    DarcyForchheimer,
)

__all__ = [
    # conditions
    "Wall",
    "Inlet",
    "Outlet",
    "SymmetryPlane",
    "Periodic",
    "Opening",
    "apply_boundary_conditions",
    # turbulent_inlet
    "SyntheticEddyMethod",
    "VortexMethod2D",
    "DigitalFilterMethod",
    "prescribed_reynolds_stresses",
    # fsi_coupling
    "FSICoupling",
    "InterfaceMapping",
    "LoadTransfer",
    "DisplacementTransfer",
    # conjugate_ht
    "ConjugateHeatTransfer",
    "FluidSolidInterface",
    # porous_jump
    "PorousJump",
    "DarcyForchheimer",
]
