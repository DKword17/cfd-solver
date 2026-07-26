#!/usr/bin/env python3
"""
boundary/porous_jump.py
=========================

Porous media jump boundary condition for modelling pressure drop across
thin porous interfaces, membranes, filters, and screens.

Physical background:
    A porous jump is a thin interface or diaphragm across which a sudden
    pressure drop occurs due to the presence of a porous medium. This
    boundary condition is commonly used to model:

        - Filters and sieves in industrial processes.
        - Perforated plates and screens in flow distributors.
        - Membrane separators in biomedical devices.
        - Flame arrestors and porous burners.

    The pressure drop across a porous interface is described by the
    Darcy-Forchheimer law, which combines a linear (viscous) term and
    a quadratic (inertial) term:

        Δp = -(μ/α) · u_n + C₂ · (1/2) · ρ · |u_n| · u_n

    where:
        Δp  = pressure drop across the interface [Pa]
        μ   = dynamic viscosity of the fluid [Pa·s]
        α   = permeability of the porous medium [m²]
        C₂  = inertial resistance factor (pressure-jump coefficient) [1/m]
        ρ   = fluid density [kg/m³]
        u_n = normal velocity component through the interface [m/s]

    The first term represents the viscous (Darcy) contribution, dominant
    at low Reynolds numbers. The second term represents the inertial
    (Forchheimer) contribution, significant at higher flow rates.

    It is crucial to understand that the pressure drop is negative
    (i.e., the pressure decreases) in the direction of flow. The sign
    convention is: Δp = p_downstream - p_upstream, which is negative
    for forward flow.

References:
    - Darcy, H. (1856). Les Fontaines Publiques de la Ville de Dijon.
      Dalmont, Paris.
    - Forchheimer, P. (1901). Wasserbewegung durch Boden. Zeitschrift
      des Vereins Deutscher Ingenieure, 45:1781-1788.
    - Ergun, S. (1952). Fluid flow through packed columns. Chemical
      Engineering Progress, 48(2):89-94.
    - ANSYS Fluent Theory Guide (2021). Section 7.2: Porous Media
      Conditions.

Author:     Ananya Patel
            Indian Institute of Technology Bombay
Date:       2026-07-27
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from navier_stokes import Field2D, Mesh2D


# ─── Darcy-Forchheimer Law ─────────────────────────────────────────────

@dataclass
class DarcyForchheimer:
    """Implements the Darcy-Forchheimer law for pressure drop across a
    porous interface.

    Physical background:
        The Darcy-Forchheimer law describes the relationship between the
        pressure drop across a porous medium and the flow rate through it.
        The law has two regimes:

        1. Darcy (viscous-dominated) regime:
               Δp = -(μ/α) · u_n · Δn
           This applies at low Reynolds numbers (Re < 1) where viscous
           forces dominate.

        2. Forchheimer (inertial-dominated) regime:
               Δp = -[(μ/α) · u_n + C₂ · (1/2) · ρ · |u_n| · u_n] · Δn
           This applies at higher Reynolds numbers where inertial effects
           become significant.

        The full law (including both terms) is recommended for most
        practical applications, as it smoothly transitions between the
        two regimes.

    Please note that the permeability α and the inertial coefficient C₂
    are empirical parameters that depend on the porous medium structure
    (porosity, pore size, tortuosity, etc.). For packed beds, the Ergun
    equation can be used to estimate these parameters from the bed
    porosity and particle diameter.

    Args:
        permeability:       The permeability of the porous medium α [m²].
                            A higher permeability means less resistance.
                            Typical values: 1e-12 to 1e-9 m² for sand,
                            1e-15 to 1e-12 m² for clay. Defaults to 1e-10.
        inertial_coefficient: The inertial resistance factor C₂ [1/m].
                            Typical values: 1e4 to 1e7 1/m for packed
                            beds. Defaults to 10000.0.
        thickness:          The thickness of the porous interface Δn [m].
                            This is the physical thickness over which the
                            pressure drop occurs. Defaults to 0.001 (1 mm).
        density:            Fluid density ρ [kg/m³]. Defaults to 1.0.
        viscosity:          Dynamic viscosity of the fluid μ [Pa·s].
                            Defaults to 1e-5.
    """
    permeability: float = 1e-10
    inertial_coefficient: float = 10000.0
    thickness: float = 0.001
    density: float = 1.0
    viscosity: float = 1e-5

    def __post_init__(self) -> None:
        """Validates that the porous medium parameters are physically
        meaningful.

        Raises:
            ValueError: If any parameter is non-positive.
        """
        # It is crucial to understand that all porous medium parameters
        # must be positive. Non-positive values would lead to unphysical
        # behaviour (e.g., negative pressure drop in the flow direction).
        if self.permeability <= 0.0:
            raise ValueError(
                f"Permeability must be positive. Received {self.permeability}."
            )
        if self.inertial_coefficient < 0.0:
            raise ValueError(
                f"Inertial coefficient must be non-negative. "
                f"Received {self.inertial_coefficient}."
            )
        if self.thickness <= 0.0:
            raise ValueError(
                f"Thickness must be positive. Received {self.thickness}."
            )
        if self.density <= 0.0:
            raise ValueError(
                f"Density must be positive. Received {self.density}."
            )
        if self.viscosity <= 0.0:
            raise ValueError(
                f"Viscosity must be positive. Received {self.viscosity}."
            )

    def pressure_drop(
        self,
        normal_velocity: np.ndarray,
    ) -> np.ndarray:
        """Computes the pressure drop across the porous interface using
        the Darcy-Forchheimer law.

        The pressure drop is computed as:
            Δp = -[(μ/α) · u_n + C₂ · (1/2) · ρ · |u_n| · u_n] · Δn

        where each term is evaluated pointwise on the interface.

        It is crucial to understand that the sign convention is such that
        a positive normal velocity (flow from fluid to porous medium)
        results in a negative pressure drop (pressure decreases in the
        flow direction).

        Args:
            normal_velocity: The velocity component normal to the porous
                             interface [m/s]. Positive values indicate
                             flow from the reference side through the
                             porous medium. Shape (n_points,).

        Returns:
            The pressure drop Δp across the interface [Pa]. Shape
            (n_points,). Negative values indicate a pressure decrease
            in the direction of flow.

        Raises:
            ValueError: If normal_velocity contains NaN values.
        """
        # ── Validate the input ─────────────────────────────────────────
        if np.any(np.isnan(normal_velocity)):
            raise ValueError(
                "The normal velocity array contains NaN values. "
                "Kindly ensure that the velocity field is properly "
                "initialised before computing the pressure drop."
            )

        # ── Compute the viscous (Darcy) term ───────────────────────────
        # Δp_viscous = -(μ/α) · u_n · Δn
        viscous_term: np.ndarray = (
            -(self.viscosity / self.permeability)
            * normal_velocity
            * self.thickness
        )

        # ── Compute the inertial (Forchheimer) term ────────────────────
        # Δp_inertial = -C₂ · (1/2) · ρ · |u_n| · u_n · Δn
        # Please note the absolute value of u_n ensures that the inertial
        # term always opposes the flow direction.
        inertial_term: np.ndarray = (
            -self.inertial_coefficient
            * 0.5
            * self.density
            * np.abs(normal_velocity)
            * normal_velocity
            * self.thickness
        )

        # ── Total pressure drop ────────────────────────────────────────
        # Δp_total = Δp_viscous + Δp_inertial
        delta_p: np.ndarray = viscous_term + inertial_term

        return delta_p

    def pressure_gradient(
        self,
        normal_velocity: np.ndarray,
    ) -> np.ndarray:
        """Computes the pressure gradient (pressure drop per unit thickness)
        across the porous interface.

        This is a convenience method that returns the pressure drop
        normalised by the interface thickness. It is useful for
        comparing the resistance of different porous media independent
        of their thickness.

        Args:
            normal_velocity: The normal velocity component [m/s].
                             Shape (n_points,).

        Returns:
            The pressure gradient dp/dn [Pa/m]. Shape (n_points,).
        """
        delta_p: np.ndarray = self.pressure_drop(normal_velocity)
        gradient: np.ndarray = delta_p / self.thickness
        return gradient

    def compute_permeability_from_ergun(
        self,
        porosity: float = 0.4,
        particle_diameter: float = 0.001,
    ) -> None:
        """Estimates the permeability and inertial coefficient using the
        Ergun equation for packed beds.

        The Ergun equation (Ergun, 1952) relates the pressure drop through
        a packed bed to the bed properties:

            α = d_p² · ε³ / (150 · (1 - ε)²)
            C₂ = 3.5 · (1 - ε) / (d_p · ε³)

        where:
            d_p = mean particle diameter [m]
            ε  = bed porosity (void fraction)

        This method updates the `permeability` and `inertial_coefficient`
        attributes in-place.

        It is crucial to understand that the Ergun equation is valid for
        randomly packed beds of spherical particles. For non-spherical
        particles or structured packings, the coefficients may need
        adjustment.

        Args:
            porosity:          The bed porosity (void fraction) ε.
                               Ranges from 0 to 1. Typical value for
                               random close packing is 0.4. Defaults to 0.4.
            particle_diameter: The mean particle diameter d_p [m].
                               Defaults to 0.001 (1 mm).

        Raises:
            ValueError: If porosity is outside (0, 1) or particle
                        diameter is non-positive.
        """
        # ── Validate inputs ────────────────────────────────────────────
        if porosity <= 0.0 or porosity >= 1.0:
            raise ValueError(
                f"Porosity must be in the range (0, 1). "
                f"Received {porosity}."
            )
        if particle_diameter <= 0.0:
            raise ValueError(
                f"Particle diameter must be positive. "
                f"Received {particle_diameter}."
            )

        # ── Ergun equation for permeability (viscous term) ─────────────
        # α = d_p² · ε³ / (150 · (1 - ε)²)
        one_minus_eps_sq: float = (1.0 - porosity) ** 2
        self.permeability = (
            particle_diameter ** 2 * porosity ** 3
            / (150.0 * one_minus_eps_sq)
        )

        # ── Ergun equation for inertial coefficient ────────────────────
        # C₂ = 3.5 · (1 - ε) / (d_p · ε³)
        self.inertial_coefficient = (
            3.5 * (1.0 - porosity) / (particle_diameter * porosity ** 3)
        )


# ─── Porous Jump Boundary Condition ────────────────────────────────────

@dataclass
class PorousJump:
    """Porous jump boundary condition that applies a pressure drop across
    a thin porous interface based on the Darcy-Forchheimer law.

    This class provides the boundary condition treatment for a porous
    interface in a CFD simulation. It computes the pressure jump and
    applies it as a boundary condition on the pressure field.

    Physical background:
        In the finite volume method, a porous jump is treated as an
        internal boundary condition where the pressure field is
        discontinuous across the interface. The velocity field is
        continuous (mass conservation), but the pressure suddenly drops
        according to the Darcy-Forchheimer law.

        The implementation follows these steps:
            1. Compute the normal velocity at the interface.
            2. Evaluate the pressure drop using the Darcy-Forchheimer law.
            3. Apply the pressure drop as a discontinuity in the pressure
               field across the interface.

    Args:
        porous_model:  The DarcyForchheimer model instance containing the
                       porous medium parameters.
        side:          The side of the fluid domain where the porous
                       interface is located ('west', 'east', 'south',
                       'north'). Defaults to 'west'.
        mesh:          The computational mesh. Optional, but required for
                       some operations. Defaults to None.
    """
    porous_model: DarcyForchheimer = field(default_factory=DarcyForchheimer)
    side: str = "west"
    mesh: Optional[Mesh2D] = None

    def apply(
        self,
        u: Field2D,
        v: Field2D,
        p: Field2D,
        mesh: Mesh2D,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Applies the porous jump boundary condition to the velocity and
        pressure fields.

        This method computes the normal velocity at the porous interface,
        evaluates the pressure drop, and applies the pressure discontinuity
        to the pressure field. The velocity field is left unchanged (the
        porous jump does not impose a velocity boundary condition).

        Args:
            u:    The x-velocity field (nx × ny).
            v:    The y-velocity field (nx × ny).
            p:    The pressure field (nx × ny).
            mesh: The computational mesh.

        Returns:
            A tuple (u, v, p) with the porous jump condition applied
            to the pressure field. The velocity field is returned
            unchanged.
        """
        u_arr: np.ndarray = u.data
        v_arr: np.ndarray = v.data
        p_arr: np.ndarray = p.data

        # ── Determine the normal velocity at the interface ─────────────
        # The normal velocity depends on the orientation of the interface.
        if self.side == "west":
            # West face: normal velocity is u (positive x-direction).
            normal_vel: np.ndarray = u_arr[0, :]
        elif self.side == "east":
            # East face: normal velocity is -u (positive x is outward).
            normal_vel = -u_arr[-1, :]
        elif self.side == "south":
            # South face: normal velocity is v (positive y-direction).
            normal_vel = v_arr[:, 0]
        elif self.side == "north":
            # North face: normal velocity is -v (positive y is outward).
            normal_vel = -v_arr[:, -1]
        else:
            raise ValueError(
                f"Invalid side '{self.side}'. "
                f"Kindly select 'west', 'east', 'south', or 'north'."
            )

        # ── Compute the pressure drop using the Darcy-Forchheimer law ──
        delta_p: np.ndarray = self.porous_model.pressure_drop(normal_vel)

        # ── Apply the pressure discontinuity ───────────────────────────
        # The pressure drop is applied as a sudden change across the
        # interface. The downstream pressure is set to the upstream
        # pressure plus the computed drop.
        # Please note that delta_p is negative for forward flow (pressure
        # decreases), so adding delta_p to the upstream pressure gives
        # the downstream pressure.
        if self.side == "west":
            # Upstream is interior (i=1), downstream is boundary (i=0).
            p_arr[0, :] = p_arr[1, :] + delta_p
        elif self.side == "east":
            # Upstream is interior (i=-2), downstream is boundary (i=-1).
            p_arr[-1, :] = p_arr[-2, :] + delta_p
        elif self.side == "south":
            # Upstream is interior (j=1), downstream is boundary (j=0).
            p_arr[:, 0] = p_arr[:, 1] + delta_p
        elif self.side == "north":
            # Upstream is interior (j=-2), downstream is boundary (j=-1).
            p_arr[:, -1] = p_arr[:, -2] + delta_p

        return u_arr, v_arr, p_arr

    def compute_pressure_loss_coefficient(
        self,
        normal_velocity: float = 1.0,
    ) -> float:
        """Computes the pressure loss coefficient (K-factor) for the
        porous interface.

        The pressure loss coefficient K is defined as:
            K = Δp / (0.5 · ρ · U²)

        where U is a reference velocity. This non-dimensional coefficient
        is useful for comparing the resistance of different porous
        interfaces independent of the flow velocity.

        Args:
            normal_velocity: The reference normal velocity [m/s].
                             Defaults to 1.0.

        Returns:
            The pressure loss coefficient K (non-dimensional).
        """
        vel_array: np.ndarray = np.array([normal_velocity])
        delta_p: np.ndarray = self.porous_model.pressure_drop(vel_array)
        dynamic_pressure: float = 0.5 * self.porous_model.density * normal_velocity ** 2

        if dynamic_pressure <= 0.0:
            return 0.0

        K: float = float(np.abs(delta_p[0]) / dynamic_pressure)
        return K
