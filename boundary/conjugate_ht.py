#!/usr/bin/env python3
"""
boundary/conjugate_ht.py
==========================

Conjugate heat transfer (CHT) interface treatment for coupled fluid-solid
thermal simulations.

Physical background:
    Conjugate heat transfer refers to the coupled thermal interaction
    between a fluid flow and a solid body. At the fluid-solid interface,
    two fundamental physical conditions must be satisfied simultaneously:

    1. Temperature continuity:
           T_fluid = T_solid   (at the interface)

    2. Heat flux continuity:
           q_fluid = q_solid   (at the interface)
       or equivalently:
           -k_fluid · (∂T/∂n)_fluid = -k_solid · (∂T/∂n)_solid

    where k_fluid and k_solid are the thermal conductivities of the fluid
    and solid, respectively, and n is the unit normal vector pointing from
    the fluid into the solid.

    In a partitioned approach, the fluid and solid thermal solvers run
    separately and exchange boundary conditions at the interface:
        - The fluid solver provides the heat flux to the solid solver
          (Neumann boundary condition for the solid).
        - The solid solver provides the temperature to the fluid solver
          (Dirichlet boundary condition for the fluid).

    This module implements:
        - FluidSolidInterface:  Representation of the fluid-solid interface
                                with thermal property management.
        - ConjugateHeatTransfer: High-level class managing the coupled
                                thermal solution at the interface.

References:
    - Versteeg, H. K. & Malalasekera, W. (2007). An Introduction to
      Computational Fluid Dynamics: The Finite Volume Method, 2nd ed.
      Pearson Education. Chapter 9: Conjugate Heat Transfer.
    - Patankar, S. V. (1980). Numerical Heat Transfer and Fluid Flow.
      Hemisphere Publishing Corporation.
    - Divo, E. & Kassab, A. J. (2007). An efficient method for conjugate
      heat transfer problems. International Journal of Heat and Mass
      Transfer, 50(23-24):4561-4574.

Author:     Ananya Patel
            Indian Institute of Technology Bombay
Date:       2026-07-27
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from navier_stokes import Field2D, Mesh2D


# ─── Fluid-Solid Interface ─────────────────────────────────────────────

@dataclass
class FluidSolidInterface:
    """Represents the fluid-solid interface for conjugate heat transfer
    calculations.

    This class stores the geometric and thermal properties of the interface
    and provides methods to compute the coupled temperature and heat flux
    boundary conditions.

    Physical background:
        At a conjugate heat transfer interface, the temperature field is
        continuous, but the temperature gradient is discontinuous due to
        the difference in thermal conductivities between the fluid and the
        solid. The interface temperature T_int can be computed by enforcing
        heat flux continuity:

            k_fluid · (T_int - T_fluid) / Δn_fluid =
                k_solid · (T_solid - T_int) / Δn_solid

        Solving for T_int yields the "coupling" or "interface" temperature:

            T_int = (k_f · T_f / Δn_f + k_s · T_s / Δn_s)
                  / (k_f / Δn_f + k_s / Δn_s)

        where T_f and T_s are the temperatures at the first cell centres
        on the fluid and solid sides, respectively, and Δn_f and Δn_s are
        the distances from the cell centres to the interface.

    Please note that this formulation implicitly assumes that the
    temperature profile between the cell centre and the interface is
    linear. For high-aspect-ratio grids near the interface, this
    assumption is generally accurate.

    Args:
        k_fluid:          Thermal conductivity of the fluid [W/(m·K)].
                          Defaults to 0.026 (air at room temperature).
        k_solid:          Thermal conductivity of the solid [W/(m·K)].
                          Defaults to 15.0 (stainless steel).
        delta_n_fluid:    Distance from the fluid cell centre to the
                          interface [m]. If not provided, it is computed
                          from the mesh. Defaults to None.
        delta_n_solid:    Distance from the solid cell centre to the
                          interface [m]. If not provided, assumed equal
                          to delta_n_fluid. Defaults to None.
    """
    k_fluid: float = 0.026
    k_solid: float = 15.0
    delta_n_fluid: Optional[float] = None
    delta_n_solid: Optional[float] = None

    def __post_init__(self) -> None:
        """Validates that the thermal conductivities are positive.

        Raises:
            ValueError: If either thermal conductivity is non-positive.
        """
        # It is crucial to understand that negative or zero thermal
        # conductivity is unphysical and will lead to numerical instability
        # or incorrect results.
        if self.k_fluid <= 0.0:
            raise ValueError(
                f"Fluid thermal conductivity must be positive. "
                f"Received k_fluid = {self.k_fluid}."
            )
        if self.k_solid <= 0.0:
            raise ValueError(
                f"Solid thermal conductivity must be positive. "
                f"Received k_solid = {self.k_solid}."
            )

    def compute_interface_temperature(
        self,
        temperature_fluid: np.ndarray,
        temperature_solid: np.ndarray,
        mesh: Mesh2D,
    ) -> np.ndarray:
        """Computes the temperature at the fluid-solid interface by
        enforcing heat flux continuity.

        The interface temperature is computed as a weighted average of
        the fluid and solid temperatures, where the weights are the
        thermal conductivities divided by the distances to the interface.

        Args:
            temperature_fluid:  Temperature at the first fluid cell centre
                                adjacent to the interface (n_points,).
            temperature_solid:  Temperature at the first solid cell centre
                                adjacent to the interface (n_points,).
            mesh:               The fluid mesh (used for the cell-to-interface
                                distance if delta_n_fluid is not provided).

        Returns:
            The interface temperature array (n_points,).

        Raises:
            ValueError: If the input arrays have different lengths.
        """
        # ── Validate input array dimensions ────────────────────────────
        if len(temperature_fluid) != len(temperature_solid):
            raise ValueError(
                "The fluid and solid temperature arrays must have the "
                "same length. "
                f"Got {len(temperature_fluid)} and {len(temperature_solid)}."
            )

        # ── Determine the cell-to-interface distances ──────────────────
        # If not explicitly provided, use the mesh cell size as an estimate.
        delta_f: float
        if self.delta_n_fluid is not None:
            delta_f = self.delta_n_fluid
        else:
            # Assume the interface is at the west face, so the distance
            # from the first cell centre to the face is dx/2.
            delta_f = mesh.dx / 2.0

        delta_s: float
        if self.delta_n_solid is not None:
            delta_s = self.delta_n_solid
        else:
            # If not provided, use the same distance as the fluid side.
            delta_s = delta_f

        # ── Compute the weighted interface temperature ─────────────────
        # T_int = (k_f/Δn_f · T_f + k_s/Δn_s · T_s) / (k_f/Δn_f + k_s/Δn_s)
        # This is derived from the heat flux continuity condition.
        w_fluid: float = self.k_fluid / delta_f
        w_solid: float = self.k_solid / delta_s

        denominator: float = w_fluid + w_solid
        if denominator <= 0.0:
            raise ValueError(
                "The sum of the weighting factors is non-positive. "
                "Kindly check the thermal conductivities and cell-to-"
                "interface distances."
            )

        interface_temperature: np.ndarray = (
            (w_fluid * temperature_fluid + w_solid * temperature_solid)
            / denominator
        )

        return interface_temperature

    def compute_interface_heat_flux(
        self,
        temperature_fluid: np.ndarray,
        interface_temperature: np.ndarray,
        mesh: Mesh2D,
    ) -> np.ndarray:
        """Computes the heat flux across the fluid-solid interface.

        The heat flux is computed from the fluid side using Fourier's law:
            q = -k_fluid · (T_int - T_fluid) / Δn_fluid

        The sign convention is: positive q means heat flows from the fluid
        to the solid.

        Please note that the same heat flux value is obtained from the
        solid side due to the flux continuity condition (up to the
        discretisation error).

        Args:
            temperature_fluid:     Temperature at the fluid cell centre
                                   adjacent to the interface (n_points,).
            interface_temperature: Temperature at the interface (n_points,),
                                   computed by `compute_interface_temperature`.
            mesh:                  The fluid mesh (for the cell-to-interface
                                   distance if delta_n_fluid is not provided).

        Returns:
            Heat flux array (n_points,) in [W/m²]. Positive values
            indicate heat flow from fluid to solid.
        """
        # Determine the cell-to-interface distance.
        delta_f: float
        if self.delta_n_fluid is not None:
            delta_f = self.delta_n_fluid
        else:
            delta_f = mesh.dx / 2.0

        # Fourier's law: q = -k_f · (T_int - T_f) / Δn_f
        # Please note the sign: the temperature gradient is computed as
        # (T_int - T_f) / Δn_f, and the negative sign in Fourier's law
        # gives the heat flux direction.
        heat_flux: np.ndarray = (
            -self.k_fluid * (interface_temperature - temperature_fluid) / delta_f
        )

        return heat_flux

    def compute_interface_heat_transfer_coefficient(
        self,
        mesh: Mesh2D,
    ) -> float:
        """Computes the effective heat transfer coefficient at the
        fluid-solid interface.

        The effective heat transfer coefficient h_eff is defined as:
            h_eff = 1 / (Δn_f / k_f + Δn_s / k_s)

        This represents the inverse of the total thermal resistance
        between the fluid cell centre and the solid cell centre.

        Args:
            mesh: The fluid mesh (for the cell-to-interface distance).

        Returns:
            The effective heat transfer coefficient [W/(m²·K)].
        """
        delta_f: float = self.delta_n_fluid if self.delta_n_fluid is not None else mesh.dx / 2.0
        delta_s: float = self.delta_n_solid if self.delta_n_solid is not None else delta_f

        # Total thermal resistance: R_total = Δn_f / k_f + Δn_s / k_s
        # Effective heat transfer coefficient: h_eff = 1 / R_total
        r_total: float = delta_f / self.k_fluid + delta_s / self.k_solid
        h_eff: float = 1.0 / r_total

        return h_eff


# ─── Conjugate Heat Transfer Manager ───────────────────────────────────

@dataclass
class ConjugateHeatTransfer:
    """High-level manager for conjugate heat transfer coupling between
    a fluid thermal solver and a solid thermal solver.

    This class orchestrates the coupled thermal boundary condition update
    at the fluid-solid interface. It manages the iterative coupling loop
    in which the fluid and solid solvers exchange temperature and heat
    flux information until convergence.

    The coupling algorithm is:
        1. Start with an initial guess for the interface temperature
           (or use the converged value from the previous time step).
        2. Apply T_int as the Dirichlet boundary condition for the fluid
           solver and solve for the fluid temperature field.
        3. Compute the heat flux q_int from the fluid solution.
        4. Apply q_int as the Neumann boundary condition for the solid
           solver and solve for the solid temperature field.
        5. Compute the new interface temperature from the updated solid
           solution.
        6. Check convergence of the interface temperature.
        7. If not converged, return to step 2.

    It is crucial to understand that for strongly coupled conjugate heat
    transfer problems (e.g., when k_f and k_s are of similar magnitude),
    the explicit coupling may converge slowly, and under-relaxation is
    recommended.

    Args:
        interface:          The FluidSolidInterface object containing the
                            interface properties.
        mesh_fluid:         The fluid mesh (Mesh2D).
        relaxation_factor:  Under-relaxation factor for the interface
                            temperature update (0.0 to 1.0). A value of
                            1.0 means no relaxation. Defaults to 0.8.
        max_iter:           Maximum number of coupling iterations per
                            time step. Defaults to 20.
        convergence_tol:    Convergence tolerance for the interface
                            temperature. Defaults to 1e-6.
    """
    interface: FluidSolidInterface = field(default_factory=FluidSolidInterface)
    mesh_fluid: Optional[Mesh2D] = None
    relaxation_factor: float = 0.8
    max_iter: int = 20
    convergence_tol: float = 1e-6

    # Internal state
    _interface_temperature: Optional[np.ndarray] = field(default=None, repr=False)
    _interface_heat_flux: Optional[np.ndarray] = field(default=None, repr=False)
    _convergence_history: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Validates the coupling parameters.

        Raises:
            ValueError: If the relaxation factor is outside (0, 1].
        """
        if self.relaxation_factor <= 0.0 or self.relaxation_factor > 1.0:
            raise ValueError(
                "The relaxation factor must be in the range (0, 1]. "
                f"Received {self.relaxation_factor}."
            )

    def couple(
        self,
        temperature_fluid: np.ndarray,
        temperature_solid: np.ndarray,
        fluid_solve: Optional[callable] = None,
        solid_solve: Optional[callable] = None,
    ) -> dict:
        """Performs one coupling step for conjugate heat transfer.

        This method iteratively updates the interface temperature and
        heat flux until convergence. If fluid_solve and solid_solve
        callbacks are provided, they are called within the coupling loop
        to update the respective temperature fields.

        Args:
            temperature_fluid:  Current fluid temperature at the first
                                cell centre adjacent to the interface
                                (n_points,).
            temperature_solid:  Current solid temperature at the first
                                cell centre adjacent to the interface
                                (n_points,).
            fluid_solve:        Optional callable that solves the fluid
                                thermal problem with the new interface
                                temperature (Dirichlet BC) and returns
                                the updated fluid temperature at the
                                interface-adjacent cells.
            solid_solve:        Optional callable that solves the solid
                                thermal problem with the new interface
                                heat flux (Neumann BC) and returns the
                                updated solid temperature at the
                                interface-adjacent cells.

        Returns:
            A dictionary containing:
                {
                    'interface_temperature':  Final interface temperature
                                              (n_points,),
                    'interface_heat_flux':    Final heat flux (n_points,),
                    'n_iterations':            Number of coupling iterations,
                    'converged':               Whether the coupling converged,
                }
        """
        n_points: int = len(temperature_fluid)

        # ── Initialise the interface temperature ───────────────────────
        # If not previously set, use the harmonic average as the initial
        # guess.
        if self._interface_temperature is None:
            self._interface_temperature = self.interface.compute_interface_temperature(
                temperature_fluid, temperature_solid, self.mesh_fluid
            )
        else:
            # Ensure the array has the correct length.
            if len(self._interface_temperature) != n_points:
                self._interface_temperature = self.interface.compute_interface_temperature(
                    temperature_fluid, temperature_solid, self.mesh_fluid
                )

        # ── Initialise the heat flux ───────────────────────────────────
        self._interface_heat_flux = self.interface.compute_interface_heat_flux(
            temperature_fluid, self._interface_temperature, self.mesh_fluid
        )

        self._convergence_history = []

        # ── Coupling iteration loop ─────────────────────────────────────
        for coupling_iter in range(self.max_iter):
            # ── Step 1: Solve the fluid thermal problem ────────────────
            # The interface temperature is imposed as a Dirichlet boundary
            # condition for the fluid.
            if fluid_solve is not None:
                temperature_fluid = fluid_solve(self._interface_temperature)

            # ── Step 2: Compute the interface heat flux ────────────────
            # The heat flux is computed from the fluid side using Fourier's
            # law.
            new_heat_flux: np.ndarray = self.interface.compute_interface_heat_flux(
                temperature_fluid, self._interface_temperature, self.mesh_fluid
            )

            # ── Step 3: Solve the solid thermal problem ────────────────
            # The interface heat flux is imposed as a Neumann boundary
            # condition for the solid.
            if solid_solve is not None:
                temperature_solid = solid_solve(new_heat_flux)

            # ── Step 4: Compute the new interface temperature ──────────
            # From the updated solid temperature and the heat flux
            # continuity condition.
            new_interface_temperature: np.ndarray = self.interface.compute_interface_temperature(
                temperature_fluid, temperature_solid, self.mesh_fluid
            )

            # ── Step 5: Apply under-relaxation ─────────────────────────
            # T_int^{k+1} = T_int^k + ω · (T_int_new - T_int^k)
            if self.relaxation_factor < 1.0:
                new_interface_temperature = (
                    self._interface_temperature
                    + self.relaxation_factor * (new_interface_temperature - self._interface_temperature)
                )

            # ── Step 6: Check convergence ──────────────────────────────
            delta: np.ndarray = new_interface_temperature - self._interface_temperature
            convergence: float = float(np.sqrt(np.mean(delta ** 2)))
            self._convergence_history.append(convergence)

            # Update the interface state.
            self._interface_temperature = new_interface_temperature
            self._interface_heat_flux = new_heat_flux

            # Check if convergence is achieved.
            if convergence < self.convergence_tol:
                break

        # ── Prepare the result dictionary ──────────────────────────────
        return {
            "interface_temperature": self._interface_temperature,
            "interface_heat_flux": self._interface_heat_flux,
            "n_iterations": coupling_iter + 1,
            "converged": convergence < self.convergence_tol,
        }

    @property
    def convergence_history(self) -> list[float]:
        """Returns the convergence history of the conjugate heat transfer
        coupling iterations.

        Returns:
            A list of residual values from each coupling iteration.
        """
        return self._convergence_history

    def update_temperature(
        self,
        temperature_fluid: np.ndarray,
        temperature_solid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Performs a single explicit update of the interface temperature
        and heat flux without the iterative coupling loop.

        This is useful for loosely coupled simulations where the interface
        condition is updated once per time step without inner iterations.

        Please note that this method does not call any fluid or solid
        solvers; it only updates the interface condition based on the
        current temperature fields.

        Args:
            temperature_fluid:  Fluid temperature at the interface-adjacent
                                cells (n_points,).
            temperature_solid:  Solid temperature at the interface-adjacent
                                cells (n_points,).

        Returns:
            A tuple (interface_temperature, interface_heat_flux).
        """
        self._interface_temperature = self.interface.compute_interface_temperature(
            temperature_fluid, temperature_solid, self.mesh_fluid
        )
        self._interface_heat_flux = self.interface.compute_interface_heat_flux(
            temperature_fluid, self._interface_temperature, self.mesh_fluid
        )

        return self._interface_temperature, self._interface_heat_flux
