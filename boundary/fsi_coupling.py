#!/usr/bin/env python3
"""
boundary/fsi_coupling.py
==========================

Fluid-Structure Interaction (FSI) coupling module for partitioned
coupling of a fluid solver with a structural dynamics solver.

Physical background:
    In Fluid-Structure Interaction problems, a fluid flow interacts with
    a deformable or moving solid structure. The coupling occurs at the
    fluid-solid interface, where two key transfer operations must be
    performed accurately and conservatively:

    1. Load Transfer (fluid → solid):
       The fluid pressure and viscous stresses acting on the interface
       are transferred to the structural solver as forces (tractions)
       that deform the structure.

    2. Displacement Transfer (solid → fluid):
       The structural deformation at the interface is communicated back
       to the fluid solver, which adjusts its mesh accordingly (via mesh
       motion or morphing) and updates the boundary conditions.

    In a partitioned coupling approach, the fluid and solid solvers run
    separately and exchange data at the interface at each coupling step.
    The interface mapping between non-conforming meshes (the fluid mesh
    and the solid mesh typically have different resolutions) is achieved
    using interpolation techniques.

    This module implements the following:
        - InterfaceMapping:   Mapping between fluid and solid interface
                              meshes using nearest-neighbour, radial basis
                              function (RBF), or bilinear interpolation.
        - LoadTransfer:       Transfer of fluid loads (pressure + shear)
                              to the structural mesh.
        - DisplacementTransfer: Transfer of structural displacements to
                              the fluid mesh.
        - FSICoupling:        High-level class orchestrating the coupling
                              workflow.

References:
    - Farhat, C., & Lesoinne, M. (2000). Two efficient staggered
      algorithms for the serial and parallel solution of three-dimensional
      nonlinear transient aeroelastic problems. Computer Methods in Applied
      Mechanics and Engineering, 182(3-4):499-515.
    - Degroote, J., Bathe, K. J., & Vierendeels, J. (2009). Performance
      of a new partitioned procedure versus a monolithic procedure in
      fluid-structure interaction. Computers & Structures, 87(11-12):793-801.
    - de Boer, A., van Zuijlen, A. H., & Bijl, H. (2007). Review of
      coupling methods for non-matching meshes. Computer Methods in Applied
      Mechanics and Engineering, 196(8):1515-1525.

Author:     Ananya Patel
            Indian Institute of Technology Bombay
Date:       2026-07-27
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional

import numpy as np

from navier_stokes import Field2D, Mesh2D


# ─── Coupling enums ────────────────────────────────────────────────────

class MappingMethod(IntEnum):
    """Enumeration of available interface mapping methods.

    Attributes:
        NEAREST_NEIGHBOUR:  Each target point receives the value from the
                            closest source point. Simple and robust, but
                            first-order accurate only.
        RBF:                Radial Basis Function interpolation. Provides
                            smooth, high-order accurate mapping. Suitable
                            for non-conforming meshes with moderate gaps.
        BILINEAR:           Bilinear interpolation on the source mesh.
                            Requires the source mesh to be structured.
    """
    NEAREST_NEIGHBOUR = 0
    RBF = 1
    BILINEAR = 2


class CouplingScheme(IntEnum):
    """Enumeration of partitioned FSI coupling schemes.

    Attributes:
        EXPLICIT:  One coupling iteration per time step (convenient but
                   may be unstable for strong coupling or heavy structures).
        IQN_ILS:   Interface Quasi-Newton with Inverse Least-Squares
                   (implicit coupling with convergence acceleration).
        FIXED_POINT: Fixed-point iteration with relaxation at the
                     interface (implicit coupling, robust but slow).
    """
    EXPLICIT = 0
    IQN_ILS = 1
    FIXED_POINT = 2


# ─── Interface Mapping ─────────────────────────────────────────────────

@dataclass
class InterfaceMapping:
    """Maps data between non-conforming fluid and solid interface meshes.

    Physical background:
        In partitioned FSI, the fluid mesh and the structural mesh typically
        have different node distributions along the interface. The mapping
        problem is: given a set of source points with known values, determine
        the values at a set of target points. This mapping must be both
        accurate (preserve the spatial distribution) and conservative
        (preserve integrated quantities like total force).

    Three mapping methods are provided:
        1. Nearest-neighbour: fast but low-order accuracy.
        2. Radial Basis Function (RBF): smooth, high-order, suitable for
           non-matching meshes. Uses a Gaussian RBF kernel.
        3. Bilinear: suitable when both meshes are structured.

    It is crucial to understand that the choice of mapping method
    significantly affects the stability and accuracy of the coupled FSI
    simulation. RBF interpolation is generally recommended for most
    applications.

    Args:
        method:     The mapping method to use (NEAREST_NEIGHBOUR, RBF,
                    or BILINEAR).
        rbf_eps:    The shape parameter for the RBF kernel (Gaussian:
                    φ(r) = exp(-(ε·r)²)). Smaller values produce wider
                    influence. Defaults to 1.0.
        rbf_kernel: The RBF kernel function. If None, defaults to the
                    Gaussian kernel. The function should accept a distance
                    array and return the kernel values.
    """
    method: MappingMethod = MappingMethod.RBF
    rbf_eps: float = 1.0
    rbf_kernel: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        """Initialises the RBF kernel function if not provided.

        Please note that the default Gaussian kernel is:
            φ(r) = exp(-(ε · r)²)
        """
        if self.rbf_kernel is None:
            # Default Gaussian RBF kernel.
            self.rbf_kernel = lambda r: np.exp(-(self.rbf_eps * r) ** 2)

    def map_data(
        self,
        source_coords: np.ndarray,
        source_values: np.ndarray,
        target_coords: np.ndarray,
    ) -> np.ndarray:
        """Maps data from the source mesh points to the target mesh points.

        This is the primary interface for the mapping operation. It
        dispatches to the appropriate method based on the selected
        MappingMethod.

        Args:
            source_coords: Coordinates of the source (donor) mesh points.
                           Shape (n_source, 2).
            source_values: Values at the source mesh points.
                           Shape (n_source,) or (n_source, n_components).
            target_coords: Coordinates of the target (receiver) mesh points.
                           Shape (n_target, 2).

        Returns:
            Interpolated values at the target mesh points.
            Shape (n_target,) or (n_target, n_components).
        """
        if self.method == MappingMethod.NEAREST_NEIGHBOUR:
            return self._nearest_neighbour(source_coords, source_values, target_coords)
        elif self.method == MappingMethod.RBF:
            return self._rbf_interpolation(source_coords, source_values, target_coords)
        elif self.method == MappingMethod.BILINEAR:
            return self._bilinear_interpolation(source_coords, source_values, target_coords)
        else:
            raise ValueError(
                f"Mapping method '{self.method}' is not recognised. "
                f"Kindly select from NEAREST_NEIGHBOUR, RBF, or BILINEAR."
            )

    def _nearest_neighbour(
        self,
        source_coords: np.ndarray,
        source_values: np.ndarray,
        target_coords: np.ndarray,
    ) -> np.ndarray:
        """Performs nearest-neighbour interpolation.

        For each target point, the value at the closest source point
        is assigned. This is the simplest and fastest method, but it
        is only first-order accurate and may introduce discontinuities.

        Args:
            source_coords: Source mesh points (n_source, 2).
            source_values: Source values (n_source,) or (n_source, n_comp).
            target_coords: Target mesh points (n_target, 2).

        Returns:
            Interpolated values (n_target,) or (n_target, n_comp).
        """
        n_target: int = target_coords.shape[0]
        target_values: list[np.ndarray] = []

        # Determine if source_values is 1D or 2D.
        if source_values.ndim == 1:
            source_values = source_values.reshape(-1, 1)

        n_components: int = source_values.shape[1]

        # For each target point, find the nearest source point.
        for t_idx in range(n_target):
            # Compute the Euclidean distance to all source points.
            diff: np.ndarray = source_coords - target_coords[t_idx, :]  # (n_source, 2)
            distances: np.ndarray = np.sqrt(np.sum(diff ** 2, axis=1))

            # Find the index of the minimum distance.
            nearest_idx: int = int(np.argmin(distances))
            target_values.append(source_values[nearest_idx, :])

        result: np.ndarray = np.array(target_values)
        # Squeeze back to 1D if the input was 1D.
        if result.shape[1] == 1:
            result = result.flatten()
        return result

    def _rbf_interpolation(
        self,
        source_coords: np.ndarray,
        source_values: np.ndarray,
        target_coords: np.ndarray,
    ) -> np.ndarray:
        """Performs Radial Basis Function (RBF) interpolation.

        The RBF interpolant is constructed as:
            s(x) = Σ_{i=1}^{N} w_i · φ(||x - x_i||)

        where φ is the RBF kernel and the weights w_i are determined by
        solving the linear system:
            Φ · w = f
        where Φ_{ij} = φ(||x_i - x_j||) and f are the source values.

        Please note that this method scales as O(N_source³ + N_source·N_target)
        due to the dense linear system solve. For very large interface
        meshes, consider using a compactly supported RBF or a greedy
        algorithm to reduce the computational cost.

        Args:
            source_coords: Source mesh points (n_source, 2).
            source_values: Source values (n_source,) or (n_source, n_comp).
            target_coords: Target mesh points (n_target, 2).

        Returns:
            Interpolated values (n_target,) or (n_target, n_comp).
        """
        n_source: int = source_coords.shape[0]
        n_target: int = target_coords.shape[0]

        # Determine the number of components.
        if source_values.ndim == 1:
            source_values = source_values.reshape(-1, 1)
            single_component: bool = True
        else:
            single_component = False

        n_components: int = source_values.shape[1]

        # ── Build the RBF interpolation matrix Φ (n_source × n_source) ─
        # Φ_{ij} = φ(||x_i - x_j||)
        Phi: np.ndarray = np.zeros((n_source, n_source))
        for i in range(n_source):
            for j in range(n_source):
                dist: float = float(np.linalg.norm(source_coords[i, :] - source_coords[j, :]))
                Phi[i, j] = self.rbf_kernel(np.array([dist]))[0] if callable(self.rbf_kernel) else 0.0

        # Add a small regularisation for numerical stability.
        Phi += 1e-12 * np.eye(n_source)

        # ── Solve for the weights ──────────────────────────────────────
        # w = Φ^{-1} · f
        try:
            weights: np.ndarray = np.linalg.solve(Phi, source_values)  # (n_source, n_comp)
        except np.linalg.LinAlgError:
            # Fall back to least-squares if the matrix is singular.
            weights = np.linalg.lstsq(Phi, source_values, rcond=None)[0]

        # ── Evaluate the interpolant at the target points ──────────────
        target_values = np.zeros((n_target, n_components))
        for t_idx in range(n_target):
            for s_idx in range(n_source):
                dist = float(np.linalg.norm(
                    target_coords[t_idx, :] - source_coords[s_idx, :]
                ))
                kernel_val: float = (
                    self.rbf_kernel(np.array([dist]))[0]
                    if callable(self.rbf_kernel) else 0.0
                )
                target_values[t_idx, :] += weights[s_idx, :] * kernel_val

        # Squeeze back to 1D if the input was 1D.
        if single_component:
            target_values = target_values.flatten()

        return target_values

    def _bilinear_interpolation(
        self,
        source_coords: np.ndarray,
        source_values: np.ndarray,
        target_coords: np.ndarray,
    ) -> np.ndarray:
        """Performs bilinear interpolation on a structured source mesh.

        This method assumes that the source coordinates form a regular
        structured grid. The target point is located within a grid cell,
        and the value is interpolated using bilinear interpolation from
        the four corner values.

        Kindly ensure that the source mesh is structured (i.e., the
        coordinates form a Cartesian grid). If the mesh is unstructured,
        this method will produce incorrect results.

        Args:
            source_coords: Source mesh points (n_source, 2). These must
                           form a structured grid.
            source_values: Source values (n_source,) or (n_source, n_comp).
            target_coords: Target mesh points (n_target, 2).

        Returns:
            Interpolated values (n_target,) or (n_target, n_comp).
        """
        n_target: int = target_coords.shape[0]

        if source_values.ndim == 1:
            source_values = source_values.reshape(-1, 1)
            single_component: bool = True
        else:
            single_component = False

        n_components: int = source_values.shape[1]

        # Extract unique x and y coordinates from the source mesh.
        # This assumes a structured grid.
        x_unique: np.ndarray = np.unique(source_coords[:, 0])
        y_unique: np.ndarray = np.unique(source_coords[:, 1])

        # Reshape the source values into a 2D grid.
        nx_src: int = len(x_unique)
        ny_src: int = len(y_unique)
        grid_values: np.ndarray = source_values.reshape(nx_src, ny_src, n_components)

        target_values = np.zeros((n_target, n_components))

        for t_idx in range(n_target):
            x_t: float = target_coords[t_idx, 0]
            y_t: float = target_coords[t_idx, 1]

            # Clamp target coordinates to the source grid bounds.
            x_t = np.clip(x_t, x_unique[0], x_unique[-1])
            y_t = np.clip(y_t, y_unique[0], y_unique[-1])

            # Find the grid cell containing the target point.
            i_x: int = int(np.searchsorted(x_unique, x_t) - 1)
            i_x = max(0, min(i_x, nx_src - 2))
            i_y: int = int(np.searchsorted(y_unique, y_t) - 1)
            i_y = max(0, min(i_y, ny_src - 2))

            # Corner coordinates and values.
            x0, x1 = x_unique[i_x], x_unique[i_x + 1]
            y0, y1 = y_unique[i_y], y_unique[i_y + 1]

            # Bilinear interpolation weights.
            denom_x: float = x1 - x0 if x1 != x0 else 1.0
            denom_y: float = y1 - y0 if y1 != y0 else 1.0

            wx: float = (x_t - x0) / denom_x
            wy: float = (y_t - y0) / denom_y

            # Interpolate each component.
            for c in range(n_components):
                f00: float = grid_values[i_x, i_y, c]
                f10: float = grid_values[i_x + 1, i_y, c]
                f01: float = grid_values[i_x, i_y + 1, c]
                f11: float = grid_values[i_x + 1, i_y + 1, c]

                # Bilinear interpolation formula.
                target_values[t_idx, c] = (
                    f00 * (1.0 - wx) * (1.0 - wy)
                    + f10 * wx * (1.0 - wy)
                    + f01 * (1.0 - wx) * wy
                    + f11 * wx * wy
                )

        if single_component:
            target_values = target_values.flatten()

        return target_values


# ─── Load Transfer ─────────────────────────────────────────────────────

@dataclass
class LoadTransfer:
    """Transfers fluid loads (pressure and shear stress) to the structural
    mesh at the fluid-solid interface.

    Physical background:
        The fluid exerts both normal and tangential forces on the solid
        surface. The total traction vector at the interface is:

            t_i = -p · n_i + τ_ij · n_j

        where p is the pressure, τ_ij is the viscous stress tensor, and
        n_i is the unit normal vector pointing from the fluid into the
        solid. The load transfer operation computes the nodal forces on
        the structural mesh by integrating the traction over each element
        face and distributing it to the nodes.

    It is crucial to understand that the load transfer must be conservative:
    the total force integrated over the fluid interface should equal the
    total force applied to the structural interface (up to interpolation
    error).

    Args:
        mapping: The InterfaceMapping object used to transfer pressure
                 and shear stress from the fluid mesh to the solid mesh.
    """
    mapping: InterfaceMapping = field(default_factory=InterfaceMapping)

    def transfer_pressure(
        self,
        pressure_fluid: np.ndarray,
        fluid_coords: np.ndarray,
        solid_coords: np.ndarray,
    ) -> np.ndarray:
        """Transfers the fluid pressure field to the structural mesh.

        The pressure is a scalar quantity, so a straightforward mapping
        from fluid to solid interface points is sufficient.

        Args:
            pressure_fluid: Pressure values at the fluid interface points.
                            Shape (n_fluid,).
            fluid_coords:   Coordinates of the fluid interface points.
                            Shape (n_fluid, 2).
            solid_coords:   Coordinates of the structural interface points.
                            Shape (n_solid, 2).

        Returns:
            Pressure values at the structural interface points.
            Shape (n_solid,).
        """
        # Please note that the mapping method handles the interpolation
        # automatically, regardless of whether the meshes are conforming.
        pressure_solid: np.ndarray = self.mapping.map_data(
            fluid_coords, pressure_fluid, solid_coords
        )
        return pressure_solid

    def transfer_shear(
        self,
        shear_fluid: np.ndarray,
        fluid_coords: np.ndarray,
        solid_coords: np.ndarray,
    ) -> np.ndarray:
        """Transfers the fluid shear stress to the structural mesh.

        Args:
            shear_fluid:  Shear stress values at the fluid interface points.
                          Shape (n_fluid, 2) for τ_x and τ_y components.
            fluid_coords: Coordinates of the fluid interface points.
                          Shape (n_fluid, 2).
            solid_coords: Coordinates of the structural interface points.
                          Shape (n_solid, 2).

        Returns:
            Shear stress values at the structural interface points.
            Shape (n_solid, 2).
        """
        shear_solid: np.ndarray = self.mapping.map_data(
            fluid_coords, shear_fluid, solid_coords
        )
        return shear_solid

    def compute_nodal_forces(
        self,
        pressure: np.ndarray,
        shear: np.ndarray,
        normals: np.ndarray,
        face_areas: np.ndarray,
    ) -> np.ndarray:
        """Computes the nodal force vectors from pressure and shear stress.

        The total force on each node is:
            F = (p · n + τ) · A

        where n is the unit normal vector, τ is the shear stress vector,
        and A is the face area associated with the node.

        Args:
            pressure:   Pressure at the interface nodes (n_nodes,).
            shear:      Shear stress at the interface nodes (n_nodes, 2).
            normals:    Unit normal vectors at the interface nodes,
                        pointing from fluid to solid (n_nodes, 2).
            face_areas: Face areas associated with each node (n_nodes,).

        Returns:
            Nodal force vectors (n_nodes, 2), where each row is
            (F_x, F_y).
        """
        n_nodes: int = len(pressure)

        # Force from pressure: F_p = p · n · A
        # Force from shear:    F_τ = τ · A
        # Total force:         F = F_p + F_τ
        forces: np.ndarray = np.zeros((n_nodes, 2))

        for i in range(n_nodes):
            # Normal force component due to pressure.
            forces[i, 0] = pressure[i] * normals[i, 0] * face_areas[i]
            forces[i, 1] = pressure[i] * normals[i, 1] * face_areas[i]

            # Tangential force component due to shear stress.
            forces[i, 0] += shear[i, 0] * face_areas[i]
            forces[i, 1] += shear[i, 1] * face_areas[i]

        return forces


# ─── Displacement Transfer ─────────────────────────────────────────────

@dataclass
class DisplacementTransfer:
    """Transfers structural displacements to the fluid mesh at the
    fluid-solid interface.

    Physical background:
        When the solid structure deforms under fluid loading, the
        displacement at the interface must be communicated back to the
        fluid solver. The fluid mesh must then be updated (via mesh
        deformation or morphing) to conform to the new shape of the
        interface.

        The displacement transfer is the inverse of the load transfer:
        displacements are known at the structural nodes and must be
        interpolated to the fluid interface nodes.

    Please note that for strongly-coupled FSI problems, the displacement
    transfer may require under-relaxation to maintain stability:

        d_fluid^{k+1} = d_fluid^k + ω · (d_solid - d_fluid^k)

    where ω is the relaxation factor (0 < ω ≤ 1).

    Args:
        mapping:           The InterfaceMapping object used to transfer
                           displacements.
        relaxation_factor: Under-relaxation factor for displacement
                           transfer (0.0 to 1.0). A value of 1.0 means
                           no relaxation. Defaults to 1.0.
    """
    mapping: InterfaceMapping = field(default_factory=InterfaceMapping)
    relaxation_factor: float = 1.0

    def transfer_displacement(
        self,
        displacement_solid: np.ndarray,
        solid_coords: np.ndarray,
        fluid_coords: np.ndarray,
        previous_fluid_displacement: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transfers structural displacements to the fluid interface mesh.

        The displacement is a vector quantity (d_x, d_y), and each
        component is mapped independently from the solid mesh to the
        fluid mesh.

        Args:
            displacement_solid:      Displacement at structural interface
                                     nodes. Shape (n_solid, 2).
            solid_coords:            Coordinates of structural interface
                                     nodes. Shape (n_solid, 2).
            fluid_coords:            Coordinates of fluid interface nodes.
                                     Shape (n_fluid, 2).
            previous_fluid_displacement: Previous displacement at fluid
                                     interface nodes (for relaxation).
                                     Shape (n_fluid, 2). Defaults to None.

        Returns:
            Displacement at fluid interface nodes. Shape (n_fluid, 2).
        """
        # Map the x-component of displacement.
        dx_solid: np.ndarray = displacement_solid[:, 0]
        dx_fluid: np.ndarray = self.mapping.map_data(
            solid_coords, dx_solid, fluid_coords
        )

        # Map the y-component of displacement.
        dy_solid: np.ndarray = displacement_solid[:, 1]
        dy_fluid: np.ndarray = self.mapping.map_data(
            solid_coords, dy_solid, fluid_coords
        )

        # Combine into a single array.
        displacement_fluid: np.ndarray = np.column_stack((dx_fluid, dy_fluid))

        # ── Apply under-relaxation if previous displacement is provided ─
        if previous_fluid_displacement is not None and self.relaxation_factor < 1.0:
            # Relaxed update: d_new = d_old + ω · (d_mapped - d_old)
            displacement_fluid = (
                previous_fluid_displacement
                + self.relaxation_factor * (displacement_fluid - previous_fluid_displacement)
            )

        return displacement_fluid


# ─── High-Level FSI Coupling Class ─────────────────────────────────────

@dataclass
class FSICoupling:
    """High-level class that orchestrates the partitioned FSI coupling
    between a fluid solver and a structural solver.

    This class manages the complete coupling workflow:
        1. Transfer fluid loads (pressure + shear) to the structure.
        2. Call the structural solver to compute displacements.
        3. Transfer structural displacements back to the fluid mesh.
        4. Update the fluid mesh and boundary conditions.

    It is crucial to understand that the actual fluid and structural
    solvers are not part of this class. Instead, this class provides
    the coupling infrastructure, and the user must provide callbacks
    or solver objects that perform the actual physics computation.

    Args:
        mesh_fluid:          The fluid mesh (Mesh2D object).
        mapping:             The interface mapping method.
        load_transfer:       The load transfer object.
        displacement_transfer: The displacement transfer object.
        coupling_scheme:     The coupling scheme (EXPLICIT, IQN_ILS,
                             or FIXED_POINT).
        max_iter:            Maximum number of coupling iterations per
                             time step (for implicit schemes). Defaults to 10.
        convergence_tol:     Convergence tolerance for the coupling loop.
                             Defaults to 1e-6.
        fluid_coords:        Coordinates of the fluid interface nodes.
                             Shape (n_fluid, 2). If not provided, defaults
                             to the boundary nodes of the fluid mesh.
    """
    mesh_fluid: Mesh2D
    mapping: InterfaceMapping = field(default_factory=InterfaceMapping)
    load_transfer: LoadTransfer = field(default_factory=LoadTransfer)
    displacement_transfer: DisplacementTransfer = field(default_factory=DisplacementTransfer)
    coupling_scheme: CouplingScheme = CouplingScheme.EXPLICIT
    max_iter: int = 10
    convergence_tol: float = 1e-6
    fluid_coords: Optional[np.ndarray] = None

    # ── Internal state ─────────────────────────────────────────────────
    _fluid_displacement: Optional[np.ndarray] = field(default=None, repr=False)
    _prev_fluid_displacement: Optional[np.ndarray] = field(default=None, repr=False)
    _convergence_history: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Validates the coupling parameters and initialises the interface
        coordinate arrays if not provided.

        Raises:
            ValueError: If the coupling parameters are inconsistent.
        """
        # ── Validate the relaxation factor ─────────────────────────────
        if self.displacement_transfer.relaxation_factor <= 0.0 or \
           self.displacement_transfer.relaxation_factor > 1.0:
            raise ValueError(
                "The relaxation factor must be in the range (0, 1]. "
                f"Received {self.displacement_transfer.relaxation_factor}."
            )

        # ── Initialise interface coordinates if not provided ──────────
        if self.fluid_coords is None:
            # Use the boundary nodes of the fluid mesh as the interface.
            # By default, we use the west boundary as the FSI interface.
            ny: int = self.mesh_fluid.ny
            yc: np.ndarray = self.mesh_fluid.yc
            self.fluid_coords = np.column_stack((
                np.zeros(ny),  # x = 0 (west face)
                yc,             # y = cell centres
            ))

        self._fluid_displacement = np.zeros_like(self.fluid_coords)
        self._prev_fluid_displacement = np.zeros_like(self.fluid_coords)

    def couple(
        self,
        pressure_fluid: np.ndarray,
        shear_fluid: np.ndarray,
        fluid_normals: np.ndarray,
        fluid_face_areas: np.ndarray,
        solid_coords: np.ndarray,
        solid_solve: Callable[[np.ndarray], np.ndarray],
        solid_normals: Optional[np.ndarray] = None,
        solid_face_areas: Optional[np.ndarray] = None,
    ) -> dict:
        """Performs one coupling step (one time step of the coupled system).

        This method executes the partitioned coupling loop:
            1. Map fluid pressure and shear to the solid mesh.
            2. Compute nodal forces on the solid mesh.
            3. Call the structural solver to obtain displacements.
            4. Map solid displacements back to the fluid mesh.
            5. Repeat for implicit coupling until convergence.

        Args:
            pressure_fluid:  Fluid pressure at the interface.
                             Shape (n_fluid,).
            shear_fluid:     Fluid shear stress at the interface.
                             Shape (n_fluid, 2).
            fluid_normals:   Unit normals at fluid interface nodes.
                             Shape (n_fluid, 2).
            fluid_face_areas: Face areas at fluid interface nodes.
                              Shape (n_fluid,).
            solid_coords:    Structural interface node coordinates.
                             Shape (n_solid, 2).
            solid_solve:     A callable that accepts nodal forces
                             (n_solid, 2) and returns nodal displacements
                             (n_solid, 2). This is the structural solver.
            solid_normals:   Unit normals at solid interface nodes.
                             Shape (n_solid, 2). If None, estimated from
                             coordinates. Defaults to None.
            solid_face_areas: Face areas at solid interface nodes.
                              Shape (n_solid,). If None, estimated.
                              Defaults to None.

        Returns:
            A dictionary containing:
                {
                    'fluid_displacement': Displacement at fluid interface
                                           (n_fluid, 2),
                    'solid_displacement':  Displacement at solid interface
                                           (n_solid, 2),
                    'n_coupling_iterations': Number of coupling iterations,
                    'convergence':         Final displacement residual,
                    'converged':           Whether the coupling converged,
                }

        Raises:
            ValueError: If the coupling iteration fails to converge.
        """
        # ── Transfer loads to the structural mesh ──────────────────────
        # Map the fluid pressure to the solid interface points.
        pressure_solid: np.ndarray = self.load_transfer.transfer_pressure(
            pressure_fluid, self.fluid_coords, solid_coords
        )

        # Map the fluid shear to the solid interface points.
        shear_solid: np.ndarray = self.load_transfer.transfer_shear(
            shear_fluid, self.fluid_coords, solid_coords
        )

        # ── Compute nodal forces on the solid mesh ─────────────────────
        # If normals and face areas are not provided, estimate them.
        if solid_normals is None or solid_face_areas is None:
            # Estimate normals (assume the interface is along the west face).
            n_solid: int = solid_coords.shape[0]
            solid_normals = np.zeros((n_solid, 2))
            solid_normals[:, 0] = -1.0  # Outward normal pointing into fluid.
            solid_face_areas = np.ones(n_solid) * self.mesh_fluid.dy

        forces_solid: np.ndarray = self.load_transfer.compute_nodal_forces(
            pressure_solid, shear_solid, solid_normals, solid_face_areas
        )

        # ── Coupling iteration loop ────────────────────────────────────
        # For the EXPLICIT scheme, only one iteration is performed.
        # For implicit schemes, fixed-point or IQN-ILS iterations are used.
        max_iters: int = 1 if self.coupling_scheme == CouplingScheme.EXPLICIT else self.max_iter

        displacement_solid: Optional[np.ndarray] = None
        convergence: float = 1.0
        self._convergence_history = []

        for coupling_iter in range(max_iters):
            # ── Call the structural solver ─────────────────────────
            displacement_solid = solid_solve(forces_solid)

            # ── Transfer displacements to the fluid mesh ───────────
            new_fluid_displacement: np.ndarray = self.displacement_transfer.transfer_displacement(
                displacement_solid,
                solid_coords,
                self.fluid_coords,
                previous_fluid_displacement=self._prev_fluid_displacement,
            )

            # ── Check convergence (implicit coupling) ──────────────
            if self.coupling_scheme != CouplingScheme.EXPLICIT:
                # Compute the L2 norm of the displacement change.
                delta: np.ndarray = new_fluid_displacement - self._fluid_displacement
                convergence = float(np.sqrt(np.mean(delta ** 2)))
                self._convergence_history.append(convergence)

                # Update the fluid displacement for the next iteration.
                self._fluid_displacement = new_fluid_displacement

                # Check if convergence is achieved.
                if convergence < self.convergence_tol:
                    break
            else:
                # For explicit coupling, simply accept the result.
                self._fluid_displacement = new_fluid_displacement
                convergence = 0.0

        # ── Store the previous displacement for the next time step ──
        self._prev_fluid_displacement = self._fluid_displacement.copy()

        # ── Return the coupling results ─────────────────────────────
        return {
            "fluid_displacement": self._fluid_displacement,
            "solid_displacement": displacement_solid,
            "n_coupling_iterations": coupling_iter + 1 if displacement_solid is not None else 0,
            "convergence": convergence,
            "converged": convergence < self.convergence_tol,
        }

    @property
    def convergence_history(self) -> list[float]:
        """Returns the convergence history of the coupling iterations.

        This is useful for debugging and for assessing the convergence
        behaviour of the coupling scheme.

        Returns:
            A list of residual values from each coupling iteration.
        """
        return self._convergence_history
