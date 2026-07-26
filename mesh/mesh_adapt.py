#!/usr/bin/env python3
"""
cfd_solver/mesh/mesh_adapt.py
===============================

Adaptive mesh refinement (AMR) for 2-D structured grids.

Two refinement indicators are provided:

1. **Gradient-based** — cell-wise L² norm of the solution gradient,
   normalised by the maximum over the domain.  Cells with large
   gradients are subdivided to capture flow features such as shear
   layers and boundary layers (Löhner, 1987).

2. **Curvature-based** — second-derivative magnitude normalised by
   the first derivative, which detects regions of high solution
   curvature (e.g. shock waves, separation points) irrespective of
   the absolute gradient magnitude (Babuska & Rheinboldt, 1978).

The refinement operates by bisecting cells in the direction of the
largest gradient — an h-refinement strategy suitable for structured
quadrilateral grids.

References
----------
- Löhner, R. (1987). An adaptive finite element scheme for transient
  problems in CFD. *Computer Methods in Applied Mechanics and
  Engineering*, 61(3), 323–338.
- Babuška, I., & Rheinboldt, W. C. (1978). Error estimates for
  adaptive finite element computations. *SIAM Journal on Numerical
  Analysis*, 15(4), 736–754.
- Thompson, J. F., Warsi, Z. U. A., & Mastin, C. W. (1985).
  *Numerical Grid Generation: Foundations and Applications*.
  North-Holland.

Author: James Hargreaves
        Department of Aeronautics, Imperial College London
Branch: dev/mesh-james
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional

import numpy as np

from .mesh_2d import Mesh2D


# ─── Refinement Direction ─────────────────────────────────────────────

class RefineDirection(IntEnum):
    """Direction in which to bisect a cell."""
    X = 0       # Split along x-direction (add an i-line)
    Y = 1       # Split along y-direction (add a j-line)
    BOTH = 2    # Split in both directions (add both i- and j-lines)


# ─── Refinement Indicator ─────────────────────────────────────────────

@dataclass
class RefinementMap:
    """
    Per-cell refinement decision produced by an indicator.

    Attributes
    ----------
    markers : np.ndarray
        Boolean array, shape ``(nx, ny)`` — ``True`` where the cell
        should be refined.
    directions : np.ndarray
        Integer array of ``RefineDirection``, shape ``(nx, ny)``.
    indicator : np.ndarray
        Raw indicator field before thresholding, shape ``(nx, ny)``.
    threshold : float
        Threshold value used for marking.
    """
    markers: np.ndarray
    directions: np.ndarray
    indicator: np.ndarray
    threshold: float


# ─── Gradient-Based Indicator ─────────────────────────────────────────

def gradient_indicator(
    field: np.ndarray,
    threshold: float = 0.2,
    normalise: bool = True,
) -> RefinementMap:
    """
    Mark cells for refinement based on the normalised solution gradient.

    The indicator is the cell-wise L² norm of the gradient:

        ε_i = ‖∇φ‖_i / max(‖∇φ‖)

    Cells where ε_i > threshold are marked for refinement in the
    direction of the larger gradient component.

    Parameters
    ----------
    field : np.ndarray
        Solution field on the current mesh, shape ``(nx, ny)``.
    threshold : float
        Refinement threshold (default: 0.2).  Lower values produce
        more refined cells.
    normalise : bool
        If ``True`` (default), normalise by the maximum gradient.

    Returns
    -------
    RefinementMap
    """
    nx, ny = field.shape
    # Central-difference gradients (interior only, padded at boundaries)
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)

    grad_x[1:-1, :] = 0.5 * (field[2:, :] - field[:-2, :])
    grad_y[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])

    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Normalise
    if normalise:
        g_max = np.max(grad_mag)
        indicator = grad_mag / max(g_max, 1e-30)
    else:
        indicator = grad_mag

    # Mark
    markers = indicator > threshold

    # Direction: refine along the axis with the larger gradient
    directions = np.full((nx, ny), RefineDirection.X, dtype=int)
    abs_gx = np.abs(grad_x)
    abs_gy = np.abs(grad_y)
    both = (abs_gx > threshold * 0.5) & (abs_gy > threshold * 0.5)

    directions[abs_gy > abs_gx] = RefineDirection.Y
    directions[both] = RefineDirection.BOTH

    return RefinementMap(
        markers=markers,
        directions=directions,
        indicator=indicator,
        threshold=threshold,
    )


# ─── Curvature-Based Indicator ────────────────────────────────────────

def curvature_indicator(
    field: np.ndarray,
    threshold: float = 0.3,
    normalise: bool = True,
) -> RefinementMap:
    """
    Mark cells for refinement based on the normalised curvature.

    The indicator measures the second-derivative magnitude relative
    to the first derivative:

        ε_i = |∇²φ| / (|∇φ| + ε₀)

    where ε₀ is a small constant to avoid division by zero.  This
    metric is particularly effective at detecting flow features with
    high curvature (shocks, separation points) while ignoring large
    but linear gradients.

    Parameters
    ----------
    field : np.ndarray
        Solution field on the current mesh, shape ``(nx, ny)``.
    threshold : float
        Refinement threshold (default: 0.3).
    normalise : bool
        If ``True`` (default), normalise by the maximum over the domain.

    Returns
    -------
    RefinementMap
    """
    nx, ny = field.shape

    # First derivatives (central differences, interior)
    grad_x = np.zeros_like(field)
    grad_y = np.zeros_like(field)
    grad_x[1:-1, :] = 0.5 * (field[2:, :] - field[:-2, :])
    grad_y[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])

    grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-30

    # Second derivatives (central)
    grad_xx = np.zeros_like(field)
    grad_yy = np.zeros_like(field)
    grad_xy = np.zeros_like(field)

    grad_xx[1:-1, :] = field[2:, :] - 2.0 * field[1:-1, :] + field[:-2, :]
    grad_yy[:, 1:-1] = field[:, 2:] - 2.0 * field[:, 1:-1] + field[:, :-2]
    grad_xy[1:-1, 1:-1] = 0.25 * (
        field[2:, 2:] - field[2:, :-2] - field[:-2, 2:] + field[:-2, :-2]
    )

    # Curvature magnitude (Frobenius norm of Hessian)
    curv_mag = np.sqrt(grad_xx**2 + grad_yy**2 + 2.0 * grad_xy**2)

    indicator = curv_mag / grad_mag

    # Normalise
    if normalise:
        c_max = np.max(indicator)
        indicator = indicator / max(c_max, 1e-30)

    # Mark
    markers = indicator > threshold

    # Direction: refine along axis of larger second derivative
    directions = np.full((nx, ny), RefineDirection.X, dtype=int)
    abs_xx = np.abs(grad_xx)
    abs_yy = np.abs(grad_yy)
    both = (abs_xx > threshold * 0.5) & (abs_yy > threshold * 0.5)

    directions[abs_yy > abs_xx] = RefineDirection.Y
    directions[both] = RefineDirection.BOTH

    return RefinementMap(
        markers=markers,
        directions=directions,
        indicator=indicator,
        threshold=threshold,
    )


# ─── Refinement Application ───────────────────────────────────────────

def refine_mesh(
    mesh: Mesh2D,
    ref_map: RefinementMap,
    field: Optional[np.ndarray] = None,
    min_cells: int = 2,
    max_level: int = 5,
) -> tuple[Mesh2D, Optional[np.ndarray]]:
    """
    Apply a refinement map to a structured mesh.

    Cells marked for refinement are bisected in the specified direction.
    The output mesh is constructed by inserting new grid lines at the
    midpoints of marked cell edges.

    Parameters
    ----------
    mesh : Mesh2D
        The current mesh to refine.
    ref_map : RefinementMap
        Per-cell refinement decisions.
    field : np.ndarray, optional
        Solution field to interpolate onto the refined mesh.
        If ``None``, only the mesh is returned.
    min_cells : int
        Minimum number of cells in each direction (default: 2).
    max_level : int
        Maximum number of refinement levels (default: 5).  This is a
        safety limit — the function does not track the current level
        automatically.

    Returns
    -------
    refined_mesh : Mesh2D
        The refined mesh.
    refined_field : np.ndarray or None
        Solution field interpolated onto the refined mesh, or ``None``
        if no field was provided.
    """
    if not np.any(ref_map.markers):
        # No refinement needed
        return mesh, field

    # Build refined 1-D node distributions
    nx, ny = mesh.nx, mesh.ny

    # Count how many new lines to insert in each direction
    # A cell at (i, j) marked for X-refinement needs a new vertical
    # line through its centre.  We can aggregate these into the
    # existing xn / yn arrays.

    # Determine which i-lines and j-lines to insert
    insert_i = np.zeros(nx, dtype=bool)   # New vertical lines (i-faces)
    insert_j = np.zeros(ny, dtype=bool)   # New horizontal lines (j-faces)

    for i in range(nx):
        for j in range(ny):
            if ref_map.markers[i, j]:
                if (ref_map.directions[i, j] in (RefineDirection.X,
                                                  RefineDirection.BOTH)):
                    insert_i[i] = True
                if (ref_map.directions[i, j] in (RefineDirection.Y,
                                                  RefineDirection.BOTH)):
                    insert_j[j] = True

    # Build new node coordinate arrays
    # Original i-nodes are at positions 0, 1, ..., nx (i.e. nx + 1 nodes)
    # We insert midpoints between existing i-nodes.

    # Compute new xi positions for nodes in x-direction
    xi_old = mesh.xn[:, 0]  # x-node positions (nx + 1)
    xi_new_list = [xi_old[0]]
    for i in range(nx):
        x_left = xi_old[i]
        x_right = xi_old[i + 1]
        if insert_i[i]:
            x_mid = 0.5 * (x_left + x_right)
            xi_new_list.append(x_mid)
        xi_new_list.append(x_right)
    xi_new = np.array(xi_new_list)
    nx_new = len(xi_new) - 1

    # Same for y
    eta_old = mesh.yn[0, :]  # y-node positions (ny + 1)
    eta_new_list = [eta_old[0]]
    for j in range(ny):
        y_bottom = eta_old[j]
        y_top = eta_old[j + 1]
        if insert_j[j]:
            y_mid = 0.5 * (y_bottom + y_top)
            eta_new_list.append(y_mid)
        eta_new_list.append(y_top)
    eta_new = np.array(eta_new_list)
    ny_new = len(eta_new) - 1

    # Build the refined mesh as a tensor product
    # For a simple Cartesian mesh this is exact.  For general
    # geometries we would need transfinite interpolation to place
    # the new nodes — here we use the uniform mapping.
    Xn_new, Yn_new = np.meshgrid(xi_new, eta_new, indexing="ij")

    # Create the refined Mesh2D via dataclass replace
    # (We'll construct it manually since Mesh2D builds from a 1-D distribution)
    refined = Mesh2D.__new__(Mesh2D)
    refined.nx = nx_new
    refined.ny = ny_new
    refined.lx = mesh.lx
    refined.ly = mesh.ly
    refined.clustering_x = mesh.clustering_x  # informational only
    refined.clustering_y = mesh.clustering_y
    refined.s0_x = mesh.s0_x
    refined.s0_y = mesh.s0_y
    refined.s1_x = mesh.s1_x
    refined.s1_y = mesh.s1_y

    refined.xn = Xn_new
    refined.yn = Yn_new
    # Cell centres (average of four nodes)
    refined.xc = 0.25 * (
        Xn_new[:-1, :-1] + Xn_new[1:, :-1]
        + Xn_new[:-1, 1:] + Xn_new[1:, 1:]
    )
    refined.yc = 0.25 * (
        Yn_new[:-1, :-1] + Yn_new[1:, :-1]
        + Yn_new[:-1, 1:] + Yn_new[1:, 1:]
    )
    refined.dx = np.diff(Xn_new, axis=0)
    refined.dy = np.diff(Yn_new, axis=1)

    # Interpolate field onto refined mesh
    refined_field = None
    if field is not None:
        refined_field = _interpolate_field(
            mesh.xc, mesh.yc, field,
            refined.xc, refined.yc,
        )

    return refined, refined_field


def _interpolate_field(
    x_old: np.ndarray,
    y_old: np.ndarray,
    field_old: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation of a field from old to new cell centres.

    For Cartesian grids this reduces to linear interpolation in each
    direction.  For non-Cartesian grids a full 2-D interpolation
    would be required; here we implement a simple inverse-distance
    weighted average from the four nearest old cell centres.
    """
    nx_new, ny_new = x_new.shape
    field_new = np.zeros((nx_new, ny_new))

    # For each new cell centre, find the nearest old cell centre
    # and average the surrounding 2×2 block using inverse-distance weights.
    for i in range(nx_new):
        for j in range(ny_new):
            # Find nearest old cell in x
            i_old = np.argmin(np.abs(x_old[:, 0] - x_new[i, j]))
            i_old = max(1, min(i_old, x_old.shape[0] - 2))
            # Nearest in y
            j_old = np.argmin(np.abs(y_old[0, :] - y_new[i, j]))
            j_old = max(1, min(j_old, y_old.shape[1] - 2))

            # 2×2 block
            x_block = x_old[i_old - 1:i_old + 2, j_old - 1:j_old + 2]
            y_block = y_old[i_old - 1:i_old + 2, j_old - 1:j_old + 2]
            f_block = field_old[i_old - 1:i_old + 2, j_old - 1:j_old + 2]

            # Inverse distance weights
            dx = x_block - x_new[i, j]
            dy = y_block - y_new[i, j]
            dist = np.sqrt(dx**2 + dy**2) + 1e-30
            weights = 1.0 / dist
            field_new[i, j] = np.sum(weights * f_block) / np.sum(weights)

    return field_new


# ─── Coarsening ──────────────────────────────────────────────────────

def coarsen_mesh(
    mesh: Mesh2D,
    field: Optional[np.ndarray] = None,
    coarsen_x: int = 2,
    coarsen_y: int = 2,
    field_aggregator: Callable = np.mean,
) -> tuple[Mesh2D, Optional[np.ndarray]]:
    """
    Coarsen the mesh by merging every ``coarsen_x × coarsen_y``
    block of cells into a single cell.

    This is the inverse operation of refinement and may be used to
    derefine regions where the solution is smooth.

    Parameters
    ----------
    mesh : Mesh2D
        The current mesh.
    field : np.ndarray, optional
        Solution field to coarsen.
    coarsen_x, coarsen_y : int
        Coarsening factor in each direction (default: 2).
    field_aggregator : Callable
        Function to combine field values (default: ``np.mean``).

    Returns
    -------
    coarse_mesh : Mesh2D
        The coarsened mesh.
    coarse_field : np.ndarray or None
    """
    nx = mesh.nx // coarsen_x
    ny = mesh.ny // coarsen_y

    # Build reduced node grid
    xn_coarse = mesh.xn[::coarsen_x, ::coarsen_y]
    yn_coarse = mesh.yn[::coarsen_x, ::coarsen_y]

    # Validate dimensions
    if xn_coarse.shape[0] < 2 or xn_coarse.shape[1] < 2:
        msg = "Coarsening factor too large for the current mesh dimensions."
        raise ValueError(msg)

    refined = Mesh2D.__new__(Mesh2D)
    refined.nx = nx
    refined.ny = ny
    refined.lx = mesh.lx
    refined.ly = mesh.ly
    refined.clustering_x = mesh.clustering_x
    refined.clustering_y = mesh.clustering_y
    refined.s0_x = mesh.s0_x
    refined.s0_y = mesh.s0_y
    refined.s1_x = mesh.s1_x
    refined.s1_y = mesh.s1_y
    refined.xn = xn_coarse
    refined.yn = yn_coarse
    refined.xc = 0.25 * (
        xn_coarse[:-1, :-1] + xn_coarse[1:, :-1]
        + xn_coarse[:-1, 1:] + xn_coarse[1:, 1:]
    )
    refined.yc = 0.25 * (
        yn_coarse[:-1, :-1] + yn_coarse[1:, :-1]
        + yn_coarse[:-1, 1:] + yn_coarse[1:, 1:]
    )
    refined.dx = np.diff(xn_coarse, axis=0)
    refined.dy = np.diff(yn_coarse, axis=1)

    coarse_field = None
    if field is not None:
        # Aggregate blocks
        coarse_field = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                block = field[
                    i * coarsen_x:(i + 1) * coarsen_x,
                    j * coarsen_y:(j + 1) * coarsen_y,
                ]
                coarse_field[i, j] = field_aggregator(block)

    return refined, coarse_field


# ─── Adaptive Loop ───────────────────────────────────────────────────

def adapt_mesh(
    mesh: Mesh2D,
    field: np.ndarray,
    indicator: str = "gradient",
    threshold: float = 0.2,
    max_refine_iterations: int = 3,
) -> tuple[Mesh2D, np.ndarray]:
    """
    Perform multiple refinement passes on a mesh based on a solution
    field, producing an adapted mesh.

    Parameters
    ----------
    mesh : Mesh2D
        Initial mesh.
    field : np.ndarray
        Solution field on the initial mesh.
    indicator : str
        Either ``"gradient"`` or ``"curvature"``.
    threshold : float
        Refinement threshold passed to the indicator.
    max_refine_iterations : int
        Maximum number of sequential refinement passes.

    Returns
    -------
    adapted_mesh : Mesh2D
    adapted_field : np.ndarray
    """
    indicator_fn = {
        "gradient": gradient_indicator,
        "curvature": curvature_indicator,
    }

    if indicator not in indicator_fn:
        msg = f"Unknown indicator '{indicator}'.  Choose from {list(indicator_fn.keys())}."
        raise ValueError(msg)

    current_mesh = mesh
    current_field = field

    for _ in range(max_refine_iterations):
        ref_map = indicator_fn[indicator](current_field, threshold=threshold)

        if not np.any(ref_map.markers):
            break  # Converged — no more cells need refinement

        current_mesh, current_field = refine_mesh(
            current_mesh, ref_map, current_field,
        )

    return current_mesh, current_field


if __name__ == "__main__":
    from .mesh_2d import uniform_mesh

    # Demonstration with a synthetic 2-D Gaussian
    mesh = uniform_mesh(16, 16, lx=2.0, ly=2.0)
    xc, yc = mesh.cell_centres()

    # A Gaussian bump — high gradient and curvature near the centre
    sigma = 0.3
    field = np.exp(-((xc - 1.0)**2 + (yc - 1.0)**2) / (2 * sigma**2))

    print(f"Initial mesh: {mesh.nx}×{mesh.ny}  ({mesh.n_cells} cells)")

    ref_map = gradient_indicator(field, threshold=0.15)
    print(f"Gradient indicator: {np.sum(ref_map.markers)} cells marked "
          f"out of {mesh.n_cells}")

    adapted, adapted_field = adapt_mesh(mesh, field, "gradient", 0.15, 3)
    print(f"Adapted mesh: {adapted.nx}×{adapted.ny}  "
          f"({adapted.n_cells} cells)")

    ref_map_c = curvature_indicator(field, threshold=0.2)
    print(f"Curvature indicator: {np.sum(ref_map_c.markers)} cells marked "
          f"out of {mesh.n_cells}")
