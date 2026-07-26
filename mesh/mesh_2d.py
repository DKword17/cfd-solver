#!/usr/bin/env python3
"""
cfd_solver/mesh/mesh_2d.py
============================

Structured 2D mesh generation for finite-volume CFD.

Provides the ``Mesh2D`` class with two construction strategies:

1. **Uniform** — equal spacing in both directions.
2. **Hyperbolic-tangent clustering** — one- or two-sided boundary
   clustering following Vinokur's (1983) method, which guarantees
   monotonic spacing and permits precise specification of the
   near-wall cell size or stretching ratio.

The mesh is collocated: velocities, pressure, and scalar fields
share the same cell-centred storage.  Face locations are stored
separately for flux computations.

References
----------
- Thompson, J. F., Warsi, Z. U. A., & Mastin, C. W. (1985).
  *Numerical Grid Generation: Foundations and Applications*.
  North-Holland.
- Vinokur, M. (1983). On one-dimensional stretching functions
  for finite-difference calculations. *Journal of Computational
  Physics*, 50(2), 215–234.
- Ferziger, J. H., & Perić, M. (2002). *Computational Methods
  for Fluid Dynamics* (3rd ed.). Springer.

Author: James Hargreaves
        Department of Aeronautics, Imperial College London
Branch: dev/mesh-james
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


# ─── Clustering Type ──────────────────────────────────────────────────

class ClusteringType(IntEnum):
    """Strategy for distributing grid points along an axis."""
    UNIFORM = 0          # Equally-spaced cells
    TANH_BOTH = 1        # Hyperbolic tangent, both ends clustered
    TANH_LEFT = 2        # Hyperbolic tangent, left-end clustering only
    TANH_RIGHT = 3       # Hyperbolic tangent, right-end clustering only


# ─── Helper: 1-D stretching (Vinokur, 1983) ──────────────────────────

def _vinokur_stretching(
    n: int,
    s0: float,
    s1: float,
) -> np.ndarray:
    """
    Compute a 1-D Vinokur hyperbolic-tangent stretching.

    Returns normalised node positions in [0, 1] such that the first
    cell width is approximately ``s0`` and the last cell width is
    approximately ``s1`` (relative to the total length).

    If both ``s0`` and ``s1`` are provided a symmetric two-sided
    clustering is produced.  If only ``s0`` is non-zero a one-sided
    clustering is produced.

    Parameters
    ----------
    n : int
        Number of cells.
    s0 : float
        Desired relative first-cell width  (Δξ₀ / L).
    s1 : float
        Desired relative last-cell width   (Δξ_{n-1} / L).

    Returns
    -------
    np.ndarray
        Normalised node positions ξ ∈ [0, 1], shape ``(n + 1,)``.
    """
    if s0 <= 0.0 or s1 <= 0.0:
        msg = "Stretching ratios must be positive."
        raise ValueError(msg)

    # Vinokur's method — solve for the stretching parameter δ such that
    # the ratio of the first to last cell matches the target.
    #
    #   f(ξ) = 0.5 * [1 + tanh(δ * (ξ - 0.5)) / tanh(δ/2)]
    #
    # We use a bisection solve for δ.

    target_ratio = s0 / s1

    def _ratio(delta: float) -> float:
        """Return the first:last cell-width ratio for a given δ."""
        if abs(delta) < 1e-12:
            return 1.0
        # Cell-centre positions
        xi = np.linspace(0.0, 1.0, n + 1)
        f = 0.5 * (1.0 + np.tanh(delta * (xi - 0.5)) / np.tanh(0.5 * delta))
        # Cell widths
        dw = np.diff(f)
        return dw[0] / dw[-1]

    # Bisection for δ
    delta_low, delta_high = 1e-6, 12.0
    r_low = _ratio(delta_low)
    r_high = _ratio(delta_high)

    if not (min(r_low, r_high) <= target_ratio <= max(r_low, r_high)):
        # Target outside achievable range — clamp
        delta = delta_high if target_ratio < r_high else delta_low
    else:
        for _ in range(60):
            delta_mid = 0.5 * (delta_low + delta_high)
            r_mid = _ratio(delta_mid)
            if r_mid > target_ratio:
                delta_high = delta_mid
                r_high = r_mid
            else:
                delta_low = delta_mid
                r_low = r_mid
            if abs(r_mid - target_ratio) < 1e-10:
                break
        delta = 0.5 * (delta_low + delta_high)

    # Build final distribution
    xi = np.linspace(0.0, 1.0, n + 1)
    if abs(delta) < 1e-12:
        return xi
    xi_clustered = 0.5 * (
        1.0 + np.tanh(delta * (xi - 0.5)) / np.tanh(0.5 * delta)
    )
    return xi_clustered


def _one_sided_stretching(n: int, s0: float) -> np.ndarray:
    """
    One-sided Vinokur stretching (only the left end is clustered).

    Parameters
    ----------
    n : int
        Number of cells.
    s0 : float
        Desired relative first-cell width (Δξ₀ / L).

    Returns
    -------
    np.ndarray
        Normalised node positions ξ ∈ [0, 1], shape ``(n + 1,)``.
    """
    # Use two-sided with a very loose last-cell constraint
    return _vinokur_stretching(n, s0, 1.0)


# ─── Mesh2D ────────────────────────────────────────────────────────────

@dataclass
class Mesh2D:
    """
    Two-dimensional structured, collocated grid.

    Cell indexing follows the CFD convention:

    * ``i`` runs in the **x**-direction (west → east),  ``i = 0 … nx - 1``.
    * ``j`` runs in the **y**-direction (south → north), ``j = 0 … ny - 1``.

    Node (vertex) coordinates are stored in arrays of shape ``(nx + 1, ny + 1)``
    while cell-centre coordinates have shape ``(nx, ny)``.

    Parameters
    ----------
    nx : int
        Number of cells in the x-direction.
    ny : int
        Number of cells in the y-direction.
    lx : float, optional
        Domain length in x (default: 1.0).
    ly : float, optional
        Domain length in y (default: 1.0).
    clustering_x : ClusteringType, optional
        Distribution strategy for x-direction.
    clustering_y : ClusteringType, optional
        Distribution strategy for y-direction.
    s0_x : float, optional
        Relative first-cell width in x (for clustering).
    s0_y : float, optional
        Relative first-cell width in y (for clustering).
    s1_x : float, optional
        Relative last-cell width in x (for two-sided clustering).
    s1_y : float, optional
        Relative last-cell width in y (for two-sided clustering).

    Attributes
    ----------
    xn : np.ndarray
        Node x-coordinates, shape ``(nx + 1, ny + 1)``.
    yn : np.ndarray
        Node y-coordinates, shape ``(nx + 1, ny + 1)``.
    xc : np.ndarray
        Cell-centre x-coordinates, shape ``(nx, ny)``.
    yc : np.ndarray
        Cell-centre y-coordinates, shape ``(nx, ny)``.
    dx : np.ndarray
        Cell width  in x, shape ``(nx, ny)``.
    dy : np.ndarray
        Cell height in y, shape ``(nx, ny)``.
    """

    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0
    clustering_x: ClusteringType = ClusteringType.UNIFORM
    clustering_y: ClusteringType = ClusteringType.UNIFORM
    s0_x: float = 0.01
    s0_y: float = 0.01
    s1_x: float = 0.01
    s1_y: float = 0.01

    xn: np.ndarray = field(init=False, repr=False)
    yn: np.ndarray = field(init=False, repr=False)
    xc: np.ndarray = field(init=False, repr=False)
    yc: np.ndarray = field(init=False, repr=False)
    dx: np.ndarray = field(init=False, repr=False)
    dy: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._build()

    # ── Public interface ──────────────────────────────────────────────

    @property
    def n_cells(self) -> int:
        """Total number of cells in the mesh."""
        return self.nx * self.ny

    @property
    def shape(self) -> tuple[int, int]:
        """Cell count per direction ``(nx, ny)``."""
        return (self.nx, self.ny)

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(xc, yc)`` — convenience access to centre coordinates."""
        return self.xc, self.yc

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(xn, yn)`` — convenience access to node coordinates."""
        return self.xn, self.yn

    def cell_volumes(self) -> np.ndarray:
        """
        Compute cell volumes (areas in 2-D).

        The volume of each quadrilateral cell is computed via the
        cross-product of its diagonal vectors, which works correctly
        for both Cartesian and body-fitted grids.

        Returns
        -------
        np.ndarray
            Cell volume array, shape ``(nx, ny)``.
        """
        # Quadrilateral area via cross-product of diagonals
        x_ac = self.xn[1:, 1:] - self.xn[:-1, :-1]
        y_ac = self.yn[1:, 1:] - self.yn[:-1, :-1]
        x_bd = self.xn[:-1, 1:] - self.xn[1:, :-1]
        y_bd = self.yn[:-1, 1:] - self.yn[1:, :-1]
        return 0.5 * np.abs(x_ac * y_bd - x_bd * y_ac)

    def jacobian(self) -> np.ndarray:
        """
        Compute the Jacobian determinant of the mapping at cell centres.

        For a mapping (x(ξ, η), y(ξ, η)) the Jacobian is:

            J = x_ξ · y_η - x_η · y_ξ

        Returns
        -------
        np.ndarray
            Jacobian determinant, shape ``(nx, ny)``.
        """
        # Metric derivatives at cell centres (central differences)
        x_xi = np.zeros((self.nx, self.ny))
        x_eta = np.zeros((self.nx, self.ny))
        y_xi = np.zeros((self.nx, self.ny))
        y_eta = np.zeros((self.nx, self.ny))

        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                x_xi[i, j] = 0.5 * (self.xc[i + 1, j] - self.xc[i - 1, j])
                x_eta[i, j] = 0.5 * (self.xc[i, j + 1] - self.xc[i, j - 1])
                y_xi[i, j] = 0.5 * (self.yc[i + 1, j] - self.yc[i, j - 1])
                y_eta[i, j] = 0.5 * (self.yc[i, j + 1] - self.yc[i, j - 1])

        J = x_xi * y_eta - x_eta * y_xi
        return J

    def compute_metrics(self) -> dict[str, np.ndarray]:
        """
        Compute all metric terms for the coordinate transformation.

        Returns
        -------
        dict
            Keys ``xi_x, xi_y, eta_x, eta_y, J`` — the contravariant
            metric components and the Jacobian determinant.
        """
        J = self.jacobian()
        # Avoid division by zero
        eps = 1e-30
        J_safe = np.where(np.abs(J) > eps, J, eps)

        # Metric derivatives at cell centres
        x_xi = np.zeros((self.nx, self.ny))
        x_eta = np.zeros((self.nx, self.ny))
        y_xi = np.zeros((self.nx, self.ny))
        y_eta = np.zeros((self.nx, self.ny))

        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                x_xi[i, j] = 0.5 * (self.xc[i + 1, j] - self.xc[i - 1, j])
                x_eta[i, j] = 0.5 * (self.xc[i, j + 1] - self.xc[i, j - 1])
                y_xi[i, j] = 0.5 * (self.yc[i + 1, j] - self.yc[i, j - 1])
                y_eta[i, j] = 0.5 * (self.yc[i, j + 1] - self.yc[i, j - 1])

        # Contravariant metrics (inverse mapping derivatives)
        xi_x = y_eta / J_safe
        xi_y = -x_eta / J_safe
        eta_x = -y_xi / J_safe
        eta_y = x_xi / J_safe

        return {
            "xi_x": xi_x,
            "xi_y": xi_y,
            "eta_x": eta_x,
            "eta_y": eta_y,
            "J": J,
        }

    # ── Construction ──────────────────────────────────────────────────

    def _build(self) -> None:
        """Construct the grid by computing node and centre coordinates."""
        # 1-D distributions (normalised)
        xi_n = self._distribute_1d(self.nx, self.lx, self.clustering_x,
                                    self.s0_x, self.s1_x)
        eta_n = self._distribute_1d(self.ny, self.ly, self.clustering_y,
                                     self.s0_y, self.s1_y)

        # Node coordinates (tensor-product grid)
        self.xn, self.yn = np.meshgrid(xi_n, eta_n, indexing="ij")

        # Cell-centre coordinates — average of four surrounding nodes
        self.xc = 0.25 * (
            self.xn[:-1, :-1] + self.xn[1:, :-1]
            + self.xn[:-1, 1:] + self.xn[1:, 1:]
        )
        self.yc = 0.25 * (
            self.yn[:-1, :-1] + self.yn[1:, :-1]
            + self.yn[:-1, 1:] + self.yn[1:, 1:]
        )

        # Cell dimensions — averaged at cell centres (shape nx × ny)
        dx_full = np.diff(self.xn, axis=0)        # (nx, ny+1)
        dy_full = np.diff(self.yn, axis=1)        # (nx+1, ny)
        self.dx = 0.5 * (dx_full[:, :-1] + dx_full[:, 1:])   # (nx, ny)
        self.dy = 0.5 * (dy_full[:-1, :] + dy_full[1:, :])   # (nx, ny)

    @staticmethod
    def _distribute_1d(
        n: int,
        length: float,
        clustering: ClusteringType,
        s0: float,
        s1: float,
    ) -> np.ndarray:
        """
        Generate a normalised 1-D node distribution.

        Parameters
        ----------
        n : int
            Number of cells.
        length : float
            Physical length of the domain in this direction.
        clustering : ClusteringType
            The clustering strategy.
        s0, s1 : float
            Relative first- and last-cell widths.

        Returns
        -------
        np.ndarray
            Physical coordinate positions, shape ``(n + 1,)``.
        """
        if clustering == ClusteringType.UNIFORM:
            xi = np.linspace(0.0, length, n + 1)
        elif clustering == ClusteringType.TANH_BOTH:
            xi_norm = _vinokur_stretching(n, s0, s1)
            xi = xi_norm * length
        elif clustering == ClusteringType.TANH_LEFT:
            xi_norm = _one_sided_stretching(n, s0)
            xi = xi_norm * length
        elif clustering == ClusteringType.TANH_RIGHT:
            # Reverse of left-only clustering
            xi_norm = _one_sided_stretching(n, s0)
            xi = (1.0 - xi_norm[::-1]) * length
        else:
            msg = f"Unknown clustering type: {clustering}"
            raise ValueError(msg)

        return xi


# ─── Convenience constructors ─────────────────────────────────────────

def uniform_mesh(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> Mesh2D:
    """
    Create a uniformly-spaced Cartesian mesh.

    Parameters
    ----------
    nx, ny : int
        Number of cells in each direction.
    lx, ly : float
        Domain extents.

    Returns
    -------
    Mesh2D
    """
    return Mesh2D(
        nx=nx, ny=ny, lx=lx, ly=ly,
        clustering_x=ClusteringType.UNIFORM,
        clustering_y=ClusteringType.UNIFORM,
    )


def clustered_mesh(
    nx: int,
    ny: int,
    lx: float = 1.0,
    ly: float = 1.0,
    s0_x: float = 0.01,
    s0_y: float = 0.01,
    cluster_x: bool = True,
    cluster_y: bool = True,
    both_ends: bool = False,
) -> Mesh2D:
    """
    Create a mesh with hyperbolic-tangent clustering near boundaries.

    Parameters
    ----------
    nx, ny : int
        Number of cells in each direction.
    lx, ly : float
        Domain extents.
    s0_x, s0_y : float
        Relative first-cell widths (fraction of domain length).
    cluster_x, cluster_y : bool
        Whether to cluster in each direction.
    both_ends : bool
        If ``True``, cluster at **both** ends; otherwise only at the
        start (left / south) boundary.

    Returns
    -------
    Mesh2D
    """
    cx = ClusteringType.TANH_BOTH if (cluster_x and both_ends) else \
         (ClusteringType.TANH_LEFT if cluster_x else ClusteringType.UNIFORM)
    cy = ClusteringType.TANH_BOTH if (cluster_y and both_ends) else \
         (ClusteringType.TANH_LEFT if cluster_y else ClusteringType.UNIFORM)

    return Mesh2D(
        nx=nx, ny=ny, lx=lx, ly=ly,
        clustering_x=cx, clustering_y=cy,
        s0_x=s0_x, s0_y=s0_y,
        s1_x=s0_x, s1_y=s0_y,
    )


if __name__ == "__main__":
    # Quick demonstration
    mesh = uniform_mesh(8, 6, lx=2.0, ly=1.0)
    print(f"Uniform mesh:  {mesh.nx}×{mesh.ny}  cells,  "
          f"volumes range = [{mesh.cell_volumes().min():.4f}, "
          f"{mesh.cell_volumes().max():.4f}]")

    mesh_c = clustered_mesh(16, 16, s0_x=0.005, s0_y=0.005, both_ends=True)
    J = mesh_c.jacobian()
    print(f"Clustered mesh:  |J| range = [{J.min():.4f}, {J.max():.4f}]")
    print(f"xc shape: {mesh_c.xc.shape},  xn shape: {mesh_c.xn.shape}")
