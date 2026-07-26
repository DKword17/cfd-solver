#!/usr/bin/env python3
"""
cfd_solver/mesh/transform.py
==============================

Body-fitted coordinate generation for 2-D structured grids.

Two complementary methods are provided:

1. **Algebraic (Transfinite Interpolation, TFI)** — constructs the
   interior grid by interpolating between four prescribed boundary
   curves.  TFI guarantees that the grid exactly matches the boundary
   and produces a valid mesh for convex domains with reasonable
   boundary distributions (Gordon & Hall, 1973).

2. **Elliptic generation** — solves a system of Poisson equations for
   the interior node positions, with user-specified source terms to
   control grid spacing and orthogonality near boundaries.  Elliptic
   grids are guaranteed to be smooth and one-to-one (no folding) for
   convex domains (Thompson et al., 1985).

The output is a ``Mesh2D`` instance suitable for use with the solver.

References
----------
- Gordon, W. J., & Hall, C. A. (1973). Construction of curvilinear
  co-ordinate systems and applications to mesh generation.
  *International Journal for Numerical Methods in Engineering*,
  7(4), 461–477.
- Thompson, J. F., Warsi, Z. U. A., & Mastin, C. W. (1985).
  *Numerical Grid Generation: Foundations and Applications*.
  North-Holland.
- Steger, J. L., & Chaussee, D. S. (1981). Generation of body-fitted
  coordinates using hyperbolic partial differential equations.
  *SIAM Journal on Scientific and Statistical Computing*, 2(1), 89–100.

Author: James Hargreaves
        Department of Aeronautics, Imperial College London
Branch: dev/mesh-james
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .mesh_2d import Mesh2D


# ─── Boundary Representation ──────────────────────────────────────────

class BoundaryCurves:
    """
    Four boundary curves defining a simply-connected 2-D domain.

    Each boundary is parameterised by a normalised coordinate (0 → 1)
    and returns arrays of x- and y-coordinates with ``n_points`` values.

    The convention follows the computational space (ξ, η):

    * West  (ξ = 0):   ``(x(0, η), y(0, η))``   for η ∈ [0, 1].
    * East  (ξ = 1):   ``(x(1, η), y(1, η))``   for η ∈ [0, 1].
    * South (η = 0):   ``(x(ξ, 0), y(ξ, 0))``   for ξ ∈ [0, 1].
    * North (η = 1):   ``(x(ξ, 1), y(ξ, 1))``   for ξ ∈ [0, 1].

    Parameters
    ----------
    west, east, south, north : Callable[[int], tuple[np.ndarray, np.ndarray]]
        Each boundary function takes an integer ``n_points`` and
        returns ``(x, y)`` arrays of that length.
    """
    def __init__(
        self,
        west: Callable[[int], tuple[np.ndarray, np.ndarray]],
        east: Callable[[int], tuple[np.ndarray, np.ndarray]],
        south: Callable[[int], tuple[np.ndarray, np.ndarray]],
        north: Callable[[int], tuple[np.ndarray, np.ndarray]],
    ):
        self._west = west
        self._east = east
        self._south = south
        self._north = north

    def west(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """West boundary (ξ = 0)."""
        return self._west(n)

    def east(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """East boundary (ξ = 1)."""
        return self._east(n)

    def south(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """South boundary (η = 0)."""
        return self._south(n)

    def north(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """North boundary (η = 1)."""
        return self._north(n)

    @staticmethod
    def from_arrays(
        x_west: np.ndarray, y_west: np.ndarray,
        x_east: np.ndarray, y_east: np.ndarray,
        x_south: np.ndarray, y_south: np.ndarray,
        x_north: np.ndarray, y_north: np.ndarray,
    ) -> "BoundaryCurves":
        """Construct from pre-computed coordinate arrays."""
        def _wrap(x, y):
            return lambda n: (x[:n], y[:n])
        return BoundaryCurves(
            west=_wrap(x_west, y_west),
            east=_wrap(x_east, y_east),
            south=_wrap(x_south, y_south),
            north=_wrap(x_north, y_north),
        )


# ─── Algebraic Grid: Transfinite Interpolation (TFI) ────────────────

def algebraic_grid(
    boundaries: BoundaryCurves,
    nx: int,
    ny: int,
    clustering_xi: Optional[float] = None,
    clustering_eta: Optional[float] = None,
) -> Mesh2D:
    """
    Generate a body-fitted grid using transfinite interpolation (TFI).

    TFI constructs the interior grid from the four boundaries using
    a bilinear blending of the boundary data:

        x(ξ, η) = (1 - η) · xₛ(ξ) + η · xₙ(ξ)
                + (1 - ξ) · x_w(η) + ξ · xₑ(η)
                - [ (1 - ξ)(1 - η) · x_sw + ξ(1 - η) · x_se
                  + (1 - ξ)η · x_nw + ξη · x_ne ]

    where xₛ, xₙ, x_w, xₑ are the boundary curves and x_sw, ... are
    the four corner points.

    Parameters
    ----------
    boundaries : BoundaryCurves
        The four boundary curves.
    nx, ny : int
        Number of cells in the ξ- and η-directions.
    clustering_xi, clustering_eta : float, optional
        If provided, controls Vinokur-style hyperbolic tangent clustering
        in the computational directions.  The value is the relative
        first-cell width (fraction of 1.0 in computational space).

    Returns
    -------
    Mesh2D
    """
    # Computational-space distribution
    if clustering_xi is not None:
        from .mesh_2d import _vinokur_stretching
        xi_n = _vinokur_stretching(nx, clustering_xi, clustering_xi)
        xi_n *= boundaries.south(nx + 1)[0][-1]  # scale to [0, 1] — actually no
        # Reset: _vinokur_stretching returns in [0, 1]
        xi_n = _vinokur_stretching(nx, clustering_xi, clustering_xi)
    else:
        xi_n = np.linspace(0.0, 1.0, nx + 1)

    if clustering_eta is not None:
        from .mesh_2d import _vinokur_stretching
        eta_n = _vinokur_stretching(ny, clustering_eta, clustering_eta)
    else:
        eta_n = np.linspace(0.0, 1.0, ny + 1)

    # Sample boundaries at the same distributions
    # (interpolation may be needed if boundaries are not sampled
    #  at exactly these points; for simplicity we sample at linspace)
    n_xi = nx + 1
    n_eta = ny + 1

    x_s, y_s = boundaries.south(n_xi)
    x_n, y_n = boundaries.north(n_xi)
    x_w, y_w = boundaries.west(n_eta)
    x_e, y_e = boundaries.east(n_eta)

    # Ensure corner consistency
    x_sw, y_sw = x_s[0], y_s[0]
    x_se, y_se = x_s[-1], y_s[-1]
    x_nw, y_nw = x_n[0], y_n[0]
    x_ne, y_ne = x_n[-1], y_n[-1]

    # Interpolate boundary data onto the stretched computational grid
    # using linear interpolation
    xi_uniform = np.linspace(0.0, 1.0, n_xi)
    eta_uniform = np.linspace(0.0, 1.0, n_eta)

    x_s_interp = np.interp(xi_n, xi_uniform, x_s)
    y_s_interp = np.interp(xi_n, xi_uniform, y_s)
    x_n_interp = np.interp(xi_n, xi_uniform, x_n)
    y_n_interp = np.interp(xi_n, xi_uniform, y_n)
    x_w_interp = np.interp(eta_n, eta_uniform, x_w)
    y_w_interp = np.interp(eta_n, eta_uniform, y_w)
    x_e_interp = np.interp(eta_n, eta_uniform, x_e)
    y_e_interp = np.interp(eta_n, eta_uniform, y_e)

    # Build 2-D grid via TFI
    Xn = np.zeros((n_xi, n_eta))
    Yn = np.zeros((n_xi, n_eta))

    for i in range(n_xi):
        xi = xi_n[i]
        for j in range(n_eta):
            eta = eta_n[j]

            # Blending functions
            phi_0 = 1.0 - xi
            phi_1 = xi
            psi_0 = 1.0 - eta
            psi_1 = eta

            # TFI formula
            Xn[i, j] = (
                psi_0 * x_s_interp[i] + psi_1 * x_n_interp[i]
                + phi_0 * x_w_interp[j] + phi_1 * x_e_interp[j]
                - (phi_0 * psi_0 * x_sw + phi_1 * psi_0 * x_se
                   + phi_0 * psi_1 * x_nw + phi_1 * psi_1 * x_ne)
            )
            Yn[i, j] = (
                psi_0 * y_s_interp[i] + psi_1 * y_n_interp[i]
                + phi_0 * y_w_interp[j] + phi_1 * y_e_interp[j]
                - (phi_0 * psi_0 * y_sw + phi_1 * psi_0 * y_se
                   + phi_0 * psi_1 * y_nw + phi_1 * psi_1 * y_ne)
            )

    # Construct Mesh2D
    mesh = Mesh2D.__new__(Mesh2D)
    mesh.nx = nx
    mesh.ny = ny
    mesh.lx = 1.0
    mesh.ly = 1.0
    mesh.clustering_x = mesh.clustering_x if hasattr(mesh, 'clustering_x') else 0
    mesh.clustering_y = mesh.clustering_y if hasattr(mesh, 'clustering_y') else 0
    mesh.s0_x = clustering_xi or 0.01
    mesh.s0_y = clustering_eta or 0.01
    mesh.s1_x = clustering_xi or 0.01
    mesh.s1_y = clustering_eta or 0.01
    mesh.xn = Xn
    mesh.yn = Yn
    mesh.xc = 0.25 * (Xn[:-1, :-1] + Xn[1:, :-1]
                       + Xn[:-1, 1:] + Xn[1:, 1:])
    mesh.yc = 0.25 * (Yn[:-1, :-1] + Yn[1:, :-1]
                       + Yn[:-1, 1:] + Yn[1:, 1:])
    mesh.dx = np.diff(Xn, axis=0)
    mesh.dy = np.diff(Yn, axis=1)

    return mesh


# ─── Elliptic Grid Generation ────────────────────────────────────────

def elliptic_grid(
    boundaries: BoundaryCurves,
    nx: int,
    ny: int,
    source_x: Optional[np.ndarray] = None,
    source_y: Optional[np.ndarray] = None,
    max_iter: int = 2000,
    tol: float = 1e-8,
    omega: float = 1.5,
) -> Mesh2D:
    """
    Generate a body-fitted grid by solving Poisson equations.

    The interior node positions satisfy:

        ∇²x = P(ξ, η)
        ∇²y = Q(ξ, η)

    where P and Q are user-specified source terms that control grid
    spacing and orthogonality near boundaries (Thompson et al., 1985).

    The equations are discretised with second-order central differences
    and solved by successive over-relaxation (SOR).

    Parameters
    ----------
    boundaries : BoundaryCurves
        The four boundary curves specifying the domain.
    nx, ny : int
        Number of cells in the ξ- and η-directions.
    source_x, source_y : np.ndarray, optional
        Source terms P and Q evaluated on the computational grid,
        shape ``(nx + 1, ny + 1)``.  If ``None``, the homogeneous
        Laplace equation is solved (minimum-curvature grid).
    max_iter : int
        Maximum SOR iterations (default: 2000).
    tol : float
        Convergence tolerance on the L₂ norm of the update
        (default: 1e-8).
    omega : float
        SOR over-relaxation factor (default: 1.5).

    Returns
    -------
    Mesh2D
    """
    nxi = nx + 1
    neta = ny + 1

    # Set source terms to zero if not provided
    if source_x is None:
        source_x = np.zeros((nxi, neta))
    if source_y is None:
        source_y = np.zeros((nxi, neta))

    # Sample boundary data
    xi_n = np.linspace(0.0, 1.0, nxi)
    eta_n = np.linspace(0.0, 1.0, neta)

    x_s, y_s = boundaries.south(nxi)
    x_n, y_n = boundaries.north(nxi)
    x_w, y_w = boundaries.west(neta)
    x_e, y_e = boundaries.east(neta)

    # Initialise with TFI as a starting guess
    b_tfi = BoundaryCurves.from_arrays(
        x_w, y_w, x_e, y_e, x_s, y_s, x_n, y_n,
    )
    tfi_mesh = algebraic_grid(b_tfi, nx, ny)
    x = tfi_mesh.xn.copy()
    y = tfi_mesh.yn.copy()

    # Impose exact boundary values
    x[:, 0] = np.interp(xi_n, np.linspace(0, 1, nxi), x_s)
    x[:, -1] = np.interp(xi_n, np.linspace(0, 1, nxi), x_n)
    x[0, :] = np.interp(eta_n, np.linspace(0, 1, neta), x_w)
    x[-1, :] = np.interp(eta_n, np.linspace(0, 1, neta), x_e)
    y[:, 0] = np.interp(xi_n, np.linspace(0, 1, nxi), y_s)
    y[:, -1] = np.interp(xi_n, np.linspace(0, 1, nxi), y_n)
    y[0, :] = np.interp(eta_n, np.linspace(0, 1, neta), y_w)
    y[-1, :] = np.interp(eta_n, np.linspace(0, 1, neta), y_e)

    # Pre-compute metric coefficients on the computational grid
    d_xi = 1.0 / (nxi - 1)
    d_eta = 1.0 / (neta - 1)
    inv_dxi2 = 1.0 / (d_xi * d_xi)
    inv_deta2 = 1.0 / (d_eta * d_eta)

    # SOR iteration
    for iteration in range(max_iter):
        delta_norm = 0.0

        for i in range(1, nxi - 1):
            for j in range(1, neta - 1):
                # Metric coefficients (computed from current x, y)
                x_xi = 0.5 * (x[i + 1, j] - x[i - 1, j])
                x_eta = 0.5 * (x[i, j + 1] - x[i, j - 1])
                y_xi = 0.5 * (y[i + 1, j] - y[i - 1, j])
                y_eta = 0.5 * (y[i, j + 1] - y[i, j - 1])

                # Jacobian
                g11 = x_xi**2 + y_xi**2      # g_{11}
                g22 = x_eta**2 + y_eta**2     # g_{22}
                g12 = x_xi * x_eta + y_xi * y_eta  # g_{12}

                # Laplace operator in computational space
                # (central differences, accounting for cross-derivatives)
                lap_x = (x[i + 1, j] - 2.0 * x[i, j] + x[i - 1, j]) * inv_dxi2 \
                        + (x[i, j + 1] - 2.0 * x[i, j] + x[i, j - 1]) * inv_deta2
                lap_y = (y[i + 1, j] - 2.0 * y[i, j] + y[i - 1, j]) * inv_dxi2 \
                        + (y[i, j + 1] - 2.0 * y[i, j] + y[i, j - 1]) * inv_deta2

                # Source terms
                P = source_x[i, j]
                Q = source_y[i, j]

                # Update: discrete Poisson equation
                #    g22 · x_ξξ + g11 · x_ηη - 2·g12 · x_ξη
                #        + J² · (P · x_ξ + Q · x_η) = 0
                # We use a simplified update treating the cross-term
                # explicitly and solving for x[i, j].
                coeff = 2.0 * (g22 * inv_dxi2 + g11 * inv_deta2)

                if coeff > 1e-30:
                    # Cross-derivative (explicit)
                    x_xi_eta = 0.25 * (
                        x[i + 1, j + 1] - x[i + 1, j - 1]
                        - x[i - 1, j + 1] + x[i - 1, j - 1]
                    )
                    y_xi_eta = 0.25 * (
                        y[i + 1, j + 1] - y[i + 1, j - 1]
                        - y[i - 1, j + 1] + y[i - 1, j - 1]
                    )

                    # RHS
                    rhs_x = (g22 * (x[i + 1, j] + x[i - 1, j]) * inv_dxi2
                             + g11 * (x[i, j + 1] + x[i, j - 1]) * inv_deta2
                             - 2.0 * g12 * x_xi_eta)
                    rhs_y = (g22 * (y[i + 1, j] + y[i - 1, j]) * inv_dxi2
                             + g11 * (y[i, j + 1] + y[i, j - 1]) * inv_deta2
                             - 2.0 * g12 * y_xi_eta)

                    # Source contributions (using first derivatives)
                    J2 = g11 * g22 - g12**2
                    rhs_x += J2 * (P * x_xi + Q * x_eta)
                    rhs_y += J2 * (P * y_xi + Q * y_eta)

                    x_new = rhs_x / coeff
                    y_new = rhs_y / coeff

                    # SOR update
                    dx = omega * (x_new - x[i, j])
                    dy = omega * (y_new - y[i, j])
                    x[i, j] += dx
                    y[i, j] += dy

                    delta_norm = max(delta_norm, dx**2 + dy**2)

        if np.sqrt(delta_norm) < tol:
            break

    # Construct Mesh2D
    mesh = Mesh2D.__new__(Mesh2D)
    mesh.nx = nx
    mesh.ny = ny
    mesh.lx = 1.0
    mesh.ly = 1.0
    mesh.clustering_x = 0
    mesh.clustering_y = 0
    mesh.s0_x = 0.01
    mesh.s0_y = 0.01
    mesh.s1_x = 0.01
    mesh.s1_y = 0.01
    mesh.xn = x
    mesh.yn = y
    mesh.xc = 0.25 * (x[:-1, :-1] + x[1:, :-1]
                       + x[:-1, 1:] + x[1:, 1:])
    mesh.yc = 0.25 * (y[:-1, :-1] + y[1:, :-1]
                       + y[:-1, 1:] + y[1:, 1:])
    mesh.dx = np.diff(x, axis=0)
    mesh.dy = np.diff(y, axis=1)

    return mesh


# ─── Source-Term Construction ─────────────────────────────────────────

def boundary_orthogonality_source(
    mesh: Mesh2D,
    wall_side: str = "south",
    target_spacing: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct source terms P, Q for elliptic grid generation that
    enforce orthogonality and specified spacing near a boundary.

    This implements the approach of Thompson et al. (1985, §VI) for
    controlling the grid near a specified wall boundary.

    Parameters
    ----------
    mesh : Mesh2D
        Initial mesh (e.g. from TFI) used to estimate derivatives.
    wall_side : str
        One of ``"south"``, ``"north"``, ``"west"``, ``"east"``.
    target_spacing : float
        Desired normal spacing at the wall in physical units.

    Returns
    -------
    source_x, source_y : np.ndarray
        Source terms P and Q, shape ``(nx + 1, ny + 1)``.
    """
    nx, ny = mesh.nx, mesh.ny
    nxi, neta = nx + 1, ny + 1

    P = np.zeros((nxi, neta))
    Q = np.zeros((nxi, neta))

    x = mesh.xn
    y = mesh.yn

    if wall_side == "south":
        j_wall = 0
        for i in range(1, nx):
            # Tangent vector along wall
            tx = x[i + 1, j_wall] - x[i - 1, j_wall]
            ty = y[i + 1, j_wall] - y[i - 1, j_wall]
            t_norm = np.sqrt(tx**2 + ty**2) + 1e-30

            # Normal vector (pointing inward, i.e. northward)
            nx_vec = -ty / t_norm
            ny_vec = tx / t_norm

            # Desired second derivative in η-direction
            x_eta = target_spacing * nx_vec
            y_eta = target_spacing * ny_vec

            # Assign source at the wall (to drive the elliptic solver)
            P[i, j_wall] = x_eta / (target_spacing**2 + 1e-30)
            Q[i, j_wall] = y_eta / (target_spacing**2 + 1e-30)

    return P, Q


if __name__ == "__main__":
    # Demonstration: generate a grid around a circular arc (C-grid)
    import math

    # Define a C-grid around a unit circle
    # South: the body surface (circle lower half)
    # North: outer boundary (larger circle)
    # West/East: wake cut

    r_inner = 1.0
    r_outer = 5.0

    def circle_segment(r, theta_start, theta_end, n):
        theta = np.linspace(theta_start, theta_end, n)
        return r * np.cos(theta), r * np.sin(theta)

    def south_curve(n):
        return circle_segment(r_inner, -math.pi, 0, n)

    def north_curve(n):
        return circle_segment(r_outer, -math.pi, 0, n)

    def west_curve(n):
        # Straight line from ( -r_inner, 0 ) to ( -r_outer, 0 )
        y = np.linspace(0, 0, n)
        x = np.linspace(-r_inner, -r_outer, n)
        return x, y

    def east_curve(n):
        x = np.linspace(r_inner, r_outer, n)
        y = np.linspace(0, 0, n)
        return x, y

    boundaries = BoundaryCurves(west_curve, east_curve, south_curve, north_curve)

    print("Generating algebraic (TFI) C-grid...")
    mesh_tfi = algebraic_grid(boundaries, 32, 24)
    print(f"  → TFI mesh: {mesh_tfi.nx}×{mesh_tfi.ny} cells")

    print("Generating elliptic C-grid (Laplace smoothing)...")
    mesh_ell = elliptic_grid(boundaries, 32, 24, max_iter=500)
    print(f"  → Elliptic mesh: {mesh_ell.nx}×{mesh_ell.ny} cells")

    # Quality check
    from .quality import assess_mesh_quality
    q = assess_mesh_quality(mesh_ell)
    print(f"  Elliptic quality:")
    for name, r in q.items():
        print(f"    {name:20s}  min={r.minimum:.4f}  mean={r.mean:.4f}  "
              f"max={r.maximum:.4f}")
