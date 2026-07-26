"""
mesh/transform.py

Coordinate transformations for body-fitted mesh generation.

Implements algebraic and elliptic grid generation methods:
    - Algebraic: transfinite interpolation (TFI)
    - Elliptic: Winslow generator (Poisson system with control functions)

Reference:
    Thompson, J. F., Soni, B. K., & Weatherill, N. P. (1999).
    *Handbook of Grid Generation*. CRC Press.
"""

from __future__ import annotations

import math

import numpy as np

from .mesh_2d import Mesh2D


def elliptic_c_grid(nx: int, ny: int,
                    r_min: float = 1.0, r_max: float = 5.0,
                    theta_max: float = 360.0,
                    n_iter: int = 1000) -> Mesh2D:
    """
    Generate an elliptic O-type (C-type) grid around a cylinder.

    The physical domain is an annulus sector: r in [r_min, r_max],
    theta in [0, theta_max] degrees.  The elliptic solver smooths
    the algebraically-initialised grid, producing near-orthogonal
    cells desirable for boundary-layer resolution.

    Parameters:
        nx:       Number of cells radially
        ny:       Number of cells circumferentially
        r_min:    Inner radius (cylinder surface)
        r_max:    Outer radius (far-field boundary)
        theta_max: Angular extent in degrees (360 = full circle)
        n_iter:   SOR iterations for elliptic smoothing

    Returns:
        A Mesh2D with the elliptic C-grid.
    """
    # Algebraic initialisation
    mesh = Mesh2D(nx=nx, ny=ny)
    mesh.x = np.zeros((nx + 1, ny + 1, 2))

    theta_rad = math.radians(theta_max)
    for i in range(nx + 1):
        r = r_min + (r_max - r_min) * (i / nx)
        for j in range(ny + 1):
            theta = theta_rad * j / ny
            mesh.x[i, j, 0] = r * math.cos(theta)
            mesh.x[i, j, 1] = r * math.sin(theta)

    # Elliptic smoothing (Winslow generator via SOR)
    # Solve: x_ξξ + x_ηη = 0,  y_ξξ + y_ηη = 0
    omega = 1.5  # Over-relaxation factor

    for _ in range(n_iter):
        max_change = 0.0

        for i in range(1, nx):
            for j in range(1, ny):
                for d in range(2):
                    lap = (mesh.x[i+1, j, d] + mesh.x[i-1, j, d]
                           + mesh.x[i, j+1, d] + mesh.x[i, j-1, d])
                    old = mesh.x[i, j, d]
                    new = 0.25 * lap
                    mesh.x[i, j, d] = (1 - omega) * old + omega * new

                    change = abs(mesh.x[i, j, d] - old)
                    if change > max_change:
                        max_change = change

        if max_change < 1e-8:
            break

    return mesh


def transfinite_interpolation(west: np.ndarray, east: np.ndarray,
                               south: np.ndarray, north: np.ndarray,
                               nx: int, ny: int) -> Mesh2D:
    """
    Generate a structured mesh via transfinite interpolation (TFI).

    Given four boundary curves (each defined at the four edges
    of the computational domain), TFI blends them to produce
    a smooth interior mesh.

    The blending functions are linear:
        phi(xi)  = 1 - xi
        psi(eta) = 1 - eta

    for computational coordinates xi, eta in [0, 1] × [0, 1].

    Parameters:
        west, east:   Arrays of shape (ny+1, 2) — curves at xi=0, xi=1
        south, north: Arrays of shape (nx+1, 2) — curves at eta=0, eta=1
        nx:           Number of interior cells in xi
        ny:           Number of interior cells in eta

    Returns:
        A Mesh2D with the interpolated grid.
    """
    mesh = Mesh2D(nx=nx, ny=ny)
    mesh.x = np.zeros((nx + 1, ny + 1, 2))

    for i in range(nx + 1):
        xi = i / nx
        phi = 1 - xi

        for j in range(ny + 1):
            eta = j / ny
            psi = 1 - eta

            # Bilinear blending
            mesh.x[i, j] = (
                phi * west[j] + xi * east[j]
                + psi * south[i] + eta * north[i]
                - phi * psi * south[0] - phi * eta * north[0]
                - xi * psi * south[nx] - xi * eta * north[nx]
            ) / 1.0

    return mesh
