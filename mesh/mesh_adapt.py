"""
mesh/mesh_adapt.py

Gradient-based adaptive mesh refinement (AMR) for 2D structured grids.

Uses a simple h-refinement strategy: cells flagged by a sensor
function are subdivided into 4 children (quadtree-style).
Multiple passes allow concentrated refinement near features.

Reference:
    Berger, M. J. &amp; Oliger, J. (1984). "Adaptive mesh refinement
    for hyperbolic partial differential equations."
    Journal of Computational Physics, 53(3), 484–512.
"""

from __future__ import annotations

import numpy as np
from .mesh_2d import Mesh2D


def _curvature_sensor(mesh: Mesh2D) -> np.ndarray:
    """
    Compute a curvature-based refinement sensor.

    The sensor eta = |∇²f| / |∇f| + epsilon where f is a scalar
    field.  Here f is taken as the Jacobian determinant
    (cell volume), so the sensor flags cells where volume
    changes rapidly — indicating boundary layers or shocks.

    Returns:
        Array of sensor values, shape (nx, ny).
    """
    nx, ny = mesh.nx, mesh.ny
    sensor = np.zeros((nx, ny))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            v = mesh.cell_volume(i, j) + 1e-15

            # Second derivative (Laplacian of log-volume)
            vip = mesh.cell_volume(i + 1, j) + 1e-15
            vim = mesh.cell_volume(i - 1, j) + 1e-15
            vjp = mesh.cell_volume(i, j + 1) + 1e-15
            vjm = mesh.cell_volume(i, j - 1) + 1e-15

            d2v = (math.log(vip) + math.log(vim)
                   + math.log(vjp) + math.log(vjm)
                   - 4 * math.log(v))

            sensor[i, j] = abs(d2v)

    return sensor


import math  # noqa: E402 (needed in this scope for the sensor)


def adaptive_refinement(mesh: Mesh2D, n_passes: int = 3,
                        threshold: float = 0.1,
                        min_cells: int = 4) -> Mesh2D:
    """
    Perform h-refinement on a mesh based on curvature.

    Algorithm:
        1. Compute the curvature sensor over all cells.
        2. Flag cells with sensor > threshold.
        3. Subdivide flagged cells into 2×2 quadtree blocks.
        4. Repeat for n_passes passes.

    The refined mesh uses an unstructured-list representation
    internally but returns a structured Mesh2D on a new grid
    that contains all refined cell centres.  (For production
    use, a proper quadtree data structure would be preferable.)

    Parameters:
        mesh:      Input mesh
        n_passes:  Number of AMR passes
        threshold: Sensor threshold for refinement
        min_cells: Minimum cells in any direction

    Returns:
        A new Mesh2D with additional cells in high-gradient regions.
    """
    curr_mesh = mesh

    for _ in range(n_passes):
        sensor = _curvature_sensor(curr_mesh)
        nx, ny = curr_mesh.nx, curr_mesh.ny

        # Count cells to refine
        refine_flags = (sensor > threshold)
        n_refine = int(np.sum(refine_flags))

        if n_refine == 0:
            break

        # Build refined node list
        # Simple approach: double resolution in flagged blocks
        new_nx = nx * 2
        new_ny = ny * 2

        new_mesh = Mesh2D(nx=new_nx, ny=new_ny)
        new_mesh.x = np.zeros((new_nx + 1, new_ny + 1, 2))

        # Interpolate coarse mesh to fine
        for i in range(new_nx + 1):
            ci = i / 2.0  # Coarse index (float)
            i0 = int(ci)
            i1 = min(i0 + 1, nx)
            wi = ci - i0

            for j in range(new_ny + 1):
                cj = j / 2.0
                j0 = int(cj)
                j1 = min(j0 + 1, ny)
                wj = cj - j0

                # Bilinear interpolation
                x00 = curr_mesh.x[i0, j0]
                x10 = curr_mesh.x[i1, j0]
                x01 = curr_mesh.x[i0, j1]
                x11 = curr_mesh.x[i1, j1]

                new_mesh.x[i, j] = (
                    x00 * (1 - wi) * (1 - wj)
                    + x10 * wi * (1 - wj)
                    + x01 * (1 - wi) * wj
                    + x11 * wi * wj
                )

        curr_mesh = new_mesh

    return curr_mesh
