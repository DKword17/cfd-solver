"""
mesh/quality.py

Metrics for assessing the quality of a 2D structured mesh.

Poor quality cells (extreme skewness, very high aspect ratio,
or near-zero orthogonality) degrade solver accuracy and stability.
These routines flag such cells before the flow solver runs.

Reference:
    Knupp, P. M. (2007). "Algebraic mesh quality metrics."
    Engineering with Computers, 23(2), 109–115.
"""

from __future__ import annotations

import numpy as np

from .mesh_2d import Mesh2D


def compute_orthogonality(mesh: Mesh2D) -> np.ndarray:
    """
    Compute the orthogonality angle (degrees) for each cell.

    Defined as the deviation of the mesh from 90° at each cell
    centre, measured between the xi and eta grid lines.

    For a cell (i, j), the two direction vectors are:

        r_xi  = x[i+1,j] - x[i-1,j]
        r_eta = x[i,j+1] - x[i,j-1]

    Orthogonality = arccos(|r_xi · r_eta| / |r_xi| |r_eta|)
    converted from radians to degrees.  A value of 90° is ideal;
    values below 45° indicate problematic cells.

    Parameters:
        mesh: A 2D structured Mesh2D instance.

    Returns:
        Array of orthogonality angles in degrees, shape (nx, ny).
    """
    nx, ny = mesh.nx, mesh.ny
    x = mesh.x
    orth = np.full((nx, ny), 90.0)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # xi-direction vector (central difference)
            r_xi = np.array([
                x[i+1, j, 0] - x[i-1, j, 0],
                x[i+1, j, 1] - x[i-1, j, 1]
            ])
            # eta-direction vector
            r_eta = np.array([
                x[i, j+1, 0] - x[i, j-1, 0],
                x[i, j+1, 1] - x[i, j-1, 1]
            ])

            norm_r_xi = np.linalg.norm(r_xi)
            norm_r_eta = np.linalg.norm(r_eta)

            if norm_r_xi < 1e-15 or norm_r_eta < 1e-15:
                orth[i, j] = 0.0  # Degenerate
            else:
                cos_theta = abs(np.dot(r_xi, r_eta)) / (norm_r_xi * norm_r_eta)
                cos_theta = min(cos_theta, 1.0)
                orth[i, j] = np.degrees(np.arccos(cos_theta))

    return orth


def compute_aspect_ratio(mesh: Mesh2D) -> np.ndarray:
    """
    Compute the aspect ratio (AR) for each cell.

    AR = longest edge / shortest edge.

    For structured CFD, AR up to 1000 is acceptable in boundary
    layers, but sudden changes in AR across adjacent cells cause
    truncation error.  Ideally AR is close to unity in isotropic
    regions.

    Parameters:
        mesh: A 2D structured Mesh2D instance.

    Returns:
        Array of aspect ratios, shape (nx, ny).
    """
    nx, ny = mesh.nx, mesh.ny
    x = mesh.x
    ar = np.ones((nx, ny))

    for i in range(nx):
        for j in range(ny):
            edges = [
                np.linalg.norm(x[i+1, j] - x[i, j]),      # South
                np.linalg.norm(x[i+1, j+1] - x[i, j+1]),  # North
                np.linalg.norm(x[i, j+1] - x[i, j]),      # West
                np.linalg.norm(x[i+1, j+1] - x[i+1, j]),  # East
            ]
            ar[i, j] = max(edges) / (min(edges) + 1e-15)

    return ar


def compute_skewness(mesh: Mesh2D) -> np.ndarray:
    """
    Compute the skewness angle (deviation from 90°) for each cell.

    Uses the included angle method: skewness is the maximum
    deviation of any interior angle from 90°.

    Ideal = 0°, poor > 45°, invalid > 90°.

    Parameters:
        mesh: A 2D structured Mesh2D instance.

    Returns:
        Array of skewness values in degrees, shape (nx, ny).
    """
    nx, ny = mesh.nx, mesh.ny
    x = mesh.x
    skew = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            # Four corners of cell (i,j)
            corners = [
                (i, j), (i+1, j), (i+1, j+1), (i, j+1)
            ]
            angles = []

            for k in range(4):
                p0 = corners[k]
                p1 = corners[(k + 1) % 4]
                p2 = corners[(k - 1) % 4]

                v1 = x[p1[0], p1[1]] - x[p0[0], p0[1]]
                v2 = x[p2[0], p2[1]] - x[p0[0], p0[1]]

                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)

                if n1 > 1e-15 and n2 > 1e-15:
                    cos_theta = np.dot(v1, v2) / (n1 * n2)
                    cos_theta = max(-1.0, min(1.0, cos_theta))
                    theta = np.degrees(np.arccos(cos_theta))
                    angles.append(theta)

            if angles:
                skew[i, j] = max(abs(a - 90.0) for a in angles)

    return skew
