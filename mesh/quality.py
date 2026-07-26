#!/usr/bin/env python3
"""
cfd_solver/mesh/quality.py
============================

Mesh quality metrics for 2-D structured grids.

Provides functions to evaluate the standard quality measures that
directly affect solution accuracy and numerical stability:

1. **Orthogonality** — deviation of cell face angles from 90°.
2. **Aspect ratio** — ratio of the longest to shortest cell dimension.
3. **Skewness** — deviation from a perfect parallelogram (equiangular
   skew).
4. **Expansion ratio** — ratio of adjacent cell volumes (smoothness).

Each function accepts a ``Mesh2D`` instance (or equivalently-structured
coordinate arrays) and returns a field of per-cell values together
with summary statistics.

References
----------
- Knupp, P. M. (2000). Achieving finite element mesh quality via
  optimization of the Jacobian matrix norm and associated quantities.
  *International Journal for Numerical Methods in Engineering*,
  48(8), 1165–1185.
- Thompson, J. F., Warsi, Z. U. A., & Mastin, C. W. (1985).
  *Numerical Grid Generation: Foundations and Applications*.
  North-Holland.
- Baker, T. J. (2005). Mesh generation: Art or science?
  *Progress in Aerospace Sciences*, 41(1), 29–63.

Author: James Hargreaves
        Department of Aeronautics, Imperial College London
Branch: dev/mesh-james
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mesh_2d import Mesh2D


# ─── Result container ─────────────────────────────────────────────────

@dataclass
class QualityReport:
    """
    Summary of a single mesh quality metric.

    Attributes
    ----------
    field : np.ndarray
        Per-cell values, shape ``(nx, ny)``.
    minimum : float
        Minimum value over all interior cells.
    maximum : float
        Maximum value over all interior cells.
    mean : float
        Arithmetic mean over all interior cells.
    std : float
        Standard deviation over all interior cells.
    """
    field: np.ndarray
    minimum: float
    maximum: float
    mean: float
    std: float


# ─── Internal helpers ─────────────────────────────────────────────────

def _edge_vectors(mesh: Mesh2D) -> tuple[np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray]:
    """
    Compute the four edge vectors for each cell.

    Returns
    -------
    e_s, e_n, e_w, e_e : np.ndarray
        South, north, west, east edge vectors, each shape ``(nx, ny, 2)``.
    """
    nx, ny = mesh.nx, mesh.ny
    xn, yn = mesh.xn, mesh.yn

    # Edge vectors (node-based)
    # South edge: (i, j) → (i+1, j)
    e_s = np.dstack((np.diff(xn, axis=0)[:, :-1],
                     np.diff(yn, axis=0)[:, :-1]))

    # North edge: (i, j+1) → (i+1, j+1)
    e_n = np.dstack((np.diff(xn, axis=0)[:, 1:],
                     np.diff(yn, axis=0)[:, 1:]))

    # West edge: (i, j) → (i, j+1)
    e_w = np.dstack((np.diff(xn, axis=1)[:-1, :],
                     np.diff(yn, axis=1)[:-1, :]))

    # East edge: (i+1, j) → (i+1, j+1)
    e_e = np.dstack((np.diff(xn, axis=1)[1:, :],
                     np.diff(yn, axis=1)[1:, :]))

    return e_s, e_n, e_w, e_e


# ─── Orthogonality ────────────────────────────────────────────────────

def compute_orthogonality(
    mesh: Mesh2D,
    aggregate: bool = True,
) -> QualityReport:
    """
    Compute the orthogonality angle (in degrees) for each cell.

    Orthogonality is defined as the angle between the vectors from
    the cell centre to the east and north face midpoints.  A value of
    90° indicates perfect orthogonality; values below 45° indicate
    severely skewed cells that may compromise solution accuracy.

    Parameters
    ----------
    mesh : Mesh2D
        The structured mesh to evaluate.
    aggregate : bool
        If ``True`` (default), return summary statistics.
        If ``False``, the report statistics are all set to NaN.

    Returns
    -------
    QualityReport
    """
    # Face-centre coordinates
    # East face centre = midpoint of SE and NE corners  →  shape (nx, ny)
    x_e = 0.5 * (mesh.xn[1:, :-1] + mesh.xn[1:, 1:])
    y_e = 0.5 * (mesh.yn[1:, :-1] + mesh.yn[1:, 1:])
    # North face centre = midpoint of NW and NE corners  →  shape (nx, ny)
    x_n = 0.5 * (mesh.xn[:-1, 1:] + mesh.xn[1:, 1:])
    y_n = 0.5 * (mesh.yn[:-1, 1:] + mesh.yn[1:, 1:])

    # Vectors from cell centre to face centres
    dx_e = mesh.xc - x_e
    dy_e = mesh.yc - y_e
    dx_n = mesh.xc - x_n
    dy_n = mesh.yc - y_n

    # Angle between east and north face vectors
    dot = dx_e * dx_n + dy_e * dy_n
    mag_e = np.sqrt(dx_e**2 + dy_e**2)
    mag_n = np.sqrt(dx_n**2 + dy_n**2)

    cos_theta = np.clip(dot / (mag_e * mag_n + 1e-30), -1.0, 1.0)
    # Orthogonality angle in degrees (90° = perfect)
    orth_angle = 90.0 - np.degrees(np.arccos(cos_theta))

    # Interior only for statistics
    interior = orth_angle[1:-1, 1:-1]
    return QualityReport(
        field=orth_angle,
        minimum=float(np.min(interior)) if aggregate else float("nan"),
        maximum=float(np.max(interior)) if aggregate else float("nan"),
        mean=float(np.mean(interior)) if aggregate else float("nan"),
        std=float(np.std(interior)) if aggregate else float("nan"),
    )


# ─── Aspect Ratio ─────────────────────────────────────────────────────

def compute_aspect_ratio(
    mesh: Mesh2D,
    aggregate: bool = True,
) -> QualityReport:
    """
    Compute the cell aspect ratio (maximum / minimum dimension).

    For structured grids the aspect ratio is defined as:

        AR = max(Δx_cell, Δy_cell) / min(Δx_cell, Δy_cell)

    An aspect ratio of 1.0 indicates a perfectly isotropic cell.
    Values above 10 suggest highly-stretched cells which can degrade
    iterative solver convergence.

    Parameters
    ----------
    mesh : Mesh2D
        The structured mesh to evaluate.
    aggregate : bool
        If ``True`` (default), return summary statistics.

    Returns
    -------
    QualityReport
    """
    # Cell dimensions from node coordinates
    dx_cell = 0.5 * (np.diff(mesh.xn, axis=0)[:, :-1]
                     + np.diff(mesh.xn, axis=0)[:, 1:])
    dy_cell = 0.5 * (np.diff(mesh.yn, axis=1)[:-1, :]
                     + np.diff(mesh.yn, axis=1)[1:, :])

    # Guard against zero dimensions
    dx_cell = np.maximum(dx_cell, 1e-30)
    dy_cell = np.maximum(dy_cell, 1e-30)

    aspect = np.maximum(dx_cell / dy_cell, dy_cell / dx_cell)

    interior = aspect[1:-1, 1:-1]
    return QualityReport(
        field=aspect,
        minimum=float(np.min(interior)) if aggregate else float("nan"),
        maximum=float(np.max(interior)) if aggregate else float("nan"),
        mean=float(np.mean(interior)) if aggregate else float("nan"),
        std=float(np.std(interior)) if aggregate else float("nan"),
    )


# ─── Skewness (Equiangular Skew) ──────────────────────────────────────

def compute_skewness(
    mesh: Mesh2D,
    aggregate: bool = True,
) -> QualityReport:
    """
    Compute the equiangular skewness for each cell.

    Equiangular skew is defined as:

        Q_EAS = max( (θ_max - θ_e) / (180° - θ_e),
                     (θ_e - θ_min) / θ_e )

    where θ_max, θ_min are the maximum and minimum internal angles
    of the cell, and θ_e = 90° is the ideal angle for a quadrilateral.

    * Q_EAS = 0  → perfectly regular quadrilateral.
    * Q_EAS → 1  → highly degenerate cell.

    Parameters
    ----------
    mesh : Mesh2D
        The structured mesh to evaluate.
    aggregate : bool
        If ``True`` (default), return summary statistics.

    Returns
    -------
    QualityReport
    """
    nx, ny = mesh.nx, mesh.ny
    xn, yn = mesh.xn, mesh.yn

    # Compute internal angles at the four corners of each cell
    # by evaluating the angle between edge vectors meeting at each node.

    # Node indices for cell (i, j) corners:
    #   SW = (i, j),   SE = (i+1, j)
    #   NW = (i, j+1), NE = (i+1, j+1)

    def _interior_angle(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
        """Angle (deg) between two 2-D vectors."""
        dot = va[..., 0] * vb[..., 0] + va[..., 1] * vb[..., 1]
        mag = (np.linalg.norm(va, axis=-1) * np.linalg.norm(vb, axis=-1)
               + 1e-30)
        return np.degrees(np.arccos(np.clip(dot / mag, -1.0, 1.0)))

    # SW corner (i, j): edges along +x (→ SE) and +y (→ NW)
    vx_sw = xn[1:, :-1] - xn[:-1, :-1]  # eastward
    vy_sw = yn[1:, :-1] - yn[:-1, :-1]
    ux_sw = xn[:-1, 1:] - xn[:-1, :-1]  # northward
    uy_sw = yn[:-1, 1:] - yn[:-1, :-1]
    v_sw = np.dstack((vx_sw, vy_sw))
    u_sw = np.dstack((ux_sw, uy_sw))
    ang_sw = _interior_angle(v_sw, u_sw)

    # SE corner (i+1, j): edges along -x and +y
    vx_se = xn[:-1, :-1] - xn[1:, :-1]  # westward
    vy_se = yn[:-1, :-1] - yn[1:, :-1]
    ux_se = xn[1:, 1:] - xn[1:, :-1]    # northward
    uy_se = yn[1:, 1:] - yn[1:, :-1]
    v_se = np.dstack((vx_se, vy_se))
    u_se = np.dstack((ux_se, uy_se))
    ang_se = _interior_angle(v_se, u_se)

    # NW corner (i, j+1): edges along +x and -y
    vx_nw = xn[1:, 1:] - xn[:-1, 1:]   # eastward
    vy_nw = yn[1:, 1:] - yn[:-1, 1:]
    ux_nw = xn[:-1, :-1] - xn[:-1, 1:]  # southward
    uy_nw = yn[:-1, :-1] - yn[:-1, 1:]
    v_nw = np.dstack((vx_nw, vy_nw))
    u_nw = np.dstack((ux_nw, uy_nw))
    ang_nw = _interior_angle(v_nw, u_nw)

    # NE corner (i+1, j+1): edges along -x and -y
    vx_ne = xn[:-1, 1:] - xn[1:, 1:]    # westward
    vy_ne = yn[:-1, 1:] - yn[1:, 1:]
    ux_ne = xn[1:, :-1] - xn[1:, 1:]    # southward
    uy_ne = yn[1:, :-1] - yn[1:, 1:]
    v_ne = np.dstack((vx_ne, vy_ne))
    u_ne = np.dstack((ux_ne, uy_ne))
    ang_ne = _interior_angle(v_ne, u_ne)

    # Stack angles per cell: shape (nx, ny, 4)
    angles = np.dstack((ang_sw, ang_se, ang_nw, ang_ne))

    theta_e = 90.0  # ideal quadrilateral
    ang_max = np.max(angles, axis=-1)
    ang_min = np.min(angles, axis=-1)

    skew = np.maximum(
        (ang_max - theta_e) / (180.0 - theta_e),
        (theta_e - ang_min) / theta_e,
    )

    interior = skew[1:-1, 1:-1]
    return QualityReport(
        field=skew,
        minimum=float(np.min(interior)) if aggregate else float("nan"),
        maximum=float(np.max(interior)) if aggregate else float("nan"),
        mean=float(np.mean(interior)) if aggregate else float("nan"),
        std=float(np.std(interior)) if aggregate else float("nan"),
    )


# ─── Expansion Ratio ──────────────────────────────────────────────────

def compute_expansion_ratio(
    mesh: Mesh2D,
    aggregate: bool = True,
) -> QualityReport:
    """
    Compute the maximum expansion ratio (adjacent cell-volume ratio).

    For each interior cell, the expansion ratio is the maximum of the
    volume ratios with its four neighbours (east, west, north, south).
    A value of 1.0 indicates perfectly uniform spacing; values above
    1.3 suggest excessively rapid grid stretching.

    Parameters
    ----------
    mesh : Mesh2D
        The structured mesh to evaluate.
    aggregate : bool
        If ``True`` (default), return summary statistics.

    Returns
    -------
    QualityReport
    """
    vol = mesh.cell_volumes()

    # Compute neighbour ratios
    re = np.zeros_like(vol)
    rw = np.zeros_like(vol)
    rn = np.zeros_like(vol)
    rs = np.zeros_like(vol)

    re[:, :-1] = np.maximum(vol[:, 1:] / np.maximum(vol[:, :-1], 1e-30),
                             vol[:, :-1] / np.maximum(vol[:, 1:], 1e-30))
    rw[:, 1:] = re[:, :-1]  # symmetric
    rn[:-1, :] = np.maximum(vol[1:, :] / np.maximum(vol[:-1, :], 1e-30),
                             vol[:-1, :] / np.maximum(vol[1:, :], 1e-30))
    rs[1:, :] = rn[:-1, :]

    expansion = np.maximum.reduce([re, rw, rn, rs])

    interior = expansion[1:-1, 1:-1]
    return QualityReport(
        field=expansion,
        minimum=float(np.min(interior)) if aggregate else float("nan"),
        maximum=float(np.max(interior)) if aggregate else float("nan"),
        mean=float(np.mean(interior)) if aggregate else float("nan"),
        std=float(np.std(interior)) if aggregate else float("nan"),
    )


# ─── Combined Quality Assessment ──────────────────────────────────────

def assess_mesh_quality(mesh: Mesh2D) -> dict[str, QualityReport]:
    """
    Run the full suite of quality metrics on a mesh and return a report.

    Parameters
    ----------
    mesh : Mesh2D
        The structured mesh to evaluate.

    Returns
    -------
    dict[str, QualityReport]
        Keys: ``"orthogonality"``, ``"aspect_ratio"``, ``"skewness"``,
        ``"expansion_ratio"``.
    """
    return {
        "orthogonality": compute_orthogonality(mesh),
        "aspect_ratio": compute_aspect_ratio(mesh),
        "skewness": compute_skewness(mesh),
        "expansion_ratio": compute_expansion_ratio(mesh),
    }


if __name__ == "__main__":
    from .mesh_2d import uniform_mesh, clustered_mesh

    mesh_u = uniform_mesh(32, 32)
    report = assess_mesh_quality(mesh_u)
    print("Uniform 32×32 mesh:")
    for name, q in report.items():
        print(f"  {name:20s}  min={q.minimum:.4f}  max={q.maximum:.4f}  "
              f"mean={q.mean:.4f}  σ={q.std:.4f}")

    mesh_c = clustered_mesh(32, 32, s0_x=1e-3, s0_y=1e-3, both_ends=True)
    report_c = assess_mesh_quality(mesh_c)
    print("\nClustered 32×32 mesh (s0=1e-3, both ends):")
    for name, q in report_c.items():
        print(f"  {name:20s}  min={q.minimum:.4f}  max={q.maximum:.4f}  "
              f"mean={q.mean:.4f}  σ={q.std:.4f}")
