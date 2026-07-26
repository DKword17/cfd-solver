"""
cfd_solver/mesh/__init__.py
============================

Mesh generation, transformation, adaptation, quality assessment, and I/O
for the CFD solver's 2D structured-grid framework.

Modules
-------
mesh_2d      — Structured grid generation (uniform and hyperbolic-tangent clustering).
mesh_adapt   — Gradient- and curvature-based adaptive mesh refinement.
transform    — Body-fitted coordinate generation (algebraic TFI and elliptic).
quality      — Mesh quality metrics (orthogonality, aspect ratio, skewness).
io           — CGNS and PLOT3D format import / export.

Author: James Hargreaves
        Department of Aeronautics, Imperial College London
Branch: dev/mesh-james
"""

from .mesh_2d import Mesh2D, ClusteringType
from .quality import compute_orthogonality, compute_aspect_ratio, compute_skewness
from .io import read_plot3d_grid, write_plot3d_grid

__all__ = [
    "Mesh2D",
    "ClusteringType",
    "compute_orthogonality",
    "compute_aspect_ratio",
    "compute_skewness",
    "read_plot3d_grid",
    "write_plot3d_grid",
]
