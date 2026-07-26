"""
mesh/mesh_2d.py

2D structured mesh generation for finite-volume CFD.

Provides:
    - Uniform Cartesian meshes
    - Hyperbolic tangent clustering (wall-normal stretching)
    - General structured block interface
    - Cell volume and face area computation

Reference:
    Thompson, J. F., Warsi, Z. U. A., &amp; Mastin, C. W. (1985).
    *Numerical Grid Generation: Foundations and Applications*.
    North-Holland.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh2D:
    """
    A 2D structured quadrilateral mesh.

    Attributes:
        nx:       Number of cells in the xi (i) direction
        ny:       Number of cells in the eta (j) direction
        x:        Node coordinates [nx+1, ny+1, 2] — (x, y) per node
        dx:       Uniform spacing approximation (mean dx)
        dy:       Uniform spacing approximation (mean dy)
    """
    nx: int
    ny: int
    x: np.ndarray = field(init=False)       # shape (nx+1, ny+1, 2)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def cell_volume(self, i: int, j: int) -> float:
        """
        Compute the volume (area) of quadrilateral cell (i, j).

        Uses the cross-product of the two diagonals for a general
        quadrilateral, giving O(h²) accuracy for mildly distorted meshes.

        Parameters:
            i: Cell index in xi-direction (0 .. nx-1)
            j: Cell index in eta-direction (0 .. ny-1)

        Returns:
            Area of the cell.
        """
        x = self.x
        # Four nodes of cell (i, j) — counterclockwise
        x1, y1 = x[i,   j]
        x2, y2 = x[i+1, j]
        x3, y3 = x[i+1, j+1]
        x4, y4 = x[i,   j+1]

        # Shoelace formula
        area = 0.5 * abs(
            x1*y2 + x2*y3 + x3*y4 + x4*y1
            - (y1*x2 + y2*x3 + y3*x4 + y4*x1)
        )
        return area

    def face_area_xi(self, i: int, j: int) -> tuple[float, float]:
        """
        Area vector (nx, ny) for the xi = constant face at (i+1/2, j).
        The face runs from node (i+1, j) to node (i+1, j+1).
        """
        dx = self.x[i+1, j+1, 0] - self.x[i+1, j, 0]
        dy = self.x[i+1, j+1, 1] - self.x[i+1, j, 1]
        return (dy, -dx)   # Outward normal (rotated 90°)

    def face_area_eta(self, i: int, j: int) -> tuple[float, float]:
        """
        Area vector for the eta = constant face at (i, j+1/2).
        The face runs from node (i, j+1) to node (i+1, j+1).
        """
        dx = self.x[i+1, j+1, 0] - self.x[i, j+1, 0]
        dy = self.x[i+1, j+1, 1] - self.x[i, j+1, 1]
        return (-dy, dx)


def create_uniform_mesh(nx: int, ny: int,
                        xmin: float = 0.0, xmax: float = 1.0,
                        ymin: float = 0.0, ymax: float = 1.0) -> Mesh2D:
    """
    Create a uniform Cartesian mesh.

    All cells are rectangles of equal size.  Useful for verification
    and method-of-manufactured-solutions testing.

    Parameters:
        nx:    Number of cells in the x-direction
        ny:    Number of cells in the y-direction
        xmin:  Domain minimum x
        xmax:  Domain maximum x
        ymin:  Domain minimum y
        ymax:  Domain maximum y

    Returns:
        A Mesh2D instance with uniform spacing.
    """
    mesh = Mesh2D(nx=nx, ny=ny)
    xv = np.linspace(xmin, xmax, nx + 1)
    yv = np.linspace(ymin, ymax, ny + 1)
    X, Y = np.meshgrid(xv, yv, indexing='ij')
    mesh.x = np.stack([X, Y], axis=-1)
    return mesh


def create_clustered_mesh(nx: int, ny: int,
                          xmin: float = 0.0, xmax: float = 1.0,
                          ymin: float = 0.0, ymax: float = 1.0,
                          beta: float = 1.5,
                          wall: str = 'bottom') -> Mesh2D:
    """
    Create a mesh with hyperbolic tangent clustering near a wall.

    Uses Vinokur's stretching function (Vinokur, 1983, JCP 50:215):

        s(eta) = 1 + tanh(beta * (eta - 1)) / tanh(beta)

    where eta in [0, 1] is the normalised coordinate and beta
    controls the clustering强度.  Higher beta = more clustering.

    Parameters:
        nx:    Number of cells in the x-direction
        ny:    Number of cells in the y-direction
        beta:  Stretching parameter (1.0 = mild, 3.0 = severe)
        wall:  'bottom' or 'top' — which boundary to cluster towards

    Returns:
        A Mesh2D with refined spacing near the specified wall.
    """
    mesh = Mesh2D(nx=nx, ny=ny)
    xv = np.linspace(xmin, xmax, nx + 1)

    # Normalised coordinate
    eta = np.linspace(0, 1, ny + 1)

    # Vinokur stretching
    tanh_beta = math.tanh(beta)
    s = 1.0 + np.tanh(beta * (eta - 1.0)) / tanh_beta
    s = (s - s[0]) / (s[-1] - s[0])  # Normalise to [0, 1]

    if wall == 'top':
        s = 1.0 - s  # Flip

    yv = ymin + (ymax - ymin) * s
    X, Y = np.meshgrid(xv, yv, indexing='ij')
    mesh.x = np.stack([X, Y], axis=-1)
    return mesh
