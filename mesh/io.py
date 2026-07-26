"""
mesh/io.py

Mesh I/O routines for standard CFD formats.

Supports:
    - PLOT3D (ASCII, structured-block): standard NASA format
    - CGNS (via h5py): CFD General Notation System (skeleton)

Reference:
    NASA Ames Research Center. "PLOT3D File Format Description."
    https://www.nas.nasa.gov/publications/plot3d.html
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .mesh_2d import Mesh2D


def write_plot3d(mesh: Mesh2D, path: str) -> None:
    """
    Write a 2D structured mesh in PLOT3D ASCII format.

    PLOT3D format header (for a single block 2D mesh):
        nx+1, ny+1
        x(1,1) ... x(nx+1,ny+1)
        y(1,1) ... y(nx+1,ny+1)

    All values are space-delimited, 6 values per line.

    Parameters:
        mesh: Mesh to write
        path: Output file path
    """
    nx, ny = mesh.nx, mesh.ny
    x = mesh.x

    lines = [f"{nx+1} {ny+1}"]

    # Write x-coordinates
    row_vals = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            row_vals.append(f"{x[i, j, 0]:.12e}")
            if len(row_vals) == 6:
                lines.append(" ".join(row_vals))
                row_vals = []
    if row_vals:
        lines.append(" ".join(row_vals))

    # Write y-coordinates
    row_vals = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            row_vals.append(f"{x[i, j, 1]:.12e}")
            if len(row_vals) == 6:
                lines.append(" ".join(row_vals))
                row_vals = []
    if row_vals:
        lines.append(" ".join(row_vals))

    Path(path).write_text("\n".join(lines))


def read_plot3d(path: str) -> Mesh2D:
    """
    Read a 2D structured mesh from PLOT3D ASCII format.

    Parameters:
        path: Input file path

    Returns:
        A Mesh2D instance.

    Raises:
        ValueError: If the file cannot be parsed.
    """
    text = Path(path).read_text().strip()
    tokens = text.split()

    if len(tokens) < 2:
        raise ValueError("PLOT3D file too short")

    nx = int(tokens[0]) - 1
    ny = int(tokens[1]) - 1

    n_nodes = (nx + 1) * (ny + 1)
    expected = 2 + 2 * n_nodes

    if len(tokens) < expected:
        raise ValueError(
            f"PLOT3D file truncated: expected {expected} tokens, "
            f"got {len(tokens)}"
        )

    mesh = Mesh2D(nx=nx, ny=ny)
    mesh.x = np.zeros((nx + 1, ny + 1, 2))

    idx = 2
    for d in range(2):
        for i in range(nx + 1):
            for j in range(ny + 1):
                mesh.x[i, j, d] = float(tokens[idx])
                idx += 1

    return mesh
