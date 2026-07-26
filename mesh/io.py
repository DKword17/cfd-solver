#!/usr/bin/env python3
"""
cfd_solver/mesh/io.py
========================

Mesh I/O for structured 2-D grids — CGNS and PLOT3D formats.

Two formats are supported:

1. **PLOT3D** (ASCII) — the classic NASA format for structured grids
   (Walatka et al., 1990).  Both the grid file (``.xyz`` or ``.grd``)
   and the solution file (``.q``) are implemented.  This is a simple,
   portable format that requires no external dependencies.

2. **CGNS** — the CFD General Notation System (Poinot et al., 2020),
   an industry-standard format based on HDF5.  Requires ``h5py``.
   If ``h5py`` is not available, a warning is issued and CGNS I/O
   is disabled.

All functions accept and return ``Mesh2D`` instances for consistency
with the rest of the mesh module.

References
----------
- Walatka, P. P., Buning, P. G., Pierce, L., & Elson, P. A. (1990).
  *PLOT3D User's Manual*. NASA Technical Memorandum 101067.
- Poinot, M., Rumsey, C., & Mani, M. (2020). CGNS: The CFD General
  Notation System. *AIAA Paper 2020-1553*.

Author: James Hargreaves
        Department of Aeronautics, Imperial College London
Branch: dev/mesh-james
"""

from __future__ import annotations

import os
import struct
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .mesh_2d import Mesh2D

# ─── Optional CGNS / HDF5 support ────────────────────────────────────

_HDF5_AVAILABLE: bool = False
try:
    import h5py
    _HDF5_AVAILABLE = True
except ImportError:
    warnings.warn(
        "h5py is not installed — CGNS I/O will be unavailable.  "
        "Install with: pip install h5py"
    )


# ╔════════════════════════════════════════════════════════════════════╗
# ║  PLOT3D Format                                                    ║
# ╚════════════════════════════════════════════════════════════════════╝

def read_plot3d_grid(filepath: Union[str, Path]) -> Mesh2D:
    """
    Read a 2-D structured grid from a PLOT3D ASCII file.

    The file is expected to contain:

        nx  ny
        x(0,0)  x(1,0)  ...  x(nx,0)
        x(0,1)  x(1,1)  ...  x(nx,1)
        ...
        y(0,0)  y(1,0)  ...  y(nx,0)
        ...

    in standard PLOT3D node-ordering: the i-index (ξ) varies fastest,
    the j-index (η) slowest.

    Parameters
    ----------
    filepath : str or Path
        Path to the PLOT3D grid file (``.xyz``, ``.grd``, or ``.p3d``).

    Returns
    -------
    Mesh2D
    """
    filepath = Path(filepath)
    data = np.loadtxt(filepath, skiprows=1)

    with open(filepath, "r") as f:
        header = f.readline().strip()
    dims = [int(d) for d in header.split()]
    if len(dims) == 2:
        nxi, neta = dims
    elif len(dims) == 4:
        # Multi-block: take the first block
        nxi, neta = dims[0], dims[1]
    else:
        msg = f"Unrecognised PLOT3D header: {header}"
        raise ValueError(msg)

    nx = nxi - 1
    ny = neta - 1

    n_expected = 2 * nxi * neta
    data = data.ravel()
    if data.size != n_expected:
        msg = (
            f"Data size mismatch: expected {n_expected} values "
            f"(2 × {nxi} × {neta}), got {data.size}."
        )
        raise ValueError(msg)

    # PLOT3D stores nodes: all x values (i-fast, j-slow), then all y values
    x_flat = data[:nxi * neta]
    y_flat = data[nxi * neta:]

    xn = x_flat.reshape((nxi, neta), order="F")  # Fortran ordering
    yn = y_flat.reshape((nxi, neta), order="F")

    return _build_from_nodes(xn, yn)


def write_plot3d_grid(
    mesh: Mesh2D,
    filepath: Union[str, Path],
    precision: int = 12,
) -> None:
    """
    Write a 2-D structured grid to a PLOT3D ASCII file.

    Parameters
    ----------
    mesh : Mesh2D
        The mesh to write.
    filepath : str or Path
        Output file path (conventionally ``.xyz`` or ``.grd``).
    precision : int
        Number of decimal places for coordinate output (default: 12).
    """
    nxi, neta = mesh.xn.shape
    fmt = f"%.{precision}e"

    with open(filepath, "w") as f:
        # Header
        f.write(f"{nxi} {neta}\n")
        # All x nodes (Fortran ordering: i varies fastest)
        for j in range(neta):
            np.savetxt(f, mesh.xn[:, j:j + 1].T, fmt=fmt)
        # All y nodes
        for j in range(neta):
            np.savetxt(f, mesh.yn[:, j:j + 1].T, fmt=fmt)


def read_plot3d_solution(
    filepath: Union[str, Path],
    n_variables: int = 5,
) -> dict[str, np.ndarray]:
    """
    Read a PLOT3D solution (``.q``) file.

    The solution file stores primitive variables in the order:
    density, u-velocity, v-velocity, w-velocity, pressure, ...
    (For 2-D, w-velocity is typically zero.)

    Parameters
    ----------
    filepath : str or Path
        Path to the PLOT3D solution file.
    n_variables : int
        Number of variables per node (default: 5).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``"density"``, ``"u"``, ``"v"``, ``"w"``,
        ``"pressure"``, and optionally extra variables as ``"var_5"``,
        ``"var_6"``, etc.
    """
    filepath = Path(filepath)
    data = np.loadtxt(filepath, skiprows=1)

    with open(filepath, "r") as f:
        header = f.readline().strip()
    dims = [int(d) for d in header.split()]
    if len(dims) == 2:
        nxi, neta = dims
    else:
        nxi, neta = dims[0], dims[1]

    nnodes = nxi * neta
    data = data.ravel()

    if data.size != nnodes * n_variables:
        msg = (
            f"Data size mismatch: expected {nnodes * n_variables} "
            f"values, got {data.size}."
        )
        raise ValueError(msg)

    var_names = ["density", "u", "v", "w", "pressure"]
    result = {}
    for k in range(min(n_variables, 5)):
        arr = data[k * nnodes:(k + 1) * nnodes].reshape((nxi, neta), order="F")
        result[var_names[k]] = arr
    for k in range(5, n_variables):
        arr = data[k * nnodes:(k + 1) * nnodes].reshape((nxi, neta), order="F")
        result[f"var_{k}"] = arr

    return result


def write_plot3d_solution(
    filepath: Union[str, Path],
    mesh: Mesh2D,
    density: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pressure: np.ndarray,
    w: Optional[np.ndarray] = None,
    precision: int = 12,
) -> None:
    """
    Write a 2-D solution to a PLOT3D ``.q`` file.

    Variables are written at **nodes** in the order:
    ρ, u, v, w (or 0), p.

    Parameters
    ----------
    filepath : str or Path
        Output solution file path.
    mesh : Mesh2D
        The mesh (used for dimensions only).
    density, u, v, pressure : np.ndarray
        Solution fields at **nodes**, shape ``(nx + 1, ny + 1)``.
    w : np.ndarray, optional
        Spanwise velocity (zeros if ``None``).
    precision : int
        Decimal places (default: 12).
    """
    nxi, neta = mesh.xn.shape
    fmt = f"%.{precision}e"

    if w is None:
        w = np.zeros_like(density)

    with open(filepath, "w") as f:
        f.write(f"{nxi} {neta}\n")

        for arr in [density, u, v, w, pressure]:
            for j in range(neta):
                np.savetxt(f, arr[:, j:j + 1].T, fmt=fmt)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  CGNS Format (via HDF5)                                           ║
# ╚════════════════════════════════════════════════════════════════════╝

CGNS_PATH = "/GridCoordinates"
CGNS_X_NAME = "CoordinateX"
CGNS_Y_NAME = "CoordinateY"


def write_cgns_grid(
    mesh: Mesh2D,
    filepath: Union[str, Path],
    base_name: str = "Base",
    zone_name: str = "Zone_1",
) -> None:
    """
    Write a 2-D structured grid to CGNS format.

    The file is written using the HDF5-based CGNS storage convention.
    Requires ``h5py``.

    Parameters
    ----------
    mesh : Mesh2D
        The mesh to write.
    filepath : str or Path
        Output file path (``.cgns`` or ``.h5``).
    base_name : str
        CGNS database name (default: ``"Base"``).
    zone_name : str
        CGNS zone name (default: ``"Zone_1"``).
    """
    if not _HDF5_AVAILABLE:
        msg = (
            "Cannot write CGNS: h5py is not installed.  "
            "Use write_plot3d_grid() instead."
        )
        raise ImportError(msg)

    filepath = Path(filepath)
    nxi, neta = mesh.xn.shape

    with h5py.File(filepath, "w") as f:
        # CGNS root attributes (required by standard)
        f.attrs["CGNS version"] = np.float32(3.4)
        f.attrs["dimensions"] = np.int32(2)

        # Base node
        base = f.create_group(base_name)
        base.attrs["type"] = np.array([b"Structured"], dtype="S12")

        # Zone node
        zone_dims = np.array([nxi, neta, 0], dtype=np.int32)
        zone = base.create_group(zone_name)
        zone.create_dataset("ZoneType", data=np.bytes_("Structured"))
        zone.create_dataset("GridDimensions", data=zone_dims)

        # Grid coordinates
        coords = zone.create_group("GridCoordinates")
        coords.create_dataset(
            "CoordinateX",
            data=mesh.xn,
            dtype=np.float64,
        )
        coords.create_dataset(
            "CoordinateY",
            data=mesh.yn,
            dtype=np.float64,
        )

        # Add minimal metadata
        coords["CoordinateX"].attrs["units"] = np.bytes_("meters")
        coords["CoordinateY"].attrs["units"] = np.bytes_("meters")


def read_cgns_grid(
    filepath: Union[str, Path],
    base_name: str = "Base",
    zone_name: str = "Zone_1",
) -> Mesh2D:
    """
    Read a 2-D structured grid from a CGNS file.

    Requires ``h5py``.

    Parameters
    ----------
    filepath : str or Path
        Input CGNS file (``.cgns`` or ``.h5``).
    base_name : str
        CGNS database name.
    zone_name : str
        CGNS zone name.

    Returns
    -------
    Mesh2D
    """
    if not _HDF5_AVAILABLE:
        msg = (
            "Cannot read CGNS: h5py is not installed.  "
            "Use read_plot3d_grid() instead."
        )
        raise ImportError(msg)

    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        zone_path = f"{base_name}/{zone_name}"
        coords_path = f"{zone_path}/GridCoordinates"

        xn = f[f"{coords_path}/{CGNS_X_NAME}"][:]
        yn = f[f"{coords_path}/{CGNS_Y_NAME}"][:]

    return _build_from_nodes(xn, yn)


# ─── Internal helpers ─────────────────────────────────────────────────

def _build_from_nodes(xn: np.ndarray, yn: np.ndarray) -> Mesh2D:
    """
    Construct a ``Mesh2D`` from node coordinate arrays.

    Parameters
    ----------
    xn, yn : np.ndarray
        Node coordinates, shape ``(nx + 1, ny + 1)``.

    Returns
    -------
    Mesh2D
    """
    nxi, neta = xn.shape
    nx = nxi - 1
    ny = neta - 1

    mesh = Mesh2D.__new__(Mesh2D)
    mesh.nx = nx
    mesh.ny = ny
    mesh.lx = float(xn[-1, 0] - xn[0, 0])
    mesh.ly = float(yn[0, -1] - yn[0, 0])
    mesh.clustering_x = 0
    mesh.clustering_y = 0
    mesh.s0_x = 0.01
    mesh.s0_y = 0.01
    mesh.s1_x = 0.01
    mesh.s1_y = 0.01
    mesh.xn = xn
    mesh.yn = yn
    mesh.xc = 0.25 * (xn[:-1, :-1] + xn[1:, :-1]
                       + xn[:-1, 1:] + xn[1:, 1:])
    mesh.yc = 0.25 * (yn[:-1, :-1] + yn[1:, :-1]
                       + yn[:-1, 1:] + yn[1:, 1:])
    mesh.dx = np.diff(xn, axis=0)
    mesh.dy = np.diff(yn, axis=1)
    return mesh


if __name__ == "__main__":
    from .mesh_2d import uniform_mesh

    mesh = uniform_mesh(8, 6, lx=2.0, ly=1.0)

    # PLOT3D I/O round-trip
    print("Testing PLOT3D I/O...")
    write_plot3d_grid(mesh, "_test_plot3d.xyz")
    mesh_in = read_plot3d_grid("_test_plot3d.xyz")
    print(f"  Written: {mesh.nx}×{mesh.ny}  →  Read: {mesh_in.nx}×{mesh_in.ny}")
    assert mesh.nx == mesh_in.nx and mesh.ny == mesh_in.ny
    assert np.allclose(mesh.xn, mesh_in.xn)
    assert np.allclose(mesh.yn, mesh_in.yn)
    print("  PLOT3D round-trip: PASSED")

    # Solution file
    print("Testing PLOT3D solution file...")
    density = np.ones(mesh.xn.shape)
    u = np.zeros(mesh.xn.shape)
    v = np.zeros(mesh.xn.shape)
    pressure = np.ones(mesh.xn.shape)
    write_plot3d_solution("_test_plot3d.q", mesh, density, u, v, pressure)
    sol = read_plot3d_solution("_test_plot3d.q")
    print(f"  Read {len(sol)} variables from solution file")
    print("  PLOT3D solution: PASSED")

    # Cleanup
    Path("_test_plot3d.xyz").unlink()
    Path("_test_plot3d.q").unlink()

    # CGNS I/O (if available)
    if _HDF5_AVAILABLE:
        print("\nTesting CGNS I/O...")
        try:
            write_cgns_grid(mesh, "_test_cgns.cgns")
            mesh_cgns = read_cgns_grid("_test_cgns.cgns")
            print(f"  Written: {mesh.nx}×{mesh.ny}  "
                  f"→  Read: {mesh_cgns.nx}×{mesh_cgns.ny}")
            assert np.allclose(mesh.xn, mesh_cgns.xn)
            assert np.allclose(mesh.yn, mesh_cgns.yn)
            print("  CGNS round-trip: PASSED")
            Path("_test_cgns.cgns").unlink()
        except Exception as e:
            print(f"  CGNS test skipped ({e})")
    else:
        print("\nCGNS I/O unavailable — h5py not installed.")
        print("  Install with: pip install h5py")
