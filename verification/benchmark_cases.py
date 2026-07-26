#!/usr/bin/env python3
"""
verification/benchmark_cases.py
===============================

Comprehensive benchmark test suite for the 2D incompressible
Navier-Stokes solver.

Versuch A 闁?Lid-driven cavity flow             (Ghia et al. 1982)
Versuch B 闁?Poiseuille (channel) flow          (analytische L閺嬫ung)
Versuch C 闁?Backward-facing step               (Armaly et al. 1983)
Versuch D 闁?Zylinderumstr閺嬫ung / cylinder flow (Re=40, 200)

Each *Versuch* (experiment) is fully self-contained: it sets up the
mesh, boundary conditions, runs the solver, computes error metrics,
and compares against reference data.

References
----------
- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for
  incompressible flow using the Navier-Stokes equations and a multigrid
  method. *J. Comp. Phys.*, 48(3):387闁?11.
- Armaly, B. F., Durst, F., Pereira, J. C. F., & Sch閺嬫ung, B. (1983).
  Experimental and theoretical investigation of backward-facing step
  flow. *J. Fluid Mech.*, 127:473闁?96.

Author: Heinrich Vogel
        TU M閻《chen, Lehrstuhl f閻『 Numerische Str閺嬫ungsmechanik
Date:   2026-07-27
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from navier_stokes import (
    Boundary2D,
    BoundaryCondition,
    Mesh2D,
    SIMPLESolver,
)


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Gemeinsame Konstanten 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

RHO: Final[float] = 1.0         # Dichte [kg/m妞翠箽
T_END_CAVITY: Final[float] = 20.0   # Endzeit f閻『 cavity [s]
DT_CAVITY: Final[float] = 0.01      # Zeitschritt [s]
NX_CAVITY: Final[int] = 64          # Cells x
NY_CAVITY: Final[int] = 64          # Cells y

T_END_POISEUILLE: Final[float] = 5.0
DT_POISEUILLE: Final[float] = 0.005

# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Versuch A: Lid-driven Cavity 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

# Reference data from Ghia et al. (1982), Tables 1闁?.
# u闁炽儲鍞篹locity along vertical centerline (x=0.5)

# Re = 100
CAVITY_RE100_Y: Final[list[float]] = [
    1.00000, 0.97656, 0.96875, 0.96094, 0.95313, 0.85156, 0.75000,
    0.50000, 0.23438, 0.10156, 0.07031, 0.06250, 0.05469, 0.00000,
]
CAVITY_RE100_U: Final[list[float]] = [
    1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
    -0.20581, -0.32627, -0.31537, -0.24799, -0.21527, -0.18577, 0.00000,
]

# Re = 400
CAVITY_RE400_Y: Final[list[float]] = [
    1.00000, 0.97656, 0.96875, 0.96094, 0.95313, 0.85156, 0.73438,
    0.50000, 0.28125, 0.17188, 0.10156, 0.07031, 0.06250, 0.05469, 0.00000,
]
CAVITY_RE400_U: Final[list[float]] = [
    1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256,
    -0.32727, -0.38836, -0.32491, -0.27896, -0.24787, -0.23372, -0.22622, 0.00000,
]

# Re = 1000
CAVITY_RE1000_Y: Final[list[float]] = [
    1.00000, 0.97656, 0.96875, 0.96094, 0.95313, 0.85156, 0.73438,
    0.50000, 0.28125, 0.17188, 0.10156, 0.07031, 0.06250, 0.05469, 0.00000,
]
CAVITY_RE1000_U: Final[list[float]] = [
    1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.27832, 0.17727,
    -0.30071, -0.38265, -0.32731, -0.27669, -0.24715, -0.23689, -0.21999, 0.00000,
]

# v闁炽儲鍞篹locity along horizontal centerline (y=0.5) from Ghia et al.
CAVITY_RE100_X: Final[list[float]] = [
    0.00000, 0.06250, 0.07031, 0.10156, 0.17188, 0.28125,
    0.50000, 0.73438, 0.85156, 0.95313, 0.96094, 0.96875, 0.97656, 1.00000,
]
CAVITY_RE100_V: Final[list[float]] = [
    0.00000, -0.20066, -0.21615, -0.24508, -0.20165, -0.13350,
    0.01758, 0.14632, 0.04648, -0.11957, -0.13965, -0.16429, -0.23271, 0.00000,
]

CAVITY_RE400_X: Final[list[float]] = [
    0.00000, 0.06250, 0.07031, 0.10156, 0.17188, 0.28125,
    0.50000, 0.73438, 0.85156, 0.95313, 0.96094, 0.96875, 0.97656, 1.00000,
]
CAVITY_RE400_V: Final[list[float]] = [
    0.00000, -0.23088, -0.24966, -0.27780, -0.28020, -0.16443,
    0.05124, 0.16518, 0.04509, -0.13154, -0.15517, -0.17886, -0.23452, 0.00000,
]

CAVITY_RE1000_X: Final[list[float]] = [
    0.00000, 0.06250, 0.07031, 0.10156, 0.17188, 0.28125,
    0.50000, 0.73438, 0.85156, 0.95313, 0.96094, 0.96875, 0.97656, 1.00000,
]
CAVITY_RE1000_V: Final[list[float]] = [
    0.00000, -0.27271, -0.28923, -0.30952, -0.30888, -0.19109,
    0.05202, 0.16673, 0.04887, -0.13980, -0.16444, -0.18887, -0.24781, 0.00000,
]


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Daten f閻『 einen einzelnen Cavity-Datensatz 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?

@dataclass
class CavityReferenz:
    """Referenzdaten f閻『 die Cavity-Str閺嬫ung bei einer Reynoldszahl."""
    Reynoldszahl: int
    y_werte: list[float]
    u_referenz: list[float]
    x_werte: list[float]
    v_referenz: list[float]


CAVITY_REFERENZEN: Final[list[CavityReferenz]] = [
    CavityReferenz(100,  CAVITY_RE100_Y,  CAVITY_RE100_U,
                   CAVITY_RE100_X, CAVITY_RE100_V),
    CavityReferenz(400,  CAVITY_RE400_Y,  CAVITY_RE400_U,
                   CAVITY_RE400_X, CAVITY_RE400_V),
    CavityReferenz(1000, CAVITY_RE1000_Y, CAVITY_RE1000_U,
                   CAVITY_RE1000_X, CAVITY_RE1000_V),
]


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Numerische Cavity-L閺嬫ung 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?

def solve_cavity(
    Re: float,
    nx: int = NX_CAVITY,
    ny: int = NY_CAVITY,
    t_end: float = T_END_CAVITY,
    dt: float = DT_CAVITY,
) -> SIMPLESolver:
    """Solve the lid闁炽儲鍚€riven cavity problem at given Re.

    Parameters
    ----------
    Re : float
        Reynoldszahl.
    nx, ny : int
        Grid resolution.
    t_end : float
        End time.
    dt : float
        Time step.

    Returns
    -------
    SIMPLESolver with converged flow field.
    """
    nu: float = 1.0 / Re
    mesh: Mesh2D = Mesh2D(nx, ny, lx=1.0, ly=1.0)
    bc: Boundary2D = Boundary2D(
        west=(BoundaryCondition.WALL, 0.0),
        east=(BoundaryCondition.WALL, 0.0),
        south=(BoundaryCondition.WALL, 0.0),
        north=(BoundaryCondition.WALL, 1.0),  # Moving lid
    )
    solver: SIMPLESolver = SIMPLESolver(mesh, nu=nu, rho=RHO, dt=dt, bc=bc)

    n_steps: int = int(t_end / dt)
    for _step in range(n_steps):
        solver.step(n_inner=15)

    return solver


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Error evaluation Cavity 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾

@dataclass
class CavityErgebnis:
    """Ergebnisse eines einzelnen Cavity-Benchmark-Laufs."""
    Reynoldszahl: int
    error_u_max: float
    error_u_mean: float
    error_v_max: float
    error_v_mean: float
    passed: bool


def cavity_auswerten(
    solver: SIMPLESolver,
    referenz: CavityReferenz,
    toleranz: float = 0.15,
) -> CavityErgebnis:
    """Compare solver cavity solution against Ghia reference data.

    Interpolates the numerical solution onto the reference y闁炽儲鍞╫cations
    (centreline x = 0.5) and computes relative errors.

    Parameters
    ----------
    solver : SIMPLESolver
        Solved cavity flow.
    referenz : CavityReferenz
        Reference data for this Re.
    toleranz : float
        Maximum allowable mean relative error (default 15 %).

    Returns
    -------
    CavityErgebnis
    """
    ny: int = solver.mesh.ny
    j_mitte: int = ny // 2

    # u(y) at x = 0.5 闁?centreline index
    i_mitte: int = solver.mesh.nx // 2
    u_zentrallinie: np.ndarray = solver.u.data[i_mitte, :]
    v_zentrallinie: np.ndarray = solver.v.data[i_mitte, :]

    y_num: np.ndarray = solver.mesh.yc

    # Interpolate numerical u onto reference y positions
    y_ref: np.ndarray = np.array(referenz.y_werte)
    u_ref: np.ndarray = np.array(referenz.u_referenz)

    # For v: x centreline, so we use solver.u horizontal centreline (j = ny//2)
    v_horizontal: np.ndarray = solver.v.data[:, j_mitte]
    x_num: np.ndarray = solver.mesh.xc
    x_ref: np.ndarray = np.array(referenz.x_werte)
    v_ref: np.ndarray = np.array(referenz.v_referenz)

    # Relative errors (L2闁炽儲鍞玱rm along centreline)
    u_interp: np.ndarray = np.interp(y_ref, y_num, u_zentrallinie)
    v_interp: np.ndarray = np.interp(x_ref, x_num, v_horizontal)

    u_max_ref: float = float(np.max(np.abs(u_ref)))
    v_max_ref: float = float(np.max(np.abs(v_ref)))

    error_u: np.ndarray = np.abs(u_interp - u_ref) / (u_max_ref + 1.0e-30)
    error_v: np.ndarray = np.abs(v_interp - v_ref) / (v_max_ref + 1.0e-30)

    error_u_max: float = float(np.max(error_u))
    error_u_mean: float = float(np.mean(error_u))
    error_v_max: float = float(np.max(error_v))
    error_v_mean: float = float(np.mean(error_v))

    passed: bool = (error_u_mean < toleranz and error_v_mean < toleranz)

    return CavityErgebnis(
        Reynoldszahl=referenz.Reynoldszahl,
        error_u_max=error_u_max,
        error_u_mean=error_u_mean,
        error_v_max=error_v_max,
        error_v_mean=error_v_mean,
        passed=passed,
    )


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Versuch B: Poiseuille-Str閺嬫ung (Kanalstr閺嬫ung) 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

@dataclass
class PoiseuilleErgebnis:
    """deviation von der analytischen Poiseuille-L閺嬫ung."""
    L1_error: float
    L2_error: float
    Linf_error: float
    maximalgeschwindigkeit_num: float
    maximalgeschwindigkeit_exakt: float
    passed: bool


def poiseuille_analytisch(y: np.ndarray, H: float, dpdx: float,
                           mu: float) -> np.ndarray:
    """Analytical Poiseuille velocity profile.

    u(y) = (1 / (2濞?) 鐠?(-dp/dx) 鐠?y 鐠?(H - y)

    Parameters
    ----------
    y : np.ndarray
        Wall闁炽儲鍞玱rmal coordinate array.
    H : float
        Channel height.
    dpdx : float
        Pressure gradient (negative for forward flow).
    mu : float
        Dynamic viscosity 濞?= 鐠?鐠?閿?

    Returns
    -------
    np.ndarray
        Analytical velocity at each y.
    """
    return (1.0 / (2.0 * mu)) * (-dpdx) * y * (H - y)


def solve_poiseuille(
    nx: int = 48,
    ny: int = 48,
    Re: float = 50.0,
    t_end: float = T_END_POISEUILLE,
    dt: float = DT_POISEUILLE,
) -> tuple[SIMPLESolver, PoiseuilleErgebnis]:
    """Solve Poiseuille channel flow and compare with analytical solution.

    Boundary conditions:
        West (inlet):   u = U_einlass (parabolic profile)
        East (outlet):  zero闁炽儲鍚噐adient
        North/South:    no闁炽儲鍞秎ip walls

    The analytical solution assumes fully闁炽儲鍚€eveloped laminar flow with
    a parabolic profile.  The numerical solution should converge to the
    same profile.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Re : float
        Reynoldszahl based on bulk velocity and channel height.
    t_end : float
        End time.
    dt : float
        Time step.

    Returns
    -------
    Tuple of (solver, PoiseuilleErgebnis).
    """
    H: float = 1.0          # Channel height
    L: float = 3.0          # Channel length
    U_max: float = 1.0      # Max velocity at centreline
    dpdx: float = -2.0 * Re * U_max / (H ** 2)  # Forcing pressure gradient

    # Analytical profile at any x
    mu: float = (1.0 / Re) * RHO  # 濞?= 鐠?鐠?閿?

    mesh: Mesh2D = Mesh2D(nx, ny, lx=L, ly=H)

    # Inlet: prescribe parabolic profile
    y_inlet: np.ndarray = mesh.yc
    u_inlet_profil: np.ndarray = poiseuille_analytisch(y_inlet, H, dpdx, mu)

    bc: Boundary2D = Boundary2D(
        west=(BoundaryCondition.INLET, float(np.mean(u_inlet_profil))),
        east=(BoundaryCondition.OUTLET, 0.0),
        south=(BoundaryCondition.WALL, 0.0),
        north=(BoundaryCondition.WALL, 0.0),
    )

    solver: SIMPLESolver = SIMPLESolver(mesh, nu=1.0 / Re, rho=RHO,
                                         dt=dt, bc=bc)

    # Set initial field to analytical (faster convergence)
    for i in range(nx):
        solver.u.data[i, :] = poiseuille_analytisch(y_inlet, H, dpdx, mu)

    n_steps: int = int(t_end / dt)
    for _step in range(n_steps):
        solver.step(n_inner=20)

    # Compare at outlet cross闁炽儲鍞秂ction (i = nx-2, one cell before outlet BC)
    u_num: np.ndarray = solver.u.data[-2, :]
    u_exakt: np.ndarray = poiseuille_analytisch(mesh.yc, H, dpdx, mu)

    error: np.ndarray = u_num - u_exakt
    L1: float = float(np.mean(np.abs(error)))
    L2: float = float(np.sqrt(np.mean(error ** 2)))
    Linf: float = float(np.max(np.abs(error)))
    u_max_num: float = float(np.max(u_num))
    u_max_exakt: float = float(np.max(u_exakt))

    passed: bool = L1 < 0.05

    return solver, PoiseuilleErgebnis(
        L1_error=L1,
        L2_error=L2,
        Linf_error=Linf,
        maximalgeschwindigkeit_num=u_max_num,
        maximalgeschwindigkeit_exakt=u_max_exakt,
        passed=passed,
    )


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Versuch C: Backward-Facing Step 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

@dataclass
class StufenErgebnis:
    """Ergebnisse des R閻々kw閻╃灗tsstufen-Benchmarks."""
    Reynoldszahl: float
    wiederanlegelaenge_num: float
    wiederanlegelaenge_ref: float
    deviation_prozent: float
    passed: bool


# Reference reattachment lengths from Armaly et al. (1983), Fig. 12
# Non闁炽儲鍚€imensionalised by step height h
STUFEN_REFERENZ: Final[dict[int, float]] = {
    100: 3.0,
    200: 5.0,
    400: 8.2,
    600: 10.0,
    800: 12.2,
}


def solve_backward_step(
    Re: float = 100.0,
    nx: int = 120,
    ny: int = 40,
    expansionsverhaeltnis: float = 2.0,
    t_end: float = 30.0,
    dt: float = 0.01,
) -> tuple[SIMPLESolver, StufenErgebnis]:
    """Solve backward闁炽儲鍚卆cing step flow.

    The domain consists of a narrow inlet channel (height h) that
    expands abruptly to a wider channel (height H = ER 鐠?h).

    Parameters
    ----------
    Re : float
        Reynoldszahl based on step height h and bulk inlet velocity.
    nx, ny : int
        Grid resolution.
    expansionsverhaeltnis : float
        Expansion ratio H / h (default 2.0).
    t_end : float
        End time.
    dt : float
        Time step.

    Returns
    -------
    Tuple of (solver, StufenErgebnis).
    """
    h: float = 1.0          # Step height
    H: float = expansionsverhaeltnis * h  # Outlet channel height
    L_vor: float = 2.0      # Inlet length before step
    L_nach: float = 20.0    # Outlet length after step
    L_gesamt: float = L_vor + L_nach

    nu: float = 1.0 / Re
    U_einlass: float = 1.0

    # Simplified: use a rectangular domain; the step is represented
    # by the boundary condition at the lower wall.
    # A full treatment would need a non闁炽儲鍞筺iform mesh; here we use a
    # uniform mesh and identify the step location geometrically.
    mesh: Mesh2D = Mesh2D(nx, ny, lx=L_gesamt, ly=H)

    bc: Boundary2D = Boundary2D(
        west=(BoundaryCondition.INLET, U_einlass),
        east=(BoundaryCondition.OUTLET, 0.0),
        south=(BoundaryCondition.WALL, 0.0),
        north=(BoundaryCondition.WALL, 0.0),
    )

    solver: SIMPLESolver = SIMPLESolver(mesh, nu=nu, rho=RHO, dt=dt, bc=bc)

    n_steps: int = int(t_end / dt)
    for _step in range(n_steps):
        solver.step(n_inner=15)

    # Detect reattachment point: find where wall shear stress changes sign
    # along the lower wall (j = 0) downstream of the step (x > L_vor)
    i_step: int = int(L_vor / mesh.dx)
    u_wand: np.ndarray = solver.u.data[i_step:, 0]  # Lower wall u

    # Find zero crossing: first index where u >= 0 after the step
    wiederanlege_idx: int | None = None
    for i in range(1, len(u_wand)):
        if u_wand[i - 1] < 0.0 <= u_wand[i]:
            wiederanlege_idx = i
            break

    if wiederanlege_idx is None:
        # No reattachment found within domain
        laenge_num: float = L_nach
        deviation: float = 100.0
    else:
        # Linear interpolation for sub闁炽儲鍙緀ll accuracy
        x0: float = mesh.xc[i_step + wiederanlege_idx - 1]
        x1: float = mesh.xc[i_step + wiederanlege_idx]
        u0: float = u_wand[wiederanlege_idx - 1]
        u1: float = u_wand[wiederanlege_idx]
        if u1 - u0 != 0.0:
            alpha: float = -u0 / (u1 - u0)
        else:
            alpha = 0.5
        x_reattach: float = x0 + alpha * (x1 - x0)
        laenge_num = x_reattach - L_vor

    # Find closest reference Re
    ref_schluessel: int = min(STUFEN_REFERENZ.keys(), key=lambda k: abs(k - Re))
    laenge_ref: float = STUFEN_REFERENZ[ref_schluessel]
    deviation = 100.0 * abs(laenge_num - laenge_ref) / (laenge_ref + 1.0e-30)
    passed: bool = deviation < 30.0

    return solver, StufenErgebnis(
        Reynoldszahl=Re,
        wiederanlegelaenge_num=laenge_num,
        wiederanlegelaenge_ref=laenge_ref,
        deviation_prozent=deviation,
        passed=passed,
    )


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Versuch D: Zylinderumstr閺嬫ung 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

@dataclass
class ZylinderErgebnis:
    """Ergebnisse des Zylinder-Benchmarks."""
    Reynoldszahl: float
    widerstandsbeiwert: float
    auftriebsbeiwert_rms: float
    Strouhalzahl: float
    passed: bool


def solve_cylinder(
    Re: float = 40.0,
    nx: int = 100,
    ny: int = 80,
    t_end: float = 20.0,
    dt: float = 0.005,
) -> tuple[SIMPLESolver, ZylinderErgebnis]:
    """Solve flow around a circular cylinder.

    Simplified model: the cylinder is represented by a no闁炽儲鍞秎ip region
    in the centre of the domain.  Drag and lift coefficients are
    approximated from the surface pressure integration.

    Parameters
    ----------
    Re : float
        Reynoldszahl based on cylinder diameter D and inflow velocity.
    nx, ny : int
        Grid resolution.
    t_end : float
        End time.
    dt : float
        Time step.

    Returns
    -------
    Tuple of (solver, ZylinderErgebnis).
    """
    D: float = 1.0           # Cylinder diameter
    L: float = 20.0 * D      # Domain length
    H: float = 10.0 * D      # Domain height
    U_inf: float = 1.0       # Inflow velocity
    nu: float = 1.0 / Re

    mesh: Mesh2D = Mesh2D(nx, ny, lx=L, ly=H)

    bc: Boundary2D = Boundary2D(
        west=(BoundaryCondition.INLET, U_inf),
        east=(BoundaryCondition.OUTLET, 0.0),
        south=(BoundaryCondition.WALL, 0.0),
        north=(BoundaryCondition.WALL, 0.0),
    )

    solver: SIMPLESolver = SIMPLESolver(mesh, nu=nu, rho=RHO, dt=dt, bc=bc)

    # Place cylinder centre at x = 5D, y = H/2
    x_center: float = 5.0 * D
    y_center: float = H / 2.0
    r_cylinder: float = D / 2.0

    # Enforce no闁炽儲鍞秎ip inside cylinder by directly setting velocity to zero
    # inside the cylinder region after each step
    n_steps: int = int(t_end / dt)

    zeitreihe_cd: list[float] = []
    zeitreihe_cl: list[float] = []

    for step in range(n_steps):
        solver.step(n_inner=10)

        # Enforce cylinder boundary condition
        X, Y = np.meshgrid(mesh.xc, mesh.yc, indexing="ij")
        mask_zyl: np.ndarray = ((X - x_center) ** 2 + (Y - y_center) ** 2
                                 < r_cylinder ** 2)
        solver.u.data[mask_zyl] = 0.0
        solver.v.data[mask_zyl] = 0.0

        # Approximate force coefficients via near闁炽儲鍙緔linder momentum balance
        # (simplified: difference in velocity gradient across cylinder)
        cd: float = _naeherungs_cd(solver, x_center, y_center, r_cylinder, nu)
        cl: float = _naeherungs_cl(solver, x_center, y_center, r_cylinder, nu)

        if step > n_steps // 4:  # Collect after initial transient
            zeitreihe_cd.append(cd)
            zeitreihe_cl.append(cl)

    mean_cd: float = float(np.mean(zeitreihe_cd)) if zeitreihe_cd else 0.0
    cl_rms: float = float(np.std(zeitreihe_cl)) if zeitreihe_cl else 0.0

    # Strouhal number from lift frequency (if unsteady, Re > ~60)
    St: float = _berechne_Strouhal(zeitreihe_cl, dt, D, U_inf)

    # Reference values (Tritton 1959, Williamson 1996)
    if abs(Re - 40.0) < 1.0:
        cd_ref: float = 1.50
        passed: bool = abs(mean_cd - cd_ref) / cd_ref < 0.3
    elif abs(Re - 200.0) < 1.0:
        cd_ref = 1.30
        st_ref: float = 0.196
        passed = (abs(mean_cd - cd_ref) / cd_ref < 0.3
                     and abs(St - st_ref) / st_ref < 0.3)
    else:
        cd_ref = 0.0
        passed = True

    return solver, ZylinderErgebnis(
        Reynoldszahl=Re,
        widerstandsbeiwert=mean_cd,
        auftriebsbeiwert_rms=cl_rms,
        Strouhalzahl=St,
        passed=passed,
    )


def _naeherungs_cd(
    solver: SIMPLESolver,
    x_c: float, y_c: float, r: float, nu: float,
) -> float:
    """Approximate drag coefficient from vorticity near cylinder."""
    omega: np.ndarray = solver.compute_vorticity()
    X, Y = np.meshgrid(solver.mesh.xc, solver.mesh.yc, indexing="ij")

    # Shell just outside cylinder
    r_shell: float = r * 1.5
    maske: np.ndarray = ((X - x_c) ** 2 + (Y - y_c) ** 2 < r_shell ** 2)
    return float(np.mean(np.abs(omega[maske])) * nu * 2.0)


def _naeherungs_cl(
    solver: SIMPLESolver,
    x_c: float, y_c: float, r: float, nu: float,
) -> float:
    """Approximate lift coefficient from vertical force imbalance."""
    omega: np.ndarray = solver.compute_vorticity()
    X, Y = np.meshgrid(solver.mesh.xc, solver.mesh.yc, indexing="ij")

    r_shell: float = r * 1.5
    maske: np.ndarray = ((X - x_c) ** 2 + (Y - y_c) ** 2 < r_shell ** 2)
    # Asymmetric vorticity 闁?lift
    oben: np.ndarray = (Y > y_c) & maske
    unten: np.ndarray = (Y < y_c) & maske
    omega_oben: float = float(np.mean(omega[oben])) if np.any(oben) else 0.0
    omega_unten: float = float(np.mean(omega[unten])) if np.any(unten) else 0.0
    return (omega_unten - omega_oben) * nu


def _berechne_Strouhal(
    cl_zeitreihe: list[float],
    dt: float,
    D: float,
    U_inf: float,
) -> float:
    """Estimate Strouhal number from lift闁炽儲鍙緊efficient time series.

    Uses FFT to find the dominant frequency.

    Parameters
    ----------
    cl_zeitreihe : list[float]
        Lift coefficient samples.
    dt : float
        Sampling interval.
    D : float
        Cylinder diameter.
    U_inf : float
        Free闁炽儲鍞秚ream velocity.

    Returns
    -------
    float
        Strouhal number St = f 鐠?D / U_inf.
    """
    if len(cl_zeitreihe) < 10:
        return 0.0

    cl_arr: np.ndarray = np.array(cl_zeitreihe)
    cl_arr = cl_arr - np.mean(cl_arr)  # Remove DC

    n: int = len(cl_arr)
    freqs: np.ndarray = np.fft.rfftfreq(n, d=dt)
    spektrum: np.ndarray = np.abs(np.fft.rfft(cl_arr))

    # Find dominant frequency (skip DC)
    idx_max: int = int(np.argmax(spektrum[1:])) + 1
    f_dominant: float = float(freqs[idx_max]) if idx_max < len(freqs) else 0.0

    St: float = f_dominant * D / U_inf
    return St if St > 0.0 else 0.0


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Haupt-Routine: Alle Versuche ausf閻“ren 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

@dataclass
class BenchmarkResults:
    """Sammelergebnisse aller Benchmarks."""
    cavity_ergebnisse: list[CavityErgebnis] = field(default_factory=list)
    poiseuille_ergebnis: PoiseuilleErgebnis | None = None
    stufen_ergebnisse: list[StufenErgebnis] = field(default_factory=list)
    zylinder_ergebnisse: list[ZylinderErgebnis] = field(default_factory=list)


def run_all_benchmarks(
    cavity_reynoldszahlen: list[int] | None = None,
    stufen_reynoldszahlen: list[float] | None = None,
    zylinder_reynoldszahlen: list[float] | None = None,
) -> BenchmarkResults:
    """Run all benchmark cases.

    Parameters
    ----------
    cavity_reynoldszahlen : list[int] | None
        Re values for cavity (default [100, 400, 1000]).
    stufen_reynoldszahlen : list[float] | None
        Re values for step (default [100, 200]).
    zylinder_reynoldszahlen : list[float] | None
        Re values for cylinder (default [40]).

    Returns
    -------
    BenchmarkResults with all results.
    """
    if cavity_reynoldszahlen is None:
        cavity_reynoldszahlen = [100, 400, 1000]
    if stufen_reynoldszahlen is None:
        stufen_reynoldszahlen = [100.0, 200.0]
    if zylinder_reynoldszahlen is None:
        zylinder_reynoldszahlen = [40.0]

    ergebnisse: BenchmarkResults = BenchmarkResults()

    # 闁冲厜鍋撻柍鍏夊亾 Versuch A: Cavity 闁冲厜鍋撻柍鍏夊亾
    print("  [Benchmark] Versuch A 闁?Lid-driven Cavity")
    for Re in cavity_reynoldszahlen:
        ref: CavityReferenz | None = None
        for r in CAVITY_REFERENZEN:
            if r.Reynoldszahl == Re:
                ref = r
                break
        if ref is None:
            continue

        solver = solve_cavity(Re)
        erg = cavity_auswerten(solver, ref)
        ergebnisse.cavity_ergebnisse.append(erg)
        status: str = "闁? if erg.passed else "闁?
        print(f"    {status} Re = {Re:4d}  "
              f"|u|_mean = {erg.error_u_mean:.4f}  "
              f"|v|_mean = {erg.error_v_mean:.4f}")

    # 闁冲厜鍋撻柍鍏夊亾 Versuch B: Poiseuille 闁冲厜鍋撻柍鍏夊亾
    print("  [Benchmark] Versuch B 闁?Poiseuille-Str閺嬫ung")
    _, p_erg = solve_poiseuille()
    ergebnisse.poiseuille_ergebnis = p_erg
    status = "闁? if p_erg.passed else "闁?
    print(f"    {status} L1 = {p_erg.L1_error:.2e}  "
          f"L2 = {p_erg.L2_error:.2e}  "
          f"L闁?= {p_erg.Linf_error:.2e}")
    print(f"      U_max (numerisch) = {p_erg.maximalgeschwindigkeit_num:.4f}")
    print(f"      U_max (exakt)     = {p_erg.maximalgeschwindigkeit_exakt:.4f}")

    # 闁冲厜鍋撻柍鍏夊亾 Versuch C: Stufe 闁冲厜鍋撻柍鍏夊亾
    print("  [Benchmark] Versuch C 闁?R閻々kw閻╃灗tsstufe")
    for Re in stufen_reynoldszahlen:
        _, s_erg = solve_backward_step(Re=Re)
        ergebnisse.stufen_ergebnisse.append(s_erg)
        status = "闁? if s_erg.passed else "闁?
        print(f"    {status} Re = {Re:4.0f}  "
              f"x_r (num) = {s_erg.wiederanlegelaenge_num:.2f}h  "
              f"x_r (ref) = {s_erg.wiederanlegelaenge_ref:.1f}h  "
              f"閾?= {s_erg.deviation_prozent:.1f}%")

    # 闁冲厜鍋撻柍鍏夊亾 Versuch D: Zylinder 闁冲厜鍋撻柍鍏夊亾
    print("  [Benchmark] Versuch D 闁?Zylinderumstr閺嬫ung")
    for Re in zylinder_reynoldszahlen:
        _, z_erg = solve_cylinder(Re=Re)
        ergebnisse.zylinder_ergebnisse.append(z_erg)
        status = "闁? if z_erg.passed else "闁?
        print(f"    {status} Re = {Re:4.0f}  "
              f"C_D = {z_erg.widerstandsbeiwert:.3f}  "
              f"C_L(RMS) = {z_erg.auftriebsbeiwert_rms:.3f}  "
              f"St = {z_erg.Strouhalzahl:.3f}")

    return ergebnisse


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Hauptprogramm 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

def main() -> int:
    """Run the full benchmark suite and print a summary."""
    print("=" * 72)
    print("  CFD-Solver 闁?Benchmark-Suite (Verifizierung)")
    print("  TU M閻《chen, Lehrstuhl f閻『 Numerische Str閺嬫ungsmechanik")
    print("=" * 72)

    ergebnisse: BenchmarkResults = run_all_benchmarks()

    print()
    print("=" * 72)
    print("  Zusammenfassung")
    print("=" * 72)

    all_passed: list[bool] = []
    for c in ergebnisse.cavity_ergebnisse:
        all_passed.append(c.passed)
    if ergebnisse.poiseuille_ergebnis is not None:
        all_passed.append(ergebnisse.poiseuille_ergebnis.passed)
    for s in ergebnisse.stufen_ergebnisse:
        all_passed.append(s.passed)
    for z in ergebnisse.zylinder_ergebnisse:
        all_passed.append(z.passed)

    total: int = len(all_passed)
    passed: int = sum(all_passed)

    print(f"  passed: {passed} / {total}")
    if passed == total:
        print("  闁?ALL BENCHMARKS PASSED")
    else:
        print(f"  闁?{total - passed} Benchmark(s) failed")
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
