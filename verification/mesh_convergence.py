#!/usr/bin/env python3
"""
verification/mesh_convergence.py
================================

Systematic mesh convergence study for the 2D incompressible
Navier-Stokes solver using the Grid Convergence Index (GCI)
methodology of Roache (1998).

The study computes solutions across 5 grid levels with a constant
refinement factor r = 闁? 闁?1.414.  For each grid level the
discretisation error is quantified, the apparent convergence order
is estimated, and the asymptotic range is verified.

Method (Roache 1998)
--------------------
For a triplet of grids (1 = fine, 2 = medium, 3 = coarse) with
refinement factor r = h闁?h闁?= h闁?h闁?

    p = ln((f闁?- f闁? / (f闁?- f闁?) / ln(r)

    f_{h=0} = f闁?+ (f闁?- f闁? / (r^p - 1)

    GCI_{fine} = F_s 鐠?|(f闁?- f闁? / f闁逞傜盃 / (r^p - 1)

where F_s = 1.25 (safety factor for 3-grid comparisons).

Asymptotic range is confirmed when:

    GCI_{23} / (r^p 鐠?GCI_{12}) 闁?1.0

References
----------
- Roache, P. J. (1998). Verification and Validation in Computational
  Science and Engineering. Hermosa Publishers.
- Celik, I. B. et al. (2008). Procedure for Estimation and Reporting
  of Uncertainty Due to Discretization in CFD Applications.
  *ASME J. Fluids Eng.*, 130(7):078001.

Author: Heinrich Vogel
        TU M閻《chen, Lehrstuhl f閻『 Numerische Str閺嬫ungsmechanik
Date:   2026-07-27
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from math import log as mlog
from pathlib import Path
from typing import Final

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from navier_stokes import Boundary2D, BoundaryCondition, Mesh2D, SIMPLESolver


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Konstanten 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

RHO: Final[float] = 1.0                        # Dichte [kg/m妞翠箽
NU: Final[float] = 1.0e-3                      # Viskosit閻╃灜 [m閾?s]
REFINEMENT_FACTOR: Final[float] = 2.0 ** 0.5   # r = 闁? 闁?1.414
NUM_GRIDS: Final[int] = 5                  # refinements
F_SICHERHEIT: Final[float] = 1.25              # Sicherheitsfaktor F_s
T_END_CONV: Final[float] = 5.0                 # End time per grid [s]
DT_KONV: Final[float] = 0.01                   # Zeitschritt [s]

# Base grid: coarsest level
NX_BASIS: Final[int] = 8
NY_BASIS: Final[int] = 8


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Dataclasses 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

@dataclass
class GridLevel:
    """Eine einzelne GridLevel der Verfeinerungsstudie.

    Attributes
    ----------
    stufe : int
        Level index (1 = finest, NUM_GRIDS = coarsest).
    nx : int
        Number of cells in x.
    ny : int
        Number of cells in y.
    grid_spacing : float
        Characteristic cell size h = sqrt(1/(nx鐠虹棴y)).
    loesung : float
        Solution value (e.g. max u闁炽儲鍞篹locity at centreline).
    """
    stufe: int
    nx: int
    ny: int
    grid_spacing: float
    loesung: float


@dataclass
class GCIErgebnis:
    """GCI闁炽儲鍘渦swertung f閻『 ein Triplett von Gittern.

    Attributes
    ----------
    fine_level : int
        Level of the finest grid in the triplet.
    convergence_order_p : float
        Apparent convergence order p.
    extrapolierter_wert : float
        Richardson extrapolation f_{h=0}.
    GCI_fein : float
        GCI on the fine grid (fractional, not %).
    GCI_grob : float
        GCI on the coarse grid (fractional).
    asymptotischer_index : float
        GCI_23 / (r^p 鐠?GCI_12).  闁?1.0 confirms asymptotic range.
    """
    fine_level: int
    convergence_order_p: float
    extrapolierter_wert: float
    GCI_fein: float
    GCI_grob: float
    asymptotischer_index: float


@dataclass
class ConvergenceStudy:
    """Complete mesh convergence study results.

    Attributes
    ----------
    GridLeveln : list[GridLevel]
        Solutions on each grid level.
    gci_ergebnisse : list[GCIErgebnis]
        GCI results for each consecutive triplet.
    intervall_unter : float
        Lower bound of 95 % confidence interval.
    intervall_ober : float
        Upper bound of 95 % confidence interval.
    im_asymptotischen_bereich : bool
        Whether asymptotic range is confirmed.
    """
    GridLeveln: list[GridLevel] = field(default_factory=list)
    gci_ergebnisse: list[GCIErgebnis] = field(default_factory=list)
    intervall_unter: float = 0.0
    intervall_ober: float = 0.0
    im_asymptotischen_bereich: bool = False


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Problem-definierende Funktion 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

def _solve_on_grid(nx: int, ny: int, Re: float = 100.0,
                      t_end: float = T_END_CONV,
                      dt: float = DT_KONV) -> float:
    """Solve a reference problem on a given grid.

    Returns a scalar quantity of interest (here: peak u闁炽儲鍞篹locity
    along the cavity vertical centreline at x = 0.5).  This quantity
    is tracked across grid levels for the GCI study.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Re : float
        Reynoldszahl.
    t_end : float
        End time.
    dt : float
        Time step.

    Returns
    -------
    float
        Peak u闁炽儲鍞篹locity on centreline (quantity of interest).
    """
    nu: float = 1.0 / Re
    mesh: Mesh2D = Mesh2D(nx, ny, lx=1.0, ly=1.0)

    bc: Boundary2D = Boundary2D(
        west=(BoundaryCondition.WALL, 0.0),
        east=(BoundaryCondition.WALL, 0.0),
        south=(BoundaryCondition.WALL, 0.0),
        north=(BoundaryCondition.WALL, 1.0),
    )

    solver: SIMPLESolver = SIMPLESolver(mesh, nu=nu, rho=RHO, dt=dt, bc=bc)

    n_steps: int = int(t_end / dt)
    for _step in range(n_steps):
        solver.step(n_inner=15)

    # Quantity of interest: max u at centreline (x = 0.5)
    i_mitte: int = nx // 2
    u_zentrum: np.ndarray = solver.u.data[i_mitte, :]
    return float(np.max(np.abs(u_zentrum)))


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Grid creation 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

def create_grid_levels(
    count: int = NUM_GRIDS,
    nx_basis: int = NX_BASIS,
    ny_basis: int = NY_BASIS,
    Re: float = 100.0,
) -> list[GridLevel]:
    """Generate and solve on all grid levels.

    Grid i (0 = coarsest) has  nx = nx_basis 鐠?r^{i}
    and ny = ny_basis 鐠?r^{i}, where r = REFINEMENT_FACTOR.

    The finest grid (stufe = 1) has the largest nx, ny.

    Parameters
    ----------
    count : int
        Number of grid levels (default 5).
    nx_basis, ny_basis : int
        Coarsest grid resolution.
    Re : float
        Reynoldszahl.

    Returns
    -------
    list[GridLevel]
        Solutions on each grid level, sorted finest 闁?coarsest.
    """
    levels: list[GridLevel] = []

    for stufe in range(count):
        idx: int = count - 1 - stufe  # 0 = finest, count-1 = coarsest
        nx: int = int(round(nx_basis * (REFINEMENT_FACTOR ** idx)))
        ny: int = int(round(ny_basis * (REFINEMENT_FACTOR ** idx)))
        nx = max(nx, 4)
        ny = max(ny, 4)

        grid_spacing: float = np.sqrt(1.0 / (nx * ny))
        loesung: float = _solve_on_grid(nx, ny, Re=Re)

        levels.append(GridLevel(
            stufe=stufe + 1,
            nx=nx,
            ny=ny,
            grid_spacing=grid_spacing,
            loesung=loesung,
        ))

    return levels


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?GCI-Berechnung (Roache 1998) 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

def berechne_gci(
    f1: float, f2: float, f3: float,
    r: float = REFINEMENT_FACTOR,
    Fs: float = F_SICHERHEIT,
) -> GCIErgebnis:
    """Compute GCI quantities for a grid triplet.

    Parameters
    ----------
    f1 : float
        Solution on fine grid.
    f2 : float
        Solution on medium grid.
    f3 : float
        Solution on coarse grid.
    r : float
        Refinement factor.
    Fs : float
        Safety factor (1.25 for 3闁炽儲鍚噐id studies).

    Returns
    -------
    GCIErgebnis
    """
    eps12: float = f2 - f1
    eps23: float = f3 - f2

    # Apparent order p
    if abs(eps12) < 1.0e-30 or abs(eps23) < 1.0e-30:
        return GCIErgebnis(
            fine_level=0,
            convergence_order_p=0.0,
            extrapolierter_wert=f1,
            GCI_fein=0.0,
            GCI_grob=0.0,
            asymptotischer_index=-1.0,
        )

    try:
        p: float = mlog(abs(eps23 / eps12)) / mlog(r)
    except (ValueError, ZeroDivisionError):
        p = 0.0

    # Clamp p to plausible range
    p = max(0.1, min(p, 6.0))

    # Richardson extrapolation
    rp: float = r ** p
    f_exact: float = f1 + (f1 - f2) / (rp - 1.0)

    # GCI
    GCI_fine: float = Fs * abs((f1 - f2) / f1) / (rp - 1.0) if abs(rp - 1.0) > 1.0e-15 else 0.0
    GCI_coarse: float = Fs * abs((f2 - f3) / f2) / (rp - 1.0) if abs(rp - 1.0) > 1.0e-15 else 0.0

    # Asymptotic range index
    asym_index: float = GCI_coarse / (rp * GCI_fine) if GCI_fine > 1.0e-15 else -1.0

    return GCIErgebnis(
        fine_level=1,
        convergence_order_p=p,
        extrapolierter_wert=f_exact,
        GCI_fein=GCI_fine,
        GCI_grob=GCI_coarse,
        asymptotischer_index=asym_index,
    )


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Hauptstudie 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

def run_grid_convergence_study(
    Re: float = 100.0,
    NUM_GRIDS: int = NUM_GRIDS,
) -> ConvergenceStudy:
    """Run a complete mesh convergence study.

    Parameters
    ----------
    Re : float
        Reynoldszahl for the cavity flow.
    NUM_GRIDS : int
        Number of grid levels (default 5).

    Returns
    -------
    ConvergenceStudy
        Full results with GCI, asymptotic range check, and
        95 % Konfidenzintervall.
    """
    levels: list[GridLevel] = create_grid_levels(
        count=NUM_GRIDS, Re=Re,
    )

    gci_liste: list[GCIErgebnis] = []
    for i in range(len(levels) - 2):
        gci = berechne_gci(
            f1=levels[i].loesung,
            f2=levels[i + 1].loesung,
            f3=levels[i + 2].loesung,
        )
        gci.fine_level = levels[i].stufe
        gci_liste.append(gci)

    # Best estimate from finest闁炽儲鍚噐id triplet
    finest_gci: GCIErgebnis = gci_liste[0] if gci_liste else GCIErgebnis(
        fine_level=0, convergence_order_p=0.0,
        extrapolierter_wert=levels[0].loesung if levels else 0.0,
        GCI_fein=0.0, GCI_grob=0.0, asymptotischer_index=-1.0,
    )

    # 95 % Konfidenzintervall (approximate)
    f_best: float = finest_gci.extrapolierter_wert
    GCI_best: float = finest_gci.GCI_fein
    intervall_unter: float = f_best * (1.0 - 2.0 * GCI_best)
    intervall_ober: float = f_best * (1.0 + 2.0 * GCI_best)

    # Asymptotic range detection
    im_asymptotischen: bool = False
    for gci in gci_liste:
        if 0.8 <= gci.asymptotischer_index <= 1.2:
            im_asymptotischen = True
            break

    return ConvergenceStudy(
        GridLeveln=levels,
        gci_ergebnisse=gci_liste,
        intervall_unter=intervall_unter,
        intervall_ober=intervall_ober,
        im_asymptotischen_bereich=im_asymptotischen,
    )


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Bericht 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

def print_GCI_report(studie: ConvergenceStudy) -> str:
    """Format the GCI study as a printable report.

    Parameters
    ----------
    studie : ConvergenceStudy
        Completed convergence study.

    Returns
    -------
    str
        Formatted report.
    """
    linie: str = "闁冲厜鍋? * 72
    bericht: str = f"\n{linie}\n"
    bericht += "  Grid convergence study (GCI 闁?Roache 1998)\n"
    bericht += f"{linie}\n"
    bericht += (
        f"  {'Stufe':>5s}  {'nx':>4s}  {'ny':>4s}  "
        f"{'h':>10s}  {'L閺嬫ung':>12s}\n"
    )
    bericht += f"{linie}\n"

    for level in studie.GridLeveln:
        bericht += (
            f"  {level.stufe:5d}  {level.nx:4d}  {level.ny:4d}  "
            f"{level.grid_spacing:10.3e}  {level.loesung:12.6e}\n"
        )

    bericht += f"{linie}\n"
    bericht += "  GCI闁炽儲鍘渦swertung (Tripletts):\n"
    bericht += f"{linie}\n"

    for gci in studie.gci_ergebnisse:
        bericht += (
            f"  Stufe {gci.fine_level} (fein):  p = {gci.convergence_order_p:.4f}"
            f"  f_extrap = {gci.extrapolierter_wert:.6e}\n"
            f"    GCI_fine = {gci.GCI_fein * 100.0:.3f}%  "
            f"GCI_coarse = {gci.GCI_grob * 100.0:.3f}%  "
            f"Asymp. = {gci.asymptotischer_index:.4f}\n"
        )

    bericht += f"{linie}\n"
    bericht += (
        f"  95 % Konfidenzintervall (aus feinstem Triplett):\n"
        f"    [{studie.intervall_unter:.6e}, {studie.intervall_ober:.6e}]\n"
    )

    if studie.im_asymptotischen_bereich:
        bericht += "  闁?Asymptotischer Bereich erreicht (0.8 闁?GCI_23/(r^p鐠虹枔CI_12) 闁?1.2)\n"
    else:
        bericht += "  闁?Asymptotischer Bereich NOCH NICHT erreicht\n"

    bericht += f"{linie}\n"
    return bericht


# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?
# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Hauptprogramm 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾
# 闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩￠幇銊︽珳闁崇儤鍔忛弲鏌ュ煛閹般劍娅滈柍鐑樺姀閺呮煡鍩?

def main() -> int:
    """Run the mesh convergence study and print the GCI report."""
    print("=" * 72)
    print("  Systematische Grid convergence study")
    print("  TU M閻《chen, Lehrstuhl f閻『 Numerische Str閺嬫ungsmechanik")
    print("=" * 72)
    print(f"  Refinement factor r = {REFINEMENT_FACTOR:.4f}")
    print(f"  Grid levels         = {NUM_GRIDS}")
    print(f"  鐠?                  = {NU:.1e}  m閾?s")
    print(f"  閿?                  = {RHO:.1f}  kg/m妞?)
    print(f"  T_end               = {T_END_CONV:.1f}  s")
    print()

    studie: ConvergenceStudy = run_grid_convergence_study(Re=100.0)
    bericht: str = print_GCI_report(studie)
    print(bericht)

    # Acceptance criterion: asymptotic range confirmed or GCI < 5 %
    finest_gci: float = (
        studie.gci_ergebnisse[0].GCI_fein if studie.gci_ergebnisse else 1.0
    )
    PASSED: bool = (
        studie.im_asymptotischen_bereich or finest_gci < 0.05
    )

    if PASSED:
        print("  闁?PASSED 闁?Grid convergence demonstrated")
    else:
        print("  闁?FAILED 閳?insufficient grid convergence")
    print()

    return 0 if PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
