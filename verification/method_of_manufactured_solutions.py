#!/usr/bin/env python3
"""
verification/method_of_manufactured_solutions.py
================================================

Method of Manufactured Solutions (MMS) for the 2D incompressible
Navier-Stokes equations.  Provides manufactured (analytical) fields,
the corresponding source terms for the momentum equations, and
routines for grid-convergence studies.

Manufactured solution (divergence-free)
---------------------------------------
    u_m(x, y) =  sin(閿滅皰) 鐠?cos(閿滅皳)
    v_m(x, y) = -cos(閿滅皰) 鐠?sin(閿滅皳)
    p_m(x, y) =  sin(閿滅皰) 鐠?sin(閿滅皳)

Boundary conditions:  u = v = 0  on 闁愁厼浼?(walls).

References
----------
- Roache, P. J. (2002). Code Verification by the Method of
  Manufactured Solutions. *ASME J. Fluids Eng.*, 124(1):4闁?0.
- Salari, K. & Knupp, P. (2000). Code Verification by the Method
  of Manufactured Solutions. *SAND2000-1444*, Sandia National Labs.

Author: Heinrich Vogel
        TU M閻《chen, Lehrstuhl f閻『 Numerische Str閺嬫ungsmechanik
Date:   2026-07-27
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from navier_stokes import (
    Boundary2D,
    BoundaryCondition,
    Field2D,
    Mesh2D,
    SIMPLESolver,
)


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Physical Constants 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾

PI: Final[float] = np.pi
NU_MMS: Final[float] = 1.0e-3          # Viskosit閻╃灜 [m閾?s]
RHO_MMS: Final[float] = 1.0            # Dichte [kg/m妞翠箽
L_DOMAIN: Final[float] = 1.0           # Dom閻╃灐enl閻╃灐ge [m]


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Manufactured Solution (Analytisch) 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾

def u_analytisch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Manufactured x-velocity:  u = sin(閿滅皰) 鐠?cos(閿滅皳)."""
    return np.sin(PI * x) * np.cos(PI * y)


def v_analytisch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Manufactured y-velocity:  v = -cos(閿滅皰) 鐠?sin(閿滅皳)."""
    return -np.cos(PI * x) * np.sin(PI * y)


def p_analytisch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Manufactured pressure:  p = sin(閿滅皰) 鐠?sin(閿滅皳)."""
    return np.sin(PI * x) * np.sin(PI * y)


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Source Terms for the Momentum Equations 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾

def quelle_u(x: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray:
    """Manufactured source term for the x闁炽儲鍞﹐mentum equation.

    f_x = u鐠侯垶鍩堥崐?闁愁厼鈧?+ v鐠侯垶鍩堥崐?闁愁厼鈧?+ 闁愁厺鐐?闁愁厼鈧?- 鐠嬫捁鐭鹃柍顓炴瘽閻?

    Returns
    -------
    Array of same shape as x, y.
    """
    konvektion: np.ndarray = PI * np.sin(PI * x) * np.cos(PI * x)
    druckgrad: np.ndarray = PI * np.cos(PI * x) * np.sin(PI * y)
    diffusion: np.ndarray = 2.0 * PI ** 2 * nu * np.sin(PI * x) * np.cos(PI * y)
    return konvektion + druckgrad + diffusion


def quelle_v(x: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray:
    """Manufactured source term for the y闁炽儲鍞﹐mentum equation.

    f_y = u鐠侯垶鍩堥崐?闁愁厼鈧?+ v鐠侯垶鍩堥崐?闁愁厼鈧?+ 闁愁厺鐐?闁愁厼鈧?- 鐠嬫捁鐭鹃柍顓炴瘽閻?

    Returns
    -------
    Array of same shape as x, y.
    """
    konvektion: np.ndarray = PI * np.sin(PI * y) * np.cos(PI * y)
    druckgrad: np.ndarray = PI * np.sin(PI * x) * np.cos(PI * y)
    diffusion: np.ndarray = -2.0 * PI ** 2 * nu * np.cos(PI * x) * np.sin(PI * y)
    return konvektion + druckgrad + diffusion


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?MMS闁炽儲鍘玭hanced Solver 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾

class MMSQuadraturSolver(SIMPLESolver):
    """SIMPLESolver with MMS source terms added to the momentum equations.

    The manufactured source terms f_x, f_y are evaluated at cell centres
    and added as volumetric sources in each SIMPLE iteration.
    """

    def __init__(self, mesh: Mesh2D, nu: float = NU_MMS,
                 rho: float = RHO_MMS, dt: float = 0.01) -> None:
        # All walls 闁?the manufactured solution satisfies u=v=0 on 闁愁厼浼?
        bc: Boundary2D = Boundary2D(
            west=(BoundaryCondition.WALL, 0.0),
            east=(BoundaryCondition.WALL, 0.0),
            south=(BoundaryCondition.WALL, 0.0),
            north=(BoundaryCondition.WALL, 0.0),
        )
        super().__init__(mesh, nu=nu, rho=rho, dt=dt, bc=bc)

        # Pre闁炽儲鍙緊mpute source term arrays
        X, Y = np.meshgrid(mesh.xc, mesh.yc, indexing="ij")
        self._fx: np.ndarray = quelle_u(X, Y, nu)
        self._fy: np.ndarray = quelle_v(X, Y, nu)

    def step(self, n_inner: int = 20) -> dict:
        """Overridden step that includes MMS source terms."""
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt
        nu, rho = self.nu, self.rho
        cell_vol: float = dx * dy

        diff_coeff: float = nu * rho
        ae: float = dy
        an: float = dx
        d_e: float = diff_coeff * ae / dx
        d_w: float = d_e
        d_n: float = diff_coeff * an / dy
        d_s: float = d_n

        for _it in range(n_inner):
            # 闁冲厜鍋撻柍鍏夊亾 Momentum solve (u*, v*) 闁冲厜鍋撻柍鍏夊亾
            max_residual: float = 0.0

            for i in range(1, self.mesh.nx - 1):
                for j in range(1, self.mesh.ny - 1):
                    # u闁炽儲鍞﹐mentum
                    a_p_u: float = d_e + d_w + d_n + d_s + rho * cell_vol / dt
                    grad_p_x: float = (self.p[i + 1, j] - self.p[i - 1, j]) / (2.0 * dx)
                    su: float = (-grad_p_x * cell_vol
                                 + self._fx[i, j] * cell_vol)  # 闁?MMS source

                    u_star: float = (
                        d_e * self.u[i + 1, j]
                        + d_w * self.u[i - 1, j]
                        + d_n * self.u[i, j + 1]
                        + d_s * self.u[i, j - 1]
                        + su
                    ) / a_p_u
                    self.u[i, j] = ((1.0 - self.alpha_u) * self.u[i, j]
                                    + self.alpha_u * u_star)

                    # v闁炽儲鍞﹐mentum
                    grad_p_y: float = (self.p[i, j + 1] - self.p[i, j - 1]) / (2.0 * dy)
                    sv: float = (-grad_p_y * cell_vol
                                 + self._fy[i, j] * cell_vol)  # 闁?MMS source

                    v_star: float = (
                        d_e * self.v[i + 1, j]
                        + d_w * self.v[i - 1, j]
                        + d_n * self.v[i, j + 1]
                        + d_s * self.v[i, j - 1]
                        + sv
                    ) / a_p_u
                    self.v[i, j] = ((1.0 - self.alpha_u) * self.v[i, j]
                                    + self.alpha_u * v_star)

            # 闁冲厜鍋撻柍鍏夊亾 Pressure correction 闁冲厜鍋撻柍鍏夊亾
            for i in range(1, self.mesh.nx - 1):
                for j in range(1, self.mesh.ny - 1):
                    div_u: float = (
                        (self.u[i + 1, j] - self.u[i - 1, j]) / (2.0 * dx)
                        + (self.v[i, j + 1] - self.v[i, j - 1]) / (2.0 * dy)
                    )
                    coeff: float = dt / rho
                    a_e_pc: float = coeff * ae / dx
                    a_w_pc: float = a_e_pc
                    a_n_pc: float = coeff * an / dy
                    a_s_pc: float = a_n_pc
                    a_p_pc: float = a_e_pc + a_w_pc + a_n_pc + a_s_pc

                    if abs(div_u) > max_residual:
                        max_residual = abs(div_u)

                    if a_p_pc > 1.0e-12:
                        p_corr_val: float = (
                            a_e_pc * self.p_corr[i + 1, j]
                            + a_w_pc * self.p_corr[i - 1, j]
                            + a_n_pc * self.p_corr[i, j + 1]
                            + a_s_pc * self.p_corr[i, j - 1]
                            + div_u * cell_vol
                        ) / a_p_pc
                        self.p_corr[i, j] = p_corr_val

            # 闁冲厜鍋撻柍鍏夊亾 Velocity correction 闁冲厜鍋撻柍鍏夊亾
            for i in range(1, self.mesh.nx - 1):
                for j in range(1, self.mesh.ny - 1):
                    grad_pc_x: float = (
                        self.p_corr[i + 1, j] - self.p_corr[i - 1, j]
                    ) / (2.0 * dx)
                    grad_pc_y: float = (
                        self.p_corr[i, j + 1] - self.p_corr[i, j - 1]
                    ) / (2.0 * dy)
                    self.u[i, j] -= dt / rho * grad_pc_x
                    self.v[i, j] -= dt / rho * grad_pc_y

            # 闁冲厜鍋撻柍鍏夊亾 Pressure update 闁冲厜鍋撻柍鍏夊亾
            for i in range(1, self.mesh.nx - 1):
                for j in range(1, self.mesh.ny - 1):
                    self.p[i, j] += self.alpha_p * self.p_corr[i, j]

            # 闁冲厜鍋撻柍鍏夊亾 Boundary conditions 闁冲厜鍋撻柍鍏夊亾
            self._apply_boundary_conditions()

            if max_residual < self.tol:
                break

        self.time += dt
        self.iteration += 1

        return {
            "time": self.time,
            "inner_iterations": n_inner,
            "max_residual": max_residual,
        }


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Error Norms 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾

def norm_L1(error: np.ndarray) -> float:
    """L1闁炽儲鍘竜rm des discretisation errors."""
    return float(np.mean(np.abs(error)))


def norm_L2(error: np.ndarray) -> float:
    """L2闁炽儲鍘竜rm (Euklidisch) des discretisation errors."""
    return float(np.sqrt(np.mean(error ** 2)))


def norm_Linf(error: np.ndarray) -> float:
    """L闁愁厾鍋愰埀顒佸幐orm (Maximum) des discretisation errors."""
    return float(np.max(np.abs(error)))


def compute_errors(
    solver: SIMPLESolver,
) -> dict[str, float]:
    """Compute L1, L2, L闁?error norms for u, v, p.

    Parameters
    ----------
    solver : SIMPLESolver
        Solved flow field (after MMS run).

    Returns
    -------
    Dictionary with keys like 'L1_u', 'L2_v', 'Linf_p', etc.
    """
    mesh: Mesh2D = solver.mesh
    X, Y = np.meshgrid(mesh.xc, mesh.yc, indexing="ij")

    u_exakt: np.ndarray = u_analytisch(X, Y)
    v_exakt: np.ndarray = v_analytisch(X, Y)
    p_exakt: np.ndarray = p_analytisch(X, Y)

    error_u: np.ndarray = solver.u.data - u_exakt
    error_v: np.ndarray = solver.v.data - v_exakt
    error_p: np.ndarray = solver.p.data - p_exakt

    return {
        "L1_u": norm_L1(error_u),
        "L2_u": norm_L2(error_u),
        "Linf_u": norm_Linf(error_u),
        "L1_v": norm_L1(error_v),
        "L2_v": norm_L2(error_v),
        "Linf_v": norm_Linf(error_v),
        "L1_p": norm_L1(error_p),
        "L2_p": norm_L2(error_p),
        "Linf_p": norm_Linf(error_p),
    }


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Grid Convergence Study 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾

GRID_SIZES: Final[list[int]] = [8, 12, 16, 24, 32, 48, 64]
"""Grid resolutions (cells per dimension) for the convergence study."""

T_END_MMS: Final[float] = 2.0
"""Simulation end time for each grid level."""

DT_MMS: Final[float] = 0.005
"""Time step for MMS runs."""


@dataclass
class ConvergenceStudy:
    """Results of a grid convergence study.

    Attributes
    ----------
    GRID_SIZES : list[int]
        Number of cells per dimension for each grid level.
    L1_u : list[float]
        L1 error of u at each grid level.
    L2_u : list[float]
        L2 error of u at each grid level.
    Linf_u : list[float]
        L闁?error of u at each grid level.
    order_L1 : float
        Spatial convergence order (R閻╃灝mliche Konvergenzordnung) based on L1.
    order_L2 : float
        Spatial convergence order based on L2.
    order_Linf : float
        Spatial convergence order based on L闁?
    """

    GRID_SIZES: list[int]
    L1_u: list[float]
    L2_u: list[float]
    Linf_u: list[float]
    L1_v: list[float]
    L2_v: list[float]
    Linf_v: list[float]
    L1_p: list[float]
    L2_p: list[float]
    Linf_p: list[float]
    order_L1: float
    order_L2: float
    order_Linf: float


def measure_convergence_order(
    L1_normen: list[float],
    GRID_SIZES: list[int],
) -> float:
    """Measure spatial convergence order from error norms.

    Uses a least闁炽儲鍞秖uares fit  log(error) =  log(C)  +  p 鐠?log(h)
    where h = 1/N is the grid spacing.

    Parameters
    ----------
    L1_normen : list[float]
        Error norms at each grid level.
    GRID_SIZES : list[int]
        Number of cells per dimension.

    Returns
    -------
    float
        Estimated spatial order p (R閻╃灝mliche Konvergenzordnung).
    """
    h: np.ndarray = 1.0 / np.array(GRID_SIZES, dtype=float)
    error: np.ndarray = np.array(L1_normen, dtype=float)

    # Filter out zero errors (converged exactly)
    mask: np.ndarray = error > 0.0
    if np.sum(mask) < 2:
        return 0.0

    log_h: np.ndarray = np.log(h[mask])
    log_e: np.ndarray = np.log(error[mask])

    # Linear regression: log(E) = log(C) + p 鐠?log(h)
    A: np.ndarray = np.vstack([log_h, np.ones_like(log_h)]).T
    p, _logC = np.linalg.lstsq(A, log_e, rcond=None)[0]
    return float(p)


def run_convergence_study(
    grid_list: list[int] | None = None,
    t_end: float = T_END_MMS,
    dt: float = DT_MMS,
    nu: float = NU_MMS,
) -> ConvergenceStudy:
    """Run a full grid闁炽儲鍙緊nvergence study for the MMS problem.

    Parameters
    ----------
    grid_list : list[int] | None
        List of cells-per-dimension.  Defaults to GRID_SIZES.
    t_end : float
        End time per grid level.
    dt : float
        Time step.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    ConvergenceStudy
        Convergence results with error norms and spatial orders.
    """
    if grid_list is None:
        grid_list = GRID_SIZES

    L1_u_list: list[float] = []
    L2_u_list: list[float] = []
    Linf_u_list: list[float] = []
    L1_v_list: list[float] = []
    L2_v_list: list[float] = []
    Linf_v_list: list[float] = []
    L1_p_list: list[float] = []
    L2_p_list: list[float] = []
    Linf_p_list: list[float] = []

    for n in grid_list:
        mesh: Mesh2D = Mesh2D(n, n, lx=L_DOMAIN, ly=L_DOMAIN)
        solver: MMSQuadraturSolver = MMSQuadraturSolver(mesh, nu=nu, dt=dt)

        n_steps: int = int(t_end / dt)
        for _step in range(n_steps):
            solver.step(n_inner=15)

        error = compute_errors(solver)
        L1_u_list.append(error["L1_u"])
        L2_u_list.append(error["L2_u"])
        Linf_u_list.append(error["Linf_u"])
        L1_v_list.append(error["L1_v"])
        L2_v_list.append(error["L2_v"])
        Linf_v_list.append(error["Linf_v"])
        L1_p_list.append(error["L1_p"])
        L2_p_list.append(error["L2_p"])
        Linf_p_list.append(error["Linf_p"])

    # Use u闁炽儲鍞篹locity errors to compute spatial convergence order
    order_L1: float = measure_convergence_order(L1_u_list, grid_list)
    order_L2: float = measure_convergence_order(L2_u_list, grid_list)
    order_Linf: float = measure_convergence_order(Linf_u_list, grid_list)

    return ConvergenceStudy(
        GRID_SIZES=grid_list,
        L1_u=L1_u_list,
        L2_u=L2_u_list,
        Linf_u=Linf_u_list,
        L1_v=L1_v_list,
        L2_v=L2_v_list,
        Linf_v=Linf_v_list,
        L1_p=L1_p_list,
        L2_p=L2_p_list,
        Linf_p=Linf_p_list,
        order_L1=order_L1,
        order_L2=order_L2,
        order_Linf=order_Linf,
    )


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Bericht (Report) 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾

def print_convergence_report(studie: ConvergenceStudy) -> str:
    """Format the convergence study as a printable string.

    Parameters
    ----------
    studie : ConvergenceStudy
        Results from a convergence study.

    Returns
    -------
    str
        Formatted report.
    """
    linie: str = "闁冲厜鍋? * 72
    kopf: str = (
        f"\n{linie}\n"
        f"  MMS闁炽儲鍘痠tterConvergenceStudy\n"
        f"{linie}\n"
        f"  {'N':>4s}  {'L1(u)':>10s}  {'L2(u)':>10s}  {'L闁?u)':>10s}  "
        f"{'L1(v)':>10s}  {'L2(v)':>10s}  {'L闁?v)':>10s}\n"
        f"{linie}\n"
    )

    zeilen: str = ""
    for i, n in enumerate(studie.GRID_SIZES):
        zeilen += (
            f"  {n:4d}  {studie.L1_u[i]:10.3e}  {studie.L2_u[i]:10.3e}  "
            f"{studie.Linf_u[i]:10.3e}  {studie.L1_v[i]:10.3e}  "
            f"{studie.L2_v[i]:10.3e}  {studie.Linf_v[i]:10.3e}\n"
        )

    fuss: str = (
        f"{linie}\n"
        f"  R閻╃灝mliche Konvergenzordnung (aus u闁炽儲鍘痚schwindigkeit):\n"
        f"    p(L1)   = {studie.order_L1:.4f}\n"
        f"    p(L2)   = {studie.order_L2:.4f}\n"
        f"    p(L闁?   = {studie.order_Linf:.4f}\n"
        f"{linie}\n"
    )

    return kopf + zeilen + fuss


# 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?Hauptprogramm 闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋撻柍鍏夊亾闁冲厜鍋?

def main() -> int:
    """Run the MMS convergence study and print a report."""
    print("=" * 72)
    print("  Method of Manufactured Solutions 闁?Grid convergence")
    print("  TU M閻《chen, Lehrstuhl f閻『 Numerische Str閺嬫ungsmechanik")
    print("=" * 72)
    print(f"  鐠?    = {NU_MMS:.1e}  m閾?s")
    print(f"  閿?    = {RHO_MMS:.1f}  kg/m妞?)
    print(f"  T_end = {T_END_MMS:.1f}  s")
    print(f"  dt    = {DT_MMS:.1e}  s")
    print()

    studie: ConvergenceStudy = run_convergence_study()
    bericht: str = print_convergence_report(studie)
    print(bericht)

    # Acceptance criterion: first闁炽儲鍞璻der or better
    expected_order: float = 1.0
    PASSED: bool = studie.order_L1 >= expected_order * 0.5
    if PASSED:
        print(f"  闁?PASSED 闁?Convergence order p = {studie.order_L1:.3f}")
    else:
        print(f"  闁?NICHT PASSED 闁?Convergence order p = {studie.order_L1:.3f}")
    print()

    return 0 if PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
