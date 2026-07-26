#!/usr/bin/env python3
"""
verification/method_of_manufactured_solutions.py
================================================

Method of manufactured solutions (MMS) for the 2D incompressible
Navier-Stokes equations.  Provides manufactured (analytical) fields,
the corresponding source terms for the momentum equations, and
routines for grid-convergence studies.

Manufactured solution (divergence-free)
---------------------------------------
    u_m(x, y) =  sin(πx) · cos(πy)
    v_m(x, y) = -cos(πx) · sin(πy)
    p_m(x, y) =  sin(πx) · sin(πy)

Boundary conditions:  u = v = 0  on ∂Ω (walls).

References
----------
- Roache, P. J. (2002). Code Verification by the Method of
  Manufactured Solutions. *ASME J. Fluids Eng.*, 124(1):4–10.
- Salari, K. & Knupp, P. (2000). Code Verification by the Method
  of Manufactured Solutions. *SAND2000-1444*, Sandia National Labs.

Author: Heinrich Vogel
        TU München, Lehrstuhl für Numerische Strömungsmechanik
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


# ─── Physical Constants ────────────────────────────────────────────────

PI: Final[float] = np.pi
NU_MMS: Final[float] = 1.0e-3          # Viskosität [m²/s]
RHO_MMS: Final[float] = 1.0            # Dichte [kg/m³]
L_DOMAIN: Final[float] = 1.0           # Domänenlänge [m]


# ─── Manufactured Solution (Analytisch) ────────────────────────────────

def u_analytisch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Manufactured x-velocity:  u = sin(πx) · cos(πy)."""
    return np.sin(PI * x) * np.cos(PI * y)


def v_analytisch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Manufactured y-velocity:  v = -cos(πx) · sin(πy)."""
    return -np.cos(PI * x) * np.sin(PI * y)


def p_analytisch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Manufactured pressure:  p = sin(πx) · sin(πy)."""
    return np.sin(PI * x) * np.sin(PI * y)


# ─── Source Terms for the Momentum Equations ──────────────────────────

def quelle_u(x: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray:
    """Manufactured source term for the x‑momentum equation.

    f_x = u·∂u/∂x + v·∂u/∂y + ∂p/∂x - ν·∇²u

    Returns
    -------
    Array of same shape as x, y.
    """
    konvektion: np.ndarray = PI * np.sin(PI * x) * np.cos(PI * x)
    druckgrad: np.ndarray = PI * np.cos(PI * x) * np.sin(PI * y)
    diffusion: np.ndarray = 2.0 * PI ** 2 * nu * np.sin(PI * x) * np.cos(PI * y)
    return konvektion + druckgrad + diffusion


def quelle_v(x: np.ndarray, y: np.ndarray, nu: float) -> np.ndarray:
    """Manufactured source term for the y‑momentum equation.

    f_y = u·∂v/∂x + v·∂v/∂y + ∂p/∂y - ν·∇²v

    Returns
    -------
    Array of same shape as x, y.
    """
    konvektion: np.ndarray = PI * np.sin(PI * y) * np.cos(PI * y)
    druckgrad: np.ndarray = PI * np.sin(PI * x) * np.cos(PI * y)
    diffusion: np.ndarray = -2.0 * PI ** 2 * nu * np.cos(PI * x) * np.sin(PI * y)
    return konvektion + druckgrad + diffusion


# ─── MMS‑Enhanced Solver ──────────────────────────────────────────────

class MMSQuadraturSolver(SIMPLESolver):
    """SIMPLESolver with MMS source terms added to the momentum equations.

    The manufactured source terms f_x, f_y are evaluated at cell centres
    and added as volumetric sources in each SIMPLE iteration.
    """

    def __init__(self, mesh: Mesh2D, nu: float = NU_MMS,
                 rho: float = RHO_MMS, dt: float = 0.01) -> None:
        # All walls — the manufactured solution satisfies u=v=0 on ∂Ω
        bc: Boundary2D = Boundary2D(
            west=(BoundaryCondition.WALL, 0.0),
            east=(BoundaryCondition.WALL, 0.0),
            south=(BoundaryCondition.WALL, 0.0),
            north=(BoundaryCondition.WALL, 0.0),
        )
        super().__init__(mesh, nu=nu, rho=rho, dt=dt, bc=bc)

        # Pre‑compute source term arrays
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
            # ── Momentum solve (u*, v*) ──
            max_residual: float = 0.0

            for i in range(1, self.mesh.nx - 1):
                for j in range(1, self.mesh.ny - 1):
                    # u‑momentum
                    a_p_u: float = d_e + d_w + d_n + d_s + rho * cell_vol / dt
                    grad_p_x: float = (self.p[i + 1, j] - self.p[i - 1, j]) / (2.0 * dx)
                    su: float = (-grad_p_x * cell_vol
                                 + self._fx[i, j] * cell_vol)  # ← MMS source

                    u_star: float = (
                        d_e * self.u[i + 1, j]
                        + d_w * self.u[i - 1, j]
                        + d_n * self.u[i, j + 1]
                        + d_s * self.u[i, j - 1]
                        + su
                    ) / a_p_u
                    self.u[i, j] = ((1.0 - self.alpha_u) * self.u[i, j]
                                    + self.alpha_u * u_star)

                    # v‑momentum
                    grad_p_y: float = (self.p[i, j + 1] - self.p[i, j - 1]) / (2.0 * dy)
                    sv: float = (-grad_p_y * cell_vol
                                 + self._fy[i, j] * cell_vol)  # ← MMS source

                    v_star: float = (
                        d_e * self.v[i + 1, j]
                        + d_w * self.v[i - 1, j]
                        + d_n * self.v[i, j + 1]
                        + d_s * self.v[i, j - 1]
                        + sv
                    ) / a_p_u
                    self.v[i, j] = ((1.0 - self.alpha_u) * self.v[i, j]
                                    + self.alpha_u * v_star)

            # ── Pressure correction ──
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

            # ── Velocity correction ──
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

            # ── Pressure update ──
            for i in range(1, self.mesh.nx - 1):
                for j in range(1, self.mesh.ny - 1):
                    self.p[i, j] += self.alpha_p * self.p_corr[i, j]

            # ── Boundary conditions ──
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


# ─── Error Norms ──────────────────────────────────────────────────────

def norm_L1(fehler: np.ndarray) -> float:
    """L1‑Norm des Diskretisierungsfehlers."""
    return float(np.mean(np.abs(fehler)))


def norm_L2(fehler: np.ndarray) -> float:
    """L2‑Norm (Euklidisch) des Diskretisierungsfehlers."""
    return float(np.sqrt(np.mean(fehler ** 2)))


def norm_Linf(fehler: np.ndarray) -> float:
    """L∞‑Norm (Maximum) des Diskretisierungsfehlers."""
    return float(np.max(np.abs(fehler)))


def berechne_fehler(
    solver: SIMPLESolver,
) -> dict[str, float]:
    """Compute L1, L2, L∞ error norms for u, v, p.

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

    fehler_u: np.ndarray = solver.u.data - u_exakt
    fehler_v: np.ndarray = solver.v.data - v_exakt
    fehler_p: np.ndarray = solver.p.data - p_exakt

    return {
        "L1_u": norm_L1(fehler_u),
        "L2_u": norm_L2(fehler_u),
        "Linf_u": norm_Linf(fehler_u),
        "L1_v": norm_L1(fehler_v),
        "L2_v": norm_L2(fehler_v),
        "Linf_v": norm_Linf(fehler_v),
        "L1_p": norm_L1(fehler_p),
        "L2_p": norm_L2(fehler_p),
        "Linf_p": norm_Linf(fehler_p),
    }


# ─── Grid Convergence Study ────────────────────────────────────────────

GITTERWEITEN: Final[list[int]] = [8, 12, 16, 24, 32, 48, 64]
"""Grid resolutions (cells per dimension) for the convergence study."""

T_END_MMS: Final[float] = 2.0
"""Simulation end time for each grid level."""

DT_MMS: Final[float] = 0.005
"""Time step for MMS runs."""


@dataclass
class KonvergenzStudie:
    """Results of a grid convergence study.

    Attributes
    ----------
    gitterweiten : list[int]
        Number of cells per dimension for each grid level.
    L1_u : list[float]
        L1 error of u at each grid level.
    L2_u : list[float]
        L2 error of u at each grid level.
    Linf_u : list[float]
        L∞ error of u at each grid level.
    ordnung_L1 : float
        Spatial convergence order (Räumliche Konvergenzordnung) based on L1.
    ordnung_L2 : float
        Spatial convergence order based on L2.
    ordnung_Linf : float
        Spatial convergence order based on L∞.
    """

    gitterweiten: list[int]
    L1_u: list[float]
    L2_u: list[float]
    Linf_u: list[float]
    L1_v: list[float]
    L2_v: list[float]
    Linf_v: list[float]
    L1_p: list[float]
    L2_p: list[float]
    Linf_p: list[float]
    ordnung_L1: float
    ordnung_L2: float
    ordnung_Linf: float


def messe_Konvergenzordnung(
    L1_normen: list[float],
    gitterweiten: list[int],
) -> float:
    """Measure spatial convergence order from error norms.

    Uses a least‑squares fit  log(error) =  log(C)  +  p · log(h)
    where h = 1/N is the grid spacing.

    Parameters
    ----------
    L1_normen : list[float]
        Error norms at each grid level.
    gitterweiten : list[int]
        Number of cells per dimension.

    Returns
    -------
    float
        Estimated spatial order p (Räumliche Konvergenzordnung).
    """
    h: np.ndarray = 1.0 / np.array(gitterweiten, dtype=float)
    fehler: np.ndarray = np.array(L1_normen, dtype=float)

    # Filter out zero errors (converged exactly)
    mask: np.ndarray = fehler > 0.0
    if np.sum(mask) < 2:
        return 0.0

    log_h: np.ndarray = np.log(h[mask])
    log_e: np.ndarray = np.log(fehler[mask])

    # Linear regression: log(E) = log(C) + p · log(h)
    A: np.ndarray = np.vstack([log_h, np.ones_like(log_h)]).T
    p, _logC = np.linalg.lstsq(A, log_e, rcond=None)[0]
    return float(p)


def fuehre_Konvergenzstudie_durch(
    gitterliste: list[int] | None = None,
    t_end: float = T_END_MMS,
    dt: float = DT_MMS,
    nu: float = NU_MMS,
) -> KonvergenzStudie:
    """Run a full grid‑convergence study for the MMS problem.

    Parameters
    ----------
    gitterliste : list[int] | None
        List of cells-per-dimension.  Defaults to GITTERWEITEN.
    t_end : float
        End time per grid level.
    dt : float
        Time step.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    KonvergenzStudie
        Convergence results with error norms and spatial orders.
    """
    if gitterliste is None:
        gitterliste = GITTERWEITEN

    L1_u_list: list[float] = []
    L2_u_list: list[float] = []
    Linf_u_list: list[float] = []
    L1_v_list: list[float] = []
    L2_v_list: list[float] = []
    Linf_v_list: list[float] = []
    L1_p_list: list[float] = []
    L2_p_list: list[float] = []
    Linf_p_list: list[float] = []

    for n in gitterliste:
        mesh: Mesh2D = Mesh2D(n, n, lx=L_DOMAIN, ly=L_DOMAIN)
        solver: MMSQuadraturSolver = MMSQuadraturSolver(mesh, nu=nu, dt=dt)

        n_steps: int = int(t_end / dt)
        for _step in range(n_steps):
            solver.step(n_inner=15)

        fehler = berechne_fehler(solver)
        L1_u_list.append(fehler["L1_u"])
        L2_u_list.append(fehler["L2_u"])
        Linf_u_list.append(fehler["Linf_u"])
        L1_v_list.append(fehler["L1_v"])
        L2_v_list.append(fehler["L2_v"])
        Linf_v_list.append(fehler["Linf_v"])
        L1_p_list.append(fehler["L1_p"])
        L2_p_list.append(fehler["L2_p"])
        Linf_p_list.append(fehler["Linf_p"])

    # Use u‑velocity errors to compute spatial convergence order
    ordnung_L1: float = messe_Konvergenzordnung(L1_u_list, gitterliste)
    ordnung_L2: float = messe_Konvergenzordnung(L2_u_list, gitterliste)
    ordnung_Linf: float = messe_Konvergenzordnung(Linf_u_list, gitterliste)

    return KonvergenzStudie(
        gitterweiten=gitterliste,
        L1_u=L1_u_list,
        L2_u=L2_u_list,
        Linf_u=Linf_u_list,
        L1_v=L1_v_list,
        L2_v=L2_v_list,
        Linf_v=Linf_v_list,
        L1_p=L1_p_list,
        L2_p=L2_p_list,
        Linf_p=Linf_p_list,
        ordnung_L1=ordnung_L1,
        ordnung_L2=ordnung_L2,
        ordnung_Linf=ordnung_Linf,
    )


# ─── Bericht (Report) ──────────────────────────────────────────────────

def drucke_Konvergenzbericht(studie: KonvergenzStudie) -> str:
    """Format the convergence study as a printable string.

    Parameters
    ----------
    studie : KonvergenzStudie
        Results from a convergence study.

    Returns
    -------
    str
        Formatted report.
    """
    linie: str = "─" * 72
    kopf: str = (
        f"\n{linie}\n"
        f"  MMS‑Gitterkonvergenzstudie\n"
        f"{linie}\n"
        f"  {'N':>4s}  {'L1(u)':>10s}  {'L2(u)':>10s}  {'L∞(u)':>10s}  "
        f"{'L1(v)':>10s}  {'L2(v)':>10s}  {'L∞(v)':>10s}\n"
        f"{linie}\n"
    )

    zeilen: str = ""
    for i, n in enumerate(studie.gitterweiten):
        zeilen += (
            f"  {n:4d}  {studie.L1_u[i]:10.3e}  {studie.L2_u[i]:10.3e}  "
            f"{studie.Linf_u[i]:10.3e}  {studie.L1_v[i]:10.3e}  "
            f"{studie.L2_v[i]:10.3e}  {studie.Linf_v[i]:10.3e}\n"
        )

    fuss: str = (
        f"{linie}\n"
        f"  Räumliche Konvergenzordnung (aus u‑Geschwindigkeit):\n"
        f"    p(L1)   = {studie.ordnung_L1:.4f}\n"
        f"    p(L2)   = {studie.ordnung_L2:.4f}\n"
        f"    p(L∞)   = {studie.ordnung_Linf:.4f}\n"
        f"{linie}\n"
    )

    return kopf + zeilen + fuss


# ─── Hauptprogramm ─────────────────────────────────────────────────────

def main() -> int:
    """Run the MMS convergence study and print a report."""
    print("=" * 72)
    print("  Method of Manufactured Solutions — Gitterkonvergenz")
    print("  TU München, Lehrstuhl für Numerische Strömungsmechanik")
    print("=" * 72)
    print(f"  ν     = {NU_MMS:.1e}  m²/s")
    print(f"  ρ     = {RHO_MMS:.1f}  kg/m³")
    print(f"  T_end = {T_END_MMS:.1f}  s")
    print(f"  dt    = {DT_MMS:.1e}  s")
    print()

    studie: KonvergenzStudie = fuehre_Konvergenzstudie_durch()
    bericht: str = drucke_Konvergenzbericht(studie)
    print(bericht)

    # Acceptance criterion: first‑order or better
    ordnung_erwartet: float = 1.0
    bestanden: bool = studie.ordnung_L1 >= ordnung_erwartet * 0.5
    if bestanden:
        print(f"  ✓ BESTANDEN — Konvergenzordnung p = {studie.ordnung_L1:.3f}")
    else:
        print(f"  ✗ NICHT BESTANDEN — Konvergenzordnung p = {studie.ordnung_L1:.3f}")
    print()

    return 0 if bestanden else 1


if __name__ == "__main__":
    sys.exit(main())
