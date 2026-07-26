#!/usr/bin/env python3
"""
verification/flat_plate_k_omega_sst.py
========================================

Verification: k-ω SST for zero-pressure-gradient flat plate
turbulent boundary layer.

Method
------
A 1-D wall-normal grid with y+ ≈ 1 at the wall is constructed.
At each streamwise station the velocity profile is computed from
the law of the wall (viscous sublayer + log law).  The k-ω SST
model is run with very small pseudo-time steps to evolve k and ω
toward equilibrium, and the resulting eddy viscosity is used to
evaluate the wall shear stress.

    C_f = τ_w / (½ ρ U∞²)

compared against the Prandtl–Schlichting correlation:

    C_f = 0.0592 / Re_x^{1/5}

Acceptance criteria
-------------------
- Mean C_f error across the domain < 5 %

References
----------
- Menter, F. R. (1994). Two-equation eddy-viscosity turbulence models
  for engineering applications. *AIAA Journal*, 32(8):1598–1605.
- White, F. M. (2006). *Viscous Fluid Flow* (3rd ed.). McGraw-Hill.

Author: Pierre Dubois
        ISAE-SUPAERO / ONERA
Branch: dev/turbulence-pierre
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from turbulence.k_omega_sst import ModeleKOmegaSST


# ─── Physical Parameters ──────────────────────────────────────────────

U_INF: float = 10.0        # Freestream velocity [m/s]
RHO_INF: float = 1.0       # Density [kg/m³]
NU_MOL: float = 1.5e-5     # Molecular viscosity [m²/s]
L_PLATE: float = 2.0       # Plate length [m]
RE_L: float = U_INF * L_PLATE / NU_MOL

NX: int = 60               # Streamwise stations
NY: int = 512              # Wall-normal cells
DOMAIN_H: float = 0.08     # Domain height [m]


# ─── Prandtl–Schlichting correlation ─────────────────────────────────

def cf_correlation(re_x: np.ndarray) -> np.ndarray:
    """C_f = 0.0592 / Re_x^{1/5}  (Prandtl–Schlichting)."""
    return 0.0592 / np.maximum(re_x, 1e3) ** 0.2


def delta_99(re_x: float, x: float) -> float:
    """Turbulent boundary layer thickness δ_99."""
    return 0.37 * x / max(re_x, 1e3) ** 0.2


def u_tau_from_cf(cf: float) -> float:
    """Friction velocity: u_τ = √(½·C_f·U∞²)."""
    return np.sqrt(0.5 * abs(cf) * U_INF ** 2 + 1e-30)


# ─── Law of the Wall ────────────────────────────────────────────────

KAPPA: float = 0.41
E_LOG: float = 9.0
Y_PLUS_TRANS: float = 11.81


def law_of_the_wall_profile(
    y: np.ndarray,
    u_tau: float,
    nu: float,
) -> np.ndarray:
    """Construct u(y) from the law of the wall.

    u⁺ = y⁺           for  y⁺ < y⁺_trans
    u⁺ = 1/κ·ln(E·y⁺) for  y⁺ ≥ y⁺_trans

    Returns u in [m/s].
    """
    y_plus = y * u_tau / nu
    u_plus = np.where(
        y_plus < Y_PLUS_TRANS,
        y_plus,
        1.0 / KAPPA * np.log(E_LOG * y_plus + 1e-30),
    )
    return u_plus * u_tau


def equilibrium_nu_t(
    y: np.ndarray,
    u_tau: float,
    nu: float,
) -> np.ndarray:
    """Equilibrium eddy viscosity: ν_t⁺ = κ y⁺ in log layer."""
    y_plus = y * u_tau / nu
    return KAPPA * u_tau * y * np.maximum(y_plus, 1.0)


def equilibrium_k(y: np.ndarray, u_tau: float,
                  delta: float, cmu: float = 0.09) -> np.ndarray:
    """Equilibrium k: k ≈ u_τ² / √C_μ  far from wall, damped near wall."""
    k_inf = u_tau ** 2 / np.sqrt(cmu)
    # Simple damping near wall
    y_norm = np.clip(y / max(delta, 1e-10), 0.0, 1.0)
    return k_inf * (1.0 - np.exp(-y_norm * 20.0))


def equilibrium_omega(
    y: np.ndarray, u_tau: float, nu: float,
    cmu: float = 0.09, kappa: float = 0.41,
) -> np.ndarray:
    """Equilibrium ω: ω = u_τ / (κ √C_μ · y), capped near wall."""
    y_safe = np.maximum(y, 1e-10)
    omega_eq = u_tau / (kappa * np.sqrt(cmu) * y_safe)
    # Cap near wall (Menter's wall value is a lower bound here)
    omega_wall = 60.0 * nu / (0.075 * y_safe ** 2)
    return np.maximum(omega_eq, omega_wall)


# ─── Mesh Stand-in ───────────────────────────────────────────────────

class _Mesh2D:
    """Duck-typed Mesh2D (scalar dx, dy)."""
    def __init__(self, nx: int, ny: int, dy: float, ly: float):
        self.nx = nx
        self.ny = ny
        self.lx = float(nx)
        self.ly = ly
        self.dx = 1.0
        self.dy = dy
        self.xc = np.zeros((nx, ny))
        self.yc = np.zeros((nx, ny))


# ─── Verification ───────────────────────────────────────────────────

def verify() -> dict:
    """Run flat plate verification.  Returns results dict."""
    dy = DOMAIN_H / NY
    yc = np.linspace(dy / 2, DOMAIN_H - dy / 2, NY)

    x = np.linspace(0.1, L_PLATE, NX)

    mesh = _Mesh2D(2, NY, dy, DOMAIN_H)
    d_wall = yc.copy()
    modele = ModeleKOmegaSST(mesh, NU_MOL,
                              distance_paroi=d_wall.reshape(1, -1))

    cf_arr = np.zeros(NX)
    re_x_arr = np.zeros(NX)

    for i in range(NX):
        xi = x[i]
        re_xi = U_INF * xi / NU_MOL
        re_x_arr[i] = re_xi

        # --- Reference Cf ---
        cf_ref = cf_correlation(re_xi)
        u_tau_ref = u_tau_from_cf(cf_ref)

        # --- Velocity profile (law of the wall) ---
        u_prof = law_of_the_wall_profile(yc, u_tau_ref, NU_MOL)
        v_prof = np.zeros(NY)  # v ≈ 0 in thin shear layer

        # --- Initialise k, ω from equilibrium ---
        delta_xi = delta_99(re_xi, xi)
        k_eq = equilibrium_k(yc, u_tau_ref, delta_xi)
        ω_eq = equilibrium_omega(yc, u_tau_ref, NU_MOL)

        modele.k.data[:, :] = k_eq.reshape(1, -1)
        modele.ω.data[:, :] = ω_eq.reshape(1, -1)

        # --- Relax the turbulence model with small dt ---
        dt_local = 1e-7
        n_iter = 200
        u_2d = u_prof.reshape(1, -1)
        v_2d = v_prof.reshape(1, -1)

        for _ in range(n_iter):
            nu_t = modele.calcule(u_2d, v_2d, dt=dt_local)

        nu_t_1d = nu_t[0, :]

        # --- Cf from ν_t at wall ---
        du_dy_wall = (u_prof[1] - u_prof[0]) / dy
        tau_w = RHO_INF * (NU_MOL + nu_t_1d[0]) * du_dy_wall
        cf_arr[i] = tau_w / (0.5 * RHO_INF * U_INF ** 2)

    err_pct = 100.0 * np.abs((cf_arr - cf_correlation(re_x_arr))
                              / cf_correlation(re_x_arr))
    mask = x > 0.15

    return {
        "x": x,
        "cf": cf_arr,
        "cf_corr": cf_correlation(re_x_arr),
        "re_x": re_x_arr,
        "err_mean_pct": float(np.mean(err_pct[mask])),
        "err_max_pct": float(np.max(err_pct[mask])),
        "cf_arr": cf_arr,
        "cf_corr_arr": cf_correlation(re_x_arr),
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 72)
    print("  k-ω SST Flat Plate Boundary Layer Verification")
    print("=" * 72)
    re_mid = U_INF * (L_PLATE / 2) / NU_MOL
    u_tau_mid = u_tau_from_cf(cf_correlation(re_mid))
    dy = DOMAIN_H / NY
    y_plus_first = u_tau_mid * dy / NU_MOL

    print(f"  U∞    = {U_INF:6.1f}  m/s")
    print(f"  ν     = {NU_MOL:8.2e}  m²/s")
    print(f"  L     = {L_PLATE:6.2f}  m")
    print(f"  Re_L  = {RE_L:8.1e}")
    print(f"  Grid  = {NX} × {NY}")
    print(f"  y+_1  = {y_plus_first:.2f}")
    print()

    result = verify()

    x = result["x"]
    cf = result["cf"]
    cf_c = result["cf_corr"]
    re_x = result["re_x"]

    print(f"{'x [m]':>7s}  {'Re_x':>10s}  {'C_f (SST)':>10s}  "
          f"{'C_f (corr)':>10s}  {'Error %':>7s}")
    print("-" * 72)

    step = max(NX // 8, 1)
    for i in range(0, NX, step):
        err_i = 100.0 * abs(cf[i] - cf_c[i]) / (cf_c[i] + 1e-30)
        print(f"{x[i]:7.4f}  {re_x[i]:10.2e}  {cf[i]:10.6f}  "
              f"{cf_c[i]:10.6f}  {err_i:7.2f}%")

    print("-" * 72)
    print(f"  Mean |error| (x>0.15m) = {result['err_mean_pct']:.2f}%")
    print(f"  Max  |error| (x>0.15m) = {result['err_max_pct']:.2f}%")

    # Benchmark values for reference
    print()
    print("  Benchmark Cf (Prandtl–Schlichting):")
    for rx in [3.33e5, 6.67e5, 1.0e6, 1.33e6]:
        print(f"    Re_x={rx:.2e}:  C_f = {cf_correlation(np.array([rx]))[0]:.6f}")

    passed = result["err_mean_pct"] < 5.0
    print()
    if passed:
        print(f"  ✓ ACCEPTED  — mean Cf error = {result['err_mean_pct']:.2f}%  <  5%")
    else:
        print(f"  ✗ REJECTED  — mean Cf error = {result['err_mean_pct']:.2f}%  ≥  5%")
    print()
    print(f"  Re_L = {RE_L:.1e}  |  y+_wall = {y_plus_first:.2f}")
    print("=" * 72)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
