#!/usr/bin/env python3
"""
solver/gpu_simple.py — GPU-accelerated SIMPLE solver.

Uses CuPy + CUDA kernels to replace CPU numpy loops.
Inherits same API as SIMPLESolver for drop-in compatibility.

Author: Ming Zhou (USTC)
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
from navier_stokes import Mesh2D, Field2D, Boundary2D, BoundaryCondition

try:
    import cupy as cp
except ImportError:
    raise ImportError("Requires CuPy: pip install cupy-cuda12x")

import sys
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from cuda_bridge import CUDABridge, CUDAPoissonSolver, CUDAFaceInterp, CUDAGradient


class GPUSIMPLESolver:
    """
    GPU-accelerated SIMPLE solver.

    All fields stored as CuPy arrays on GPU.
    Access via .u, .v, .p properties (returns Field2D-compatible views).

    Parameters:
        mesh:    Mesh2D instance
        nu:      kinematic viscosity
        rho:     density
        dt:      time step
        bc:      Boundary2D
        device:  GPU device ID
    """

    def __init__(self, mesh: Mesh2D, nu: float = 1e-3, rho: float = 1.0,
                 dt: float = 0.01, bc: Optional[Boundary2D] = None,
                 device: int = 0):
        self.mesh = mesh
        self.nu = nu
        self.rho = rho
        self.dt = dt
        self.bc = bc or Boundary2D()

        nx, ny = mesh.nx, mesh.ny

        with cp.cuda.Device(device):
            self._u = cp.zeros((nx, ny), dtype=cp.float32)
            self._v = cp.zeros((nx, ny), dtype=cp.float32)
            self._p = cp.zeros((nx, ny), dtype=cp.float32)
            self._p_corr = cp.zeros((nx, ny), dtype=cp.float32)

        self.alpha_u = 0.7
        self.alpha_p = 0.3
        self.tol = 1e-6
        self.n_inner = 20
        self.time = 0.0
        self.iteration = 0
        self.history: list[dict] = []

        self._bridge = CUDABridge(nx, ny, mesh.dx, mesh.dy, dt, rho, device)
        self._poisson = self._bridge.poisson
        self._gradient = self._bridge.gradient

    @property
    def u(self) -> Field2D:
        f = Field2D(self.mesh.nx, self.mesh.ny)
        f.data = cp.asnumpy(self._u)
        return f

    @u.setter
    def u(self, val):
        if isinstance(val, Field2D):
            self._u = cp.asarray(val.data, dtype=cp.float32)
        else:
            self._u = cp.asarray(val, dtype=cp.float32)

    @property
    def v(self) -> Field2D:
        f = Field2D(self.mesh.nx, self.mesh.ny)
        f.data = cp.asnumpy(self._v)
        return f

    @v.setter
    def v(self, val):
        if isinstance(val, Field2D):
            self._v = cp.asarray(val.data, dtype=cp.float32)
        else:
            self._v = cp.asarray(val, dtype=cp.float32)

    @property
    def p(self) -> Field2D:
        f = Field2D(self.mesh.nx, self.mesh.ny)
        f.data = cp.asnumpy(self._p)
        return f

    @p.setter
    def p(self, val):
        if isinstance(val, Field2D):
            self._p = cp.asarray(val.data, dtype=cp.float32)
        else:
            self._p = cp.asarray(val, dtype=cp.float32)

    @property
    def p_corr(self) -> Field2D:
        f = Field2D(self.mesh.nx, self.mesh.ny)
        f.data = cp.asnumpy(self._p_corr)
        return f

    @p_corr.setter
    def p_corr(self, val):
        if isinstance(val, Field2D):
            self._p_corr = cp.asarray(val.data, dtype=cp.float32)
        else:
            self._p_corr = cp.asarray(val, dtype=cp.float32)

    def _apply_u_bc(self, u):
        nx, ny = self.mesh.nx, self.mesh.ny
        bc_w, bc_wv = self.bc.west
        if bc_w == BoundaryCondition.WALL:
            u[0, :] = 0.0
        elif bc_w == BoundaryCondition.INLET:
            u[0, :] = bc_wv
        elif bc_w == BoundaryCondition.OUTLET:
            u[0, :] = u[1, :]
        bc_e, bc_ev = self.bc.east
        if bc_e == BoundaryCondition.WALL:
            u[-1, :] = 0.0
        elif bc_e == BoundaryCondition.INLET:
            u[-1, :] = bc_ev
        elif bc_e == BoundaryCondition.OUTLET:
            u[-1, :] = u[-2, :]
        if self.bc.south[0] == BoundaryCondition.WALL:
            u[:, 0] = 0.0
        else:
            u[:, 0] = u[:, 1]
        if self.bc.north[0] == BoundaryCondition.WALL:
            u[:, -1] = 0.0
        else:
            u[:, -1] = u[:, -2]

    def _apply_v_bc(self, v):
        nx, ny = self.mesh.nx, self.mesh.ny
        if self.bc.west[0] == BoundaryCondition.WALL:
            v[0, :] = 0.0
        else:
            v[0, :] = v[1, :]
        if self.bc.east[0] == BoundaryCondition.WALL:
            v[-1, :] = 0.0
        else:
            v[-1, :] = v[-2, :]
        bc_s, bc_sv = self.bc.south
        if bc_s == BoundaryCondition.WALL:
            v[:, 0] = 0.0
        elif bc_s == BoundaryCondition.INLET:
            v[:, 0] = bc_sv
        elif bc_s == BoundaryCondition.OUTLET:
            v[:, 0] = v[:, 1]
        bc_n, bc_nv = self.bc.north
        if bc_n == BoundaryCondition.WALL:
            v[:, -1] = 0.0
        elif bc_n == BoundaryCondition.INLET:
            v[:, -1] = bc_nv
        elif bc_n == BoundaryCondition.OUTLET:
            v[:, -1] = v[:, -2]

    def _apply_p_corr_bc(self, p_corr):
        p_corr[0, :] = p_corr[1, :]
        p_corr[-1, :] = p_corr[-2, :]
        p_corr[:, 0] = p_corr[:, 1]
        p_corr[:, -1] = p_corr[:, -2]

    def _build_momentum_coeffs(self, u_old, v_old):
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        cell_vol = dx * dy
        mu = self.nu * self.rho
        dt = self.dt

        de = cp.float32(mu * dy / dx)
        dn = cp.float32(mu * dx / dy)

        u_faces = cp.zeros((nx + 1, ny), dtype=cp.float32)
        u_faces[1:-1, :] = 0.5 * (u_old[:-1, :] + u_old[1:, :])
        u_faces[0, :] = u_old[0, :]
        u_faces[-1, :] = u_old[-1, :]
        mf_x = self.rho * u_faces * dy

        v_faces = cp.zeros((nx, ny + 1), dtype=cp.float32)
        v_faces[:, 1:-1] = 0.5 * (v_old[:, :-1] + v_old[:, 1:])
        v_faces[:, 0] = v_old[:, 0]
        v_faces[:, -1] = v_old[:, -1]
        mf_y = self.rho * v_faces * dx

        aE = cp.zeros((nx, ny), dtype=cp.float32)
        aW = cp.zeros((nx, ny), dtype=cp.float32)
        aN = cp.zeros((nx, ny), dtype=cp.float32)
        aS = cp.zeros((nx, ny), dtype=cp.float32)
        aP = cp.zeros((nx, ny), dtype=cp.float32)

        mf_x_e = mf_x[1:, :]
        aE[:-1, :] = de + cp.maximum(-mf_x_e, 0.0)
        mf_x_w = mf_x[:-1, :]
        aW[1:, :] = de + cp.maximum(mf_x_w, 0.0)
        aP[:-1, :] += de + cp.maximum(mf_x_e, 0.0)
        aP[1:, :]  += de + cp.maximum(-mf_x_w, 0.0)

        mf_y_n = mf_y[:, 1:]
        aN[:, :-1] = dn + cp.maximum(-mf_y_n, 0.0)
        mf_y_s = mf_y[:, :-1]
        aS[:, 1:] = dn + cp.maximum(mf_y_s, 0.0)
        aP[:, :-1] += dn + cp.maximum(mf_y_n, 0.0)
        aP[:, 1:]  += dn + cp.maximum(-mf_y_s, 0.0)

        aP += self.rho * cell_vol / dt

        grad_p_x = cp.zeros((nx, ny), dtype=cp.float32)
        grad_p_x[1:-1, :] = (self._p[2:, :] - self._p[:-2, :]) / (2.0 * dx)
        grad_p_x[0, :] = (self._p[1, :] - self._p[0, :]) / dx
        grad_p_x[-1, :] = (self._p[-1, :] - self._p[-2, :]) / dx

        grad_p_y = cp.zeros((nx, ny), dtype=cp.float32)
        grad_p_y[:, 1:-1] = (self._p[:, 2:] - self._p[:, :-2]) / (2.0 * dy)
        grad_p_y[:, 0] = (self._p[:, 1] - self._p[:, 0]) / dy
        grad_p_y[:, -1] = (self._p[:, -1] - self._p[:, -2]) / dy

        Su = -grad_p_x * cell_vol + self.rho * cell_vol / dt * u_old
        Sv = -grad_p_y * cell_vol + self.rho * cell_vol / dt * v_old

        return aP, aE, aW, aN, aS, Su, Sv

    def _jacobi_momentum(self, aP, aE, aW, aN, aS, Su, Sv, u_old, v_old):
        nx, ny = self.mesh.nx, self.mesh.ny

        u_neighbors = cp.zeros((nx, ny), dtype=cp.float32)
        v_neighbors = cp.zeros((nx, ny), dtype=cp.float32)
        u_neighbors[:-1, :] += aE[:-1, :] * u_old[1:, :]
        v_neighbors[:-1, :] += aE[:-1, :] * v_old[1:, :]
        u_neighbors[1:, :]  += aW[1:, :]  * u_old[:-1, :]
        v_neighbors[1:, :]  += aW[1:, :]  * v_old[:-1, :]
        u_neighbors[:, :-1] += aN[:, :-1] * u_old[:, 1:]
        v_neighbors[:, :-1] += aN[:, :-1] * v_old[:, 1:]
        u_neighbors[:, 1:]  += aS[:, 1:]  * u_old[:, :-1]
        v_neighbors[:, 1:]  += aS[:, 1:]  * v_old[:, :-1]

        u_star = (u_neighbors + Su) / (aP + 1e-30)
        v_star = (v_neighbors + Sv) / (aP + 1e-30)

        u_star[0, :] = u_old[0, :]
        u_star[-1, :] = u_old[-1, :]
        u_star[:, 0] = u_old[:, 0]
        u_star[:, -1] = u_old[:, -1]
        v_star[0, :] = v_old[0, :]
        v_star[-1, :] = v_old[-1, :]
        v_star[:, 0] = v_old[:, 0]
        v_star[:, -1] = v_old[:, -1]

        return u_star, v_star

    def _build_pressure_correction(self, u_star, v_star, aP):
        nx, ny = self.mesh.nx, self.mesh.ny
        cell_vol = self.mesh.dx * self.mesh.dy
        dx, dy = self.mesh.dx, self.mesh.dy
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        div_u = cp.zeros((nx, ny), dtype=cp.float32)
        u_e = 0.5 * (u_star[1:, :] + u_star[:-1, :])
        v_n = 0.5 * (v_star[:, 1:] + v_star[:, :-1])
        div_u[1:-1, 1:-1] = (
            (u_e[1:, 1:-1] - u_e[:-1, 1:-1]) * inv_dx
            + (v_n[1:-1, 1:] - v_n[1:-1, :-1]) * inv_dy
        )

        rhs_p = -self.rho * div_u * cell_vol
        Df = cell_vol / (aP + 1e-30)
        return rhs_p, Df

    def _velocity_correction(self, u_star, v_star, p_corr, Df):
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy

        grad_pc_x = cp.zeros((nx, ny), dtype=cp.float32)
        grad_pc_y = cp.zeros((nx, ny), dtype=cp.float32)
        grad_pc_x[1:-1, :] = (p_corr[2:, :] - p_corr[:-2, :]) / (2.0 * dx)
        grad_pc_y[:, 1:-1] = (p_corr[:, 2:] - p_corr[:, :-2]) / (2.0 * dy)

        u_new = u_star - Df * grad_pc_x
        v_new = v_star - Df * grad_pc_y
        return u_new, v_new

    def step(self, n_inner: Optional[int] = None) -> dict:
        if n_inner is None:
            n_inner = self.n_inner

        u_old = self._u.copy()
        v_old = self._v.copy()
        max_residual = 0.0

        for inner_it in range(n_inner):
            aP, aE, aW, aN, aS, Su, Sv = self._build_momentum_coeffs(u_old, v_old)
            u_star, v_star = self._jacobi_momentum(aP, aE, aW, aN, aS, Su, Sv, u_old, v_old)
            self._apply_u_bc(u_star)
            self._apply_v_bc(v_star)

            rhs_p, Df = self._build_pressure_correction(u_star, v_star, aP)
            self._p_corr.fill(0.0)
            self._p_corr, res = self._poisson.solve(rhs_p, self._p_corr, n_iter=50, tol=1e-6)
            self._apply_p_corr_bc(self._p_corr)

            u_new, v_new = self._velocity_correction(u_star, v_star, self._p_corr, Df)
            self._p += self.alpha_p * self._p_corr
            self._u = (1.0 - self.alpha_u) * u_old + self.alpha_u * u_new
            self._v = (1.0 - self.alpha_u) * v_old + self.alpha_u * v_new

            self._apply_u_bc(self._u)
            self._apply_v_bc(self._v)

            div = self._gradient.divergence(self._u, self._v)
            max_residual = float(cp.max(cp.abs(div)))
            if max_residual < self.tol:
                break

        self.time += self.dt
        self.iteration += 1
        stats = {
            'time': self.time,
            'inner_iterations': inner_it + 1,
            'max_residual': max_residual,
            'u_mean': float(cp.mean(cp.abs(self._u))),
            'v_mean': float(cp.mean(cp.abs(self._v))),
            'p_range': (float(cp.min(self._p)), float(cp.max(self._p))),
            'continuity_error': max_residual,
        }
        self.history.append(stats)
        return stats

    def compute_vorticity(self) -> cp.ndarray:
        return self._gradient.vorticity(self._u, self._v)

    def compute_divergence(self) -> cp.ndarray:
        return self._gradient.divergence(self._u, self._v)

    def compute_kinetic_energy(self) -> float:
        cell_vol = self.mesh.dx * self.mesh.dy
        ke = 0.5 * self.rho * (self._u ** 2 + self._v ** 2) * cell_vol
        return float(cp.sum(ke))


def cavity_flow_gpu(Re=100, nx=32, ny=32, t_end=10.0, dt=0.01, device=0, report=True):
    nu = 1.0 / Re
    mesh = Mesh2D(nx, ny, lx=1.0, ly=1.0)
    bc = Boundary2D(
        west=(BoundaryCondition.WALL, 0.0),
        east=(BoundaryCondition.WALL, 0.0),
        south=(BoundaryCondition.WALL, 0.0),
        north=(BoundaryCondition.WALL, 1.0),
    )
    solver = GPUSIMPLESolver(mesh, nu=nu, rho=1.0, dt=dt, bc=bc, device=device)
    n_steps = int(t_end / dt)

    if report:
        print(f"  [GPU-SIMPLE] Cavity flow Re={Re}, grid {nx}×{ny}, {n_steps} steps")
    for step in range(n_steps):
        stats = solver.step(n_inner=10)
        if report and step % 100 == 0:
            print(f"  [Step {step:4d}] t={stats['time']:.2f}  res={stats['max_residual']:.2e}")
    return solver


if __name__ == "__main__":
    solver = cavity_flow_gpu(Re=100, nx=32, ny=32, t_end=1.0, dt=0.01, report=True)
    print(f"  Done. KE={solver.compute_kinetic_energy():.4f}")
