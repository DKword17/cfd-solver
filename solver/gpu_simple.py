#!/usr/bin/env python3
"""solver/gpu_simple.py — GPU-accelerated SIMPLE solver."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from navier_stokes import Mesh2D, Field2D, Boundary2D, BoundaryCondition

try:
    import cupy as cp
except ImportError:
    raise ImportError("pip install cupy-cuda12x")

import sys
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from cuda_bridge import CUDABridge, CUDAPoissonSolver, CUDAGradient


class GPUSIMPLESolver:
    def __init__(self, mesh: Mesh2D, nu=1e-3, rho=1.0, dt=0.01,
                 bc: Boundary2D = None, device=0):
        self.mesh = mesh
        self.nu, self.rho, self.dt = nu, rho, dt
        self.bc = bc or Boundary2D()
        self.device = device
        nx, ny = mesh.nx, mesh.ny

        with cp.cuda.Device(device):
            self._u = cp.zeros((nx, ny), dtype=cp.float32)
            self._v = cp.zeros((nx, ny), dtype=cp.float32)
            self._p = cp.zeros((nx, ny), dtype=cp.float32)
            self._p_corr = cp.zeros((nx, ny), dtype=cp.float32)

        self.alpha_u, self.alpha_p = 0.7, 0.3
        self.tol = 1e-6
        self.n_inner = 20
        self.time = 0.0
        self.iteration = 0
        self.history = []

        self._bridge = CUDABridge(nx, ny, mesh.dx, mesh.dy, dt, rho, device)
        self._poisson = self._bridge.poisson
        self._gradient = self._bridge.gradient

    @property
    def u(self):
        f = Field2D(self.mesh.nx, self.mesh.ny); f.data = cp.asnumpy(self._u); return f
    @u.setter
    def u(self, v):
        self._u = cp.asarray(v.data if isinstance(v, Field2D) else v, dtype=cp.float32)
    @property
    def v(self):
        f = Field2D(self.mesh.nx, self.mesh.ny); f.data = cp.asnumpy(self._v); return f
    @v.setter
    def v(self, v):
        self._v = cp.asarray(v.data if isinstance(v, Field2D) else v, dtype=cp.float32)
    @property
    def p(self):
        f = Field2D(self.mesh.nx, self.mesh.ny); f.data = cp.asnumpy(self._p); return f
    @p.setter
    def p(self, v):
        self._p = cp.asarray(v.data if isinstance(v, Field2D) else v, dtype=cp.float32)
    @property
    def p_corr(self):
        f = Field2D(self.mesh.nx, self.mesh.ny); f.data = cp.asnumpy(self._p_corr); return f
    @p_corr.setter
    def p_corr(self, v):
        self._p_corr = cp.asarray(v.data if isinstance(v, Field2D) else v, dtype=cp.float32)

    def _apply_u_bc(self, u):
        nx, ny = self.mesh.nx, self.mesh.ny
        b = self.bc
        if b.west[0] == BoundaryCondition.WALL: u[0, :] = 0.0
        elif b.west[0] == BoundaryCondition.INLET: u[0, :] = b.west[1]
        elif b.west[0] == BoundaryCondition.OUTLET: u[0, :] = u[1, :]
        if b.east[0] == BoundaryCondition.WALL: u[-1, :] = 0.0
        elif b.east[0] == BoundaryCondition.INLET: u[-1, :] = b.east[1]
        elif b.east[0] == BoundaryCondition.OUTLET: u[-1, :] = u[-2, :]
        if b.south[0] == BoundaryCondition.WALL: u[:, 0] = 0.0
        else: u[:, 0] = u[:, 1]
        if b.north[0] == BoundaryCondition.WALL: u[:, -1] = 0.0
        else: u[:, -1] = u[:, -2]

    def _apply_v_bc(self, v):
        nx, ny = self.mesh.nx, self.mesh.ny
        b = self.bc
        if b.west[0] == BoundaryCondition.WALL: v[0, :] = 0.0
        else: v[0, :] = v[1, :]
        if b.east[0] == BoundaryCondition.WALL: v[-1, :] = 0.0
        else: v[-1, :] = v[-2, :]
        if b.south[0] == BoundaryCondition.WALL: v[:, 0] = 0.0
        elif b.south[0] == BoundaryCondition.INLET: v[:, 0] = b.south[1]
        elif b.south[0] == BoundaryCondition.OUTLET: v[:, 0] = v[:, 1]
        if b.north[0] == BoundaryCondition.WALL: v[:, -1] = 0.0
        elif b.north[0] == BoundaryCondition.INLET: v[:, -1] = b.north[1]
        elif b.north[0] == BoundaryCondition.OUTLET: v[:, -1] = v[:, -2]

    def _apply_p_corr_bc(self, p):
        p[0, :] = p[1, :]; p[-1, :] = p[-2, :]
        p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]

    def _build_coeffs(self, u_old, v_old):
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        cell_vol = dx * dy
        mu = self.nu * self.rho
        dt = self.dt
        de = cp.float32(mu * dy / dx)
        dn = cp.float32(mu * dx / dy)

        uf = cp.zeros((nx+1, ny), dtype=cp.float32)
        uf[1:-1, :] = 0.5*(u_old[:-1, :]+u_old[1:, :]); uf[0, :]=u_old[0, :]; uf[-1, :]=u_old[-1, :]
        mfx = self.rho * uf * dy
        vf = cp.zeros((nx, ny+1), dtype=cp.float32)
        vf[:, 1:-1] = 0.5*(v_old[:, :-1]+v_old[:, 1:]); vf[:, 0]=v_old[:, 0]; vf[:, -1]=v_old[:, -1]
        mfy = self.rho * vf * dx

        aE = cp.zeros((nx, ny), dtype=cp.float32)
        aW = cp.zeros((nx, ny), dtype=cp.float32)
        aN = cp.zeros((nx, ny), dtype=cp.float32)
        aS = cp.zeros((nx, ny), dtype=cp.float32)
        aP = cp.zeros((nx, ny), dtype=cp.float32)

        mxe = mfx[1:, :]
        aE[:-1, :] = de + cp.maximum(-mxe, 0.0)
        mxw = mfx[:-1, :]
        aW[1:, :] = de + cp.maximum(mxw, 0.0)
        aP[:-1, :] += de + cp.maximum(mxe, 0.0)
        aP[1:, :] += de + cp.maximum(-mxw, 0.0)
        myn = mfy[:, 1:]
        aN[:, :-1] = dn + cp.maximum(-myn, 0.0)
        mys = mfy[:, :-1]
        aS[:, 1:] = dn + cp.maximum(mys, 0.0)
        aP[:, :-1] += dn + cp.maximum(myn, 0.0)
        aP[:, 1:] += dn + cp.maximum(-mys, 0.0)
        aP += self.rho * cell_vol / dt

        gpx = cp.zeros((nx, ny), dtype=cp.float32)
        gpx[1:-1, :] = (self._p[2:, :]-self._p[:-2, :])/(2.0*dx)
        gpx[0, :] = (self._p[1, :]-self._p[0, :])/dx
        gpx[-1, :] = (self._p[-1, :]-self._p[-2, :])/dx
        gpy = cp.zeros((nx, ny), dtype=cp.float32)
        gpy[:, 1:-1] = (self._p[:, 2:]-self._p[:, :-2])/(2.0*dy)
        gpy[:, 0] = (self._p[:, 1]-self._p[:, 0])/dy
        gpy[:, -1] = (self._p[:, -1]-self._p[:, -2])/dy

        Su = -gpx*cell_vol + self.rho*cell_vol/dt*u_old
        Sv = -gpy*cell_vol + self.rho*cell_vol/dt*v_old
        return aP, aE, aW, aN, aS, Su, Sv

    def _jacobi(self, aP, aE, aW, aN, aS, Su, Sv, uo, vo):
        nx, ny = self.mesh.nx, self.mesh.ny
        un = cp.zeros((nx, ny), dtype=cp.float32)
        vn = cp.zeros((nx, ny), dtype=cp.float32)
        un[:-1, :] += aE[:-1, :]*uo[1:, :]
        vn[:-1, :] += aE[:-1, :]*vo[1:, :]
        un[1:, :] += aW[1:, :]*uo[:-1, :]
        vn[1:, :] += aW[1:, :]*vo[:-1, :]
        un[:, :-1] += aN[:, :-1]*uo[:, 1:]
        vn[:, :-1] += aN[:, :-1]*vo[:, 1:]
        un[:, 1:] += aS[:, 1:]*uo[:, :-1]
        vn[:, 1:] += aS[:, 1:]*vo[:, :-1]
        us = (un+Su)/(aP+1e-30)
        vs = (vn+Sv)/(aP+1e-30)
        us[0,:]=uo[0,:]; us[-1,:]=uo[-1,:]; us[:,0]=uo[:,0]; us[:,-1]=uo[:,-1]
        vs[0,:]=vo[0,:]; vs[-1,:]=vo[-1,:]; vs[:,0]=vo[:,0]; vs[:,-1]=vo[:,-1]
        return us, vs

    def _build_pc(self, us, vs, aP):
        nx, ny = self.mesh.nx, self.mesh.ny
        cv = self.mesh.dx*self.mesh.dy
        id_x, id_y = 1.0/self.mesh.dx, 1.0/self.mesh.dy
        du = cp.zeros((nx, ny), dtype=cp.float32)
        ue = 0.5*(us[1:, :]+us[:-1, :])
        vn = 0.5*(vs[:, 1:]+vs[:, :-1])
        du[1:-1, 1:-1] = (ue[1:, 1:-1]-ue[:-1, 1:-1])*id_x + (vn[1:-1, 1:]-vn[1:-1, :-1])*id_y
        return -self.rho*du*cv, cv/(aP+1e-30)

    def _vel_corr(self, us, vs, pc, Df):
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        gx = cp.zeros((nx, ny), dtype=cp.float32)
        gy = cp.zeros((nx, ny), dtype=cp.float32)
        gx[1:-1, :] = (pc[2:, :]-pc[:-2, :])/(2.0*dx)
        gy[:, 1:-1] = (pc[:, 2:]-pc[:, :-2])/(2.0*dy)
        return us-Df*gx, vs-Df*gy

    def step(self, n_inner=None):
        if n_inner is None:
            n_inner = self.n_inner
        uo, vo = self._u.copy(), self._v.copy()
        mr = 0.0
        for it in range(n_inner):
            aP, aE, aW, aN, aS, Su, Sv = self._build_coeffs(uo, vo)
            us, vs = self._jacobi(aP, aE, aW, aN, aS, Su, Sv, uo, vo)
            self._apply_u_bc(us); self._apply_v_bc(vs)
            rp, Df = self._build_pc(us, vs, aP)
            self._p_corr.fill(0.0)
            self._p_corr, _ = self._poisson.solve(rp, self._p_corr, 50, 1e-6)
            self._apply_p_corr_bc(self._p_corr)
            un, vn = self._vel_corr(us, vs, self._p_corr, Df)
            self._p += self.alpha_p*self._p_corr
            self._u = (1-self.alpha_u)*uo + self.alpha_u*un
            self._v = (1-self.alpha_u)*vo + self.alpha_u*vn
            self._apply_u_bc(self._u); self._apply_v_bc(self._v)
            div = self._gradient.divergence(self._u, self._v)
            mr = float(cp.max(cp.abs(div)))
            if mr < self.tol:
                break
        self.time += self.dt; self.iteration += 1
        s = {'time': self.time, 'inner_iterations': it+1, 'max_residual': mr,
             'u_mean': float(cp.mean(cp.abs(self._u))),
             'v_mean': float(cp.mean(cp.abs(self._v)))}
        self.history.append(s)
        return s

    def compute_vorticity(self):
        return self._gradient.vorticity(self._u, self._v)

    def compute_kinetic_energy(self):
        cv = self.mesh.dx*self.mesh.dy
        return float(cp.sum(0.5*self.rho*(self._u**2+self._v**2)*cv))


def cavity_flow_gpu(Re=100, nx=32, ny=32, t_end=10.0, dt=0.01, device=0, report=True):
    nu = 1.0/Re
    mesh = Mesh2D(nx, ny, lx=1.0, ly=1.0)
    bc = Boundary2D(west=(BoundaryCondition.WALL, 0.0),
                    east=(BoundaryCondition.WALL, 0.0),
                    south=(BoundaryCondition.WALL, 0.0),
                    north=(BoundaryCondition.WALL, 1.0))
    slv = GPUSIMPLESolver(mesh, nu=nu, rho=1.0, dt=dt, bc=bc, device=device)
    ns = int(t_end/dt)
    if report:
        print(f"  [GPU] Cavity Re={Re} {nx}x{ny} {ns} steps")
    for s in range(ns):
        st = slv.step(10)
        if report and s%100==0:
            print(f"  Step {s:4d} t={st['time']:.2f} res={st['max_residual']:.2e}")
    return slv


if __name__=="__main__":
    s=cavity_flow_gpu(100,32,32,1.0,0.01,0,True)
    print(f"  KE={s.compute_kinetic_energy():.4f}")
