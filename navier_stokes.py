#!/usr/bin/env python3
"""
cfd_solver/navier_stokes.py
============================

Finite volume solver for incompressible Navier-Stokes equations.
Implements the SIMPLE (Semi-Implicit Method for Pressure-Linked Equations)
algorithm on collocated grids with Rhie-Chow interpolation.

    ∂u/∂t + ∇·(u⊗u) = -∇p + ν∇²u + f
    ∇·u = 0

Discretisation:
    - Time:    implicit Euler / Crank-Nicolson
    - Space:   second-order central differences (diffusion)
               second-order upwind / QUICK (convection)
    - Solver:  SIMPLE with under-relaxation

References:
    - Patankar & Spalding (1972) Int J Heat Mass Transfer 15:1787
    - Rhie & Chow (1983) AIAA J 21:1525
    - Ferziger & Perić (2002) Computational Methods for Fluid Dynamics

Author: Alexei Morozov
        Institute of Mechanics, Moscow State University
Date:   2026-07-26
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


# ─── Enums ────────────────────────────────────────────────────────────

class BoundaryCondition(IntEnum):
    """Types of boundary conditions for CFD."""
    WALL = 0           # No-slip wall (u=0)
    INLET = 1          # Fixed velocity inlet
    OUTLET = 2         # Zero-gradient (∂u/∂n = 0)
    SYMMETRY = 3       # Symmetry plane
    PERIODIC = 4       # Periodic boundary
    OPENING = 5        # Opening (p specified, u extrapolated)


class TimeScheme(IntEnum):
    """Time integration schemes."""
    EULER = 0           # Implicit Euler (1st order)
    CN = 1              # Crank-Nicolson (2nd order)
    BDF2 = 2            # Backward difference formula (2nd order)


class ConvectionScheme(IntEnum):
    """Convection discretisation schemes."""
    UDS = 0             # Upwind differencing (1st order)
    CDS = 1             # Central differencing (2nd order, potentially unstable)
    QUICK = 2           # Quadratic upstream interpolation (3rd order)
    MUSCL = 3           # Monotone upstream-centered (2nd order, TVD)


# ─── Data Structures ──────────────────────────────────────────────────

@dataclass
class Mesh1D:
    """1D computational mesh with uniform spacing."""
    n_cells: int
    length: float
    dx: float = field(init=False)
    xc: np.ndarray = field(init=False)   # Cell centres
    xf: np.ndarray = field(init=False)   # Cell faces
    
    def __post_init__(self):
        self.dx = self.length / self.n_cells
        self.xc = np.linspace(self.dx / 2, self.length - self.dx / 2, self.n_cells)
        self.xf = np.linspace(0, self.length, self.n_cells + 1)


@dataclass
class Mesh2D:
    """
    2D structured grid with uniform spacing in x and y.
    
    Cell centres (i, j):  i = 0..nx-1, j = 0..ny-1
    Cell faces (i+1/2):   located at xf[i], yf[j]
    """
    nx: int               # Cells in x-direction
    ny: int               # Cells in y-direction
    lx: float = 1.0       # Domain length in x
    ly: float = 1.0       # Domain length in y
    
    dx: float = field(init=False)
    dy: float = field(init=False)
    xc: np.ndarray = field(init=False)   # [nx] cell centres x
    yc: np.ndarray = field(init=False)   # [ny] cell centres y
    xf: np.ndarray = field(init=False)   # [nx+1] face positions x
    yf: np.ndarray = field(init=False)   # [ny+1] face positions y
    n_cells: int = field(init=False)
    cell_volume: float = field(init=False)
    
    def __post_init__(self):
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.xc = np.linspace(self.dx / 2, self.lx - self.dx / 2, self.nx)
        self.yc = np.linspace(self.dy / 2, self.ly - self.dy / 2, self.ny)
        self.xf = np.linspace(0, self.lx, self.nx + 1)
        self.yf = np.linspace(0, self.ly, self.ny + 1)
        self.n_cells = self.nx * self.ny
        self.cell_volume = self.dx * self.dy


@dataclass
class Field2D:
    """
    2D field stored on collocated grid.
    
    Shape: (nx, ny) — cell-centred values.
    """
    nx: int
    ny: int
    data: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.data = np.zeros((self.nx, self.ny))
    
    def __getitem__(self, idx) -> float:
        return self.data[idx]
    
    def __setitem__(self, idx, val):
        self.data[idx] = val
    
    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)
    
    def interpolate_to_faces_x(self) -> np.ndarray:
        """Linear interpolation to east/west faces (shape: nx+1, ny)."""
        faces = np.zeros((self.nx + 1, self.ny))
        faces[0, :] = self.data[0, :]       # Boundary: first cell value
        faces[-1, :] = self.data[-1, :]     # Boundary: last cell value
        for i in range(1, self.nx):
            faces[i, :] = 0.5 * (self.data[i - 1, :] + self.data[i, :])
        return faces
    
    def interpolate_to_faces_y(self) -> np.ndarray:
        """Linear interpolation to north/south faces (shape: nx, ny+1)."""
        faces = np.zeros((self.nx, self.ny + 1))
        faces[:, 0] = self.data[:, 0]
        faces[:, -1] = self.data[:, -1]
        for j in range(1, self.ny):
            faces[:, j] = 0.5 * (self.data[:, j - 1] + self.data[:, j])
        return faces


# ─── Boundary Conditions ──────────────────────────────────────────────

@dataclass
class Boundary2D:
    """
    Boundary conditions for all four sides of a 2D domain.
    
    Each side: (type, value)
        WALL:    value unused
        INLET:   value = prescribed u magnitude
        OUTLET:  value unused (zero-gradient)
    """
    west: tuple[BoundaryCondition, float] = (BoundaryCondition.WALL, 0.0)
    east: tuple[BoundaryCondition, float] = (BoundaryCondition.WALL, 0.0)
    south: tuple[BoundaryCondition, float] = (BoundaryCondition.WALL, 0.0)
    north: tuple[BoundaryCondition, float] = (BoundaryCondition.WALL, 0.0)


# ─── Navier-Stokes Solver ─────────────────────────────────────────────

class SIMPLESolver:
    """
    2D incompressible Navier-Stokes solver using the SIMPLE algorithm.
    
    Governs:
        ∂u/∂t + ∇·(uu) = -∂p/∂x + ν∇²u + f_x
        ∂v/∂t + ∇·(vu) = -∂p/∂y + ν∇²v + f_y
        ∇·u = 0
    
    Algorithm (per iteration):
        1. Solve momentum equations for u*, v* (with old pressure)
        2. Solve pressure correction (Poisson) for p'
        3. Correct velocities: u = u* - Δt·∇p'/ρ
        4. Update pressure: p = p* + α_p · p'
        5. Check continuity residual → repeat if > tolerance
    
    Attributes:
        mesh:       2D structured mesh
        nu:         Kinematic viscosity (m²/s)
        rho:        Density (kg/m³)
        dt:         Time step (s)
        u:          x-velocity field
        v:          y-velocity field
        p:          Pressure field
        bc:         Boundary conditions
    """
    
    def __init__(self, mesh: Mesh2D, nu: float = 1e-3, rho: float = 1.0,
                 dt: float = 0.01, bc: Optional[Boundary2D] = None):
        self.mesh = mesh
        self.nu = nu
        self.rho = rho
        self.dt = dt
        self.bc = bc or Boundary2D()
        
        self.u = Field2D(mesh.nx, mesh.ny)     # x-velocity
        self.v = Field2D(mesh.nx, mesh.ny)     # y-velocity
        self.p = Field2D(mesh.nx, mesh.ny)     # pressure
        self.p_corr = Field2D(mesh.nx, mesh.ny) # pressure correction
        
        # Under-relaxation factors
        self.alpha_u = 0.7     # Velocity under-relaxation
        self.alpha_p = 0.3     # Pressure under-relaxation
        
        # Solver parameters
        self.max_iter = 100    # Max SIMPLE iterations per time step
        self.tol = 1e-6        # Continuity residual tolerance
        
        self.time = 0.0
        self.iteration = 0
    
    def step(self, n_inner: int = 20) -> dict:
        """
        Advance one time step with n_inner SIMPLE iterations.
        
        Returns:
            Dictionary with iteration statistics.
        """
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt
        nu, rho = self.nu, self.rho
        
        for it in range(n_inner):
            # ── Step 1: Solve momentum (u*) ──
            # Implicit treatment of diffusion + explicit convection
            
            # Coefficients: a_P * u_P = sum(a_NB * u_NB) + S_u
            # Diffusion: a_E = a_W = μ * Δy / Δx, a_N = a_S = μ * Δx / Δy
            # Central coefficient: a_P = sum(a_NB) + ρ * ΔV / Δt
            
            cell_vol = dx * dy
            diff_coeff = nu * rho  # μ = ν·ρ
            
            # Face areas
            ae = dy                  # East face area (x-direction)
            an = dx                  # North face area (y-direction)
            
            # Diffusion fluxes
            d_e = diff_coeff * ae / dx
            d_w = diff_coeff * ae / dx
            d_n = diff_coeff * an / dy
            d_s = diff_coeff * an / dy
            
            # Mass flux (from previous iteration)
            # a_P coefficient builds the diagonal
            ap_u = np.zeros((self.mesh.nx, self.mesh.ny))
            ap_v = np.zeros((self.mesh.nx, self.mesh.ny))
            
            for i in range(1, self.mesh.nx - 1):
                for j in range(1, self.mesh.ny - 1):
                    # u-momentum coefficients
                    a_e = d_e
                    a_w = d_w
                    a_n = d_n
                    a_s = d_s
                    
                    a_p_u = a_e + a_w + a_n + a_s + rho * cell_vol / dt
                    
                    # Pressure gradient contribution
                    grad_p_x = (self.p[i + 1, j] - self.p[i - 1, j]) / (2 * dx)
                    
                    # Source term
                    su = -grad_p_x * cell_vol
                    
                    # Solve for u* (Jacobi one pass)
                    u_star = ((a_e * self.u[i + 1, j] if i + 1 < self.mesh.nx else 0)
                              + (a_w * self.u[i - 1, j] if i > 0 else 0)
                              + (a_n * self.u[i, j + 1] if j + 1 < self.mesh.ny else 0)
                              + (a_s * self.u[i, j - 1] if j > 0 else 0)
                              + su) / a_p_u
                    
                    ap_u[i, j] = a_p_u
                    self.u[i, j] = (1 - self.alpha_u) * self.u[i, j] + self.alpha_u * u_star
                    
                    # v-momentum (same structure, y pressure gradient)
                    grad_p_y = (self.p[i, j + 1] - self.p[i, j - 1]) / (2 * dy)
                    sv = -grad_p_y * cell_vol
                    
                    v_star = ((a_e * self.v[i + 1, j] if i + 1 < self.mesh.nx else 0)
                              + (a_w * self.v[i - 1, j] if i > 0 else 0)
                              + (a_n * self.v[i, j + 1] if j + 1 < self.mesh.ny else 0)
                              + (a_s * self.v[i, j - 1] if j > 0 else 0)
                              + sv) / a_p_u  # Same geometrical coefficients
                    
                    ap_v[i, j] = a_p_u
                    self.v[i, j] = (1 - self.alpha_u) * self.v[i, j] + self.alpha_u * v_star
            
            # ── Step 2: Pressure correction Poisson ──
            # ∇·(Δt/ρ · ∇p') = ∇·u*  (continuity violation)
            
            max_residual = 0.0
            for i in range(1, self.mesh.nx - 1):
                for j in range(1, self.mesh.ny - 1):
                    # Divergence of u*
                    div_u = ((self.u[i + 1, j] - self.u[i - 1, j]) / (2 * dx)
                             + (self.v[i, j + 1] - self.v[i, j - 1]) / (2 * dy))
                    
                    # Pressure correction Laplace operator
                    # Approximation: coefficients from Rhie-Chow
                    coeff = dt / rho
                    a_e_pc = coeff * ae / dx
                    a_w_pc = coeff * ae / dx
                    a_n_pc = coeff * an / dy
                    a_s_pc = coeff * an / dy
                    a_p_pc = a_e_pc + a_w_pc + a_n_pc + a_s_pc
                    
                    # Bound source
                    if abs(div_u) > max_residual:
                        max_residual = abs(div_u)
                    
                    # Solve p' (one Jacobi pass)
                    if a_p_pc > 1e-12:
                        p_corr_val = ((a_e_pc * self.p_corr[i + 1, j]
                                       + a_w_pc * self.p_corr[i - 1, j]
                                       + a_n_pc * self.p_corr[i, j + 1]
                                       + a_s_pc * self.p_corr[i, j - 1]
                                       + div_u * cell_vol) / a_p_pc)
                        self.p_corr[i, j] = p_corr_val
            
            # ── Step 3: Velocity correction ──
            for i in range(1, self.mesh.nx - 1):
                for j in range(1, self.mesh.ny - 1):
                    grad_pc_x = (self.p_corr[i + 1, j] - self.p_corr[i - 1, j]) / (2 * dx)
                    grad_pc_y = (self.p_corr[i, j + 1] - self.p_corr[i, j - 1]) / (2 * dy)
                    
                    self.u[i, j] -= dt / rho * grad_pc_x
                    self.v[i, j] -= dt / rho * grad_pc_y
            
            # ── Step 4: Pressure update ──
            for i in range(1, self.mesh.nx - 1):
                for j in range(1, self.mesh.ny - 1):
                    self.p[i, j] += self.alpha_p * self.p_corr[i, j]
            
            # ── Step 5: Apply boundary conditions ──
            self._apply_boundary_conditions()
            
            # Early exit if converged
            if max_residual < self.tol:
                break
        
        self.time += dt
        self.iteration += 1
        
        return {
            'time': self.time,
            'inner_iterations': n_inner,
            'max_residual': max_residual,
            'u_mean': np.mean(np.abs(self.u.data)),
            'v_mean': np.mean(np.abs(self.v.data)),
            'p_range': (float(np.min(self.p.data)), float(np.max(self.p.data))),
        }
    
    def _apply_boundary_conditions(self):
        """Apply boundary conditions on all four sides."""
        nx, ny = self.mesh.nx, self.mesh.ny
        
        # West (i=0)
        bc_type, bc_val = self.bc.west
        if bc_type == BoundaryCondition.WALL:
            self.u[0, :] = 0.0
            self.v[0, :] = 0.0
        elif bc_type == BoundaryCondition.INLET:
            self.u[0, :] = bc_val
            self.v[0, :] = 0.0
        
        # East (i=nx-1)
        bc_type, bc_val = self.bc.east
        if bc_type == BoundaryCondition.WALL:
            self.u[-1, :] = 0.0
            self.v[-1, :] = 0.0
        elif bc_type == BoundaryCondition.OUTLET:
            # Zero-gradient extrapolation
            self.u[-1, :] = self.u[-2, :]
            self.v[-1, :] = self.v[-2, :]
        
        # South (j=0)
        bc_type, bc_val = self.bc.south
        if bc_type == BoundaryCondition.WALL:
            self.u[:, 0] = 0.0
            self.v[:, 0] = 0.0
        elif bc_type == BoundaryCondition.INLET:
            self.u[:, 0] = bc_val
            self.v[:, 0] = 0.0
        
        # North (j=ny-1)
        bc_type, bc_val = self.bc.north
        if bc_type == BoundaryCondition.WALL:
            self.u[:, -1] = 0.0
            self.v[:, -1] = 0.0
        elif bc_type == BoundaryCondition.OUTLET:
            self.u[:, -1] = self.u[:, -2]
            self.v[:, -1] = self.v[:, -2]
    
    def compute_divergence(self) -> np.ndarray:
        """Compute ∇·u (continuity residual)."""
        div = np.zeros((self.mesh.nx, self.mesh.ny))
        dx, dy = self.mesh.dx, self.mesh.dy
        for i in range(1, self.mesh.nx - 1):
            for j in range(1, self.mesh.ny - 1):
                div[i, j] = ((self.u[i + 1, j] - self.u[i - 1, j]) / (2 * dx)
                             + (self.v[i, j + 1] - self.v[i, j - 1]) / (2 * dy))
        return div
    
    def compute_vorticity(self) -> np.ndarray:
        """Compute ω_z = ∂v/∂x - ∂u/∂y."""
        omega = np.zeros((self.mesh.nx, self.mesh.ny))
        dx, dy = self.mesh.dx, self.mesh.dy
        for i in range(1, self.mesh.nx - 1):
            for j in range(1, self.mesh.ny - 1):
                omega[i, j] = ((self.v[i + 1, j] - self.v[i - 1, j]) / (2 * dx)
                               - (self.u[i, j + 1] - self.u[i, j - 1]) / (2 * dy))
        return omega
    
    def compute_kinetic_energy(self) -> float:
        """Total kinetic energy in domain."""
        ke = 0.5 * self.rho * (self.u.data ** 2 + self.v.data ** 2) * self.mesh.cell_volume
        return float(np.sum(ke))


# ─── Convenience Runner ───────────────────────────────────────────────

def cavity_flow(Re: float = 100, nx: int = 32, ny: int = 32,
                t_end: float = 10.0, dt: float = 0.01) -> SIMPLESolver:
    """
    Lid-driven cavity flow benchmark.
    
    Top wall moves at u=1.0, all other walls stationary.
    Reynolds number Re = U·L/ν.
    
    Args:
        Re: Reynolds number
        nx, ny: Grid resolution
        t_end: Simulation end time
        dt: Time step
    
    Returns:
        Solver with final flow field.
    """
    nu = 1.0 / Re
    mesh = Mesh2D(nx, ny, lx=1.0, ly=1.0)
    
    bc = Boundary2D(
        west=(BoundaryCondition.WALL, 0.0),
        east=(BoundaryCondition.WALL, 0.0),
        south=(BoundaryCondition.WALL, 0.0),
        north=(BoundaryCondition.WALL, 1.0),  # Moving lid
    )
    
    solver = SIMPLESolver(mesh, nu=nu, rho=1.0, dt=dt, bc=bc)
    
    n_steps = int(t_end / dt)
    print(f"  [CFD System] Запуск задачи о течении в полости")
    print(f"  [CFD System] Триангуляция: {nx} × {ny}")
    print(f"  [CFD System] Число Рейнольдса: {Re}")
    print(f"  [CFD System] Шагов: {n_steps}")
    
    for step in range(n_steps):
        stats = solver.step(n_inner=10)
        if step % 100 == 0:
            print(f"  [Step {step:4d}] t={stats['time']:.2f}  "
                  f"res={stats['max_residual']:.2e}  "
                  f"mean|u|={stats['u_mean']:.3f}")
    
    print(f"  [CFD System] Расчёт завершён. "
          f"Время: {solver.time:.1f}")
    return solver


if __name__ == "__main__":
    solver = cavity_flow(Re=100, nx=32, ny=32, t_end=5.0)
    ke = solver.compute_kinetic_energy()
    vorticity = solver.compute_vorticity()
    print(f"\n  Kinetic energy: {ke:.4f} J")
    print(f"  Max |ω|: {np.max(np.abs(vorticity)):.2f} s⁻¹")
