#!/usr/bin/env python3
"""
cuda_bridge.py — CuPy Python 接口

将 kernel/*.cu 中的 CUDA 核加载、编译并暴露为 Python API.
与 navier_stokes.py 中的 Mesh2D, Field2D 类型兼容.

依赖:
    cupy >= 12.0  (推荐 13.x)
    numpy

使用方式:
    from cuda_bridge import CUDABridge

    bridge = CUDABridge(nx=128, ny=128, dx=0.01, dy=0.01, dt=0.01, rho=1.0)
    bridge.solve_poisson(rhs_gpu, p_gpu, n_iter=100)
    bridge.compute_gradient(phi_faces_x, phi_faces_y)
    ...

基准测试:
    bridge.benchmark_poisson(grid_sizes=[64, 128, 256, 512])
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

_KERNEL_DIR = Path(__file__).parent / "kernel"

# ──────────────────────────────────────────────────────────────────────
# 尝试导入 CuPy
# ──────────────────────────────────────────────────────────────────────
try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "cuda_bridge.py 需要 CuPy. 安装: pip install cupy-cuda12x  "
        "(根据你的 CUDA 版本选择)"
    )


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _grid_2d(nx: int, ny: int, tile_x: int = 16, tile_y: int = 16):
    bx = (nx + tile_x - 1) // tile_x
    by = (ny + tile_y - 1) // tile_y
    return (bx, by), (tile_x, tile_y)


def _grid_1d(n: int, block_size: int = 256):
    return (math.ceil(n / block_size),), (block_size,)


def _load_cu_source(filename: str) -> str:
    path = _KERNEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"CUDA kernel not found: {path}")
    return path.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════
# Poisson 求解器
# ══════════════════════════════════════════════════════════════════════

class CUDAPoissonSolver:
    """
    GPU accelerated pressure Poisson solver using Red-Black Gauss-Seidel.

    Parameters:
        nx, ny: grid dimensions (including boundaries)
        dx, dy: cell spacing
        dt:     time step
        rho:    fluid density
    """
    TILE_X = 16
    TILE_Y = 16

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 dt: float = 0.01, rho: float = 1.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.rho = rho
        self.n_cells = nx * ny

        source = _load_cu_source("laplace_cuda.cu")
        self._module = cp.RawModule(code=source)
        self._krn_rbgs = self._module.get_function("_kernel_poisson_rbgs")
        self._krn_residual = self._module.get_function("_kernel_poisson_residual")
        self._krn_coeff = self._module.get_function("_kernel_poisson_coefficients")

        # GPU coefficient arrays
        self._aE = cp.zeros(self.n_cells, dtype=cp.float32)
        self._aW = cp.zeros(self.n_cells, dtype=cp.float32)
        self._aN = cp.zeros(self.n_cells, dtype=cp.float32)
        self._aS = cp.zeros(self.n_cells, dtype=cp.float32)
        self._aP = cp.zeros(self.n_cells, dtype=cp.float32)
        self._allocate_coefficients()

        # grid/block config
        self._grid2d, self._block2d = _grid_2d(nx, ny, self.TILE_X, self.TILE_Y)
        self._grid1d = (math.ceil(self.n_cells / 256),)
        self._block1d = (256,)

    def _allocate_coefficients(self):
        grid, block = _grid_2d(self.nx, self.ny)
        self._krn_coeff(grid, block, (
            self._aE, self._aW, self._aN, self._aS, self._aP,
            self.nx, self.ny,
            cp.float32(self.dx), cp.float32(self.dy),
            cp.float32(self.dt), cp.float32(self.rho)))

    def update_timestep(self, dt: float):
        self.dt = dt
        self._allocate_coefficients()

    def solve(self, rhs: cp.ndarray, p: cp.ndarray,
              n_iter: int = 50, tol: float = 1e-6,
              report: bool = False) -> tuple[cp.ndarray, float]:
        for it in range(n_iter):
            # red cells (red=1)
            self._krn_rbgs(self._grid2d, self._block2d, (
                p, rhs, self._aE, self._aW, self._aN, self._aS, self._aP,
                self.nx, self.ny, 1))
            # black cells (red=0)
            self._krn_rbgs(self._grid2d, self._block2d, (
                p, rhs, self._aE, self._aW, self._aN, self._aS, self._aP,
                self.nx, self.ny, 0))

            if it % 5 == 0 or it == n_iter - 1:
                res = self._compute_residual(p, rhs)
                if report:
                    print(f"  [Poisson] iter={it:3d}  RMS res={res:.2e}")
                if res < tol:
                    break
        return p, res

    def _compute_residual(self, p: cp.ndarray, rhs: cp.ndarray) -> float:
        res_gpu = cp.zeros(1, dtype=cp.float32)
        self._krn_residual(self._grid1d, self._block1d, (
            p, rhs, self._aE, self._aW, self._aN, self._aS, self._aP,
            self.nx, self.ny, res_gpu))
        total = float(res_gpu[0])
        return math.sqrt(total / self.n_cells)


# ══════════════════════════════════════════════════════════════════════
# Rhie-Chow interpolation
# ══════════════════════════════════════════════════════════════════════

class CUDAFaceInterp:
    def __init__(self, nx: int, ny: int):
        self.nx = nx
        self.ny = ny
        self.n_cells = nx * ny

        source = _load_cu_source("face_interp.cu")
        self._module = cp.RawModule(code=source)
        self._krn_faces_x = self._module.get_function("_kernel_rhie_chow_faces_x")
        self._krn_faces_y = self._module.get_function("_kernel_rhie_chow_faces_y")
        self._krn_df = self._module.get_function("_kernel_compute_Df")

        self._grid_x, self._block_x = _grid_2d(nx + 1, ny)
        self._grid_y, self._block_y = _grid_2d(nx, ny + 1)

    def compute_Df(self, aP: cp.ndarray, cell_volume: float) -> cp.ndarray:
        Df = cp.empty_like(aP)
        grid = (math.ceil(self.n_cells / 256),)
        block = (256,)
        self._krn_df(grid, block, (aP, Df, self.n_cells, cp.float32(cell_volume)))
        return Df

    def interpolate_faces_x(self, phi_c: cp.ndarray, grad_x: cp.ndarray,
                             Df: cp.ndarray, dx: float) -> cp.ndarray:
        phi_f = cp.empty((self.ny, self.nx + 1), dtype=cp.float32)
        self._krn_faces_x(self._grid_x, self._block_x, (
            phi_c, grad_x, Df, phi_f, self.nx, self.ny, cp.float32(dx)))
        return phi_f

    def interpolate_faces_y(self, phi_c: cp.ndarray, grad_y: cp.ndarray,
                             Df: cp.ndarray, dy: float) -> cp.ndarray:
        phi_f = cp.empty((self.ny + 1, self.nx), dtype=cp.float32)
        self._krn_faces_y(self._grid_y, self._block_y, (
            phi_c, grad_y, Df, phi_f, self.nx, self.ny, cp.float32(dy)))
        return phi_f


# ══════════════════════════════════════════════════════════════════════
# TVD limiter
# ══════════════════════════════════════════════════════════════════════

class CUDALimiter:
    MINMOD = 0
    VAN_LEER = 1
    SUPERBEE = 2

    _NAMES = {0: "minmod", 1: "van Leer", 2: "superbee"}

    def __init__(self, nx: int, ny: int):
        self.nx = nx
        self.ny = ny

        source = _load_cu_source("limiter.cu")
        self._module = cp.RawModule(code=source)
        self._krn_x = self._module.get_function("_kernel_tvd_face_x")
        self._krn_y = self._module.get_function("_kernel_tvd_face_y")
        self._krn_val = self._module.get_function("_kernel_limiter_value")

        self._grid_x, self._block_x = _grid_2d(nx - 1, ny)
        self._grid_y, self._block_y = _grid_2d(nx, ny - 1)

    def apply_x(self, phi: cp.ndarray, limiter_type: int = MINMOD) -> cp.ndarray:
        phi_f = cp.empty((self.ny, self.nx + 1), dtype=cp.float32)
        self._krn_x(self._grid_x, self._block_x, (phi, phi_f, self.nx, self.ny, limiter_type))
        return phi_f

    def apply_y(self, phi: cp.ndarray, limiter_type: int = MINMOD) -> cp.ndarray:
        phi_f = cp.empty((self.ny + 1, self.nx), dtype=cp.float32)
        self._krn_y(self._grid_y, self._block_y, (phi, phi_f, self.nx, self.ny, limiter_type))
        return phi_f


# ══════════════════════════════════════════════════════════════════════
# Gradient / divergence / vorticity
# ══════════════════════════════════════════════════════════════════════

class CUDAGradient:
    def __init__(self, nx: int, ny: int, dx: float, dy: float):
        self.nx = nx
        self.ny = ny
        self.inv_dx = cp.float32(1.0 / dx)
        self.inv_dy = cp.float32(1.0 / dy)
        self.inv_vol = cp.float32(1.0 / (dx * dy))

        source = _load_cu_source("gradient.cu")
        self._module = cp.RawModule(code=source)
        self._krn_gg = self._module.get_function("_kernel_gradient_gg")
        self._krn_central = self._module.get_function("_kernel_gradient_central")
        self._krn_div = self._module.get_function("_kernel_divergence")
        self._krn_vort = self._module.get_function("_kernel_vorticity")

        self._grid2d, self._block2d = _grid_2d(nx, ny)
        self._grid_inner, self._block_inner = _grid_2d(nx - 2, ny - 2)

    def green_gauss(self, phi_f_x: cp.ndarray, phi_f_y: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        grad_x = cp.empty((self.ny, self.nx), dtype=cp.float32)
        grad_y = cp.empty((self.ny, self.nx), dtype=cp.float32)
        self._krn_gg(self._grid2d, self._block2d, (
            phi_f_x, phi_f_y, grad_x, grad_y,
            self.nx, self.ny, self.inv_dx, self.inv_dy, self.inv_vol))
        return grad_x, grad_y

    def central(self, phi: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        grad_x = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        grad_y = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self._krn_central(self._grid_inner, self._block_inner, (
            phi, grad_x, grad_y, self.nx, self.ny, self.inv_dx, self.inv_dy))
        return grad_x, grad_y

    def divergence(self, u: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
        div = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self._krn_div(self._grid_inner, self._block_inner, (
            u, v, div, self.nx, self.ny, self.inv_dx, self.inv_dy))
        return div

    def vorticity(self, u: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
        omega = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self._krn_vort(self._grid_inner, self._block_inner, (
            u, v, omega, self.nx, self.ny, self.inv_dx, self.inv_dy))
        return omega


# ══════════════════════════════════════════════════════════════════════
# Top-level bridge
# ══════════════════════════════════════════════════════════════════════

class CUDABridge:
    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 dt: float = 0.01, rho: float = 1.0, device: int = 0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

        cp.cuda.Device(device).use()

        self.poisson = CUDAPoissonSolver(nx, ny, dx, dy, dt, rho)
        self.face_interp = CUDAFaceInterp(nx, ny)
        self.gradient = CUDAGradient(nx, ny, dx, dy)
        self.limiter = CUDALimiter(nx, ny)
        self._warm_up()

    def _warm_up(self):
        dummy = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        dummy_f_x = cp.zeros((self.ny, self.nx + 1), dtype=cp.float32)
        dummy_f_y = cp.zeros((self.ny + 1, self.nx), dtype=cp.float32)
        _ = self.gradient.green_gauss(dummy_f_x, dummy_f_y)
        cp.cuda.Stream.null.synchronize()

    def to_gpu(self, arr: np.ndarray) -> cp.ndarray:
        return cp.asarray(arr, dtype=cp.float32)

    def to_cpu(self, arr: cp.ndarray) -> np.ndarray:
        return cp.asnumpy(arr)

    def solve_poisson(self, rhs: cp.ndarray, p: cp.ndarray,
                      n_iter: int = 50, tol: float = 1e-6,
                      report: bool = False) -> tuple[cp.ndarray, float]:
        return self.poisson.solve(rhs, p, n_iter, tol, report)

    def compute_gradient(self, phi_f_x: cp.ndarray, phi_f_y: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        return self.gradient.green_gauss(phi_f_x, phi_f_y)

    def compute_divergence(self, u: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
        return self.gradient.divergence(u, v)

    def compute_vorticity(self, u: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
        return self.gradient.vorticity(u, v)

    @staticmethod
    def benchmark_poisson(grid_sizes: list[int] = None):
        if grid_sizes is None:
            grid_sizes = [64, 128, 256, 512]

        print("=" * 60)
        print("  Poisson GPU solver benchmark")
        print("=" * 60)
        print(f"  {'Grid':>8s}  {'Time(ms)':>10s}  {'Cells/s':>12s}  {'Residual':>10s}")
        print("  " + "-" * 44)

        for n in grid_sizes:
            dx = dy = 1.0 / n
            solver = CUDAPoissonSolver(n, n, dx, dy, dt=0.01, rho=1.0)

            rhs = cp.random.randn(n, n).astype(cp.float32)
            p = cp.zeros((n, n), dtype=cp.float32)

            solver.solve(rhs, p, n_iter=10, tol=0.0)
            cp.cuda.Stream.null.synchronize()

            start = cp.cuda.Event(); end = cp.cuda.Event()
            start.record()
            p, res = solver.solve(rhs, p, n_iter=200, tol=0.0)
            end.record(); end.synchronize()

            elapsed_ms = cp.cuda.get_elapsed_time(start, end)
            cells_per_s = (n * n * 200) / (elapsed_ms * 1e-3)
            print(f"  {n:4d}×{n:<4d}  {elapsed_ms:8.1f}ms  {cells_per_s:12.1e}  {res:10.2e}")


if __name__ == "__main__":
    CUDABridge.benchmark_poisson([64, 128, 256, 512])
