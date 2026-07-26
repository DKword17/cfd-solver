#!/usr/bin/env python3
"""
cuda_bridge.py — CuPy Python interface for kernel/*.cu GPU kernels.

Usage:
    from cuda_bridge import CUDABridge
    bridge = CUDABridge(nx=128, ny=128, dx=0.01, dy=0.01)
    p, res = bridge.solve_poisson(rhs, p, n_iter=100)
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np

_KERNEL_DIR = Path(__file__).parent / "kernel"

try:
    import cupy as cp
except ImportError:
    raise ImportError("pip install cupy-cuda12x")


def _grid_2d(nx: int, ny: int, tx: int = 16, ty: int = 16):
    return ((nx + tx - 1) // tx, (ny + ty - 1) // ty), (tx, ty)


def _grid_1d(n: int, bs: int = 256):
    return (math.ceil(n / bs),), (bs,)


def _load_cu(name: str) -> str:
    return (_KERNEL_DIR / name).read_text(encoding="utf-8")


class CUDAPoissonSolver:
    """Poisson solver: Red-Black Gauss-Seidel on GPU."""

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 dt: float = 0.01, rho: float = 1.0):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.dt, self.rho = dt, rho
        self.n_cells = nx * ny

        mod = cp.RawModule(code=_load_cu("laplace_cuda.cu"))
        self._rbgs = mod.get_function("_kernel_poisson_rbgs")
        self._res = mod.get_function("_kernel_poisson_residual")
        self._coeff = mod.get_function("_kernel_poisson_coefficients")

        self._aE = cp.zeros(self.n_cells, dtype=cp.float32)
        self._aW = cp.zeros(self.n_cells, dtype=cp.float32)
        self._aN = cp.zeros(self.n_cells, dtype=cp.float32)
        self._aS = cp.zeros(self.n_cells, dtype=cp.float32)
        self._aP = cp.zeros(self.n_cells, dtype=cp.float32)
        self._alloc_coeffs()

        self._g2d, self._b2d = _grid_2d(nx, ny)
        self._g1d, self._b1d = _grid_1d(self.n_cells)
        # shared memory for residual kernel: blockDim.x * 4 bytes
        self._res_smem = 256 * 4

    def _alloc_coeffs(self):
        g, b = _grid_2d(self.nx, self.ny)
        self._coeff(g, b, (self._aE, self._aW, self._aN, self._aS, self._aP,
                           self.nx, self.ny,
                           cp.float32(self.dx), cp.float32(self.dy),
                           cp.float32(self.dt), cp.float32(self.rho)))

    def solve(self, rhs, p, n_iter=50, tol=1e-6, report=False):
        for it in range(n_iter):
            self._rbgs(self._g2d, self._b2d,
                       (p, rhs, self._aE, self._aW, self._aN, self._aS,
                        self._aP, self.nx, self.ny, 1))
            self._rbgs(self._g2d, self._b2d,
                       (p, rhs, self._aE, self._aW, self._aN, self._aS,
                        self._aP, self.nx, self.ny, 0))
            if it % 5 == 0 or it == n_iter - 1:
                res = self._residual(p, rhs)
                if report:
                    print(f"  [Poisson] iter={it:3d}  RMS res={res:.2e}")
                if res < tol:
                    break
        return p, res

    def _residual(self, p, rhs):
        r = cp.zeros(1, dtype=cp.float32)
        self._res(self._g1d, self._b1d,
                  (p, rhs, self._aE, self._aW, self._aN, self._aS,
                   self._aP, self.nx, self.ny, r),
                  shared_mem=self._res_smem)
        return math.sqrt(float(r[0]) / self.n_cells)


class CUDAFaceInterp:
    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        mod = cp.RawModule(code=_load_cu("face_interp.cu"))
        self._fx = mod.get_function("_kernel_rhie_chow_faces_x")
        self._fy = mod.get_function("_kernel_rhie_chow_faces_y")
        self._df = mod.get_function("_kernel_compute_Df")
        self._gx, self._bx = _grid_2d(nx + 1, ny)
        self._gy, self._by = _grid_2d(nx, ny + 1)

    def interpolate_x(self, phi_c, grad_x, Df, dx):
        f = cp.empty((self.ny, self.nx + 1), dtype=cp.float32)
        self._fx(self._gx, self._bx,
                 (phi_c, grad_x, Df, f, self.nx, self.ny, cp.float32(dx)))
        return f

    def interpolate_y(self, phi_c, grad_y, Df, dy):
        f = cp.empty((self.ny + 1, self.nx), dtype=cp.float32)
        self._fy(self._gy, self._by,
                 (phi_c, grad_y, Df, f, self.nx, self.ny, cp.float32(dy)))
        return f


class CUDAGradient:
    def __init__(self, nx, ny, dx, dy):
        self.nx, self.ny = nx, ny
        self.inv_dx, self.inv_dy = 1.0/dx, 1.0/dy
        mod = cp.RawModule(code=_load_cu("gradient.cu"))
        self._gg = mod.get_function("_kernel_gradient_gg")
        self._cent = mod.get_function("_kernel_gradient_central")
        self._div = mod.get_function("_kernel_divergence")
        self._vort = mod.get_function("_kernel_vorticity")
        self._g2d, self._b2d = _grid_2d(nx, ny)
        self._gi, self._bi = _grid_2d(nx - 2, ny - 2)

    def green_gauss(self, fx, fy):
        gx = cp.empty((self.ny, self.nx), dtype=cp.float32)
        gy = cp.empty((self.ny, self.nx), dtype=cp.float32)
        self._gg(self._g2d, self._b2d,
                 (fx, fy, gx, gy, self.nx, self.ny,
                  cp.float32(self.inv_dx), cp.float32(self.inv_dy),
                  cp.float32(1.0)))
        return gx, gy

    def divergence(self, u, v):
        d = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self._div(self._gi, self._bi,
                  (u, v, d, self.nx, self.ny,
                   cp.float32(self.inv_dx), cp.float32(self.inv_dy)))
        return d

    def vorticity(self, u, v):
        o = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self._vort(self._gi, self._bi,
                   (u, v, o, self.nx, self.ny,
                    cp.float32(self.inv_dx), cp.float32(self.inv_dy)))
        return o


class CUDALimiter:
    MINMOD, VAN_LEER, SUPERBEE = 0, 1, 2

    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        mod = cp.RawModule(code=_load_cu("limiter.cu"))
        self._x, self._y, self._v = (
            mod.get_function("_kernel_tvd_face_x"),
            mod.get_function("_kernel_tvd_face_y"),
            mod.get_function("_kernel_limiter_value"),
        )
        self._gx, self._bx = _grid_2d(nx - 1, ny)
        self._gy, self._by = _grid_2d(nx, ny - 1)

    def apply_x(self, phi, lt=MINMOD):
        f = cp.empty((self.ny, self.nx + 1), dtype=cp.float32)
        self._x(self._gx, self._bx, (phi, f, self.nx, self.ny, lt))
        return f

    def apply_y(self, phi, lt=MINMOD):
        f = cp.empty((self.ny + 1, self.nx), dtype=cp.float32)
        self._y(self._gy, self._by, (phi, f, self.nx, self.ny, lt))
        return f


class CUDABridge:
    """统一的 GPU 加速器桥接层，封装所有 CUDA 模块。"""

    def __init__(self, nx, ny, dx, dy, dt=0.01, rho=1.0, device=0):
        cp.cuda.Device(device).use()
        self.poisson = CUDAPoissonSolver(nx, ny, dx, dy, dt, rho)
        self.interp = CUDAFaceInterp(nx, ny)
        self.gradient = CUDAGradient(nx, ny, dx, dy)
        self.limiter = CUDALimiter(nx, ny)
        self._warmup()

    def _warmup(self):
        d = cp.zeros((self.poisson.ny, self.poisson.nx), dtype=cp.float32)
        self.gradient.divergence(d, d)
        cp.cuda.Stream.null.synchronize()

    def to_gpu(self, a):
        return cp.asarray(a, dtype=cp.float32)

    def to_cpu(self, a):
        return cp.asnumpy(a)

    @staticmethod
    def benchmark(grids=None):
        """Poisson 求解器性能基准测试。"""
        if grids is None:
            grids = [64, 128, 256, 512]
        print(f"{'Grid':>8s}  {'Time(ms)':>10s}  {'Cells/s':>12s}  {'Res':>10s}")
        print("-" * 44)
        for n in grids:
            dx = dy = 1.0 / n
            s = CUDAPoissonSolver(n, n, dx, dy)
            r = cp.random.randn(n, n).astype(cp.float32)
            p = cp.zeros((n, n), dtype=cp.float32)
            s.solve(r, p, 10, 0.0)
            cp.cuda.Stream.null.synchronize()
            e1 = cp.cuda.Event()
            e2 = cp.cuda.Event()
            e1.record()
            p_, res = s.solve(r, p, 200, 0.0)
            e2.record()
            e2.synchronize()
            t = cp.cuda.get_elapsed_time(e1, e2)
            cps = (n * n * 200) / (t * 1e-3)
            print(f"{n:4d}x{n:<4d}  {t:8.1f}ms  {cps:12.1e}  {res:10.2e}")


if __name__ == "__main__":
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print(f"GPU: {gpu_name}")
    CUDABridge.benchmark([64, 128, 256, 512])