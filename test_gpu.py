#!/usr/bin/env python3
"""
test_gpu.py — GPU module verification + benchmark.

Tests:
  1. Poisson solver accuracy (vs analytical)
  2. GPU gradient kernel vs numpy
  3. GPU SIMPLE vs CPU SIMPLE
  4. Performance benchmarks

Run:
    python test_gpu.py              # full
    python test_gpu.py --quick      # quick (small grids)
"""

import sys, time, argparse
from pathlib import Path
import numpy as np

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from navier_stokes import Mesh2D, Boundary2D, BoundaryCondition, cavity_flow
from cuda_bridge import CUDAPoissonSolver, CUDAGradient

try:
    import cupy as cp
except ImportError:
    print("[FAIL] CuPy not installed.")
    sys.exit(1)


def test_poisson():
    """Verify GPU Poisson solver: solve -∇²p = f with known solution."""
    print("\n" + "=" * 60)
    print("  Test 1: Poisson solver accuracy")
    print("=" * 60)

    n = 64
    dx = dy = 1.0 / n
    solver = CUDAPoissonSolver(n, n, dx, dy, dt=0.01, rho=1.0)

    x = np.linspace(dx/2, 1-dx/2, n)
    y = np.linspace(dy/2, 1-dy/2, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    p_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    rhs_np = -2.0 * np.pi**2 * p_exact

    p_gpu = cp.zeros((n, n), dtype=cp.float32)
    rhs_gpu = cp.asarray(rhs_np, dtype=cp.float32)
    p_gpu, res = solver.solve(rhs_gpu, p_gpu, n_iter=500, tol=1e-10)

    err = np.max(np.abs(p_exact[1:-1, 1:-1] - cp.asnumpy(p_gpu)[1:-1, 1:-1]))
    print(f"  max|p_exact - p_GPU| = {err:.6f}  (res={res:.2e})")
    print(f"  [{'PASS' if err < 0.01 else 'WARN'}]")
    return err


def test_gradient():
    """Verify GPU gradient vs numpy."""
    print("\n" + "=" * 60)
    print("  Test 2: Gradient kernel")
    print("=" * 60)

    n = 32
    dx = dy = 1.0 / n
    gpu = CUDAGradient(n, n, dx, dy)

    x = np.linspace(dx/2, 1-dx/2, n)
    y = np.linspace(dy/2, 1-dy/2, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    phi_np = X**2 + Y**2  # ∂φ/∂x=2x, ∂φ/∂y=2y

    phi_gpu = cp.asarray(phi_np, dtype=cp.float32)
    pf_x = cp.empty((n, n+1), dtype=cp.float32)
    pf_x[:, 1:-1] = 0.5 * (phi_gpu[:, :-1] + phi_gpu[:, 1:])
    pf_x[:, 0] = phi_gpu[:, 0]
    pf_x[:, -1] = phi_gpu[:, -1]
    pf_y = cp.empty((n+1, n), dtype=cp.float32)
    pf_y[1:-1, :] = 0.5 * (phi_gpu[:-1, :] + phi_gpu[1:, :])
    pf_y[0, :] = phi_gpu[0, :]
    pf_y[-1, :] = phi_gpu[-1, :]

    gx, gy = gpu.green_gauss(pf_x, pf_y)
    err_x = np.max(np.abs(2*X[:] - cp.asnumpy(gx)[:, :]))
    err_y = np.max(np.abs(2*Y[:] - cp.asnumpy(gy)[:, :]))
    print(f"  max|∂φ/∂x - GPU| = {err_x:.6f}")
    print(f"  max|∂φ/∂y - GPU| = {err_y:.6f}")

    u_gpu = cp.asarray(X.astype(np.float32))
    v_gpu = cp.asarray(-Y.astype(np.float32))
    div = gpu.divergence(u_gpu, v_gpu)
    err_div = np.max(np.abs(cp.asnumpy(div)[1:-1, 1:-1]))
    print(f"  max|∇·u| (should be ~0) = {err_div:.6f}")
    print(f"  [{'PASS' if max(err_x, err_y, err_div) < 0.05 else 'WARN'}]")
    return max(err_x, err_y, err_div)


def test_simple_solver():
    """GPU SIMPLE vs CPU SIMPLE on cavity flow."""
    print("\n" + "=" * 60)
    print("  Test 3: GPU SIMPLE vs CPU SIMPLE")
    print("=" * 60)

    n = 32
    n_steps = 5
    dt = 0.01
    Re = 100

    t0 = time.time()
    cpu_solver = cavity_flow(Re=Re, nx=n, ny=n, t_end=n_steps*dt, dt=dt)
    t_cpu = time.time() - t0

    from solver.gpu_simple import cavity_flow_gpu
    t0 = time.time()
    gpu_solver = cavity_flow_gpu(Re=Re, nx=n, ny=n, t_end=n_steps*dt,
                                  dt=dt, device=0, report=False)
    t_gpu = time.time() - t0

    u_cpu = cpu_solver.u.data
    v_cpu = cpu_solver.v.data
    u_gpu = cp.asnumpy(gpu_solver._u)
    v_gpu = cp.asnumpy(gpu_solver._v)

    err_u = np.max(np.abs(u_cpu - u_gpu))
    err_v = np.max(np.abs(v_cpu - v_gpu))
    print(f"  CPU={t_cpu:.3f}s  GPU={t_gpu:.3f}s  speedup={t_cpu/max(t_gpu,1e-6):.1f}×")
    print(f"  max|u_diff|={err_u:.6f}  max|v_diff|={err_v:.6f}")
    print(f"  [{'PASS' if err_u < 0.1 and err_v < 0.1 else 'DIFF'}]")
    return err_u, err_v


def benchmark():
    """Performance benchmark."""
    print("\n" + "=" * 60)
    print("  Benchmark: GPU Poisson")
    print("=" * 60)
    from cuda_bridge import CUDABridge
    CUDABridge.benchmark_poisson([64, 128, 256, 512])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("  CFD-GPU Test Suite — Ming Zhou (USTC)")
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    print(f"  GPU: {gpu_name}  CuPy: {cp.__version__}")
    print("=" * 60)

    if args.benchmark_only:
        benchmark()
        return

    test_poisson()
    test_gradient()
    if not args.quick:
        test_simple_solver()
        benchmark()

    print("\n[DONE]")


if __name__ == "__main__":
    main()
