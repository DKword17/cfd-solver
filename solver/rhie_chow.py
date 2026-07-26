#!/usr/bin/env python3
"""
solver/rhie_chow.py
=====================

Интерполяция Райса-Чоу для коллоцированных (co-located) сеток.

Проблема: на коллоцированной сетке центральные разности для градиента
давления приводят к шахматной (checkerboard) невязке — ∇·u не
"чувствует" осцилляций давления, потому что ∇p использует только
чётную/нечётную связь.

Решение Rhie & Chow (1983):
    u_f = u_f_lin + D_f · (∇p_f - ∇p_f_lin)

где:
    u_f       — скорость на грани [м/с]
    u_f_lin   — линейная интерполяция u_P, u_N [м/с]
    D_f       — коэффициент проницаемости (1/aP)_f [м⁴·с/кг]
    ∇p_f      — градиент давления на грани (центральная разность) [Па/м]
    ∇p_f_lin  — линейная интерполяция ∇p_P, ∇p_N [Па/м]

Поправка D_f · (∇p_f - ∇p_f_lin) добавляет в конвективный поток
зависимость от давления, подавляя шахматные моды.

Литература:
    Rhie & Chow (1983) AIAA J 21:1525–1532
    Ferziger & Perić (2002) §8.6
    Patankar (1980) Numerical Heat Transfer and Fluid Flow, §6.7

Автор: Alexei Morozov, Институт механики МГУ, 2026
"""

from __future__ import annotations

import numpy as np
from navier_stokes import Mesh2D, Boundary2D, BoundaryCondition


# ─── вспомогательная: давление на гранях ──────────────────────────────

def _pressure_face_gradient(
    p: np.ndarray,       # [nx, ny]  давление в центрах ячеек [Па]
    mesh: Mesh2D,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Градиент давления непосредственно на гранях (центральная разность).

        (∂p/∂x)_{i+½} ≈ (p_{i+1} - p_i) / Δx   [Па/м]
        (∂p/∂y)_{j+½} ≈ (p_{j+1} - p_j) / Δy   [Па/м]

    Returns:
        (grad_px_faces [nx+1, ny], grad_py_faces [nx, ny+1])
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # по x: (i+½)   грани
    gpx = np.zeros((nx + 1, ny))
    gpx[1:-1, :] = (p[1:, :] - p[:-1, :]) / dx
    # экстраполяция на физические границы (∂p/∂n = 0 на стенке)
    gpx[0, :] = gpx[1, :]
    gpx[-1, :] = gpx[-2, :]

    # по y: (j+½)   грани
    gpy = np.zeros((nx, ny + 1))
    gpy[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dy
    gpy[:, 0] = gpy[:, 1]
    gpy[:, -1] = gpy[:, -2]

    return gpx, gpy


# ─── интерполяция 1/aP на грани ──────────────────────────────────────

def _interp_ap_inv_face(
    ap_inv: np.ndarray,     # [nx, ny]  1/aP  [м⁴·с/кг]
    mesh: Mesh2D,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Линейная интерполяция 1/aP на грани.

    Returns:
        (ap_inv_x [nx+1, ny], ap_inv_y [nx, ny+1])
    """
    nx, ny = mesh.nx, mesh.ny

    # x-грани: гармоническое среднее для коэффициента проницаемости
    ap_x = np.zeros((nx + 1, ny))
    # гармоническое среднее: 2 * a_i * a_{i+1} / (a_i + a_{i+1})
    ap_x[1:-1, :] = 2.0 * ap_inv[:-1, :] * ap_inv[1:, :] / (
        ap_inv[:-1, :] + ap_inv[1:, :] + 1e-30
    )
    ap_x[0, :] = ap_inv[0, :]
    ap_x[-1, :] = ap_inv[-1, :]

    # y-грани
    ap_y = np.zeros((nx, ny + 1))
    ap_y[:, 1:-1] = 2.0 * ap_inv[:, :-1] * ap_inv[:, 1:] / (
        ap_inv[:, :-1] + ap_inv[:, 1:] + 1e-30
    )
    ap_y[:, 0] = ap_inv[:, 0]
    ap_y[:, -1] = ap_inv[:, -1]

    return ap_x, ap_y


def rhie_chow_mass_flux_x(
    u: np.ndarray,          # [nx, ny]  x-скорость в центрах ячеек [м/с]
    p: np.ndarray,          # [nx, ny]  давление в центрах [Па]
    ap_u: np.ndarray,       # [nx, ny]  центральный коэффициент aP для u [кг/с]
    mesh: Mesh2D,
    rho: float = 1.0,       # плотность [кг/м³]
) -> np.ndarray:
    """
    Массовый поток на x-гранях (east/west) по Rhie-Chow.

        ṁ_{i+½} = ρ · u_f · Δy                               [кг/с]

    где u_f — скорость на грани с Rhie-Chow поправкой:

        u_f = u_f_lin + D_f · ( (∂p/∂x)_f - (∂p/∂x)_f_lin )

        D_f = (ΔV / aP)_f = (Δx·Δy / aP)_f                  [м⁴·с/кг]

    Returns:
        mfx [nx+1, ny] — массовый поток на x-гранях.
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # ── 1. Линейная интерполяция u на x-грани ──
    u_f_lin = np.zeros((nx + 1, ny))
    u_f_lin[1:-1, :] = 0.5 * (u[:-1, :] + u[1:, :])
    # границы: экстраполяция (будет заменена BC в решателе)
    u_f_lin[0, :] = u[0, :]
    u_f_lin[-1, :] = u[-1, :]

    # ── 2. Коэффициент проницаемости D_f = (ΔV / aP)_f ──
    cell_vol = dx * dy
    # 1/aP [c/кг — ну, размерность:  (ΔV/aP) = м³ / (кг/с) = м⁴·с/кг ...]
    ap_inv = cell_vol / (ap_u + 1e-30)   # [м⁴·с/кг]

    D_f_x, _ = _interp_ap_inv_face(ap_inv, mesh)
    # D_f_x — это уже (ΔV / aP)_f [м⁴·с/кг]

    # ── 3. Градиент давления на гранях ──
    grad_p_x_faces, grad_p_y_faces = _pressure_face_gradient(p, mesh)
    # (∂p/∂x)_{i+½} — берём из grad_p_x_faces

    # ── 4. Линейная интерполяция градиента давления на грани ──
    grad_p_x_lin = np.zeros((nx + 1, ny))
    grad_p_x_lin[1:-1, :] = 0.5 * (
        (p[1:, :] - p[:-1, :]) / dx  # это центральная разность, но между ячейками
        # для линейной интерполяции градиента из центров
    )
    # градиент в центрах ячеек — центральная разность
    grad_p_cell = np.zeros((nx, ny))
    grad_p_cell[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * dx)
    grad_p_cell[0, :] = (p[1, :] - p[0, :]) / dx
    grad_p_cell[-1, :] = (p[-1, :] - p[-2, :]) / dx

    # интерполяция градиента на грани
    grad_p_face_lin = np.zeros((nx + 1, ny))
    grad_p_face_lin[1:-1, :] = 0.5 * (grad_p_cell[:-1, :] + grad_p_cell[1:, :])
    grad_p_face_lin[0, :] = grad_p_cell[0, :]
    grad_p_face_lin[-1, :] = grad_p_cell[-1, :]

    # ── 5. Rhie-Chow поправка ──
    # u_f = u_f_lin + D_f · ( (∂p/∂x)_f - (∂p/∂x)_f_lin )
    u_f = u_f_lin + D_f_x * (grad_p_x_faces - grad_p_face_lin)

    # ── 6. Массовый поток ──
    mfx = rho * u_f * dy   # [кг/с]

    return mfx


def rhie_chow_mass_flux_y(
    v: np.ndarray,          # [nx, ny]  y-скорость [м/с]
    p: np.ndarray,          # [nx, ny]  давление [Па]
    ap_v: np.ndarray,       # [nx, ny]  центральный коэффициент aP для v [кг/с]
    mesh: Mesh2D,
    rho: float = 1.0,
) -> np.ndarray:
    """
    Массовый поток на y-гранях (north/south) по Rhie-Chow.

        ṁ_{j+½} = ρ · v_f · Δx                               [кг/с]

    Returns:
        mfy [nx, ny+1] — массовый поток на y-гранях.
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # ── 1. Линейная интерполяция v на y-грани ──
    v_f_lin = np.zeros((nx, ny + 1))
    v_f_lin[:, 1:-1] = 0.5 * (v[:, :-1] + v[:, 1:])
    v_f_lin[:, 0] = v[:, 0]
    v_f_lin[:, -1] = v[:, -1]

    # ── 2. D_f = (ΔV / aP)_f ──
    cell_vol = dx * dy
    ap_inv = cell_vol / (ap_v + 1e-30)
    _, D_f_y = _interp_ap_inv_face(ap_inv, mesh)

    # ── 3. Градиент давления на y-гранях ──
    grad_p_x_faces, grad_p_y_faces = _pressure_face_gradient(p, mesh)

    # ── 4. Линейная интерполяция градиента давления на y-грани ──
    grad_p_cell = np.zeros((nx, ny))
    grad_p_cell[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dy)
    grad_p_cell[:, 0] = (p[:, 1] - p[:, 0]) / dy
    grad_p_cell[:, -1] = (p[:, -1] - p[:, -2]) / dy

    grad_p_face_lin = np.zeros((nx, ny + 1))
    grad_p_face_lin[:, 1:-1] = 0.5 * (grad_p_cell[:, :-1] + grad_p_cell[:, 1:])
    grad_p_face_lin[:, 0] = grad_p_cell[:, 0]
    grad_p_face_lin[:, -1] = grad_p_cell[:, -1]

    # ── 5. Rhie-Chow поправка ──
    v_f = v_f_lin + D_f_y * (grad_p_y_faces - grad_p_face_lin)

    # ── 6. Массовый поток ──
    mfy = rho * v_f * dx

    return mfy


def rhie_chow_face_velocity(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    ap_u: np.ndarray,
    ap_v: np.ndarray,
    mesh: Mesh2D,
    rho: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Полный Rhie-Chow — скорости и массовые потоки на всех гранях.

    Returns:
        (u_face_x [nx+1, ny], v_face_y [nx, ny+1],
         mfx [nx+1, ny], mfy [nx, ny+1])
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    cell_vol = dx * dy
    ap_inv_u = cell_vol / (ap_u + 1e-30)
    ap_inv_v = cell_vol / (ap_v + 1e-30)

    # ── градиент давления на гранях ──
    gpx_f, gpy_f = _pressure_face_gradient(p, mesh)

    # ── градиент давления в центрах ──
    gpx_c = np.zeros((nx, ny))
    gpy_c = np.zeros((nx, ny))
    gpx_c[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * dx)
    gpy_c[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dy)
    gpx_c[0, :] = (p[1, :] - p[0, :]) / dx
    gpx_c[-1, :] = (p[-1, :] - p[-2, :]) / dx
    gpy_c[:, 0] = (p[:, 1] - p[:, 0]) / dy
    gpy_c[:, -1] = (p[:, -1] - p[:, -2]) / dy

    # ── интерполяция D_f на грани ──
    D_x, D_y = _interp_ap_inv_face(ap_inv_u, mesh)
    # для v используем ap_v
    D_x_v, D_y_v = _interp_ap_inv_face(ap_inv_v, mesh)

    # ── u на x-гранях ──
    u_f = np.zeros((nx + 1, ny))
    u_f[1:-1, :] = 0.5 * (u[:-1, :] + u[1:, :])
    u_f[0, :] = u[0, :]
    u_f[-1, :] = u[-1, :]

    gpx_lin = np.zeros((nx + 1, ny))
    gpx_lin[1:-1, :] = 0.5 * (gpx_c[:-1, :] + gpx_c[1:, :])
    gpx_lin[0, :] = gpx_c[0, :]
    gpx_lin[-1, :] = gpx_c[-1, :]

    u_f += D_x * (gpx_f - gpx_lin)

    # ── v на y-гранях ──
    v_f = np.zeros((nx, ny + 1))
    v_f[:, 1:-1] = 0.5 * (v[:, :-1] + v[:, 1:])
    v_f[:, 0] = v[:, 0]
    v_f[:, -1] = v[:, -1]

    gpy_lin = np.zeros((nx, ny + 1))
    gpy_lin[:, 1:-1] = 0.5 * (gpy_c[:, :-1] + gpy_c[:, 1:])
    gpy_lin[:, 0] = gpy_c[:, 0]
    gpy_lin[:, -1] = gpy_c[:, -1]

    v_f += D_y_v * (gpy_f - gpy_lin)

    # ── массовые потоки ──
    mfx = rho * u_f * dy
    mfy = rho * v_f * dx

    return u_f, v_f, mfx, mfy
