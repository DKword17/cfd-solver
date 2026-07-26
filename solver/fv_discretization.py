#!/usr/bin/env python3
"""
solver/fv_discretisation.py
============================

Операторы конечного объёма на структурированной 2D коллоцированной сетке.
Все операторы возвращают массивы numpy размерности (nx, ny).

    — grad_cell_centered:  градиент φ в центре ячейки  [·/м]
    — grad_face:           градиент φ на грани (i±½)   [·/м]
    — div_face_flux:       дивергенция потока F         [·/м³]
    — laplacian:           диффузионный член ∇·(Γ∇φ)   [·/с]
    — laplacian_nonorthogonal: с коррекцией неортогональности

Литература:
    Ferziger & Perić (2002) Computational Methods for Fluid Dynamics
    Jasak (1996) PhD Thesis, Imperial College London

Автор: Alexei Morozov, Институт механики МГУ, 2026
"""

from __future__ import annotations

import numpy as np

# типы из основного модуля
from navier_stokes import Mesh2D, Field2D, Boundary2D, BoundaryCondition

# ─── Hаправления граней ───────────────────────────────────────────────
#    W      P      E        j+1 —— N ——
#    i-1    i     i+1         |         |
#                             j —— P —— j
#    S                       j-1 —— S ——
#    j-1                          i-1  i  i+1


def grad_cell_centered(
    phi: np.ndarray,
    mesh: Mesh2D,
    bc: Boundary2D | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Градиент φ в центре ячейки — метод Гаусса (теорема о дивергенции).

        (∇φ)_P = (1/ΔV) * Σ_f (φ_f · S_f)   [1/м]

    где S_f — вектор площади грани, φ_f — интерполированное на грань
    значение (линейно для внутренних граней, BC для граничных).

    Parameters:
        phi:  поле [nx, ny]
        mesh:  сетка
        bc:    граничные условия (опционально, None = нуль-градиент)

    Returns:
        (dphidx, dphidy) — массивы [nx, ny], градиент по x и y.
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    dphidx = np.zeros((nx, ny))
    dphidy = np.zeros((nx, ny))

    # Внутренние ячейки: центральные разности 2-го порядка
    dphidx[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dx)
    dphidy[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dy)

    # ── Границы — односторонние разности ──
    if bc is not None:
        # Запад (i=0)
        bc_type, _ = bc.west
        if bc_type == BoundaryCondition.WALL:
            dphidx[0, :] = (-3.0 * phi[0, :] + 4.0 * phi[1, :] - phi[2, :]) / (2.0 * dx)
        else:
            dphidx[0, :] = (phi[1, :] - phi[0, :]) / dx  # односторонняя
        # Восток (i=nx-1)
        bc_type, _ = bc.east
        if bc_type == BoundaryCondition.WALL:
            dphidx[-1, :] = (3.0 * phi[-1, :] - 4.0 * phi[-2, :] + phi[-3, :]) / (2.0 * dx)
        else:
            dphidx[-1, :] = (phi[-1, :] - phi[-2, :]) / dx
        # Юг (j=0)
        bc_type, _ = bc.south
        if bc_type == BoundaryCondition.WALL:
            dphidy[:, 0] = (-3.0 * phi[:, 0] + 4.0 * phi[:, 1] - phi[:, 2]) / (2.0 * dy)
        else:
            dphidy[:, 0] = (phi[:, 1] - phi[:, 0]) / dy
        # Север (j=ny-1)
        bc_type, _ = bc.north
        if bc_type == BoundaryCondition.WALL:
            dphidy[:, -1] = (3.0 * phi[:, -1] - 4.0 * phi[:, -2] + phi[:, -3]) / (2.0 * dy)
        else:
            dphidy[:, -1] = (phi[:, -1] - phi[:, -2]) / dy
    else:
        # Без BC — экстраполяция нуль-градиент
        dphidx[0, :] = (phi[1, :] - phi[0, :]) / dx
        dphidx[-1, :] = (phi[-1, :] - phi[-2, :]) / dx
        dphidy[:, 0] = (phi[:, 1] - phi[:, 0]) / dy
        dphidy[:, -1] = (phi[:, -1] - phi[:, -2]) / dy

    return dphidx, dphidy


def grad_face(
    phi: np.ndarray,
    mesh: Mesh2D,
    bc: Boundary2D | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Градиент φ на гранях ячеек.

    Для грани (i+½) — центральная разность:
        (∂φ/∂x)_{i+½} ≈ (φ_{i+1} - φ_i) / Δx   [·/м]
    Для грани (j+½) — аналогично по y.

    Returns:
        (grad_x_faces [nx+1, ny], grad_y_faces [nx, ny+1]).
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # x-грани (east/west) — форма (nx+1, ny)
    grad_x = np.zeros((nx + 1, ny))
    grad_x[1:-1, :] = (phi[1:, :] - phi[:-1, :]) / dx
    if bc:
        bc_type, bc_val = bc.west
        if bc_type == BoundaryCondition.WALL:
            grad_x[0, :] = 0.0
        elif bc_type == BoundaryCondition.INLET:
            grad_x[0, :] = (phi[0, :] - bc_val) / (dx * 0.5)  # [м/с]/м
        else:
            grad_x[0, :] = 0.0
        bc_type, _ = bc.east
        if bc_type == BoundaryCondition.OUTLET:
            grad_x[-1, :] = 0.0
        else:
            grad_x[-1, :] = 0.0
    else:
        grad_x[0, :] = 0.0
        grad_x[-1, :] = 0.0

    # y-грани (north/south) — форма (nx, ny+1)
    grad_y = np.zeros((nx, ny + 1))
    grad_y[:, 1:-1] = (phi[:, 1:] - phi[:, :-1]) / dy
    if bc:
        bc_type, bc_val = bc.south
        if bc_type == BoundaryCondition.WALL:
            grad_y[:, 0] = 0.0
        elif bc_type == BoundaryCondition.INLET:
            grad_y[:, 0] = (phi[:, 0] - bc_val) / (dy * 0.5)
        else:
            grad_y[:, 0] = 0.0
        bc_type, _ = bc.north
        if bc_type == BoundaryCondition.OUTLET:
            grad_y[:, -1] = 0.0
        else:
            grad_y[:, -1] = 0.0
    else:
        grad_y[:, 0] = 0.0
        grad_y[:, -1] = 0.0

    return grad_x, grad_y


def div_face_flux(
    flux_x: np.ndarray,      # [nx+1, ny]  поток на x-гранях
    flux_y: np.ndarray,      # [nx, ny+1]  поток на y-гранях
    mesh: Mesh2D,
) -> np.ndarray:
    """
    Дивергенция потока через грани — дискретный аналог теоремы Гаусса.

        (∇·F)_P = (1/ΔV) * Σ_f (F_f · n_f · S_f)   [·/м³]

    где F_f — поток на грани, n_f — нормаль, S_f — площадь грани.

                      Fx_{i+½} - Fx_{i-½}   Fy_{j+½} - Fy_{j-½}
        (∇·F)_P ≈     ──────────────────── + ────────────────────  [·/м³]
                              Δx                    Δy

    Returns:
        Массив [nx, ny] — дивергенция в центрах ячеек.
    """
    dx, dy = mesh.dx, mesh.dy
    nx, ny = mesh.nx, mesh.ny

    div = np.zeros((nx, ny))

    # Внутренние ячейки
    div[1:-1, 1:-1] = (
        (flux_x[2:, 1:-1] - flux_x[1:-1, 1:-1]) / dx
        + (flux_y[1:-1, 2:] - flux_y[1:-1, 1:-1]) / dy
    )

    # Границы — односторонние
    div[0, :] = (flux_x[1, :] - flux_x[0, :]) / dx
    div[-1, :] = (flux_x[-1, :] - flux_x[-2, :]) / dx
    div[:, 0] = (flux_y[:, 1] - flux_y[:, 0]) / dy
    div[:, -1] = (flux_y[:, -1] - flux_y[:, -2]) / dy

    return div


def laplacian(
    phi: np.ndarray,
    gamma: float | np.ndarray,   # коэффициент диффузии [м²/с] или [·/с]
    mesh: Mesh2D,
    bc: Boundary2D | None = None,
) -> np.ndarray:
    """
    Лапласиан — дискретизация диффузионного члена ∇·(Γ∇φ).

        ∇·(Γ∇φ)_P ≈ Σ_f Γ_f (∇φ)_f · S_f
                   = Σ_f Γ_f (∂φ/∂n)_f · S_f        [·/с]

    Дискретизация 2-го порядка центральными разностями:

        d_e · (φ_E - φ_P) - d_w · (φ_P - φ_W) +
        d_n · (φ_N - φ_P) - d_s · (φ_P - φ_S)

    где d_e = Γ_e · Δy / Δx, d_n = Γ_n · Δx / Δy  и т.д.

    Args:
        phi:   поле [nx, ny]
        gamma: коэффициент диффузии (скаляр или массив [nx, ny])
        mesh:  сетка
        bc:    граничные условия

    Returns:
        Массив [nx, ny] — значение лапласиана в центрах ячеек.
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    if isinstance(gamma, (int, float)):
        gamma = np.full((nx, ny), gamma)

    # Коэффициенты на гранях — гармоническое среднее
    # Γ_e = 2 * Γ_P * Γ_E / (Γ_P + Γ_E)
    gamma_e = 2.0 * gamma[:-1, :] * gamma[1:, :] / (gamma[:-1, :] + gamma[1:, :] + 1e-30)
    # gamma_e — (nx-1, ny), содержит Γ на east face (i+½) для i=0..nx-2
    gamma_w = np.zeros((nx + 1, ny))
    gamma_w[1:-1, :] = gamma_e  # (nx-1, ny) → (nx-1, ny)
    gamma_w[0, :] = gamma[0, :]
    gamma_w[-1, :] = gamma[-1, :]

    gamma_n = 2.0 * gamma[:, :-1] * gamma[:, 1:] / (gamma[:, :-1] + gamma[:, 1:] + 1e-30)
    # gamma_n — (nx, ny-1), содержит Γ на north face (j+½) для j=0..ny-2
    gamma_s = np.zeros((nx, ny + 1))
    gamma_s[:, 1:-1] = gamma_n  # (nx, ny-1) → (nx, ny-1)
    gamma_s[:, 0] = gamma[:, 0]
    gamma_s[:, -1] = gamma[:, -1]

    # Диффузионные проводимости [м²/с * м = м³/с ... ну, по смыслу — поток]
    # d_e = Γ_e * Δy / Δx  [··]
    de = gamma_e * dy / dx   # (nx-1, ny) — east face coefficients
    dw = de                  # но со сдвигом
    dn = gamma_n * dx / dy   # (nx, ny-1)
    ds = dn

    lap = np.zeros((nx, ny))

    # Внутренние ячейки
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            lap[i, j] = (
                de[i, j] * (phi[i + 1, j] - phi[i, j])
                - de[i - 1, j] * (phi[i, j] - phi[i - 1, j])
                + dn[i, j] * (phi[i, j + 1] - phi[i, j])
                - dn[i, j - 1] * (phi[i, j] - phi[i, j - 1])
            )

    if bc is not None:
        # Запад (i=0)
        bc_type, bc_val = bc.west
        if bc_type == BoundaryCondition.WALL:
            # ∂φ/∂x = 0 → flux = 0
            pass
        elif bc_type == BoundaryCondition.INLET:
            lap[0, :] += 2.0 * gamma[0, :] * dy / dx * (bc_val - phi[0, :])
        elif bc_type == BoundaryCondition.OUTLET:
            pass
        # Восток (i=nx-1)
        bc_type, bc_val = bc.east
        if bc_type == BoundaryCondition.WALL:
            pass
        elif bc_type == BoundaryCondition.OUTLET:
            pass
        elif bc_type == BoundaryCondition.INLET:
            lap[-1, :] += 2.0 * gamma[-1, :] * dy / dx * (bc_val - phi[-1, :])
        # Юг (j=0)
        bc_type, bc_val = bc.south
        if bc_type == BoundaryCondition.INLET:
            lap[:, 0] += 2.0 * gamma[:, 0] * dx / dy * (bc_val - phi[:, 0])
        # Север (j=ny-1)
        bc_type, bc_val = bc.north
        if bc_type == BoundaryCondition.INLET:
            lap[:, -1] += 2.0 * gamma[:, -1] * dx / dy * (bc_val - phi[:, -1])

    return lap


def laplacian_nonorthogonal(
    phi: np.ndarray,
    gamma: float | np.ndarray,
    mesh: Mesh2D,
    nonorth_correction: np.ndarray,   # [nx+1, ny+1] sin(θ) — мера неортогональности
    bc: Boundary2D | None = None,
) -> np.ndarray:
    """
    Лапласиан с коррекцией неортогональности сетки (Jasak, 1996).

    Основной вклад (orthogonal):
        ∇·(Γ∇φ)_orth = Σ_f Γ_f · (φ_N - φ_P) / |d| · S_f

    Коррекция (non-orthogonal) — разложение градиента на нормальную
    и касательную составляющие:

        ∇·(Γ∇φ)_total = ∇·(Γ∇φ)_orth +
                        Σ_f Γ_f · ((∇φ)_f - (∇φ)_orth) · S_f

    Args:
        phi:                поле [nx, ny]
        gamma:              коэффициент диффузии
        mesh:               сетка
        nonorth_correction: массив мер неортогональности [nx+1, ny+1]

    Returns:
        Лапласиан с коррекцией [nx, ny].
    """
    # Основная ортогональная часть
    lap_orth = laplacian(phi, gamma, mesh, bc)

    # Коррекция: запасной член для неортогональности
    # на равномерной сетке nonorth_correction = 0 → коррекция не нужна
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    grad_x_f, grad_y_f = grad_face(phi, mesh, bc)

    # Коррекция только там, где сетка неортогональна
    correction = np.zeros((nx, ny))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # x-грани
            theta_e = nonorth_correction[i + 1, j]
            theta_w = nonorth_correction[i, j]
            # y-грани
            theta_n = nonorth_correction[i, j + 1]
            theta_s = nonorth_correction[i, j]

            corr_x = (
                (grad_x_f[i + 1, j] - (phi[i + 1, j] - phi[i, j]) / dx) * dy * theta_e
                - (grad_x_f[i, j] - (phi[i, j] - phi[i - 1, j]) / dx) * dy * theta_w
            ) / dx

            corr_y = (
                (grad_y_f[i, j + 1] - (phi[i, j + 1] - phi[i, j]) / dy) * dx * theta_n
                - (grad_y_f[i, j] - (phi[i, j] - phi[i, j - 1]) / dy) * dx * theta_s
            ) / dy

            correction[i, j] = corr_x + corr_y

    return lap_orth + correction
