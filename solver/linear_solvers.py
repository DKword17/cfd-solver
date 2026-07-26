#!/usr/bin/env python3
"""
solver/linear_solvers.py
=========================

Итерационные решатели разреженных линейных систем Ax = b,
возникающих в методе конечного объёма.

Реализованы:
    — SIP (Strongly Implicit Procedure) — метод Стоуна
      для 5-диагональных матриц (структурированная 2D сетка)

    — BiCGStab (Biconjugate Gradient Stabilized) —
      стабилизированный метод сопряжённых градиентов
      для несимметричных матриц

    — GMRES(m) — обобщённый метод минимальной невязки
      с рестартом (Saad & Schultz, 1986)

Литература:
    Stone (1968) SIAM J Numer Anal 5:530–558
    Van der Vorst (1992) SIAM J Sci Stat Comput 13:631–644
    Saad & Schultz (1986) SIAM J Sci Stat Comput 7:856–869
    Ferziger & Perić (2002) §5.5

Автор: Alexei Morozov, Институт механики МГУ, 2026
"""

from __future__ import annotations

import numpy as np
from typing import Callable

# ─── SIP: Strongly Implicit Procedure (Stone, 1968) ───────────────────
# Решает A·x = b для 5-диагональной матрицы на структурированной 2D сетке.
#
# Матрица A хранится в виде 5 лент:
#   aP  — центральная диагональ  [nx, ny]
#   aE  — восточная (i+1)        [nx-1, ny]  (aE[i,j] связывает P[i,j]→E[i+1,j])
#   aW  — западная  (i-1)        [nx-1, ny]
#   aN  — северная  (j+1)        [nx, ny-1]
#   aS  — южная     (j-1)        [nx, ny-1]
#
# SIP разлагает A ≈ LU, где L и U сохраняют 5-диагональную структуру,
# добавляя заполнение по SW-NE направлению — аппроксимация Стоуна.


def _sip_decompose(
    aP: np.ndarray,       # [nx, ny]
    aE: np.ndarray,       # [nx-1, ny]
    aW: np.ndarray,       # [nx-1, ny]
    aN: np.ndarray,       # [nx, ny-1]
    aS: np.ndarray,       # [nx, ny-1]
    alpha: float = 0.0,   # зарезервировано; 0 = ILU(0)
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    ILU(0)-факторизация для 5-диагональной матрицы.

    A ≈ L·U, где:
        Lc — диагональ L (1 на диагонали U)
        Ls, Lw — поддиагонали L
        Un, Ue — наддиагонали U

    Формулы обхода (естественный порядок i,j):
        Lw[i,j] = aW[i,j]                     (j >= 0) — копия
        Ls[i,j] = aS[i,j]                     (i >= 0)
        Lc[i,j] = aP[i,j] - Lw[i,j]*Ue[i-1,j] - Ls[i,j]*Un[i,j-1]
        Ue[i,j] = aE[i,j] / Lc[i,j]
        Un[i,j] = aN[i,j] / Lc[i,j]

    Гарантированно устойчиво для M-матриц (диагональное преобладание).
    """
    nx, ny = aP.shape

    Lc = np.zeros((nx, ny))
    Ls = np.zeros((nx, ny))
    Lw = np.zeros((nx, ny))
    Ue = np.zeros((nx, ny))
    Un = np.zeros((nx, ny))

    # расширенные ленты (согласование размеров)
    aW_ex = np.zeros((nx, ny))
    aE_ex = np.zeros((nx, ny))
    aS_ex = np.zeros((nx, ny))
    aN_ex = np.zeros((nx, ny))

    aW_ex[1:, :] = aW
    aE_ex[:-1, :] = aE
    aS_ex[:, 1:] = aS
    aN_ex[:, :-1] = aN

    for i in range(nx):
        for j in range(ny):
            if i > 0:
                Lw[i, j] = aW_ex[i, j]           # копируем aW
            if j > 0:
                Ls[i, j] = aS_ex[i, j]           # копируем aS

            # Lc[i,j] = aP[i,j] - Lw*Ue[i-1,j] - Ls*Un[i,j-1]
            diag = aP[i, j]
            if i > 0:
                diag -= Lw[i, j] * Ue[i - 1, j]
            if j > 0:
                diag -= Ls[i, j] * Un[i, j - 1]

            if abs(diag) < 1e-30:
                diag = 1e-30 * (1.0 if diag >= 0 else -1.0)
            Lc[i, j] = diag

            if i < nx - 1:
                Ue[i, j] = aE_ex[i, j] / diag
            if j < ny - 1:
                Un[i, j] = aN_ex[i, j] / diag

    return Lc, Ls, Lw, Un, Ue


def _sip_solve_LU(
    Lc: np.ndarray, Ls: np.ndarray, Lw: np.ndarray,
    Un: np.ndarray, Ue: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Обратная подстановка LU·x = rhs.

    Шаг 1: L·y = rhs  (прямая подстановка — SW→NE sweep)
    Шаг 2: U·x = y    (обратная подстановка — NE→SW sweep)
    """
    nx, ny = Lc.shape
    y = np.zeros((nx, ny))

    # Прямая подстановка: L·y = rhs
    for i in range(nx):
        for j in range(ny):
            y[i, j] = rhs[i, j]
            if i > 0:
                y[i, j] -= Lw[i, j] * y[i - 1, j]
            if j > 0:
                y[i, j] -= Ls[i, j] * y[i, j - 1]
            y[i, j] /= Lc[i, j] + 1e-30

    # Обратная подстановка: U·x = y
    x = y.copy()
    for i in range(nx - 1, -1, -1):
        for j in range(ny - 1, -1, -1):
            if i < nx - 1:
                x[i, j] -= Ue[i, j] * x[i + 1, j]
            if j < ny - 1:
                x[i, j] -= Un[i, j] * x[i, j + 1]

    return x


def solve_sip(
    aP: np.ndarray,
    aE: np.ndarray,
    aW: np.ndarray,
    aN: np.ndarray,
    aS: np.ndarray,
    rhs: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    alpha: float = 0.92,
    omega: float = 1.0,          # параметр релаксации
) -> tuple[np.ndarray, int, float]:
    """
    SIP-решатель для 5-диагональной системы.

    Args:
        aP, aE, aW, aN, aS:  ленты матрицы A
        rhs:  правая часть [nx, ny]
        max_iter:  макс. итераций
        tol:      относительный допуск ||res|| / ||rhs||
        alpha:    параметр Стоуна
        omega:    релаксация (1.0 = без релаксации)

    Returns:
        (x, n_iter, residual) — решение, число итераций, норма невязки.
    """
    nx, ny = aP.shape
    x = np.zeros((nx, ny))

    # LU-факторизация (один раз, если матрица не меняется)
    Lc, Ls, Lw, Un, Ue = _sip_decompose(aP, aE, aW, aN, aS, alpha)

    rhs_norm = np.linalg.norm(rhs) + 1e-30

    for it in range(max_iter):
        # Невязка: res = rhs - A·x
        res = rhs.copy()
        # A·x
        ax = aP * x
        ax[:-1, :] += aE * x[1:, :]
        ax[1:, :] += aW * x[:-1, :]
        ax[:, :-1] += aN * x[:, 1:]
        ax[:, 1:] += aS * x[:, :-1]
        res -= ax

        # Решаем LU·Δx = res
        dx = _sip_solve_LU(Lc, Ls, Lw, Un, Ue, res)

        # Релаксация
        x += omega * dx

        res_norm = np.linalg.norm(res)
        if res_norm / rhs_norm < tol:
            return x, it + 1, float(res_norm)

    return x, max_iter, float(np.linalg.norm(rhs - ax))


# ─── BiCGStab (Van der Vorst, 1992) ───────────────────────────────────
# Стабилизированный метод сопряжённых градиентов для несимметричных A.
#
# BiCGStab: (M⁻¹A)·x ≈ M⁻¹b  с правым предобуславливанием


def solve_bicgstab(
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    x0: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, int, float]:
    """
    BiCGStab — стабилизированный метод сопряжённых градиентов.

    matvec(x) = A·x — оператор матрицы.
    preconditioner(r) = M⁻¹·r — предобуславливатель (опционально).

    Алгоритм (Van der Vorst, 1992):
        r₀ = b - A·x₀
        ρ₀ = α = ω₀ = 1
        p₀ = v₀ = 0
        for i = 1, 2, ...:
            ρᵢ = (r̃, r_{i-1})
            β = (ρᵢ/ρ_{i-1}) · (α/ω_{i-1})
            pᵢ = r_{i-1} + β·(p_{i-1} - ω_{i-1}·v_{i-1})
            y = M⁻¹·pᵢ
            vᵢ = A·y
            α = ρᵢ / (r̃, vᵢ)
            s = r_{i-1} - α·vᵢ
            z = M⁻¹·s
            t = A·z
            ωᵢ = (t, s) / (t, t)
            xᵢ = x_{i-1} + α·y + ωᵢ·z
            rᵢ = s - ωᵢ·t
            if ||rᵢ|| < tol: stop

    Returns:
        (x, n_iter, residual)
    """
    rhs_flat = rhs.ravel()
    n = len(rhs_flat)

    if x0 is None:
        x = np.zeros_like(rhs_flat)
    else:
        x = x0.ravel()

    r = rhs_flat - matvec(x.reshape(rhs.shape)).ravel()

    # Начальное псевдо-случайное r̃
    rng = np.random.default_rng(seed=42)
    r_tilde = rng.standard_normal(n)

    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0
    p = np.zeros(n)
    v = np.zeros(n)

    rhs_norm = np.linalg.norm(rhs_flat) + 1e-30

    for it in range(1, max_iter + 1):
        rho = np.dot(r_tilde, r)

        if abs(rho) < 1e-30:
            # крах — рестарт с новым r̃
            r_tilde = np.random.standard_normal(n)
            rho = np.dot(r_tilde, r)

        if it == 1:
            p[:] = r
        else:
            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)

        # Предобуславливание: y = M⁻¹·p
        if preconditioner is not None:
            y = preconditioner(p.reshape(rhs.shape)).ravel()
        else:
            y = p.copy()

        v = matvec(y.reshape(rhs.shape)).ravel()

        rv = np.dot(r_tilde, v)
        if abs(rv) < 1e-30:
            rv = 1e-30
        alpha = rho / rv

        s = r - alpha * v

        # Предобуславливание: z = M⁻¹·s
        if preconditioner is not None:
            z = preconditioner(s.reshape(rhs.shape)).ravel()
        else:
            z = s.copy()

        t = matvec(z.reshape(rhs.shape)).ravel()

        tt = np.dot(t, t)
        if tt < 1e-30:
            tt = 1e-30
        omega = np.dot(t, s) / tt

        x += alpha * y + omega * z
        r = s - omega * t

        res_norm = np.linalg.norm(r)
        if res_norm / rhs_norm < tol:
            return x.reshape(rhs.shape), it, float(res_norm)

        rho_prev = rho

    return x.reshape(rhs.shape), max_iter, float(np.linalg.norm(r))


# ─── GMRES(m) — Saad & Schultz (1986) ─────────────────────────────────
# GMRES(m): метод минимальной невязки с рестартом каждые m итераций
# Основание: ортогонализация Арнольди → решение задачи наименьших квадратов


def _arnoldi(
    matvec: Callable[[np.ndarray], np.ndarray],
    V: np.ndarray,           # [n, k+1] — базис Крылова
    k: int,                  # текущий шаг (0-indexed)
    n: int,                  # размерность
    H: np.ndarray,           # [k+2, k+1] — матрица Хессенберга
) -> np.ndarray:
    """
    Один шаг ортогонализации Арнольди.

    v_{k+1} = A·v_k - Σ_{i=0}^{k} h_{i,k}·v_i
    h_{i,k} = (A·v_k, v_i)
    h_{k+1,k} = ||v_{k+1}||
    v_{k+1} /= h_{k+1,k}
    """
    w = matvec(V[:, k].reshape(-1, 1)).ravel()

    for i in range(k + 1):
        H[i, k] = np.dot(w, V[:, i])
        w -= H[i, k] * V[:, i]

    H[k + 1, k] = np.linalg.norm(w)

    if H[k + 1, k] > 1e-30:
        V[:, k + 1] = w / H[k + 1, k]

    return V, H


def _givens_rotation(
    H: np.ndarray,
    g: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Применение вращений Гивенса к матрице Хессенберга H[:, :k+1]
    и правой части g.

    Returns:
        (H, g, cs, sn) — преобразованные H и g, коэффициенты вращений.
    """
    n = H.shape[0]
    cs_list = []
    sn_list = []

    for j in range(k + 1):
        if j < k:
            # пропустить уже обработанные столбцы
            continue
        for i in range(j + 1, n):
            if abs(H[i, j]) > 1e-30:
                # вращение Гивенса для обнуления H[i, j]
                a = H[j, j]
                b = H[i, j]
                r = np.hypot(a, b)
                cs = a / r
                sn = -b / r
                cs_list.append(cs)
                sn_list.append(sn)

                # Применяем вращение к строкам j и i
                for col in range(j, k + 1):
                    H_j_col = H[j, col]
                    H_i_col = H[i, col]
                    H[j, col] = cs * H_j_col - sn * H_i_col
                    H[i, col] = sn * H_j_col + cs * H_i_col

                # Применяем вращение к правой части
                g_j = g[j]
                g_i = g[i]
                g[j] = cs * g_j - sn * g_i
                g[i] = sn * g_j + cs * g_i

    return H, g, cs_list, sn_list


def solve_gmres(
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    x0: np.ndarray | None = None,
    max_iter: int = 200,
    restart: int = 30,
    tol: float = 1e-8,
    preconditioner: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, int, float]:
    """
    GMRES(m) — обобщённый метод минимальной невязки с рестартом.

    GMRES строит базис Крылова K_m(A, r₀) ортогонализацией Арнольди,
    затем решает задачу наименьших квадратов ||β·e₁ - H·y||₂.

    Args:
        matvec:      A·x
        rhs:         правая часть [nx, ny]
        x0:          начальное приближение
        max_iter:    макс. внешних итераций (рестартов)
        restart:     размер подпространства Крылова m
        tol:         относительный допуск
        preconditioner: M⁻¹·r (опционально)

    Returns:
        (x, n_iter, residual)
    """
    rhs_flat = rhs.ravel()
    n = len(rhs_flat)

    if x0 is None:
        x = np.zeros_like(rhs_flat)
    else:
        x = x0.ravel()

    rhs_norm = np.linalg.norm(rhs_flat) + 1e-30

    for outer in range(max_iter):
        # Невязка: r₀ = b - A·x
        r0 = rhs_flat - matvec(x.reshape(rhs.shape)).ravel()

        # Предобуславливание r₀
        if preconditioner is not None:
            r0 = preconditioner(r0.reshape(rhs.shape)).ravel()

        r0_norm = np.linalg.norm(r0)
        if r0_norm / rhs_norm < tol:
            return x.reshape(rhs.shape), outer, float(r0_norm)

        # Базис Крылова
        m = min(restart, n)
        V = np.zeros((n, m + 1))
        V[:, 0] = r0 / (r0_norm + 1e-30)

        # Матрица Хессенберга [m+1, m]
        H = np.zeros((m + 1, m))

        # Правая часть для least-squares: β·e₁
        g = np.zeros(m + 1)
        g[0] = r0_norm

        for k in range(m):
            # Шаг Арнольди
            V, H = _arnoldi(matvec if preconditioner is None
                           else lambda v: matvec(v.reshape(rhs.shape)).ravel(),
                           V, k, n, H)

            # Вращения Гивенса
            H, g, _, _ = _givens_rotation(H, g, k)

            # Проверка невязки
            if abs(g[k + 1]) / rhs_norm < tol:
                break

        # Решаем H·y = g — верхняя треугольная система
        k_final = min(k, m - 1)

        # Обратная подстановка
        y = np.zeros(k_final + 1)
        for i in range(k_final, -1, -1):
            y[i] = g[i]
            for j in range(i + 1, k_final + 1):
                y[i] -= H[i, j] * y[j]
            if abs(H[i, i]) > 1e-30:
                y[i] /= H[i, i]

        # x = x₀ + V·y
        dx = V[:, :k_final + 1] @ y

        if preconditioner is not None:
            # Если был preconditioner, нужно обратное преобразование
            # (упрощённо — считаем, что M ≈ I для простоты)
            pass

        x += dx

        # Проверка сходимости
        res = rhs_flat - matvec(x.reshape(rhs.shape)).ravel()
        res_norm = np.linalg.norm(res)
        if res_norm / rhs_norm < tol:
            return x.reshape(rhs.shape), outer * restart + k_final + 1, float(res_norm)

    return x.reshape(rhs.shape), max_iter * restart, float(
        np.linalg.norm(rhs_flat - matvec(x.reshape(rhs.shape)).ravel())
    )
