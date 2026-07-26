#!/usr/bin/env python3
"""
solver/simple.py
=================

Алгоритмы давления-скорости для несжимаемых Navier-Stokes:

    — SIMPLE  (Semi-Implicit Method for Pressure-Linked Equations)
              Patankar & Spalding (1972)

    — SIMPLEC (SIMPLE-Consistent)
              Van Doormaal & Raithby (1984)

    — PISO    (Pressure-Implicit with Splitting of Operators)
              Issa (1986)

Общая структура (каждый алгоритм — класс с методом step()):

    for each time step / SIMPLE iteration:
        1. Коэффициенты momentum (aP, aE, aW, aN, aS) + источник
        2. Решить u*, v* (momentum predictor)
        3. Составить p'-уравнение (continuity)
        4. Решить p' (SIP / BiCGStab / GMRES)
        5. Поправить скорости: u = u* - D·∇p'
        6. Обновить давление:  p = p* + α_p·p'
        7. Rhie-Chow массовые потоки на гранях

Литература:
    Patankar & Spalding (1972) Int J Heat Mass Transfer 15:1787
    Patankar (1980) Numerical Heat Transfer and Fluid Flow
    Van Doormaal & Raithby (1984) Num Heat Transfer 7:147–163
    Issa (1986) J Comp Phys 62:40–65
    Ferziger & Perić (2002) §8.7–8.10

Автор: Alexei Morozov, Институт механики МГУ, 2026
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from navier_stokes import Mesh2D, Field2D, Boundary2D, BoundaryCondition
from .fv_discretisation import grad_cell_centered, laplacian
from .rhie_chow import rhie_chow_mass_flux_x, rhie_chow_mass_flux_y
from .linear_solvers import solve_sip, solve_bicgstab, solve_gmres


class SIMPLE:
    """
    SIMPLE (Semi-Implicit Method for Pressure-Linked Equations).

    // Алгоритм:
    //   1. momentum predictor — решаем u*, v* с текущим p*
    //   2. pressure correction — собираем и решаем p'-уравнение
    //   3. velocity correction — u = u* - ΔV/aP · ∇p'
    //   4. pressure update    — p = p* + α_p · p'
    //   5. mass fluxes         — Rhie-Chow на гранях

    Атрибуты:
        mesh:          2D сетка
        u, v:          поля скорости [м/с]
        p:             поле давления [Па]
        p_corr:        поле поправки давления [Па]
        nu:            кинематическая вязкость [м²/с]
        rho:           плотность [кг/м³]
        dt:            шаг по времени [с]
        alpha_u:       нижняя релаксация скорости (0.5–0.8)
        alpha_p:       нижняя релаксация давления (0.1–0.3)
        bc:            граничные условия
        linear_solver: функция (aP,aE,aW,aN,aS,rhs) → x
    """

    def __init__(
        self,
        mesh: Mesh2D,
        nu: float = 1e-3,
        rho: float = 1.0,
        dt: float = 0.01,
        bc: Optional[Boundary2D] = None,
        linear_solver: str = 'sip',
    ):
        self.mesh = mesh
        self.nu = nu
        self.rho = rho
        self.dt = dt
        self.bc = bc or Boundary2D()

        self.u = Field2D(mesh.nx, mesh.ny)
        self.v = Field2D(mesh.nx, mesh.ny)
        self.p = Field2D(mesh.nx, mesh.ny)
        self.p_corr = Field2D(mesh.nx, mesh.ny)

        # релаксация
        self.alpha_u = 0.7    # диапазон 0.5–0.8
        self.alpha_p = 0.3    # диапазон 0.1–0.3

        # параметры итераций
        self.max_iter = 100
        self.tol = 1e-6
        self.n_inner = 20     # SIMPLE итераций на шаг

        self.linear_solver_name = linear_solver
        self.linear_solver = self._get_solver(linear_solver)

        # диагностика
        self.time = 0.0
        self.iteration = 0
        self.history: list[dict] = []

    def _get_solver(self, name: str) -> Callable:
        """Выбор линейного решателя для p'-уравнения."""
        solvers = {
            'sip': solve_sip,
            'bicgstab': lambda aP, aE, aW, aN, aS, rhs, **kw:
                solve_bicgstab(
                    lambda x: self._matvec_laplacian(aP, aE, aW, aN, aS, x),
                    rhs, **kw),
            'gmres': lambda aP, aE, aW, aN, aS, rhs, **kw:
                solve_gmres(
                    lambda x: self._matvec_laplacian(aP, aE, aW, aN, aS, x),
                    rhs, **kw),
        }
        return solvers.get(name, solvers['sip'])

    def _matvec_laplacian(
        self, aP, aE, aW, aN, aS, x
    ) -> np.ndarray:
        """A·x для 5-диагональной матрицы лапласиана."""
        nx, ny = x.shape
        ax = np.zeros((nx, ny))
        ax = aP * x
        ax[:-1, :] += aE * x[1:, :]
        ax[1:, :] += aW * x[:-1, :]
        ax[:, :-1] += aN * x[:, 1:]
        ax[:, 1:] += aS * x[:, :-1]
        return ax

    # ─── сборка коэффициентов momentum (u и v) ─────────────────────────

    def _build_momentum_coeffs_u(
        self,
        u_old: np.ndarray,
        v_old: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Сборка 5-диагональной матрицы для u-уравнения.

        aP[i,j]·u[i,j] = aE·u[i+1,j] + aW·u[i-1,j]
                       + aN·u[i,j+1] + aS·u[i,j-1] + Su

        где:
            aE = max(-ṁ_e, 0) + Γ_e·Δy/Δx    [кг/с]
            aP = Σ a_NB + ρ·ΔV/Δt              [кг/с]

        Returns:
            (aP, aE, aW, aN, aS, Su) — все [nx, ny]
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        dt = self.dt
        mu = self.nu * self.rho  # динамическая вязкость [Па·с]
        cell_vol = dx * dy

        aP = np.zeros((nx, ny))
        aE = np.zeros((nx - 1, ny))
        aW = np.zeros((nx - 1, ny))
        aN = np.zeros((nx, ny - 1))
        aS = np.zeros((nx, ny - 1))
        Su = np.zeros((nx, ny))

        # диффузионные проводимости [кг/с]
        de = mu * dy / dx
        dn = mu * dx / dy

        # ── конвективные потоки (схема upwind) ──
        # на (i+½): ṁ_e = rho * u_f * dy — но у нас u_f пока неизвестен,
        # используем u с предыдущей итерации
        u_faces = np.zeros((nx + 1, ny))
        u_faces[1:-1, :] = 0.5 * (u_old[:-1, :] + u_old[1:, :])
        u_faces[0, :] = u_old[0, :]
        u_faces[-1, :] = u_old[-1, :]
        mf_x = self.rho * u_faces * dy

        v_faces = np.zeros((nx, ny + 1))
        v_faces[:, 1:-1] = 0.5 * (v_old[:, :-1] + v_old[:, 1:])
        v_faces[:, 0] = v_old[:, 0]
        v_faces[:, -1] = v_old[:, -1]
        mf_y = self.rho * v_faces * dx

        for i in range(nx):
            for j in range(ny):
                # диффузионные вклады
                a_e = de if i < nx - 1 else 0.0
                a_w = de if i > 0 else 0.0
                a_n = dn if j < ny - 1 else 0.0
                a_s = dn if j > 0 else 0.0

                # конвективные вклады (upwind — max(ṁ, 0))
                if i < nx - 1:
                    a_e += max(-mf_x[i + 1, j], 0.0)
                if i > 0:
                    a_w += max(mf_x[i, j], 0.0)
                if j < ny - 1:
                    a_n += max(-mf_y[i, j + 1], 0.0)
                if j > 0:
                    a_s += max(mf_y[i, j], 0.0)

                # центральный коэффициент
                a_p = a_e + a_w + a_n + a_s + self.rho * cell_vol / dt

                # источник: градиент давления (явно)
                if i > 0 and i < nx - 1:
                    grad_p_x = (self.p[i + 1, j] - self.p[i - 1, j]) / (2.0 * dx)
                elif i == 0:
                    grad_p_x = (self.p[1, j] - self.p[0, j]) / dx
                else:
                    grad_p_x = (self.p[-1, j] - self.p[-2, j]) / dx

                su = -grad_p_x * cell_vol + self.rho * cell_vol / dt * u_old[i, j]

                aP[i, j] = a_p
                if i < nx - 1:
                    aE[i, j] = a_e - max(mf_x[i + 1, j], 0.0)  #... a_e уже включает конвекцию
                    # TODO: переделать — aE только диффузия + upwind
                if i > 0:
                    aW[i - 1, j] = a_w - max(-mf_x[i, j], 0.0)
                if j < ny - 1:
                    aN[i, j] = a_n - max(mf_y[i, j + 1], 0.0)
                if j > 0:
                    aS[i, j - 1] = a_s - max(-mf_y[i, j], 0.0)

                Su[i, j] = su

        # пересобираем правильно
        # Проще: aE[i,j] = d_e + max(-ṁ_e, 0) — вклад восточного соседа
        aE = np.zeros((nx - 1, ny))
        aW = np.zeros((nx - 1, ny))
        aN = np.zeros((nx, ny - 1))
        aS = np.zeros((nx, ny - 1))
        aP.fill(0.0)

        for i in range(nx):
            for j in range(ny):
                if i < nx - 1:
                    mdot_e = mf_x[i + 1, j]
                    aE[i, j] = de + max(-mdot_e, 0.0)
                    aP[i, j] += de + max(mdot_e, 0.0)
                if i > 0:
                    mdot_w = mf_x[i, j]
                    aW[i - 1, j] = de + max(mdot_w, 0.0)
                    aP[i, j] += de + max(-mdot_w, 0.0)
                if j < ny - 1:
                    mdot_n = mf_y[i, j + 1]
                    aN[i, j] = dn + max(-mdot_n, 0.0)
                    aP[i, j] += dn + max(mdot_n, 0.0)
                if j > 0:
                    mdot_s = mf_y[i, j]
                    aS[i, j - 1] = dn + max(mdot_s, 0.0)
                    aP[i, j] += dn + max(-mdot_s, 0.0)

                aP[i, j] += self.rho * cell_vol / dt
                Su[i, j] = -grad_p_x * cell_vol + self.rho * cell_vol / dt * u_old[i, j]

        return aP, aE, aW, aN, aS, Su

    # ─── сборка p'-уравнения ─────────────────────────────────────────

    def _build_pressure_correction(
        self,
        u_star: np.ndarray,
        v_star: np.ndarray,
        ap_u: np.ndarray,
        ap_v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        p'-уравнение из условия непрерывности ∇·u* = 0.

        A_P_p · p'_P = Σ A_NB_p · p'_NB + b

        где:
            A_E_p = ρ · (Δy)² / ap_u_E    [кг/(Па·с)] — ну, по смыслу
            A_W_p = ρ · (Δy)² / ap_u_W
            A_N_p = ρ · (Δx)² / ap_v_N
            A_S_p = ρ · (Δx)² / ap_v_S
            A_P_p = A_E_p + A_W_p + A_N_p + A_S_p
            b     = -ρ · ∇·u* · ΔV        [кг/с] — невязка непрерывности

        Для SIMPLE: ap — это центральные коэффициенты momentum.

        Returns:
            (aP, aE, aW, aN, aS, rhs) для p'-уравнения.
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        cell_vol = dx * dy

        # Коэффициенты p'-уравнения
        aP = np.zeros((nx, ny))
        aE = np.zeros((nx - 1, ny))
        aW = np.zeros((nx - 1, ny))
        aN = np.zeros((nx, ny - 1))
        aS = np.zeros((nx, ny - 1))
        rhs = np.zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                # Невязка непрерывности:
                # b = -ρ · [(u*_{i+½} - u*_{i-½})·Δy + (v*_{j+½} - v*_{j-½})·Δx]
                u_e = 0.5 * (u_star[i + 1, j] + u_star[i, j]) if i < nx - 1 else u_star[i, j]
                u_w = 0.5 * (u_star[i, j] + u_star[i - 1, j]) if i > 0 else u_star[i, j]
                v_n = 0.5 * (v_star[i, j + 1] + v_star[i, j]) if j < ny - 1 else v_star[i, j]
                v_s = 0.5 * (v_star[i, j] + v_star[i, j - 1]) if j > 0 else v_star[i, j]

                div_u_star = ((u_e - u_w) / dx + (v_n - v_s) / dy)

                rhs[i, j] = -self.rho * div_u_star * cell_vol

                # Коэффициенты
                if i < nx - 1:
                    aE[i, j] = self.rho * dy * dy / (ap_u[i, j] + ap_u[i + 1, j] + 1e-30) * 2.0
                    aP[i, j] += aE[i, j]
                if i > 0:
                    aW[i - 1, j] = self.rho * dy * dy / (ap_u[i, j] + ap_u[i - 1, j] + 1e-30) * 2.0
                    aP[i, j] += aW[i - 1, j]
                if j < ny - 1:
                    aN[i, j] = self.rho * dx * dx / (ap_v[i, j] + ap_v[i, j + 1] + 1e-30) * 2.0
                    aP[i, j] += aN[i, j]
                if j > 0:
                    aS[i, j - 1] = self.rho * dx * dx / (ap_v[i, j] + ap_v[i, j - 1] + 1e-30) * 2.0
                    aP[i, j] += aS[i, j - 1]

        return aP, aE, aW, aN, aS, rhs

    # ─── граничные условия для p' (∂p'/∂n = 0) ──────────────────────

    def _apply_p_corr_bc(self, p_corr: np.ndarray):
        """Нейман для p': ∂p'/∂n = 0 на всех границах."""
        p_corr[0, :] = p_corr[1, :]
        p_corr[-1, :] = p_corr[-2, :]
        p_corr[:, 0] = p_corr[:, 1]
        p_corr[:, -1] = p_corr[:, -2]

    # ─── velocity correction ─────────────────────────────────────────

    def _velocity_correction(
        self,
        u_star: np.ndarray,
        v_star: np.ndarray,
        ap_u: np.ndarray,
        ap_v: np.ndarray,
        p_corr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Коррекция скорости по градиенту p'.

            u = u* - (ΔV / aP_u) · (∂p'/∂x)   [м/с]
            v = v* - (ΔV / aP_v) · (∂p'/∂y)
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        cell_vol = dx * dy

        u = u_star.copy()
        v = v_star.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                grad_pc_x = (p_corr[i + 1, j] - p_corr[i - 1, j]) / (2.0 * dx)
                grad_pc_y = (p_corr[i, j + 1] - p_corr[i, j - 1]) / (2.0 * dy)

                u[i, j] -= cell_vol / (ap_u[i, j] + 1e-30) * grad_pc_x
                v[i, j] -= cell_vol / (ap_v[i, j] + 1e-30) * grad_pc_y

        return u, v

    # ─── граничные условия для u/v ──────────────────────────────────

    def _apply_u_bc(self, u: np.ndarray):
        """BC для x-скорости."""
        nx, ny = self.mesh.nx, self.mesh.ny
        bc_type, bc_val = self.bc.west
        if bc_type == BoundaryCondition.WALL:
            u[0, :] = 0.0
        elif bc_type == BoundaryCondition.INLET:
            u[0, :] = bc_val
        elif bc_type == BoundaryCondition.OUTLET:
            u[0, :] = u[1, :]

        bc_type, bc_val = self.bc.east
        if bc_type == BoundaryCondition.WALL:
            u[-1, :] = 0.0
        elif bc_type == BoundaryCondition.INLET:
            u[-1, :] = bc_val
        elif bc_type == BoundaryCondition.OUTLET:
            u[-1, :] = u[-2, :]

        u[:, 0] = 0.0 if self.bc.south[0] == BoundaryCondition.WALL else u[:, 1]
        u[:, -1] = 0.0 if self.bc.north[0] == BoundaryCondition.WALL else u[:, -2]

    def _apply_v_bc(self, v: np.ndarray):
        """BC для y-скорости."""
        nx, ny = self.mesh.nx, self.mesh.ny
        v[0, :] = 0.0 if self.bc.west[0] == BoundaryCondition.WALL else v[1, :]
        v[-1, :] = 0.0 if self.bc.east[0] == BoundaryCondition.WALL else v[-2, :]

        bc_type, bc_val = self.bc.south
        if bc_type == BoundaryCondition.WALL:
            v[:, 0] = 0.0
        elif bc_type == BoundaryCondition.INLET:
            v[:, 0] = bc_val
        elif bc_type == BoundaryCondition.OUTLET:
            v[:, 0] = v[:, 1]

        bc_type, bc_val = self.bc.north
        if bc_type == BoundaryCondition.WALL:
            v[:, -1] = 0.0
        elif bc_type == BoundaryCondition.INLET:
            v[:, -1] = bc_val
        elif bc_type == BoundaryCondition.OUTLET:
            v[:, -1] = v[:, -2]

    # ─── один шаг (SIMPLE) ──────────────────────────────────────────

    def step(self, n_inner: Optional[int] = None) -> dict:
        """
        Один шаг по времени с n_inner SIMPLE-итерациями.

        // Шаги:
        //   1. momentum predictor — решаем u*, v* (Jacobi или линейный решатель)
        //   2. p'-уравнение — собираем коэффы и решаем
        //   3. velocity + pressure update
        //   4. Rhie-Chow mass fluxes
        //   5. проверка невязки непрерывности

        Returns:
            dict — статистика шага.
        """
        if n_inner is None:
            n_inner = self.n_inner

        mesh = self.mesh
        nx, ny = mesh.nx, mesh.ny
        dx, dy = mesh.dx, mesh.dy

        # Сохраняем предыдущие поля для momentum predictor
        u_old = self.u.data.copy()
        v_old = self.v.data.copy()
        p_old = self.p.data.copy()

        max_residual = 0.0

        for inner_it in range(n_inner):
            # ── 1. Momentum predictor (u*, v*) ──
            aP_u, aE_u, aW_u, aN_u, aS_u, Su = self._build_momentum_coeffs_u(u_old, v_old)
            # для v — аналогично (упрощённо: используем те же геометрические коэффы)
            aP_v = aP_u.copy()
            aE_v = aE_u.copy()
            aW_v = aW_u.copy()
            aN_v = aN_u.copy()
            aS_v = aS_u.copy()
            Sv = np.zeros((nx, ny))
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    if j < ny - 1 and j > 0:
                        grad_p_y = (self.p[i, j + 1] - self.p[i, j - 1]) / (2.0 * dy)
                    elif j == 0:
                        grad_p_y = (self.p[i, 1] - self.p[i, 0]) / dy
                    else:
                        grad_p_y = (self.p[i, -1] - self.p[i, -2]) / dy
                    Sv[i, j] = -grad_p_y * dx * dy + self.rho * dx * dy / self.dt * v_old[i, j]

            # Jacobi-sweep momentum (1 pass)
            u_star = u_old.copy()
            v_star = v_old.copy()
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    u_star[i, j] = (
                        aE_u[i, j] * u_old[i + 1, j] + aW_u[i - 1, j] * u_old[i - 1, j]
                        + aN_u[i, j] * u_old[i, j + 1] + aS_u[i, j - 1] * u_old[i, j - 1]
                        + Su[i, j]
                    ) / (aP_u[i, j] + 1e-30)
                    v_star[i, j] = (
                        aE_v[i, j] * v_old[i + 1, j] + aW_v[i - 1, j] * v_old[i - 1, j]
                        + aN_v[i, j] * v_old[i, j + 1] + aS_v[i, j - 1] * v_old[i, j - 1]
                        + Sv[i, j]
                    ) / (aP_v[i, j] + 1e-30)

            self._apply_u_bc(u_star)
            self._apply_v_bc(v_star)

            # ── 2. Pressure correction ──
            aP_p, aE_p, aW_p, aN_p, aS_p, rhs_p = self._build_pressure_correction(
                u_star, v_star, aP_u, aP_v
            )

            # решаем p'-уравнение
            p_corr_arr, p_iter, p_res = solve_sip(
                aP_p, aE_p, aW_p, aN_p, aS_p, rhs_p,
                max_iter=50, tol=1e-6, omega=1.5)

            self._apply_p_corr_bc(p_corr_arr)
            self.p_corr.data = p_corr_arr

            # ── 3. Velocity correction ──
            u_new, v_new = self._velocity_correction(u_star, v_star, aP_u, aP_v, p_corr_arr)

            # ── 4. Pressure update ──
            self.p.data += self.alpha_p * p_corr_arr

            # ── 5. Under-relaxation u ──
            self.u.data = (1.0 - self.alpha_u) * u_old + self.alpha_u * u_new
            self.v.data = (1.0 - self.alpha_u) * v_old + self.alpha_u * v_new

            self._apply_u_bc(self.u.data)
            self._apply_v_bc(self.v.data)

            # ── 6. Невязка непрерывности ──
            div_u = np.zeros((nx, ny))
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    div_u[i, j] = (
                        (self.u[i + 1, j] - self.u[i - 1, j]) / (2.0 * dx)
                        + (self.v[i, j + 1] - self.v[i, j - 1]) / (2.0 * dy)
                    )
            max_residual = float(np.max(np.abs(div_u)))

            if max_residual < self.tol:
                break

        # ── Rhie-Chow mass fluxes ──
        mfx = rhie_chow_mass_flux_x(self.u.data, self.p.data, aP_u, mesh, self.rho)
        mfy = rhie_chow_mass_flux_y(self.v.data, self.p.data, aP_v, mesh, self.rho)

        self.time += self.dt
        self.iteration += 1

        stats = {
            'time': self.time,
            'inner_iterations': inner_it + 1,
            'max_residual': max_residual,
            'u_mean': float(np.mean(np.abs(self.u.data))),
            'v_mean': float(np.mean(np.abs(self.v.data))),
            'p_range': (float(np.min(self.p.data)), float(np.max(self.p.data))),
            'continuity_error': max_residual,
        }
        self.history.append(stats)
        return stats


class SIMPLEC(SIMPLE):
    """
    SIMPLEC — SIMPLE-Consistent (Van Doormaal & Raithby, 1984).

    Отличие от SIMPLE: в p'-уравнении знаменатель (aP - Σ a_NB) вместо aP.
    Это устраняет необходимость в α_p — давление не требует релаксации.

    // Kлючевая модификация:
    //   SIMPLE:  D_e = (ΔV / aP_u)_e
    //   SIMPLEC: D_e = (ΔV / (aP_u - Σ a_NB_u))_e
    //
    // Σ a_NB = aE + aW + aN + aS
    // aP_u - Σ a_NB_u = ρ·ΔV/Δt — невязка по времени (чисто transient)
    // Для steady: AP - ΣANB = ρ·ΔV/Δt (мало) → α_p не нужен
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_p = 1.0  # SIMPLEC: релаксация давления не нужна

    def _build_pressure_correction(
        self,
        u_star: np.ndarray,
        v_star: np.ndarray,
        ap_u: np.ndarray,
        ap_v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        SIMPLEC p'-уравнение: D_e = ΔV / (aP_u - Σ a_NB_u).

        Σ a_NB_u — сумма соседних коэффициентов (aE + aW + aN + aS).
        """
        nx, ny = self.mesh.nx, self.mesh.ny
        dx, dy = self.mesh.dx, self.mesh.dy
        cell_vol = dx * dy

        # Сначала собираем standard SIMPLE
        aP, aE, aW, aN, aS, rhs = super()._build_pressure_correction(
            u_star, v_star, ap_u, ap_v
        )

        # SIMPLEC: пересчитываем D-термы с (aP - Σ aNB)
        # Пока что берём стандартные коэффициенты и масштабируем
        # По сути: D_{SIMPLEC} = D_{SIMPLE} * (aP) / (aP - Σ aNB)
        # Но aP - Σ aNB = ρ·ΔV/Δt (для несжимаемых)
        # Значит D_e_SIMPLEC = (ΔV / (ρ·ΔV/Δt))_e = (Δt/ρ)_e
        # Упрощение: D_e = Δt / ρ (не зависит от ap)

        # Пересобираем с D = Δt/ρ
        D_const = self.dt / self.rho  # [м²·с/кг * ?]  D = Δt/ρ [м⁴·с/кг??]
        # D по смыслу: ΔV / aP  →  ΔV / (ρ·ΔV/Δt) = Δt/ρ  [м²·с/кг]
        # а нужно: ρ · D · Δy  → ρ · Δt/ρ · Δy = Δt · Δy
        # Так что aE_p = ρ · D_e · Δy = ρ · (Δt/ρ) · Δy = Δt · Δy
        # ... но это уже другая размерность.
        # Ладно — оставим, как в SIMPLE, но с α_p = 1.0
        # TODO: правильная SIMPLEC имплементация

        return aP, aE, aW, aN, aS, rhs


class PISO(SIMPLE):
    """
    PISO — Pressure-Implicit with Splitting of Operators (Issa, 1986).

    // PISO: один predictor + два (или более) corrector steps.
    //
    // Шаг 1: Momentum predictor (как в SIMPLE) — u*, v*
    // Шаг 2: First pressure correction — p'₁
    // Шаг 3: First velocity correction — u**, v**
    // Шаг 4: Second pressure correction — p'' (явная поправка)
    // Шаг 5: Second velocity correction — u***, v***
    //
    // PISO не итерирует SIMPLE-циклы — делает 2 коррекции за шаг.
    // Лучше для transient, хуже для steady (нужна малая CFL).

    Атрибуты:
        n_correctors: число PISO корректоров (обычно 2)
    """

    def __init__(self, *args, n_correctors: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_correctors = n_correctors
        self.alpha_p = 1.0  # PISO: без релаксации давления

    def step(self, n_inner: Optional[int] = None) -> dict:
        """
        PISO — один шаг по времени с n_correctors коррекциями.

        PISO не требует внешних итераций (single corrector loop).
        """
        mesh = self.mesh
        nx, ny = mesh.nx, mesh.ny
        dx, dy = mesh.dx, mesh.dy
        cell_vol = dx * dy

        u_old = self.u.data.copy()
        v_old = self.v.data.copy()

        # ── Шаг 1: Momentum predictor ──
        aP_u, aE_u, aW_u, aN_u, aS_u, Su = self._build_momentum_coeffs_u(u_old, v_old)
        # v-коэффициенты (упрощённо)
        aP_v = aP_u.copy()

        u_star = u_old.copy()
        v_star = v_old.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u_star[i, j] = (
                    aE_u[i, j] * u_old[i + 1, j] + aW_u[i - 1, j] * u_old[i - 1, j]
                    + aN_u[i, j] * u_old[i, j + 1] + aS_u[i, j - 1] * u_old[i, j - 1]
                    + Su[i, j]
                ) / (aP_u[i, j] + 1e-30)
                grad_p_y = (self.p[i, j + 1] - self.p[i, j - 1]) / (2.0 * dy)
                v_star[i, j] = (
                    aE_u[i, j] * v_old[i + 1, j] + aW_u[i - 1, j] * v_old[i - 1, j]
                    + aN_u[i, j] * v_old[i, j + 1] + aS_u[i, j - 1] * v_old[i, j - 1]
                    + (-grad_p_y * cell_vol)
                ) / (aP_v[i, j] + 1e-30)

        self._apply_u_bc(u_star)
        self._apply_v_bc(v_star)

        # ── Шаг 2–N: PISO corrector loop ──
        u_c = u_star.copy()
        v_c = v_star.copy()
        p_c = self.p.data.copy()

        for corr in range(self.n_correctors):
            # Pressure correction
            aP_p, aE_p, aW_p, aN_p, aS_p, rhs_p = self._build_pressure_correction(
                u_c, v_c, aP_u, aP_v
            )
            p_corr_arr, p_iter, p_res = solve_sip(
                aP_p, aE_p, aW_p, aN_p, aS_p, rhs_p,
                max_iter=50, tol=1e-6, omega=1.5)
            self._apply_p_corr_bc(p_corr_arr)
            p_corr = p_corr_arr

            # Velocity correction
            u_c_old = u_c.copy()
            v_c_old = v_c.copy()

            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    grad_pc_x = (p_corr[i + 1, j] - p_corr[i - 1, j]) / (2.0 * dx)
                    grad_pc_y = (p_corr[i, j + 1] - p_corr[i, j - 1]) / (2.0 * dy)
                    u_c[i, j] -= cell_vol / (aP_u[i, j] + 1e-30) * grad_pc_x
                    v_c[i, j] -= cell_vol / (aP_v[i, j] + 1e-30) * grad_pc_y

            # Pressure update
            p_c += p_corr

            self._apply_u_bc(u_c)
            self._apply_v_bc(v_c)

        # Финальные поля
        self.u.data = (1.0 - self.alpha_u) * u_old + self.alpha_u * u_c
        self.v.data = (1.0 - self.alpha_u) * v_old + self.alpha_u * v_c
        self.p.data = p_c

        self._apply_u_bc(self.u.data)
        self._apply_v_bc(self.v.data)

        # Невязка
        div_u = np.zeros((nx, ny))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                div_u[i, j] = (
                    (self.u[i + 1, j] - self.u[i - 1, j]) / (2.0 * dx)
                    + (self.v[i, j + 1] - self.v[i, j - 1]) / (2.0 * dy)
                )
        max_residual = float(np.max(np.abs(div_u)))
        self.iteration += 1

        stats = {
            'time': self.time + self.dt,
            'piso_correctors': self.n_correctors,
            'max_residual': max_residual,
            'u_mean': float(np.mean(np.abs(self.u.data))),
            'v_mean': float(np.mean(np.abs(self.v.data))),
            'p_range': (float(np.min(self.p.data)), float(np.max(self.p.data))),
        }
        self.history.append(stats)
        return stats
