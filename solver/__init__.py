#!/usr/bin/env python3
"""
cfd_solver/solver — пакет конечного объёма для Navier-Stokes.

Модули:
    fv_discretisation.py   — операторы градиента, дивергенции, лапласиана
    linear_solvers.py      — SIP / BiCGStab / GMRES(m)
    rhie_chow.py           — интерполяция Райса-Чоу для коллокации
    simple.py              — SIMPLE / SIMPLEC / PISO

Автор: Alexei Morozov, Институт механики МГУ, 2026
"""

from .fv_discretisation import (
    grad_cell_centered,
    grad_face,
    div_face_flux,
    laplacian,
    laplacian_nonorthogonal,
)

from .linear_solvers import (
    solve_sip,
    solve_bicgstab,
    solve_gmres,
)

from .rhie_chow import (
    rhie_chow_mass_flux_x,
    rhie_chow_mass_flux_y,
    rhie_chow_face_velocity,
)

from .simple import (
    SIMPLE,
    SIMPLEC,
    PISO,
)
