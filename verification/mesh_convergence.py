#!/usr/bin/env python3
"""
verification/mesh_convergence.py
================================

Systematic mesh convergence study for the 2D incompressible
Navier-Stokes solver using the Grid Convergence Index (GCI)
methodology of Roache (1998).

The study computes solutions across 5 grid levels with a constant
refinement factor r = √2 ≈ 1.414.  For each grid level the
discretisation error is quantified, the apparent convergence order
is estimated, and the asymptotic range is verified.

Method (Roache 1998)
--------------------
For a triplet of grids (1 = fine, 2 = medium, 3 = coarse) with
refinement factor r = h₂/h₁ = h₃/h₂:

    p = ln((f₃ - f₂) / (f₂ - f₁)) / ln(r)

    f_{h=0} = f₁ + (f₁ - f₂) / (r^p - 1)

    GCI_{fine} = F_s · |(f₁ - f₂) / f₁| / (r^p - 1)

where F_s = 1.25 (safety factor for 3-grid comparisons).

Asymptotic range is confirmed when:

    GCI_{23} / (r^p · GCI_{12}) ≈ 1.0

References
----------
- Roache, P. J. (1998). Verification and Validation in Computational
  Science and Engineering. Hermosa Publishers.
- Celik, I. B. et al. (2008). Procedure for Estimation and Reporting
  of Uncertainty Due to Discretization in CFD Applications.
  *ASME J. Fluids Eng.*, 130(7):078001.

Author: Heinrich Vogel
        TU München, Lehrstuhl für Numerische Strömungsmechanik
Date:   2026-07-27
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from math import log as mlog
from pathlib import Path
from typing import Final

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from navier_stokes import Boundary2D, BoundaryCondition, Mesh2D, SIMPLESolver


# ═══════════════════════════════════════════════════════════════════════
# ─── Konstanten ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

RHO: Final[float] = 1.0                        # Dichte [kg/m³]
NU: Final[float] = 1.0e-3                      # Viskosität [m²/s]
REFINEMENT_FACTOR: Final[float] = 2.0 ** 0.5   # r = √2 ≈ 1.414
ANZAHL_GITTER: Final[int] = 5                  # Anzahl Gitterstufen
F_SICHERHEIT: Final[float] = 1.25              # Sicherheitsfaktor F_s
T_END_KONV: Final[float] = 5.0                 # Endzeit pro Gitter [s]
DT_KONV: Final[float] = 0.01                   # Zeitschritt [s]

# Basisgitter: gröbstes Gitter
NX_BASIS: Final[int] = 8
NY_BASIS: Final[int] = 8


# ═══════════════════════════════════════════════════════════════════════
# ─── Dataclasses ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Gitterebene:
    """Eine einzelne Gitterebene der Verfeinerungsstudie.

    Attributes
    ----------
    stufe : int
        Level index (1 = finest, ANZAHL_GITTER = coarsest).
    nx : int
        Number of cells in x.
    ny : int
        Number of cells in y.
    gitterweite : float
        Characteristic cell size h = sqrt(1/(nx·ny)).
    loesung : float
        Solution value (e.g. max u‑velocity at centreline).
    """
    stufe: int
    nx: int
    ny: int
    gitterweite: float
    loesung: float


@dataclass
class GCIErgebnis:
    """GCI‑Auswertung für ein Triplett von Gittern.

    Attributes
    ----------
    stufe_fein : int
        Level of the finest grid in the triplet.
    konvergenzordnung_p : float
        Apparent convergence order p.
    extrapolierter_wert : float
        Richardson extrapolation f_{h=0}.
    GCI_fein : float
        GCI on the fine grid (fractional, not %).
    GCI_grob : float
        GCI on the coarse grid (fractional).
    asymptotischer_index : float
        GCI_23 / (r^p · GCI_12).  ≈ 1.0 confirms asymptotic range.
    """
    stufe_fein: int
    konvergenzordnung_p: float
    extrapolierter_wert: float
    GCI_fein: float
    GCI_grob: float
    asymptotischer_index: float


@dataclass
class KonvergenzStudie:
    """Complete mesh convergence study results.

    Attributes
    ----------
    gitterebenen : list[Gitterebene]
        Solutions on each grid level.
    gci_ergebnisse : list[GCIErgebnis]
        GCI results for each consecutive triplet.
    intervall_unter : float
        Lower bound of 95 % confidence interval.
    intervall_ober : float
        Upper bound of 95 % confidence interval.
    im_asymptotischen_bereich : bool
        Whether asymptotic range is confirmed.
    """
    gitterebenen: list[Gitterebene] = field(default_factory=list)
    gci_ergebnisse: list[GCIErgebnis] = field(default_factory=list)
    intervall_unter: float = 0.0
    intervall_ober: float = 0.0
    im_asymptotischen_bereich: bool = False


# ═══════════════════════════════════════════════════════════════════════
# ─── Problem-definierende Funktion ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def _loese_auf_gitter(nx: int, ny: int, Re: float = 100.0,
                      t_end: float = T_END_KONV,
                      dt: float = DT_KONV) -> float:
    """Solve a reference problem on a given grid.

    Returns a scalar quantity of interest (here: peak u‑velocity
    along the cavity vertical centreline at x = 0.5).  This quantity
    is tracked across grid levels for the GCI study.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Re : float
        Reynoldszahl.
    t_end : float
        End time.
    dt : float
        Time step.

    Returns
    -------
    float
        Peak u‑velocity on centreline (quantity of interest).
    """
    nu: float = 1.0 / Re
    mesh: Mesh2D = Mesh2D(nx, ny, lx=1.0, ly=1.0)

    bc: Boundary2D = Boundary2D(
        west=(BoundaryCondition.WALL, 0.0),
        east=(BoundaryCondition.WALL, 0.0),
        south=(BoundaryCondition.WALL, 0.0),
        north=(BoundaryCondition.WALL, 1.0),
    )

    solver: SIMPLESolver = SIMPLESolver(mesh, nu=nu, rho=RHO, dt=dt, bc=bc)

    n_steps: int = int(t_end / dt)
    for _step in range(n_steps):
        solver.step(n_inner=15)

    # Quantity of interest: max u at centreline (x = 0.5)
    i_mitte: int = nx // 2
    u_zentrum: np.ndarray = solver.u.data[i_mitte, :]
    return float(np.max(np.abs(u_zentrum)))


# ═══════════════════════════════════════════════════════════════════════
# ─── Gittererzeugung ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def erzeuge_gitterebenen(
    anzahl: int = ANZAHL_GITTER,
    nx_basis: int = NX_BASIS,
    ny_basis: int = NY_BASIS,
    Re: float = 100.0,
) -> list[Gitterebene]:
    """Generate and solve on all grid levels.

    Grid i (0 = coarsest) has  nx = nx_basis · r^{i}
    and ny = ny_basis · r^{i}, where r = REFINEMENT_FACTOR.

    The finest grid (stufe = 1) has the largest nx, ny.

    Parameters
    ----------
    anzahl : int
        Number of grid levels (default 5).
    nx_basis, ny_basis : int
        Coarsest grid resolution.
    Re : float
        Reynoldszahl.

    Returns
    -------
    list[Gitterebene]
        Solutions on each grid level, sorted finest → coarsest.
    """
    ebenen: list[Gitterebene] = []

    for stufe in range(anzahl):
        idx: int = anzahl - 1 - stufe  # 0 = finest, anzahl-1 = coarsest
        nx: int = int(round(nx_basis * (REFINEMENT_FACTOR ** idx)))
        ny: int = int(round(ny_basis * (REFINEMENT_FACTOR ** idx)))
        nx = max(nx, 4)
        ny = max(ny, 4)

        gitterweite: float = np.sqrt(1.0 / (nx * ny))
        loesung: float = _loese_auf_gitter(nx, ny, Re=Re)

        ebenen.append(Gitterebene(
            stufe=stufe + 1,
            nx=nx,
            ny=ny,
            gitterweite=gitterweite,
            loesung=loesung,
        ))

    return ebenen


# ═══════════════════════════════════════════════════════════════════════
# ─── GCI-Berechnung (Roache 1998) ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def berechne_gci(
    f1: float, f2: float, f3: float,
    r: float = REFINEMENT_FACTOR,
    Fs: float = F_SICHERHEIT,
) -> GCIErgebnis:
    """Compute GCI quantities for a grid triplet.

    Parameters
    ----------
    f1 : float
        Solution on fine grid.
    f2 : float
        Solution on medium grid.
    f3 : float
        Solution on coarse grid.
    r : float
        Refinement factor.
    Fs : float
        Safety factor (1.25 for 3‑grid studies).

    Returns
    -------
    GCIErgebnis
    """
    eps12: float = f2 - f1
    eps23: float = f3 - f2

    # Apparent order p
    if abs(eps12) < 1.0e-30 or abs(eps23) < 1.0e-30:
        return GCIErgebnis(
            stufe_fein=0,
            konvergenzordnung_p=0.0,
            extrapolierter_wert=f1,
            GCI_fein=0.0,
            GCI_grob=0.0,
            asymptotischer_index=-1.0,
        )

    try:
        p: float = mlog(abs(eps23 / eps12)) / mlog(r)
    except (ValueError, ZeroDivisionError):
        p = 0.0

    # Clamp p to plausible range
    p = max(0.1, min(p, 6.0))

    # Richardson extrapolation
    rp: float = r ** p
    f_exact: float = f1 + (f1 - f2) / (rp - 1.0)

    # GCI
    GCI_fine: float = Fs * abs((f1 - f2) / f1) / (rp - 1.0) if abs(rp - 1.0) > 1.0e-15 else 0.0
    GCI_coarse: float = Fs * abs((f2 - f3) / f2) / (rp - 1.0) if abs(rp - 1.0) > 1.0e-15 else 0.0

    # Asymptotic range index
    asym_index: float = GCI_coarse / (rp * GCI_fine) if GCI_fine > 1.0e-15 else -1.0

    return GCIErgebnis(
        stufe_fein=1,
        konvergenzordnung_p=p,
        extrapolierter_wert=f_exact,
        GCI_fein=GCI_fine,
        GCI_grob=GCI_coarse,
        asymptotischer_index=asym_index,
    )


# ═══════════════════════════════════════════════════════════════════════
# ─── Hauptstudie ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def fuehre_Gitterkonvergenzstudie_durch(
    Re: float = 100.0,
    anzahl_gitter: int = ANZAHL_GITTER,
) -> KonvergenzStudie:
    """Run a complete mesh convergence study.

    Parameters
    ----------
    Re : float
        Reynoldszahl for the cavity flow.
    anzahl_gitter : int
        Number of grid levels (default 5).

    Returns
    -------
    KonvergenzStudie
        Full results with GCI, asymptotic range check, and
        95 % Konfidenzintervall.
    """
    ebenen: list[Gitterebene] = erzeuge_gitterebenen(
        anzahl=anzahl_gitter, Re=Re,
    )

    gci_liste: list[GCIErgebnis] = []
    for i in range(len(ebenen) - 2):
        gci = berechne_gci(
            f1=ebenen[i].loesung,
            f2=ebenen[i + 1].loesung,
            f3=ebenen[i + 2].loesung,
        )
        gci.stufe_fein = ebenen[i].stufe
        gci_liste.append(gci)

    # Best estimate from finest‑grid triplet
    finest_gci: GCIErgebnis = gci_liste[0] if gci_liste else GCIErgebnis(
        stufe_fein=0, konvergenzordnung_p=0.0,
        extrapolierter_wert=ebenen[0].loesung if ebenen else 0.0,
        GCI_fein=0.0, GCI_grob=0.0, asymptotischer_index=-1.0,
    )

    # 95 % Konfidenzintervall (approximate)
    f_best: float = finest_gci.extrapolierter_wert
    GCI_best: float = finest_gci.GCI_fein
    intervall_unter: float = f_best * (1.0 - 2.0 * GCI_best)
    intervall_ober: float = f_best * (1.0 + 2.0 * GCI_best)

    # Asymptotic range detection
    im_asymptotischen: bool = False
    for gci in gci_liste:
        if 0.8 <= gci.asymptotischer_index <= 1.2:
            im_asymptotischen = True
            break

    return KonvergenzStudie(
        gitterebenen=ebenen,
        gci_ergebnisse=gci_liste,
        intervall_unter=intervall_unter,
        intervall_ober=intervall_ober,
        im_asymptotischen_bereich=im_asymptotischen,
    )


# ═══════════════════════════════════════════════════════════════════════
# ─── Bericht ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def drucke_GCI_Bericht(studie: KonvergenzStudie) -> str:
    """Format the GCI study as a printable report.

    Parameters
    ----------
    studie : KonvergenzStudie
        Completed convergence study.

    Returns
    -------
    str
        Formatted report.
    """
    linie: str = "─" * 72
    bericht: str = f"\n{linie}\n"
    bericht += "  Gitterkonvergenzstudie (GCI — Roache 1998)\n"
    bericht += f"{linie}\n"
    bericht += (
        f"  {'Stufe':>5s}  {'nx':>4s}  {'ny':>4s}  "
        f"{'h':>10s}  {'Lösung':>12s}\n"
    )
    bericht += f"{linie}\n"

    for ebene in studie.gitterebenen:
        bericht += (
            f"  {ebene.stufe:5d}  {ebene.nx:4d}  {ebene.ny:4d}  "
            f"{ebene.gitterweite:10.3e}  {ebene.loesung:12.6e}\n"
        )

    bericht += f"{linie}\n"
    bericht += "  GCI‑Auswertung (Tripletts):\n"
    bericht += f"{linie}\n"

    for gci in studie.gci_ergebnisse:
        bericht += (
            f"  Stufe {gci.stufe_fein} (fein):  p = {gci.konvergenzordnung_p:.4f}"
            f"  f_extrap = {gci.extrapolierter_wert:.6e}\n"
            f"    GCI_fine = {gci.GCI_fein * 100.0:.3f}%  "
            f"GCI_coarse = {gci.GCI_grob * 100.0:.3f}%  "
            f"Asymp. = {gci.asymptotischer_index:.4f}\n"
        )

    bericht += f"{linie}\n"
    bericht += (
        f"  95 % Konfidenzintervall (aus feinstem Triplett):\n"
        f"    [{studie.intervall_unter:.6e}, {studie.intervall_ober:.6e}]\n"
    )

    if studie.im_asymptotischen_bereich:
        bericht += "  ✓ Asymptotischer Bereich erreicht (0.8 ≤ GCI_23/(r^p·GCI_12) ≤ 1.2)\n"
    else:
        bericht += "  ⚠ Asymptotischer Bereich NOCH NICHT erreicht\n"

    bericht += f"{linie}\n"
    return bericht


# ═══════════════════════════════════════════════════════════════════════
# ─── Hauptprogramm ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    """Run the mesh convergence study and print the GCI report."""
    print("=" * 72)
    print("  Systematische Gitterkonvergenzstudie")
    print("  TU München, Lehrstuhl für Numerische Strömungsmechanik")
    print("=" * 72)
    print(f"  Refinement factor r = {REFINEMENT_FACTOR:.4f}")
    print(f"  Grid levels         = {ANZAHL_GITTER}")
    print(f"  ν                   = {NU:.1e}  m²/s")
    print(f"  ρ                   = {RHO:.1f}  kg/m³")
    print(f"  T_end               = {T_END_KONV:.1f}  s")
    print()

    studie: KonvergenzStudie = fuehre_Gitterkonvergenzstudie_durch(Re=100.0)
    bericht: str = drucke_GCI_Bericht(studie)
    print(bericht)

    # Acceptance criterion: asymptotic range confirmed or GCI < 5 %
    finest_gci: float = (
        studie.gci_ergebnisse[0].GCI_fein if studie.gci_ergebnisse else 1.0
    )
    bestanden: bool = (
        studie.im_asymptotischen_bereich or finest_gci < 0.05
    )

    if bestanden:
        print("  ✓ BESTANDEN — Gitterkonvergenz nachgewiesen")
    else:
        print("  ✗ NICHT BESTANDEN — Gitterkonvergenz unzureichend")
    print()

    return 0 if bestanden else 1


if __name__ == "__main__":
    sys.exit(main())
