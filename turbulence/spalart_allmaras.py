#!/usr/bin/env python3
"""
turbulence/spalart_allmaras.py
==============================

Modèle de Spalart-Allmaras — équation de transport à une équation
pour la viscosité tourbillonnaire.

    ∂ν̃/∂t + uⱼ ∂ν̃/∂xⱼ = C_b1 S̃ ν̃
                        - C_w1 f_w (ν̃/d)²
                        + 1/σ [∇·((ν + ν̃)∇ν̃) + C_b2 (∇ν̃)²]

    ν_t = ν̃ · f_v1,   f_v1 = χ³ / (χ³ + C_v1³),   χ = ν̃ / ν

Références:
    - Spalart, P. R. & Allmaras, S. R. (1994). A one-equation turbulence
      model for aerodynamic flows. La Recherche Aérospatiale, 1:5–21.
    - https://turbmodels.larc.nasa.gov/spalart.html

Auteur: Pierre Dubois
        ISAE-SUPAERO / ONERA
Date:   2026-07-27
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from navier_stokes import Field2D, Mesh2D


# ─── Constantes du modèle SA ───────────────────────────────────────────

C_b1: float = 0.1355
C_b2: float = 0.622
C_v1: float = 7.1
C_w2: float = 0.3
C_w3: float = 2.0
σ: float = 2.0 / 3.0
κ: float = 0.41       # Constante de von Kármán

# C_w1 = C_b1/κ² + (1 + C_b2)/σ
C_w1: float = C_b1 / κ ** 2 + (1.0 + C_b2) / σ


class ModeleSpalartAllmaras:
    """
    Modèle de Spalart-Allmaras pour la viscosité tourbillonnaire.

    Stocke le champ ν̃ (ViscositéTourbillonnaire) et calcule la
    viscosité turbulente ν_t = ν̃ · f_v1.

    Paramètres:
        solveur:   Référence au solveur Navier-Stokes (ou `None` si
                   utilisation autonome).
        distance_paroi: Champ des distances à la paroi [m].
                        Si None, une approximation est construite.
        C_b1, C_b2, C_v1, σ, κ: Constantes du modèle.
    """

    def __init__(
        self,
        maillage: Mesh2D,
        viscosité_moléculaire: float = 1e-5,
        distance_paroi: Optional[np.ndarray] = None,
    ) -> None:
        self.maillage = maillage
        self.ν = viscosité_moléculaire

        nx, ny = maillage.nx, maillage.ny

        # Champ de travail : ν̃ (nu-tilde) — viscosité tourbillonnaire modifiée
        self.ν̃ = Field2D(nx, ny)

        # Distance à la paroi [m] — champ scalaire (nx, ny)
        if distance_paroi is not None:
            self.d = distance_paroi
        else:
            # Approximation : distance à la paroi la plus proche
            self.d = self._estime_distance_paroi()

        # Viscosité turbulente résultante ν_t (champ effectif)
        self.ν_t = Field2D(nx, ny)

    def _estime_distance_paroi(self) -> np.ndarray:
        """
        Estime la distance minimale à une paroi pour chaque cellule.

            d_{i,j} = min( xc_i, Lx - xc_i, yc_j, Ly - yc_j )

        Pour un domaine rectangulaire avec parois sur les 4 côtés.
        """
        xc = self.maillage.xc[:, np.newaxis]   # (nx, 1)
        yc = self.maillage.yc[np.newaxis, :]   # (1, ny)
        Lx = self.maillage.lx
        Ly = self.maillage.ly

        dx_paroi = np.minimum(xc, Lx - xc)
        dy_paroi = np.minimum(yc, Ly - yc)
        return np.minimum(dx_paroi, dy_paroi)

    def calcule(self, u: np.ndarray, v: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Avance l'équation de Spalart-Allmaras d'un pas de temps.

        Retourne le champ ν_t = ν̃ · f_v1 mis à jour [(nx, ny)].

        Formule compacte (Euler explicite + Jacobi simplifié) :
            ν̃^{(n+1)} = ν̃^{(n)} + Δt · RHS(ν̃^{(n)})

        Args:
            u:  Champ de vitesse x [(nx, ny)]
            v:  Champ de vitesse y [(nx, ny)]
            dt: Pas de temps [s]

        Retourne:
            ν_t: Viscosité tourbillonnaire [(nx, ny)]
        """
        nx, ny = self.maillage.nx, self.maillage.ny
        dx, dy = self.maillage.dx, self.maillage.dy

        ν̃ = self.ν̃.data
        ν = self.ν
        d = self.d

        # ── Fonctions auxiliaires ─────────────────────────────────────

        # χ = ν̃ / ν
        χ = ν̃ / (ν + 1e-30)

        # f_v1 = χ³ / (χ³ + C_v1³) — fonction de blocage visqueux
        χ_cube = χ ** 3
        f_v1 = χ_cube / (χ_cube + C_v1 ** 3)

        # Module du taux de rotation S = |ω| = |∂v/∂x - ∂u/∂y|
        S = np.zeros((nx, ny))
        S[1:-1, 1:-1] = np.abs(
            (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dx)
            - (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dy)
        )

        # S̃ = S + ν̃/(κ²d²) · f_v2,   f_v2 = 1 - χ/(1 + χ·f_v1)
        rapport = ν̃ / ((κ * d) ** 2 + 1e-30)
        f_v2 = 1.0 - χ / (1.0 + χ * f_v1 + 1e-30)
        S̃ = S + rapport * f_v2

        # Production : P = C_b1 · S̃ · ν̃
        production = C_b1 * S̃ * ν̃

        # ── Terme de destruction pariétale ────────────────────────────
        # f_w = g · [(1 + C_w3⁶) / (g⁶ + C_w3⁶)]^{1/6}
        # g = r + C_w2 · (r⁶ - r),   r = ν̃ / (S̃ · κ² · d²)
        r = ν̃ / (S̃ * (κ * d) ** 2 + 1e-30)
        g = r + C_w2 * (r ** 6 - r)
        g_6 = g ** 6
        f_w = g * ((1.0 + C_w3 ** 6) / (g_6 + C_w3 ** 6 + 1e-30)) ** (1.0 / 6.0)

        destruction = C_w1 * f_w * (ν̃ / (d + 1e-30)) ** 2

        # ── Terme de diffusion ────────────────────────────────────────
        # ∇·[(ν + ν̃)∇ν̃] + C_b2 (∇ν̃)²
        # Discrétisation centrée d'ordre 2
        ν_eff = ν + ν̃

        dν̃_dx = np.zeros((nx, ny))
        dν̃_dy = np.zeros((nx, ny))
        d2ν̃_dx2 = np.zeros((nx, ny))
        d2ν̃_dy2 = np.zeros((nx, ny))

        dν̃_dx[1:-1, :] = (ν̃[2:, :] - ν̃[:-2, :]) / (2 * dx)
        dν̃_dy[:, 1:-1] = (ν̃[:, 2:] - ν̃[:, :-2]) / (2 * dy)
        d2ν̃_dx2[1:-1, :] = (ν̃[2:, :] - 2 * ν̃[1:-1, :] + ν̃[:-2, :]) / (dx ** 2)
        d2ν̃_dy2[:, 1:-1] = (ν̃[:, 2:] - 2 * ν̃[:, 1:-1] + ν̃[:, :-2]) / (dy ** 2)

        diffusion = (
            ν_eff * (d2ν̃_dx2 + d2ν̃_dy2)
            + (dν̃_dx ** 2 + dν̃_dy ** 2) * (ν + C_b2 * ν̃) / ν_eff
        ) / σ

        # ── Convection (amont d'ordre 1) ──────────────────────────────
        # uⱼ ∂ν̃/∂xⱼ — schéma upwind
        convection = np.zeros((nx, ny))
        # Flux en x
        mask_u_pos = u >= 0
        mask_u_neg = ~mask_u_pos
        # i=1..nx-2
        convection[1:-1, :] += np.where(
            mask_u_pos[1:-1, :],
            u[1:-1, :] * (ν̃[1:-1, :] - ν̃[:-2, :]) / dx,
            u[1:-1, :] * (ν̃[2:, :] - ν̃[1:-1, :]) / dx,
        )
        # Flux en y
        mask_v_pos = v >= 0
        mask_v_neg = ~mask_v_pos
        convection[:, 1:-1] += np.where(
            mask_v_pos[:, 1:-1],
            v[:, 1:-1] * (ν̃[:, 1:-1] - ν̃[:, :-2]) / dy,
            v[:, 1:-1] * (ν̃[:, 2:] - ν̃[:, 1:-1]) / dy,
        )

        # ── Mise à jour Euler explicite ──────────────────────────────
        rhs = production - destruction + diffusion - convection
        ν̃_nouveau = ν̃ + dt * rhs

        # Limitation : ν̃ ≥ 0
        ν̃_nouveau = np.maximum(ν̃_nouveau, 0.0)

        # ── Conditions aux limites (paroi : ν̃ = 0) ─────────────────
        ν̃_nouveau[0, :] = 0.0
        ν̃_nouveau[-1, :] = 0.0
        ν̃_nouveau[:, 0] = 0.0
        ν̃_nouveau[:, -1] = 0.0

        self.ν̃.data = ν̃_nouveau

        # ── Viscosité tourbillonnaire effective avec f_v1 ──────────────
        χ = ν̃_nouveau / (ν + 1e-30)
        f_v1 = χ ** 3 / (χ ** 3 + C_v1 ** 3 + 1e-30)
        self.ν_t.data = ν̃_nouveau * f_v1

        return self.ν_t.data

    @property
    def viscosité_tourbillonnaire(self) -> np.ndarray:
        """ν_t — Viscosité tourbillonnaire effective [(nx, ny)]."""
        return self.ν_t.data
