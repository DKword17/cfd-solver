#!/usr/bin/env python3
"""
turbulence/les_smagorinsky.py
=============================

Modèle de Smagorinsky pour la simulation des grandes échelles (LES).

Viscosité de sous-maille :
    ν_sgs = (C_s · Δ)² · |S|

    |S| = √(2 · S_ij · S_ij)
    Δ   = (Δx · Δy)^{1/2}   (filtre isotrope en 2D)

Contrainte de sous-maille (modèle de Boussinesq) :
    τ_ij = -2 · ν_sgs · S_ij + (1/3) · τ_kk · δ_ij

Le coefficient de Smagorinsky C_s varie selon l'écoulement :
    - Cisaillement homogène : C_s ≈ 0.10
    - Turbulence isotrope   : C_s ≈ 0.20
    - Canaux / couches limites : C_s ≈ 0.12–0.15

Références :
    - Smagorinsky, J. (1963). General circulation experiments with the
      primitive equations. Monthly Weather Review, 91(3):99–164.
    - Lilly, D. K. (1966). On the application of the eddy viscosity
      concept in the inertial sub-range of turbulence. NCAR Manuscript.
    - Pope, S. B. (2000). Turbulent Flows. Cambridge Univ. Press.

Auteur: Pierre Dubois
        ISAE-SUPAERO / ONERA
Date:   2026-07-27
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from navier_stokes import Field2D, Mesh2D

# Coefficient de Smagorinsky par défaut (écoulements cisaillés)
C_s_DÉFAUT: float = 0.12


class ModeleSmagorinsky:
    """
    Modèle de Smagorinsky pour la viscosité de sous-maille.

    Calcule ν_sgs = (C_s · Δ)² · |S| à chaque pas de temps à partir
    du champ de vitesse résolu.

    Paramètres:
        maillage:             Grille de calcul 2D.
        C_s:                  Coefficient de Smagorinsky.
                              ~0.10 cisailllement, ~0.20 turbulence isotrope.
        lissage_temporel:     Facteur de relaxation temporelle (0 = aucun).
                              Utile pour stabiliser ν_sgs quand |S| fluctue.
    """

    def __init__(
        self,
        maillage: Mesh2D,
        C_s: float = C_s_DÉFAUT,
        lissage_temporel: float = 0.0,
    ) -> None:
        self.maillage = maillage
        self.C_s = C_s
        self.α = lissage_temporel

        nx, ny = maillage.nx, maillage.ny

        # Échelle de filtre Δ = (Δx · Δy)^{1/2}
        self.Δ = (maillage.dx * maillage.dy) ** 0.5

        # Viscosité de sous-maille ν_sgs
        self.ν_sgs = Field2D(nx, ny)

    def calcule(self, u: np.ndarray, v: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Calcule la viscosité de sous-maille ν_sgs à partir du champ
        de vitesse résolu.

            ν_sgs = (C_s · Δ)² · |S|

        Args:
            u:  Champ de vitesse x [(nx, ny)]
            v:  Champ de vitesse y [(nx, ny)]
            dt: Pas de temps [s] (utilisé si lissage_temporel > 0)

        Retourne:
            ν_sgs: Viscosité de sous-maille [(nx, ny)]
        """
        nx, ny = self.maillage.nx, self.maillage.ny
        dx, dy = self.maillage.dx, self.maillage.dy

        # ── Tenseur des déformations résolu S_ij ──────────────────────
        du_dx = np.zeros((nx, ny))
        du_dy = np.zeros((nx, ny))
        dv_dx = np.zeros((nx, ny))
        dv_dy = np.zeros((nx, ny))

        du_dx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
        du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dy)
        dv_dx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dx)
        dv_dy[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dy)

        # |S| = √(2·S_ij·S_ij)
        # S_11 = du/dx, S_22 = dv/dy, S_12 = ½(du/dy + dv/dx)
        # 2·S_ij·S_ij = 2·S_11² + 2·S_22² + 4·S_12²
        S_mag = np.sqrt(
            2.0 * du_dx ** 2
            + 2.0 * dv_dy ** 2
            + (du_dy + dv_dx) ** 2
        )

        # ── Viscosité de sous-maille ──────────────────────────────────
        ν_sgs_nouveau = (self.C_s * self.Δ) ** 2 * S_mag

        # Lissage temporel optionnel : ν_sgs = (1-α)·ν_sgs + α·ν_sgs_nouveau
        if self.α > 0.0:
            # α est le facteur de relaxation : 0 = tout nouveau, 1 = gelé
            facteur = dt / (self.α + dt)
            self.ν_sgs.data = (1.0 - facteur) * self.ν_sgs.data + facteur * ν_sgs_nouveau
        else:
            self.ν_sgs.data = ν_sgs_nouveau

        # Conditions aux limites : ν_sgs = 0 aux parois
        self.ν_sgs.data[0, :] = 0.0
        self.ν_sgs.data[-1, :] = 0.0
        self.ν_sgs.data[:, 0] = 0.0
        self.ν_sgs.data[:, -1] = 0.0

        return self.ν_sgs.data

    @property
    def viscosité_sous_maille(self) -> np.ndarray:
        """ν_sgs — Viscosité de sous-maille [(nx, ny)]."""
        return self.ν_sgs.data

    def contrainte_sous_maille(self, u: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
        """
        Calcule le tenseur des contraintes de sous-maille τ_ij.

            τ_ij = -2 · ν_sgs · S_ij

        Retourne un dictionnaire avec les composantes :
            'xx', 'yy', 'xy' du tenseur τ_ij.
        """
        nx, ny = self.maillage.nx, self.maillage.ny
        dx, dy = self.maillage.dx, self.maillage.dy

        du_dx = np.zeros((nx, ny))
        du_dy = np.zeros((nx, ny))
        dv_dx = np.zeros((nx, ny))
        dv_dy = np.zeros((nx, ny))

        du_dx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
        du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dy)
        dv_dx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dx)
        dv_dy[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dy)

        ν_sgs = self.ν_sgs.data

        return {
            'xx': -2.0 * ν_sgs * du_dx,
            'yy': -2.0 * ν_sgs * dv_dy,
            'xy': -ν_sgs * (du_dy + dv_dx),
        }
