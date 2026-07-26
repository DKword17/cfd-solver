#!/usr/bin/env python3
"""
turbulence/k_epsilon.py
=======================

Modèle k-ε standard et variante RNG (Renormalization Group Theory)
pour la viscosité tourbillonnaire.

    ν_t = C_μ · k² / ε

    ∂k/∂t + uⱼ ∂k/∂xⱼ = P_k - ε + ∂/∂xⱼ [(ν + ν_t/σ_k) ∂k/∂xⱼ]

    ∂ε/∂t + uⱼ ∂ε/∂xⱼ = C_ε1 P_k ε/k - C_ε2 ε²/k
                        + ∂/∂xⱼ [(ν + ν_t/σ_ε) ∂ε/∂xⱼ]

Variante RNG (Yakhot et al., 1992) :
    C_ε1^{RNG} = 1.42 - η(1 - η/η₀) / (1 + βη³)
    η = S · k / ε

Références:
    - Launder, B. E. & Spalding, D. B. (1974). The numerical computation
      of turbulent flows. Computer Methods in Applied Mechanics and
      Engineering, 3(2):269–289.
    - Yakhot, V. et al. (1992). Development of turbulence models for shear
      flows by a double expansion technique. Physics of Fluids A, 4:1510.

Auteur: Pierre Dubois
        ISAE-SUPAERO / ONERA
Date:   2026-07-27
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from navier_stokes import Field2D, Mesh2D


# ─── Constantes du modèle k-ε standard ─────────────────────────────────

C_μ: float = 0.09
C_ε1: float = 1.44
C_ε2: float = 1.92
σ_k: float = 1.0
σ_ε: float = 1.3

# Constantes RNG
C_ε1_RNG: float = 1.42
η_0: float = 4.38
β_RNG: float = 0.012


class ModeleKEpsilon:
    """
    Modèle k-ε standard ou RNG pour la viscosité tourbillonnaire.

    Transporte l'énergie cinétique turbulente k et le taux de
    dissipation ε.

    Paramètres:
        maillage:             Grille de calcul 2D.
        viscosité_moléculaire: Viscosité laminaire ν [m²/s].
        variante:             'standard' ou 'RNG'.
        C_μ, C_ε1, C_ε2:     Constantes du modèle.
    """

    def __init__(
        self,
        maillage: Mesh2D,
        viscosité_moléculaire: float = 1e-5,
        variante: Literal["standard", "RNG"] = "standard",
    ) -> None:
        self.maillage = maillage
        self.ν = viscosité_moléculaire
        self.variante = variante

        nx, ny = maillage.nx, maillage.ny

        # Énergie cinétique turbulente k [m²/s²]
        self.k = Field2D(nx, ny)

        # Taux de dissipation ε [m²/s³]
        self.ε = Field2D(nx, ny)

        # Viscosité tourbillonnaire résultante ν_t [m²/s]
        self.ν_t = Field2D(nx, ny)

        # Initialisation : petite perturbation homogène
        k_0 = 1e-6 * viscosité_moléculaire ** 2
        ε_0 = C_μ * k_0 ** 1.5 / (0.1 * min(maillage.lx, maillage.ly))
        self.k.data[:, :] = k_0
        self.ε.data[:, :] = ε_0

    def calcule(self, u: np.ndarray, v: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Avance les équations k-ε d'un pas de temps.

        Retourne ν_t = C_μ · k² / ε [(nx, ny)].

        Args:
            u:  Champ de vitesse x [(nx, ny)]
            v:  Champ de vitesse y [(nx, ny)]
            dt: Pas de temps [s]

        Retourne:
            ν_t: Viscosité tourbillonnaire [(nx, ny)]
        """
        nx, ny = self.maillage.nx, self.maillage.ny
        dx, dy = self.maillage.dx, self.maillage.dy

        k = self.k.data
        ε = self.ε.data
        ν = self.ν
        ν_t = self.ν_t.data

        # ── Tenseur des déformations S_ij ─────────────────────────────
        # S_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i)
        # |S| = √(2·S_ij·S_ij)

        du_dx = np.zeros((nx, ny))
        du_dy = np.zeros((nx, ny))
        dv_dx = np.zeros((nx, ny))
        dv_dy = np.zeros((nx, ny))

        du_dx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
        du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dy)
        dv_dx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dx)
        dv_dy[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dy)

        # |S| = √(2·S_ij·S_ij) = √(2du/dx² + (du/dy+dv/dx)² + 2dv/dy²)
        S_mag = np.sqrt(
            2.0 * du_dx ** 2
            + (du_dy + dv_dx) ** 2
            + 2.0 * dv_dy ** 2
        )

        # ── Production de k ───────────────────────────────────────────
        # P_k = ν_t · |S|²
        ν_t = np.maximum(C_μ * k ** 2 / (ε + 1e-15), 0.0)
        P_k = ν_t * S_mag ** 2

        # ── Coefficient C_ε1 modifié pour RNG ─────────────────────────
        if self.variante == "RNG":
            η = S_mag * k / (ε + 1e-15)
            # C_ε1^{RNG} = C_ε1 - η(1 - η/η_0) / (1 + βη³)
            C_ε1_eff = C_ε1_RNG - η * (1.0 - η / η_0) / (1.0 + β_RNG * η ** 3 + 1e-15)
        else:
            C_ε1_eff = np.full((nx, ny), C_ε1)

        # ── Diffusion ─────────────────────────────────────────────────
        ν_eff_k = ν + ν_t / σ_k
        ν_eff_ε = ν + ν_t / σ_ε

        d2k_dx2 = np.zeros((nx, ny))
        d2k_dy2 = np.zeros((nx, ny))
        d2ε_dx2 = np.zeros((nx, ny))
        d2ε_dy2 = np.zeros((nx, ny))

        d2k_dx2[1:-1, :] = (k[2:, :] - 2 * k[1:-1, :] + k[:-2, :]) / (dx ** 2)
        d2k_dy2[:, 1:-1] = (k[:, 2:] - 2 * k[:, 1:-1] + k[:, :-2]) / (dy ** 2)
        d2ε_dx2[1:-1, :] = (ε[2:, :] - 2 * ε[1:-1, :] + ε[:-2, :]) / (dx ** 2)
        d2ε_dy2[:, 1:-1] = (ε[:, 2:] - 2 * ε[:, 1:-1] + ε[:, :-2]) / (dy ** 2)

        diffusion_k = ν_eff_k * (d2k_dx2 + d2k_dy2)
        diffusion_ε = ν_eff_ε * (d2ε_dx2 + d2ε_dy2)

        # ── Convection (upwind 1er ordre) ─────────────────────────────
        # u ∂k/∂x + v ∂k/∂y
        conv_k = np.zeros((nx, ny))
        conv_ε = np.zeros((nx, ny))

        for champ, conv in [(k, conv_k), (ε, conv_ε)]:
            # Flux x
            conv[1:-1, :] += np.where(
                u[1:-1, :] >= 0,
                u[1:-1, :] * (champ[1:-1, :] - champ[:-2, :]) / dx,
                u[1:-1, :] * (champ[2:, :] - champ[1:-1, :]) / dx,
            )
            # Flux y
            conv[:, 1:-1] += np.where(
                v[:, 1:-1] >= 0,
                v[:, 1:-1] * (champ[:, 1:-1] - champ[:, :-2]) / dy,
                v[:, 1:-1] * (champ[:, 2:] - champ[:, 1:-1]) / dy,
            )

        # ── Mise à jour Euler explicite ──────────────────────────────
        # ∂k/∂t = P_k - ε + diffusion_k - convection_k
        rhs_k = P_k - ε + diffusion_k - conv_k
        k_nouveau = k + dt * rhs_k

        # ∂ε/∂t = C_ε1·P_k·ε/k - C_ε2·ε²/k + diffusion_ε - convection_ε
        rhs_ε = (
            C_ε1_eff * P_k * ε / (k + 1e-15)
            - C_ε2 * ε ** 2 / (k + 1e-15)
            + diffusion_ε
            - conv_ε
        )
        ε_nouveau = ε + dt * rhs_ε

        # Stabilité : k, ε ≥ 0
        k_nouveau = np.maximum(k_nouveau, 1e-12)
        ε_nouveau = np.maximum(ε_nouveau, 1e-12)

        # ── Conditions aux limites (parois : k=0, ε extrapolé) ────────
        k_nouveau[0, :] = 0.0
        k_nouveau[-1, :] = 0.0
        k_nouveau[:, 0] = 0.0
        k_nouveau[:, -1] = 0.0

        ε_nouveau[0, :] = ε_nouveau[1, :]
        ε_nouveau[-1, :] = ε_nouveau[-2, :]
        ε_nouveau[:, 0] = ε_nouveau[:, 1]
        ε_nouveau[:, -1] = ε_nouveau[:, -2]

        self.k.data = k_nouveau
        self.ε.data = ε_nouveau

        # ── Viscosité tourbillonnaire ─────────────────────────────────
        self.ν_t.data = C_μ * k_nouveau ** 2 / (ε_nouveau + 1e-15)

        return self.ν_t.data

    @property
    def viscosité_tourbillonnaire(self) -> np.ndarray:
        """ν_t — Viscosité tourbillonnaire [(nx, ny)]."""
        return self.ν_t.data
