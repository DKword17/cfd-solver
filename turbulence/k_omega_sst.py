#!/usr/bin/env python3
"""
turbulence/k_omega_sst.py
=========================

Modèle k-ω SST (Shear Stress Transport) de Menter (1994).

Mélange la formulation k-ω près de la paroi avec la formulation
k-ε transformée dans le sillage et les écoulements libres.

    ν_t = a_1·k / max(a_1·ω, |S|·F_2)

    ∂k/∂t + uⱼ ∂k/∂xⱼ = P̃_k - β*·k·ω
                        + ∂/∂xⱼ [(ν + σ_k·ν_t) ∂k/∂xⱼ]

    ∂ω/∂t + uⱼ ∂ω/∂xⱼ = α·P_k/ν_t - β·ω²
                        + ∂/∂xⱼ [(ν + σ_ω·ν_t) ∂ω/∂xⱼ]
                        + 2(1 - F_1)·σ_ω₂/ω · ∂k/∂xⱼ · ∂ω/∂xⱼ

Fonctions de mélange :
    F_1 = tanh(Γ₁⁴),   F_2 = tanh(Γ_2²)

Références:
    - Menter, F. R. (1994). Two-equation eddy-viscosity turbulence
      models for engineering applications. AIAA Journal, 32(8):1598–1605.

Auteur: Pierre Dubois
        ISAE-SUPAERO / ONERA
Date:   2026-07-27
"""

from __future__ import annotations

import numpy as np

from navier_stokes import Field2D, Mesh2D


# ─── Constantes du modèle k-ω SST (Menter, 1994) ──────────────────────
# La notation (1) désigne les constantes du régime k-ω (interne),
# la notation (2) désigne celles du régime k-ε transformé (externe).

# Constantes internes (régime k-ω)
σ_k1: float = 0.85
σ_ω1: float = 0.5
β1: float = 0.0750
α1: float = 5.0 / 9.0

# Constantes externes (régime k-ε transformé)
σ_k2: float = 1.0
σ_ω2: float = 0.856
β2: float = 0.0828
α2: float = 0.44

# Constantes universelles
β_étoile: float = 0.09
κ: float = 0.41
a_1: float = 0.31


class ModeleKOmegaSST:
    """
    Modèle k-ω SST (Menter, 1994) pour la viscosité tourbillonnaire.

    Transporte l'énergie cinétique turbulente k et la fréquence
    turbulente ω, avec fonction de mélange F_1/F_2 pour la transition
    paroi ↔ sillage.

    Paramètres:
        maillage:             Grille de calcul 2D.
        viscosité_moléculaire: Viscosité laminaire ν [m²/s].
        distance_paroi:       Champ des distances à la paroi [m].
                              Si None, approximation rectangulaire.
    """

    def __init__(
        self,
        maillage: Mesh2D,
        viscosité_moléculaire: float = 1e-5,
        distance_paroi: np.ndarray | None = None,
    ) -> None:
        self.maillage = maillage
        self.ν = viscosité_moléculaire

        nx, ny = maillage.nx, maillage.ny

        # Distance à la paroi
        if distance_paroi is not None:
            self.d = distance_paroi
        else:
            self.d = self._estime_distance_paroi()

        # Énergie cinétique turbulente k [m²/s²]
        self.k = Field2D(nx, ny)

        # Fréquence turbulente ω [1/s]
        self.ω = Field2D(nx, ny)

        # Viscosité tourbillonnaire ν_t [m²/s]
        self.ν_t = Field2D(nx, ny)

        # Initialisation
        k_0 = 1e-8
        ω_0 = 5.0 * viscosité_moléculaire / (0.1 * min(maillage.lx, maillage.ly)) ** 2
        self.k.data[:, :] = k_0
        self.ω.data[:, :] = ω_0

    def _estime_distance_paroi(self) -> np.ndarray:
        """
        Distance minimale aux parois pour un domaine rectangulaire.
            d(i,j) = min(xc_i, Lx - xc_i, yc_j, Ly - yc_j)
        """
        xc = self.maillage.xc[:, np.newaxis]
        yc = self.maillage.yc[np.newaxis, :]
        Lx, Ly = self.maillage.lx, self.maillage.ly

        dx_p = np.minimum(xc, Lx - xc)
        dy_p = np.minimum(yc, Ly - yc)
        return np.minimum(dx_p, dy_p)

    def calcule(self, u: np.ndarray, v: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Avance les équations k-ω SST d'un pas de temps.

        Retourne ν_t = a_1·k / max(a_1·ω, |S|·F_2) [(nx, ny)].

        Args:
            u:  Champ de vitesse x [(nx, ny)]
            v:  Champ de vitesse y [(nx, ny)]
            dt: Pas de temps [s]

        Retourne:
            ν_t: Viscosité tourbillonnaire [(nx, ny)]
        """
        nx, ny = self.maillage.nx, self.maillage.ny
        dx, dy = self.maillage.dx, self.maillage.dy
        d = self.d

        k = self.k.data
        ω = self.ω.data
        ν = self.ν

        # ── Tenseur des déformations et vorticité ──────────────────────
        du_dx = np.zeros((nx, ny))
        du_dy = np.zeros((nx, ny))
        dv_dx = np.zeros((nx, ny))
        dv_dy = np.zeros((nx, ny))

        du_dx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
        du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dy)
        dv_dx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dx)
        dv_dy[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dy)

        # |S| = √(2·S_ij·S_ij)
        S_mag = np.sqrt(2.0 * du_dx ** 2 + (du_dy + dv_dx) ** 2 + 2.0 * dv_dy ** 2)

        # ── Production de k (limitée) ─────────────────────────────────
        ν_t = a_1 * k / (np.maximum(a_1 * ω, S_mag * 1e-15) + 1e-15)
        P_k = ν_t * S_mag ** 2
        # Limitation de Wilcox : P̃_k = min(P_k, 20·β*·k·ω)
        P_k_tilde = np.minimum(P_k, 20.0 * β_étoile * k * ω)

        # ── Fonctions de mélange F_1, F_2 ──────────────────────────────
        # CD_kω = max( 2·σ_ω₂/ω · ∇k·∇ω, 10⁻²⁰ )
        dk_dx = np.zeros((nx, ny))
        dk_dy = np.zeros((nx, ny))
        dω_dx = np.zeros((nx, ny))
        dω_dy = np.zeros((nx, ny))

        dk_dx[1:-1, :] = (k[2:, :] - k[:-2, :]) / (2 * dx)
        dk_dy[:, 1:-1] = (k[:, 2:] - k[:, :-2]) / (2 * dy)
        dω_dx[1:-1, :] = (ω[2:, :] - ω[:-2, :]) / (2 * dx)
        dω_dy[:, 1:-1] = (ω[:, 2:] - ω[:, :-2]) / (2 * dy)

        CD_kω = np.maximum(
            2.0 * σ_ω2 / (ω + 1e-15) * (dk_dx * dω_dx + dk_dy * dω_dy),
            1e-20,
        )

        # Γ₁ = min[max( √k/(β*·ω·d), 500ν/(ω·d²) ), 4·σ_ω₂·k / CD_kω·d²]
        sqrt_k = np.sqrt(k)
        arg1_a = sqrt_k / (β_étoile * ω * d + 1e-15)
        arg1_b = 500.0 * ν / (ω * d ** 2 + 1e-15)
        arg1_c = 4.0 * σ_ω2 * k / (CD_kω * d ** 2 + 1e-15)

        Γ_1 = np.minimum(np.maximum(arg1_a, arg1_b), arg1_c)
        F_1 = np.tanh(Γ_1 ** 4)

        # Γ_2 = max( 2·√k/(β*·ω·d), 500ν/(ω·d²) )
        Γ_2 = np.maximum(arg1_a, arg1_b)
        F_2 = np.tanh(Γ_2 ** 2)

        # ── Constantes interpolées par F_1 ─────────────────────────────
        σ_k = F_1 * σ_k1 + (1.0 - F_1) * σ_k2
        σ_ω = F_1 * σ_ω1 + (1.0 - F_1) * σ_ω2
        β = F_1 * β1 + (1.0 - F_1) * β2
        α = F_1 * α1 + (1.0 - F_1) * α2

        # ── Termes de diffusion ───────────────────────────────────────
        ν_eff_k = ν + σ_k * ν_t
        ν_eff_ω = ν + σ_ω * ν_t

        d2k_dx2 = np.zeros((nx, ny))
        d2k_dy2 = np.zeros((nx, ny))
        d2ω_dx2 = np.zeros((nx, ny))
        d2ω_dy2 = np.zeros((nx, ny))

        d2k_dx2[1:-1, :] = (k[2:, :] - 2 * k[1:-1, :] + k[:-2, :]) / (dx ** 2)
        d2k_dy2[:, 1:-1] = (k[:, 2:] - 2 * k[:, 1:-1] + k[:, :-2]) / (dy ** 2)
        d2ω_dx2[1:-1, :] = (ω[2:, :] - 2 * ω[1:-1, :] + ω[:-2, :]) / (dx ** 2)
        d2ω_dy2[:, 1:-1] = (ω[:, 2:] - 2 * ω[:, 1:-1] + ω[:, :-2]) / (dy ** 2)

        diffusion_k = ν_eff_k * (d2k_dx2 + d2k_dy2)
        diffusion_ω = ν_eff_ω * (d2ω_dx2 + d2ω_dy2)

        # ── Convection (upwind 1er ordre) ─────────────────────────────
        conv_k = np.zeros((nx, ny))
        conv_ω = np.zeros((nx, ny))

        for champ, conv in [(k, conv_k), (ω, conv_ω)]:
            conv[1:-1, :] += np.where(
                u[1:-1, :] >= 0,
                u[1:-1, :] * (champ[1:-1, :] - champ[:-2, :]) / dx,
                u[1:-1, :] * (champ[2:, :] - champ[1:-1, :]) / dx,
            )
            conv[:, 1:-1] += np.where(
                v[:, 1:-1] >= 0,
                v[:, 1:-1] * (champ[:, 1:-1] - champ[:, :-2]) / dy,
                v[:, 1:-1] * (champ[:, 2:] - champ[:, 1:-1]) / dy,
            )

        # ── Terme de diffusion croisée (cross-diffusion) ──────────────
        # 2(1 - F_1) · σ_ω₂/ω · ∇k·∇ω
        cross_diff = 2.0 * (1.0 - F_1) * σ_ω2 / (ω + 1e-15) * (
            dk_dx * dω_dx + dk_dy * dω_dy
        )

        # ── Mise à jour Euler explicite ──────────────────────────────
        # ∂k/∂t = P̃_k - β*·k·ω + diffusion_k - convection_k
        rhs_k = P_k_tilde - β_étoile * k * ω + diffusion_k - conv_k
        k_nouveau = k + dt * rhs_k

        # ∂ω/∂t = α·P_k/ν_t - β·ω² + diffusion_ω - convection_ω + cross_diff
        rhs_ω = (
            α * P_k / (ν_t + 1e-15)
            - β * ω ** 2
            + diffusion_ω
            - conv_ω
            + cross_diff
        )
        ω_nouveau = ω + dt * rhs_ω

        # Stabilité
        k_nouveau = np.maximum(k_nouveau, 1e-12)
        ω_nouveau = np.maximum(ω_nouveau, 1e-12)

        # ── Conditions aux limites ─────────────────────────────────────
        k_nouveau[0, :] = 0.0
        k_nouveau[-1, :] = 0.0
        k_nouveau[:, 0] = 0.0
        k_nouveau[:, -1] = 0.0

        # ω à la paroi : ω_wall = 60ν / (β₁·d²)  (Menter, 1994)
        ω_paroi = 60.0 * ν / (β1 * (d + 1e-15) ** 2)
        ω_nouveau[0, :] = np.maximum(ω_paroi[0, :], ω_nouveau[1, :])
        ω_nouveau[-1, :] = np.maximum(ω_paroi[-1, :], ω_nouveau[-2, :])
        ω_nouveau[:, 0] = np.maximum(ω_paroi[:, 0], ω_nouveau[:, 1])
        ω_nouveau[:, -1] = np.maximum(ω_paroi[:, -1], ω_nouveau[:, -2])

        self.k.data = k_nouveau
        self.ω.data = ω_nouveau

        # ── Viscosité tourbillonnaire SST ─────────────────────────────
        # ν_t = a_1·k / max(a_1·ω, |S|·F_2)
        self.ν_t.data = a_1 * k_nouveau / (
            np.maximum(a_1 * ω_nouveau, S_mag * F_2) + 1e-15
        )

        return self.ν_t.data

    @property
    def viscosité_tourbillonnaire(self) -> np.ndarray:
        """ν_t — Viscosité tourbillonnaire [(nx, ny)]."""
        return self.ν_t.data
