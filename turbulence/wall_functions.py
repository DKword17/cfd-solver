#!/usr/bin/env python3
"""
turbulence/wall_functions.py
============================

Fonctions de paroi pour les modèles de turbulence RANS.

Deux approches :

    1. Loi de paroi standard (loi logarithmique)
        u⁺ = y⁺                          pour y⁺ < y_0_plus
        u⁺ = 1/κ · ln(E · y⁺)           pour y⁺ ≥ y_0_plus

        avec y_0_plus = 11.81 (intersection sous-couche visqueuse / log)

    2. Loi de paroi scalable (CFX, Vieser et al. 2002)
        y⁺* = max(y⁺, y_0_plus)
        u⁺  = 1/κ · ln(E · y⁺*)

        Évite la détérioration de la solution quand y⁺ < 11.81
        sur des maillages grossiers.

Références :
    - Launder, B. E. & Spalding, D. B. (1974). The numerical computation
      of turbulent flows. CMAME, 3(2):269–289.
    - Vieser, W., Esch, T., & Menter, F. R. (2002). Heat transfer
      predictions using advanced two-equation turbulence models.
      CFX Technical Memorandum CFX-VAL10/0602.

Auteur: Pierre Dubois
        ISAE-SUPAERO / ONERA
Date:   2026-07-27
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from navier_stokes import BoundaryCondition, Mesh2D

# Constantes de von Kármán et de lissage de paroi
κ: float = 0.41
E_paroi: float = 9.0   # Constante de rugosité pour paroi lisse
y_0_plus: float = 11.81     # Intersection viscosité / loi log


def calcule_y_plus(
    u_tau: np.ndarray,
    distance_paroi: np.ndarray,
    ν: float,
) -> np.ndarray:
    """
    Calcule y⁺ = u_τ · d / ν.

    Args:
        u_tau:         Vitesse de frottement u_τ = √(τ_w/ρ) [m/s]
        distance_paroi: Distance à la paroi [m]
        ν:             Viscosité cinématique moléculaire [m²/s]

    Retourne:
        y⁺ : Nombre de Reynolds de paroi (sans dimension).
    """
    return u_tau * distance_paroi / (ν + 1e-30)


def vitesse_frottement(
    u_parallèle: np.ndarray,
    distance_paroi: np.ndarray,
    ν: float,
    méthode: Literal["standard", "scalable"] = "scalable",
) -> np.ndarray:
    """
    Calcule la vitesse de frottement u_τ à partir de la vitesse
    parallèle à la paroi au centre de la première cellule.

    Résout itérativement :
        u_τ = u_∥ / u⁺(y⁺)

    Args:
        u_parallèle:   Vitesse parallèle à la paroi [m/s]
        distance_paroi: Distance de la première cellule à la paroi [m]
        ν:             Viscosité moléculaire [m²/s]
        méthode:       'standard' ou 'scalable'

    Retourne:
        u_τ : Vitesse de frottement [m/s]
    """
    # Estimation initiale — hypothèse sous-couche visqueuse
    u_tau = np.sqrt(ν * np.abs(u_parallèle) / (distance_paroi + 1e-30) + 1e-30)

    # Résolution itérative de u_τ = u_∥ / u⁺(y⁺)
    for _ in range(10):
        y_plus = calcule_y_plus(u_tau, distance_paroi, ν)

        if méthode == "scalable":
            # y⁺* = max(y⁺, y_0_plus) — loi tout-log
            y_plus_eff = np.maximum(y_plus, y_0_plus)
            u_plus = 1.0 / κ * np.log(E_paroi * y_plus_eff)
        else:
            # Loi standard avec sous-couche visqueuse
            u_plus = np.where(
                y_plus < y_0_plus,
                y_plus,
                1.0 / κ * np.log(E_paroi * y_plus),
            )

        u_tau_nouveau = np.abs(u_parallèle) / (u_plus + 1e-30)

        # Sous-relaxation pour la convergence
        u_tau = 0.5 * u_tau + 0.5 * u_tau_nouveau

    return u_tau


def profil_u_plus(
    y_plus: np.ndarray | float,
    méthode: Literal["standard", "scalable"] = "standard",
) -> np.ndarray | float:
    """
    Profil de vitesse pariétal u⁺(y⁺).

    Standard :
        u⁺ = y⁺              si y⁺ < y_0_plus
        u⁺ = 1/κ · ln(E·y⁺)  si y⁺ ≥ y_0_plus

    Scalable :
        u⁺ = 1/κ · ln(E · max(y⁺, y_0_plus))

    Args:
        y_plus:  Distance adimensionnelle à la paroi
        méthode: 'standard' ou 'scalable'

    Retourne:
        u_plus : Vitesse adimensionnelle.
    """
    if méthode == "scalable":
        y_plus_eff = np.maximum(y_plus, y_0_plus)
        return 1.0 / κ * np.log(E_paroi * y_plus_eff)

    # Standard
    if isinstance(y_plus, np.ndarray):
        return np.where(
            y_plus < y_0_plus,
            y_plus,
            1.0 / κ * np.log(E_paroi * y_plus),
        )
    return y_plus if y_plus < y_0_plus else 1.0 / κ * np.log(E_paroi * y_plus)


def contrainte_paroi(
    u_parallèle: np.ndarray,
    distance_paroi: np.ndarray,
    ν: float,
    ρ: float = 1.0,
    méthode: Literal["standard", "scalable"] = "scalable",
) -> np.ndarray:
    """
    Calcule la contrainte de cisaillement à la paroi τ_w.

        τ_w = ρ · u_τ²

    Utilise la vitesse de frottement u_τ calculée par
    résolution itérative de la loi logarithmique.

    Args:
        u_parallèle:   Vitesse parallèle à la paroi [m/s]
        distance_paroi: Distance à la paroi [m]
        ν:             Viscosité moléculaire [m²/s]
        ρ:             Masse volumique [kg/m³]
        méthode:       'standard' ou 'scalable'

    Retourne:
        τ_w : Contrainte pariétale [Pa]
    """
    u_tau = vitesse_frottement(u_parallèle, distance_paroi, ν, méthode)
    return ρ * u_tau ** 2


def coefficient_frottement(
    u_parallèle: np.ndarray,
    distance_paroi: np.ndarray,
    ν: float,
    ρ: float = 1.0,
    u_réf: float = 1.0,
    méthode: Literal["standard", "scalable"] = "scalable",
) -> np.ndarray:
    """
    Calcule le coefficient de frottement C_f = τ_w / (½·ρ·U²).

    Args:
        u_parallèle:   Vitesse parallèle à la paroi au centre [m/s]
        distance_paroi: Distance de la cellule à la paroi [m]
        ν:             Viscosité moléculaire [m²/s]
        ρ:             Masse volumique [kg/m³]
        u_réf:         Vitesse de référence de l'écoulement [m/s]
        méthode:       'standard' ou 'scalable'

    Retourne:
        C_f : Coefficient de frottement (sans dimension).
    """
    τ_w = contrainte_paroi(u_parallèle, distance_paroi, ν, ρ, méthode)
    return τ_w / (0.5 * ρ * u_réf ** 2 + 1e-30)


def distance_paroi_rectangulaire(maillage: Mesh2D) -> np.ndarray:
    """
    Calcule la distance de chaque cellule à la paroi la plus proche
    pour un domaine rectangulaire.

        d(i,j) = min(x_i, Lx - x_i, y_j, Ly - y_j)

    Args:
        maillage: Grille de calcul 2D.

    Retourne:
        d : Distance à la paroi [(nx, ny)].
    """
    xc = maillage.xc[:, np.newaxis]
    yc = maillage.yc[np.newaxis, :]
    Lx, Ly = maillage.lx, maillage.ly

    dx_p = np.minimum(xc, Lx - xc)
    dy_p = np.minimum(yc, Ly - yc)
    return np.minimum(dx_p, dy_p)


def estime_ν_t_paroi(
    y_plus: np.ndarray,
    ν: float,
    κ: float = 0.41,
) -> np.ndarray:
    """
    Estime la viscosité turbulente près de la paroi à partir de la
    loi de paroi.

    Pour la sous-couche logarithmique :
        ν_t ≈ κ · u_τ · d = κ · ν · y⁺

    Args:
        y_plus:  Distance adimensionnelle à la paroi
        ν:       Viscosité moléculaire [m²/s]

    Retourne:
        ν_t_paroi : Viscosité turbulente estimée [(nx, ny)].
    """
    ν_t = np.zeros_like(y_plus)
    mask = y_plus > y_0_plus
    ν_t[mask] = κ * ν * y_plus[mask]
    return ν_t
