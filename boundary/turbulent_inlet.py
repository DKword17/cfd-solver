#!/usr/bin/env python3
"""
boundary/turbulent_inlet.py
=============================

Turbulent inflow generation methods for direct numerical simulation (DNS)
and large eddy simulation (LES) of turbulent flows.

Physical background:
    In wall-bounded turbulent flows, the inflow boundary condition must
    provide physically realistic turbulent fluctuations that are consistent
    with the targeted turbulent statistics. Simply prescribing a mean
    velocity profile without fluctuations leads to an unphysically long
    development length before a realistic turbulent state is attained. This
    module implements four widely used techniques for generating turbulent
    inflow conditions:

    1. Synthetic Eddy Method (SEM, Jarrin et al., 2006):
       Coherent eddies are superposed on the mean velocity profile using
       a prescribed shape function and Reynolds stress tensor. The method
       reproduces one-point statistics and has been extensively validated
       for boundary layer and channel flow simulations.

    2. Vortex Method for 2D (Sergent, 2002):
       Discrete vortices with prescribed circulation and length scale are
       introduced at the inlet plane. This method is particularly suitable
       for 2D simulations where spanwise vortices are the dominant
       coherent structures.

    3. Digital Filter Method (Klein et al., 2003):
       A random field is filtered using a digital filter with prescribed
       correlation function to generate inflow fluctuations with target
       spatial and temporal correlations.

    4. Prescribed Reynolds Stresses:
       Direct specification of the full Reynolds stress tensor at the
       inlet plane, from which fluctuating velocities are reconstructed
       using a Cholesky decomposition of the Reynolds stress tensor.

References:
    - Jarrin, N., Benhamadouche, S., Laurence, D., & Prosser, R. (2006).
      A synthetic-eddy-method for generating inflow conditions for
      large-eddy simulations. International Journal of Heat and Fluid
      Flow, 27(4):585-593.
    - Klein, M., Sadiki, A., & Janicka, J. (2003). A digital filter based
      generation of inflow data for spatially developing direct numerical
      simulation or large eddy simulation. Journal of Computational
      Physics, 186(2):652-665.
    - Sergent, M. E. (2002). Vers une methodologie de couplage entre la
      Simulation des Grandes Echelles et les modeles statistiques.
      PhD thesis, Ecole Centrale de Lyon.
    - Lund, T. S., Wu, X., & Squires, K. D. (1998). Generation of
      turbulent inflow data for spatially-developing boundary layer
      simulations. Journal of Computational Physics, 140(2):233-258.

Author:     Ananya Patel
            Indian Institute of Technology Bombay
Date:       2026-07-27
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from navier_stokes import Field2D, Mesh2D


# ─── Utility functions ─────────────────────────────────────────────────

def _cholesky_reynolds_stress(
    reynolds_stress: np.ndarray,
) -> np.ndarray:
    """Performs a Cholesky decomposition of the Reynolds stress tensor.

    Given a symmetric positive-definite Reynolds stress tensor R_ij, the
    Cholesky decomposition yields a lower-triangular matrix A_ij such that
    R = A · A^T. This decomposition is used to generate fluctuating
    velocities with the prescribed Reynolds stresses from uncorrelated
    random numbers.

    Please note that the Reynolds stress tensor must be symmetric and
    positive-definite for the Cholesky decomposition to succeed. If the
    matrix is not positive-definite, a small regularisation is applied.

    Args:
        reynolds_stress: A 3×3 symmetric positive-definite matrix
                         representing the Reynolds stress tensor.

    Returns:
        A 3×3 lower-triangular matrix A such that R = A · A^T.

    Raises:
        np.linalg.LinAlgError: If the Cholesky decomposition fails even
                               after regularisation.
    """
    # It is crucial to understand that the Reynolds stress tensor should
    # have the form:
    #   R = [[u'u',  u'v',  u'w'],
    #        [v'u',  v'v',  v'w'],
    #        [w'u',  w'v',  w'w']]
    try:
        # Attempt the Cholesky decomposition directly.
        A: np.ndarray = np.linalg.cholesky(reynolds_stress)
    except np.linalg.LinAlgError:
        # If the decomposition fails, add a small diagonal regularisation
        # to ensure positive-definiteness. This is a common safeguard in
        # turbulent inflow generation.
        epsilon: float = 1e-10
        regularised: np.ndarray = reynolds_stress + epsilon * np.eye(3)
        A = np.linalg.cholesky(regularised)

    return A


def _gaussian_shape_function(
    r: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Computes the Gaussian shape function used in the Synthetic Eddy
    Method.

    The shape function defines the spatial distribution of the velocity
    fluctuation induced by a single synthetic eddy. A Gaussian shape is
    commonly used because it is smooth, differentiable, and decays rapidly
    with distance from the eddy centre.

    Args:
        r:      The distance from the eddy centre (normalised by the
                eddy length scale). Can be a scalar or numpy array.
        sigma:  The standard deviation of the Gaussian (controls the
                width of the eddy). Defaults to 1.0.

    Returns:
        The shape function value(s) at the given normalised distance(s).
    """
    # Gaussian shape function: f(r) = exp(-r² / (2σ²))
    return np.exp(-0.5 * (r / sigma) ** 2)


# ─── Synthetic Eddy Method (SEM) ───────────────────────────────────────

@dataclass
class SyntheticEddyMethod:
    """Synthetic Eddy Method (SEM) for generating turbulent inflow
    fluctuations.

    Physical background:
        The Synthetic Eddy Method, originally proposed by Jarrin et al.
        (2006), generates a turbulent velocity field by superposing
        coherent eddies (synthetic eddies) onto a mean velocity profile.
        Each eddy is characterised by:
            - A centre location (randomly distributed within the inlet
              plane and a short upstream distance).
            - A length scale (related to the turbulent integral scale).
            - A strength (determined by the prescribed Reynolds stresses).
            - A shape function (Gaussian, tent, or step function).

        The velocity fluctuation at a point x is given by:
            u'_i(x) = (1/√N) · Σ_{k=1}^{N} a_{ij} · ε_j^{(k)}
                      · f_σ(x - x^{(k)})

        where N is the number of eddies, a_{ij} is the Cholesky
        decomposition of the Reynolds stress tensor, ε_j are random
        signs, and f_σ is the shape function.

    Args:
        mesh:               The computational mesh.
        mean_velocity:      The mean velocity profile as a 2D array
                            (nx × ny) or a scalar.
        reynolds_stress:    The Reynolds stress tensor at the inlet.
                            Must be a 3×3 array.
        n_eddies:           Number of synthetic eddies to superpose.
                            A larger number yields better statistics but
                            increases computational cost. Defaults to 200.
        length_scale:       The characteristic length scale of the eddies.
                            Typically a fraction of the boundary layer
                            thickness or channel half-height. Defaults to
                            0.1 (non-dimensional).
        shape_function:     The eddy shape function. Currently supports
                            'gaussian'. Defaults to 'gaussian'.
        seed:               Random seed for reproducibility. Defaults to
                            None (unseeded).
    """
    mesh: Mesh2D
    mean_velocity: float = 1.0
    reynolds_stress: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01, 0.01]))
    n_eddies: int = 200
    length_scale: float = 0.1
    shape_function: str = "gaussian"
    seed: Optional[int] = None

    # Internal state
    _rng: np.random.Generator = field(init=False, repr=False)
    _eddy_positions: np.ndarray = field(init=False, repr=False)
    _eddy_strengths: np.ndarray = field(init=False, repr=False)
    _cholesky_A: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialises the random number generator, performs the Cholesky
        decomposition of the Reynolds stress tensor, and generates the
        initial ensemble of synthetic eddies.

        Raises:
            ValueError: If the Reynolds stress tensor is not 3×3.
        """
        # ── Validate the Reynolds stress tensor dimensions ─────────────
        if self.reynolds_stress.shape != (3, 3):
            raise ValueError(
                "The Reynolds stress tensor must be a 3×3 array. "
                f"Received shape {self.reynolds_stress.shape}. "
                "Kindly ensure that you provide the full tensor "
                "R_ij = <u'_i u'_j>."
            )

        # ── Initialise the random number generator ─────────────────────
        self._rng = np.random.default_rng(self.seed)

        # ── Perform the Cholesky decomposition ─────────────────────────
        # This gives us the lower-triangular matrix A such that R = A·A^T.
        self._cholesky_A = _cholesky_reynolds_stress(self.reynolds_stress)

        # ── Generate the initial eddy distribution ─────────────────────
        self._initialise_eddies()

    def _initialise_eddies(self) -> None:
        """Generates the positions and strengths of the synthetic eddies.

        The eddies are randomly positioned in a virtual box that extends
        upstream of the inlet plane. The size of this box is determined
        by the length scale and the number of eddies.
        """
        # Determine the virtual box dimensions.
        # The box extends one length scale upstream and spans the entire
        # inlet plane. Please note that the streamwise extent of the box
        # should be at least one eddy length scale to avoid spurious
        # correlations.
        ny: int = self.mesh.ny
        ly: float = self.mesh.ly

        # Box dimensions: [x_min, x_max] × [y_min, y_max]
        x_min: float = -self.length_scale
        x_max: float = 0.0  # Inlet plane is at x = 0
        y_min: float = 0.0
        y_max: float = ly

        # Generate random eddy centre positions within the virtual box.
        self._eddy_positions = np.zeros((self.n_eddies, 2))
        self._eddy_positions[:, 0] = self._rng.uniform(x_min, x_max, self.n_eddies)
        self._eddy_positions[:, 1] = self._rng.uniform(y_min, y_max, self.n_eddies)

        # Generate random signs for the eddy intensities.
        # Each eddy has three independent random signs (for the three
        # velocity components), each being either +1 or -1 with equal
        # probability.
        self._eddy_strengths = self._rng.choice([-1, 1], size=(self.n_eddies, 3)).astype(np.float64)

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates the fluctuating velocity field at the inlet plane
        using the Synthetic Eddy Method.

        This method computes the velocity fluctuations (u', v', w') at
        the inlet plane by superposing the contributions of all synthetic
        eddies. The fluctuations are then added to the mean velocity
        profile to obtain the instantaneous inlet velocity.

        Returns:
            A tuple (u_in, v_in, w_in) containing the instantaneous
            velocity components at the inlet plane. Each array has shape
            (ny,) corresponding to the inlet plane for a west/east face,
            or (nx,) for a south/north face.

        Raises:
            RuntimeError: If the eddy positions have not been initialised.
        """
        ny: int = self.mesh.ny
        yc: np.ndarray = self.mesh.yc  # Cell centre y-coordinates

        # Initialise the fluctuation arrays.
        up: np.ndarray = np.zeros(ny, dtype=np.float64)  # u' fluctuation
        vp: np.ndarray = np.zeros(ny, dtype=np.float64)  # v' fluctuation
        wp: np.ndarray = np.zeros(ny, dtype=np.float64)  # w' fluctuation

        # ── Loop over all synthetic eddies ─────────────────────────────
        # For each eddy, we compute its contribution to the velocity
        # fluctuation at each cell centre on the inlet plane.
        for k in range(self.n_eddies):
            # Eddy centre coordinates.
            x_k: float = self._eddy_positions[k, 0]
            y_k: float = self._eddy_positions[k, 1]

            # Compute the normalised distance from each cell centre to
            # the eddy centre. Please note that the distance is computed
            # in the (x, y) plane, with x being the streamwise direction.
            dy_grid: np.ndarray = yc - y_k
            # For the inlet plane (x = 0), the streamwise offset is:
            dx_grid: float = 0.0 - x_k
            # Normalised distance vector.
            r_norm: np.ndarray = np.sqrt(dx_grid ** 2 + dy_grid ** 2) / self.length_scale

            # Compute the shape function value.
            if self.shape_function == "gaussian":
                f_val: np.ndarray = _gaussian_shape_function(r_norm)
            else:
                # Default to Gaussian if an unrecognised shape is given.
                f_val = _gaussian_shape_function(r_norm)

            # Eddy intensity: random signs scaled by the Cholesky factor.
            # The velocity fluctuation from this eddy at position y is:
            #   u'_i(y) = (1/√N) · A_ij · ε_j · f(y - y_k)
            epsilon: np.ndarray = self._eddy_strengths[k, :]  # shape (3,)

            # Compute the contribution to each component.
            # u' component: A_0j · ε_j
            contribution: np.ndarray = self._cholesky_A @ epsilon  # shape (3,)

            # Add the weighted contribution to the fluctuation fields.
            up += contribution[0] * f_val
            vp += contribution[1] * f_val
            wp += contribution[2] * f_val

        # ── Normalise by the square root of the number of eddies ───────
        # This ensures that the variance of the fluctuations is independent
        # of the number of eddies (for a sufficiently large N).
        inv_sqrt_n: float = 1.0 / np.sqrt(float(self.n_eddies))
        up *= inv_sqrt_n
        vp *= inv_sqrt_n
        wp *= inv_sqrt_n

        # ── Add the mean velocity to obtain the instantaneous inlet velocity ─
        u_in: np.ndarray = self.mean_velocity + up
        v_in: np.ndarray = vp  # Mean v is typically zero at the inlet.
        w_in: np.ndarray = wp  # Spanwise fluctuation (only relevant for 3D).

        return u_in, v_in, w_in

    def refresh_eddies(self, fraction: float = 0.1) -> None:
        """Replaces a fraction of the synthetic eddies with new ones to
        maintain statistical stationarity.

        In long-duration simulations, the eddies that have been advected
        downstream should be replaced by new eddies entering the virtual
        box. This method randomly removes a fraction of the existing
        eddies and creates new ones to replace them.

        Please note that refreshing a small fraction (5-10%) at each time
        step is recommended to maintain a statistically steady inflow.

        Args:
            fraction: The fraction of eddies to replace (0.0 to 1.0).
                      Defaults to 0.1 (10%).
        """
        n_replace: int = max(1, int(fraction * self.n_eddies))
        indices: np.ndarray = self._rng.choice(
            self.n_eddies, size=n_replace, replace=False
        )

        # Generate new positions for the replaced eddies.
        ny: int = self.mesh.ny
        ly: float = self.mesh.ly

        self._eddy_positions[indices, 0] = self._rng.uniform(
            -self.length_scale, 0.0, n_replace
        )
        self._eddy_positions[indices, 1] = self._rng.uniform(0.0, ly, n_replace)

        # Generate new random signs for the replaced eddies.
        self._eddy_strengths[indices, :] = self._rng.choice(
            [-1, 1], size=(n_replace, 3)
        ).astype(np.float64)


# ─── Vortex Method for 2D ──────────────────────────────────────────────

@dataclass
class VortexMethod2D:
    """Vortex method for generating 2D turbulent inflow conditions.

    Physical background:
        The vortex method generates inflow fluctuations by introducing
        discrete vortices at the inlet plane. Each vortex is characterised
        by its circulation (Γ), core radius (σ), and centre position. The
        velocity fluctuation induced by a single 2D vortex at a point
        (x, y) is given by the Biot-Savart law:

            u'_θ(r) = (Γ / 2πr) · [1 - exp(-r² / 2σ²)]

        for a Gaussian vortex core. This corresponds to the velocity field
        of a Lamb-Oseen vortex, which is an exact solution of the Navier-
        Stokes equations for a decaying vortex.

        The total fluctuation is obtained by summing the contributions
        of all vortices. The vortex circulations are scaled to match the
        prescribed turbulence intensity.

    References:
        - Sergent, M. E. (2002). PhD thesis, Ecole Centrale de Lyon.
        - Mathey, F., Cokljat, D., Bertoglio, J. P., & Sergent, E.
          (2006). Assessment of the vortex method for LES of complex
          flows. Engineering Turbulence Modelling and Experiments 6.

    Args:
        mesh:                The computational mesh.
        mean_velocity:       The mean streamwise velocity at the inlet.
        turbulence_intensity: The turbulence intensity (I = u_rms / U_mean).
                              Defaults to 0.05 (5%).
        n_vortices:          The number of discrete vortices. Defaults to 100.
        core_radius:         The core radius of each vortex. Defaults to 0.05.
        integral_length:     The integral length scale of the turbulence.
                              Defaults to 0.1.
        seed:                Random seed for reproducibility. Defaults to None.
    """
    mesh: Mesh2D
    mean_velocity: float = 1.0
    turbulence_intensity: float = 0.05
    n_vortices: int = 100
    core_radius: float = 0.05
    integral_length: float = 0.1
    seed: Optional[int] = None

    # Internal state
    _rng: np.random.Generator = field(init=False, repr=False)
    _vortex_positions: np.ndarray = field(init=False, repr=False)
    _vortex_circulations: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialises the random number generator and the vortex ensemble.
        """
        self._rng = np.random.default_rng(self.seed)
        self._initialise_vortices()

    def _initialise_vortices(self) -> None:
        """Initialises the positions and circulations of the discrete
        vortices.

        The vortices are randomly positioned within a virtual box that
        extends one integral length scale upstream of the inlet. The
        circulations are randomly assigned with alternating signs to
        ensure that the net circulation is approximately zero, which
        is physically consistent with homogeneous turbulence.
        """
        ly: float = self.mesh.ly

        # Vortex positions: randomly distributed in a virtual box.
        self._vortex_positions = np.zeros((self.n_vortices, 2))
        self._vortex_positions[:, 0] = self._rng.uniform(
            -self.integral_length, 0.0, self.n_vortices
        )
        self._vortex_positions[:, 1] = self._rng.uniform(0.0, ly, self.n_vortices)

        # Vortex circulations: random signs scaled by the turbulence
        # intensity and integral length scale. The circulation of each
        # vortex is Γ ≈ ± 2π · u_rms · l_int.
        u_rms: float = self.turbulence_intensity * self.mean_velocity
        circulation_magnitude: float = 2.0 * np.pi * u_rms * self.integral_length

        self._vortex_circulations = self._rng.choice(
            [-1, 1], self.n_vortices
        ).astype(np.float64) * circulation_magnitude

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Generates the 2D fluctuating velocity field at the inlet plane
        using the vortex method.

        The velocity at each point on the inlet plane is computed as the
        sum of the mean velocity and the contributions of all discrete
        vortices.

        Returns:
            A tuple (u_in, v_in) containing the streamwise and wall-normal
            velocity components at the inlet plane. Each array has shape
            (ny,) for a west/east face inlet.
        """
        ny: int = self.mesh.ny
        yc: np.ndarray = self.mesh.yc

        # Initialise the fluctuation arrays.
        up: np.ndarray = np.zeros(ny, dtype=np.float64)
        vp: np.ndarray = np.zeros(ny, dtype=np.float64)

        # ── Loop over all discrete vortices ────────────────────────────
        for k in range(self.n_vortices):
            x_k: float = self._vortex_positions[k, 0]
            y_k: float = self._vortex_positions[k, 1]
            Gamma_k: float = self._vortex_circulations[k]

            # Compute the distance from the inlet plane point (0, y)
            # to the vortex centre (x_k, y_k).
            dx: np.ndarray = 0.0 - x_k  # Inlet plane is at x = 0
            dy: np.ndarray = yc - y_k
            r_sq: np.ndarray = dx ** 2 + dy ** 2
            r: np.ndarray = np.sqrt(r_sq)

            # ── Lamb-Oseen vortex velocity profile ─────────────────
            # The tangential velocity is given by:
            #   u_θ(r) = (Γ / 2πr) · [1 - exp(-r² / 2σ²)]
            # where σ is the core radius.
            # Please note that we regularise the denominator to avoid
            # division by zero at the vortex centre.
            mask: np.ndarray = r > 1e-12
            u_theta: np.ndarray = np.zeros_like(r)

            if np.any(mask):
                u_theta[mask] = (
                    Gamma_k / (2.0 * np.pi * r[mask])
                    * (1.0 - np.exp(-r_sq[mask] / (2.0 * self.core_radius ** 2)))
                )

            # Decompose the tangential velocity into Cartesian components.
            # At the inlet plane (x = 0), the unit tangential vector is:
            #   e_θ = (-dy/r, dx/r)  (counter-clockwise)
            u_contrib: np.ndarray = np.zeros(ny)
            v_contrib: np.ndarray = np.zeros(ny)

            if np.any(mask):
                # u component: u_θ · (-dy / r)
                u_contrib[mask] = -u_theta[mask] * dy[mask] / r[mask]
                # v component: u_θ · (dx / r)
                v_contrib[mask] = u_theta[mask] * dx[mask] / r[mask]

            # Add the contribution of this vortex to the fluctuation field.
            up += u_contrib
            vp += v_contrib

        # ── Combine with the mean velocity ─────────────────────────────
        # It is crucial to understand that the vortex method may not
        # exactly reproduce the target turbulence intensity in a single
        # realisation; statistical convergence requires ensemble averaging.
        u_in: np.ndarray = self.mean_velocity + up
        v_in: np.ndarray = vp

        return u_in, v_in

    def refresh_vortices(self, fraction: float = 0.1) -> None:
        """Replaces a fraction of the vortices with new ones to maintain
        statistical stationarity.

        Args:
            fraction: The fraction of vortices to replace (0.0 to 1.0).
                      Defaults to 0.1.
        """
        n_replace: int = max(1, int(fraction * self.n_vortices))
        indices: np.ndarray = self._rng.choice(
            self.n_vortices, size=n_replace, replace=False
        )

        ly: float = self.mesh.ly
        u_rms: float = self.turbulence_intensity * self.mean_velocity
        circulation_magnitude: float = 2.0 * np.pi * u_rms * self.integral_length

        self._vortex_positions[indices, 0] = self._rng.uniform(
            -self.integral_length, 0.0, n_replace
        )
        self._vortex_positions[indices, 1] = self._rng.uniform(0.0, ly, n_replace)
        self._vortex_circulations[indices] = (
            self._rng.choice([-1, 1], n_replace) * circulation_magnitude
        )


# ─── Digital Filter Method ─────────────────────────────────────────────

@dataclass
class DigitalFilterMethod:
    """Digital filter method for generating turbulent inflow fluctuations
    with prescribed spatial and temporal correlations.

    Physical background:
        The digital filter method, proposed by Klein et al. (2003),
        generates a turbulent velocity field with prescribed second-order
        statistics by applying a digital filter to a random white-noise
        field. The key idea is to convolve a random signal with a filter
        kernel that reproduces the target two-point correlation function.

    The procedure is as follows:
        1. Generate a random white-noise field r(x, y) with zero mean
           and unit variance.
        2. Apply a digital filter with coefficients b_k that are derived
           from the target correlation function.
        3. Scale the filtered field by the Cholesky decomposition of the
           Reynolds stress tensor to obtain the fluctuating velocity
           components.

    For a Gaussian correlation function, the filter coefficients are:
        b_k = exp(-π · k² / n²)  (for a 1D filter)
    where n controls the filter width (related to the integral length scale).

    References:
        - Klein, M., Sadiki, A., & Janicka, J. (2003). Journal of
          Computational Physics, 186(2):652-665.
        - di Mare, L., Klein, M., Jones, W. P., & Janicka, J. (2006).
          Physics of Fluids, 18(2):025107.

    Args:
        mesh:               The computational mesh.
        mean_velocity:      The mean velocity at the inlet.
        reynolds_stress:    The Reynolds stress tensor (3×3 array).
        integral_length_x:  The integral length scale in the streamwise
                            direction. Defaults to 0.1.
        integral_length_y:  The integral length scale in the wall-normal
                            direction. Defaults to 0.05.
        filter_width:       The number of filter points (must be odd).
                            Controls the spatial correlation length.
                            Defaults to 11.
        seed:               Random seed for reproducibility. Defaults to None.
    """
    mesh: Mesh2D
    mean_velocity: float = 1.0
    reynolds_stress: np.ndarray = field(default_factory=lambda: np.diag([0.01, 0.01, 0.01]))
    integral_length_x: float = 0.1
    integral_length_y: float = 0.05
    filter_width: int = 11
    seed: Optional[int] = None

    # Internal state
    _rng: np.random.Generator = field(init=False, repr=False)
    _cholesky_A: np.ndarray = field(init=False, repr=False)
    _filter_coefficients: np.ndarray = field(init=False, repr=False)
    _white_noise: np.ndarray = field(init=False, repr=False)
    _filtered_field: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validates the filter width, initialises the random number
        generator, computes the Cholesky decomposition, and builds the
        digital filter coefficients.

        Raises:
            ValueError: If the filter width is not odd and positive.
        """
        # ── Validate the filter width ──────────────────────────────────
        if self.filter_width < 3:
            raise ValueError(
                "The filter width must be at least 3. A width of "
                f"{self.filter_width} is too small to produce meaningful "
                "correlation."
            )
        if self.filter_width % 2 == 0:
            raise ValueError(
                "The filter width must be an odd number. An even filter "
                "width produces an asymmetric filter kernel, which would "
                "introduce a directional bias in the fluctuations."
            )

        # ── Validate the Reynolds stress tensor ────────────────────────
        if self.reynolds_stress.shape != (3, 3):
            raise ValueError(
                "The Reynolds stress tensor must be a 3×3 array. "
                f"Received shape {self.reynolds_stress.shape}."
            )

        # ── Initialise the random number generator ─────────────────────
        self._rng = np.random.default_rng(self.seed)

        # ── Perform the Cholesky decomposition ─────────────────────────
        self._cholesky_A = _cholesky_reynolds_stress(self.reynolds_stress)

        # ── Build the digital filter coefficients ──────────────────────
        # The coefficients are computed from a Gaussian correlation function.
        # For a 1D filter of width N (N must be odd), the coefficients are:
        #   b_k = exp(-π · k² / (2 · n²))
        # where k ranges from -M to +M, with M = (N - 1) / 2, and n is
        # related to the integral length scale.
        M: int = (self.filter_width - 1) // 2
        b: np.ndarray = np.zeros(self.filter_width)

        # Compute the normalised filter parameter n from the integral
        # length scale. Please note that for a Gaussian filter,
        # L_int = n · Δy / √(π · ln 2).
        n_y: float = self.integral_length_y / (self.mesh.dy if self.mesh.dy > 0 else 1.0)

        for k_idx, k in enumerate(range(-M, M + 1)):
            b[k_idx] = np.exp(-np.pi * k ** 2 / (2.0 * max(n_y, 0.5) ** 2))

        # Normalise the filter coefficients so that the variance of the
        # filtered field equals the variance of the input field.
        self._filter_coefficients = b / np.linalg.norm(b)

        # ── Initialise the white noise and filtered fields ─────────────
        ny: int = self.mesh.ny
        self._white_noise = np.zeros(ny)
        self._filtered_field = np.zeros(ny)

    def _apply_filter_1d(self, signal: np.ndarray) -> np.ndarray:
        """Applies the 1D digital filter to a signal using convolution.

        The convolution is performed with 'same' mode so that the output
        has the same length as the input. Boundary effects are handled
        by reflecting the signal at the boundaries.

        Args:
            signal: The 1D input signal (white noise) of length ny.

        Returns:
            The 1D filtered signal of the same length as the input.
        """
        # Use numpy's convolution with 'same' mode for convenience.
        # This pads the input signal at both ends to maintain the length.
        filtered: np.ndarray = np.convolve(
            signal, self._filter_coefficients, mode="same"
        )
        return filtered

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates the turbulent inflow velocity field using the digital
        filter method.

        This method generates a new white-noise field, applies the digital
        filter to introduce spatial correlations, and then scales the
        result by the Cholesky decomposition of the Reynolds stress tensor
        to obtain the velocity fluctuations.

        Returns:
            A tuple (u_in, v_in, w_in) containing the instantaneous
            velocity components at the inlet plane. Each array has shape
            (ny,) for a west/east face inlet.
        """
        ny: int = self.mesh.ny

        # ── Step 1: Generate a white-noise field ───────────────────────
        # The white noise has zero mean and unit variance.
        self._white_noise = self._rng.normal(0.0, 1.0, ny)

        # ── Step 2: Apply the digital filter ───────────────────────────
        self._filtered_field = self._apply_filter_1d(self._white_noise)

        # ── Step 3: Scale by the Cholesky factor of the Reynolds stress ─
        # Generate three independent filtered fields (one per velocity
        # component) and combine them using the Cholesky factor.
        # This ensures that <u'_i u'_j> = R_ij.
        filtered_fields: list[np.ndarray] = []
        for _ in range(3):
            noise: np.ndarray = self._rng.normal(0.0, 1.0, ny)
            filtered_fields.append(self._apply_filter_1d(noise))

        # Combine the filtered fields using the Cholesky factor A.
        # The fluctuating velocity is: u' = A · ξ
        # where ξ is the vector of filtered white-noise fields.
        up: np.ndarray = (
            self._cholesky_A[0, 0] * filtered_fields[0]
            + self._cholesky_A[0, 1] * filtered_fields[1]
            + self._cholesky_A[0, 2] * filtered_fields[2]
        )
        vp: np.ndarray = (
            self._cholesky_A[1, 0] * filtered_fields[0]
            + self._cholesky_A[1, 1] * filtered_fields[1]
            + self._cholesky_A[1, 2] * filtered_fields[2]
        )
        wp: np.ndarray = (
            self._cholesky_A[2, 0] * filtered_fields[0]
            + self._cholesky_A[2, 1] * filtered_fields[1]
            + self._cholesky_A[2, 2] * filtered_fields[2]
        )

        # ── Step 4: Add the mean velocity ──────────────────────────────
        u_in: np.ndarray = self.mean_velocity + up
        v_in: np.ndarray = vp
        w_in: np.ndarray = wp

        return u_in, v_in, w_in


# ─── Prescribed Reynolds Stresses ──────────────────────────────────────

def prescribed_reynolds_stresses(
    reynolds_stress: np.ndarray,
    n_points: int,
    mean_velocity: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates fluctuating velocity components that exactly satisfy a
    prescribed Reynolds stress tensor using a Cholesky decomposition
    approach.

    Physical background:
        Given a target Reynolds stress tensor R_ij = <u'_i u'_j>, the
        fluctuating velocities can be generated as:

            u'_i = A_ij · ξ_j

        where A is the Cholesky factor of R (R = A · A^T) and ξ_j are
        uncorrelated random numbers with zero mean and unit variance.

        This method ensures that the ensemble-averaged Reynolds stresses
        converge to the prescribed values. However, it does not enforce
        any spatial or temporal correlations; these must be introduced
        separately (e.g., using the digital filter method above).

    Kindly ensure that the Reynolds stress tensor is symmetric and positive-
    definite. If it is not, the Cholesky decomposition will fail, and a
    small diagonal regularisation will be applied automatically.

    Args:
        reynolds_stress: The target Reynolds stress tensor as a 3×3 array.
                         The tensor components are:
                             R[0,0] = <u'u'>  (streamwise normal stress)
                             R[1,1] = <v'v'>  (wall-normal normal stress)
                             R[2,2] = <w'w'>  (spanwise normal stress)
                             R[0,1] = <u'v'>  (shear stress)
        n_points:       The number of points at which to generate
                        fluctuations.
        mean_velocity:  The mean streamwise velocity. Added to the
                        streamwise fluctuation to obtain the total
                        instantaneous velocity. Defaults to 1.0.
        seed:           Random seed for reproducibility. Defaults to None.

    Returns:
        A tuple (u, v, w) of numpy arrays, each of length n_points,
        containing the instantaneous velocity components (mean +
        fluctuation).

    Raises:
        ValueError: If the Reynolds stress tensor is not 3×3.

    Example:
        >>> R = np.array([[0.01, 0.002, 0.0],
        ...               [0.002, 0.005, 0.0],
        ...               [0.0,   0.0,   0.003]])
        >>> u, v, w = prescribed_reynolds_stresses(R, n_points=64,
        ...                                        mean_velocity=10.0)
        >>> print(f"u_mean = {u.mean():.2f}, v_rms = {v.std():.3f}")
        u_mean = 10.00, v_rms = 0.071
    """
    # ── Validate the Reynolds stress tensor ────────────────────────────
    if reynolds_stress.shape != (3, 3):
        raise ValueError(
            "The Reynolds stress tensor must be a 3×3 array. "
            f"Received shape {reynolds_stress.shape}. "
            "Please provide the full tensor R_ij."
        )

    # ── Perform the Cholesky decomposition ─────────────────────────────
    # It is crucial to understand that the Cholesky decomposition gives
    # us the lower-triangular matrix A such that R = A · A^T.
    A: np.ndarray = _cholesky_reynolds_stress(reynolds_stress)

    # ── Generate uncorrelated random numbers ──────────────────────────
    rng: np.random.Generator = np.random.default_rng(seed)
    xi: np.ndarray = rng.normal(0.0, 1.0, size=(3, n_points))

    # ── Compute the fluctuating velocities ───────────────────────────
    # u'_i = A_ij · ξ_j  (summation over j implied).
    # Note that A is 3×3 and xi is 3×n_points.
    fluctuations: np.ndarray = A @ xi  # shape (3, n_points)

    # ── Add the mean velocity to obtain the instantaneous field ──────
    u: np.ndarray = mean_velocity + fluctuations[0, :]
    v: np.ndarray = fluctuations[1, :]
    w: np.ndarray = fluctuations[2, :]

    return u, v, w
