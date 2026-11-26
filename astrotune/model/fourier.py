"""Separable Fourier features of the spacetime coordinate.

The velocity field of the bijector is a function of ``u = (t, x_1 ... x_d)`` -- the
flow parameter and the catalog's parameters -- and a plain MLP is biased towards
smooth functions of ``u``: it needs many units to build something that varies
quickly, and those units then have to serve every catalog as well. Feeding the
first layer a fixed Fourier basis of ``u`` gives it those scales for free.

The basis here is *separable*: the modes are axis-aligned, ``k = (2 pi / lambda)
e_i``, so each feature depends on one coordinate and nothing else,

    cos(2 pi u_i / lambda),  sin(2 pi u_i / lambda),

for a log-spaced ladder of wavelengths on each axis. Separability is what makes
the block interpretable (one ladder per coordinate, no mixing of Teff with time),
needs no random seed, and keeps the feature count at ``2 * n_axes * n_modes``.

The flow parameter is treated separately from the parameters, because a frequency
along ``t`` is a frequency *along the integration*: the adaptive solver has to
resolve every oscillation, so it costs function evaluations rather than just
weights. Its wavelengths are therefore the first ``time_harmonics`` harmonics
``1, 1/2, 1/4 ...`` (in units of the flow parameter, which runs from 0 to 1)
rather than the geometric ladder, and the default of two caps it at two
oscillations over the whole flow.

That default is this block's, not the trainer's: ``LatentTrainConfig`` starts
with fourier_n_modes = fourier_time_harmonics = 0, because two oscillations along
the flow turned out to be visible in the learned map (several catalogs swinging
twice before settling) without improving the fit. Turn it on per run.
"""

from __future__ import annotations

import numpy as np
import sonnet as snt
import tensorflow as tf


def mode_wavelengths(n_modes: int, lam_min: float, lam_max: float) -> np.ndarray:
    """``n_modes`` wavelengths log-spaced from ``lam_max`` down to ``lam_min``.

    Both ends are hit exactly; one mode is placed at ``lam_max``; none at all
    (``n_modes <= 0``) gives an empty ladder, which is how the features are turned
    off axis by axis.
    """
    if n_modes <= 0:
        return np.zeros(0, dtype="f8")
    if n_modes == 1:
        return np.array([float(lam_max)])
    return np.geomspace(float(lam_max), float(lam_min), int(n_modes))


def time_wavelengths(n_harmonics: int) -> np.ndarray:
    """``2 ** -(n - 1)`` for ``n = 1 ... n_harmonics``: 1, 1/2, 1/4, ..."""
    return 2.0 ** -np.arange(max(0, int(n_harmonics)), dtype="f8")


class FourierFeatures(snt.Module):
    """``[cos, sin]`` of a separable ladder of modes of a spacetime coordinate.

    Args:
      n_in: width of the coordinate handed to :meth:`__call__`.
      n_modes: wavelengths per parameter axis, log-spaced ``lam_max`` to ``lam_min``.
        Six of them between the default 8 and 0.25 are exact octaves
        (8, 4, 2, 1, 0.5, 0.25), since the range is ``2**5``.
      lam_min, lam_max: the shortest and longest wavelength, in the units of the
        coordinate itself. There is no per-axis rescaling: with separable modes each
        axis has its own ladder, so a shared scale would only blur what "one
        wavelength" means.
      time_harmonics: harmonics of the flow parameter, ``1, 1/2, 1/4, ...`` in units
        of ``t`` (which runs 0 to 1). Zero leaves the time axis out entirely.
      time_axis: which column is the flow parameter, or ``None`` for a coordinate
        that has none (the delta model's ``(x, ln sigma)``).
    """

    def __init__(
        self,
        n_in: int,
        n_modes: int = 6,
        lam_min: float = 0.25,
        lam_max: float = 8.0,
        time_harmonics: int = 2,
        time_axis: int | None = 0,
        name=None,
    ):
        super().__init__(name=name)
        if lam_min <= 0 or lam_max < lam_min:
            raise ValueError(f"need 0 < lam_min <= lam_max, got {lam_min}, {lam_max}")
        waves: list = []
        for axis in range(int(n_in)):
            if time_axis is not None and axis == int(time_axis):
                lams = time_wavelengths(time_harmonics)
            else:
                lams = mode_wavelengths(n_modes, lam_min, lam_max)
            waves.extend((axis, float(lam)) for lam in lams)
        if not waves:
            raise ValueError(
                "the Fourier block would be empty: give n_modes or time_harmonics"
            )
        self.n_in = int(n_in)
        self.time_axis = None if time_axis is None else int(time_axis)
        self.axes = tf.constant([a for a, _ in waves], dtype=tf.int32)
        self.omega = tf.constant([2.0 * np.pi / lam for _, lam in waves], dtype=tf.float32)
        self.n_waves = len(waves)
        self.n_features = 2 * self.n_waves
        # Each cos/sin lies in [-1, 1]; scaling the block by 1/sqrt(n_waves) brings it
        # to the same order as the raw coordinate columns it is concatenated with,
        # when the arguments are spread over many radians. (They are not for the
        # longest wavelengths, where the features are nearly constant across the data
        # -- that is what makes them the low-frequency part of the basis.)
        self.scale = float(1.0 / np.sqrt(self.n_waves))

    def __call__(self, u) -> tf.Tensor:
        """``u``: ``(b, n_in)`` spacetime coordinate -> ``(b, 2 * n_waves)``."""
        angle = tf.gather(u, self.axes, axis=1) * self.omega
        return tf.concat([tf.cos(angle), tf.sin(angle)], axis=-1) * self.scale

    def wavelengths(self) -> np.ndarray:
        """The wavelength of every wave, in coordinate units (for tests and logs)."""
        return 2.0 * np.pi / self.omega.numpy()
