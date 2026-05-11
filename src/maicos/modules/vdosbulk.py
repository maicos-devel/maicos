#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing bulk vibrational density of states (VDOS)."""

import logging

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import NoDataError
from scipy.integrate import simpson

from ..core import AnalysisBase
from ..lib.util import render_docs

logger = logging.getLogger(__name__)


@render_docs
class VDOSBulk(AnalysisBase):
    r"""Bulk vibrational density of states from a streaming velocity ACF.

    Computes the normalized velocity autocorrelation function

    .. math::

        Z(\tau) = \frac{\sum_{i,d}\langle v_{i,d}(0)\,v_{i,d}(\tau)\rangle_{t_0}}
                       {\sum_{i,d}\langle v_{i,d}(0)^{2} \rangle_{t_0}}

    of the supplied ``atomgroup`` and Fourier-cosine-transforms it to obtain
    the vibrational density of states

    .. math::

        g(\omega) = \frac{2}{\pi} \int_0^{\infty} Z(\tau)\,\cos(\omega\,\tau)\,
                    \mathrm{d}\tau .

    The autocorrelation is accumulated frame by frame with the streaming
    multi-tau correlator from :class:`maicos.lib.correlator.MultiTauCorrelator`,
    so the memory cost is bounded and independent of the trajectory length.
    The cosine transform is evaluated by Simpson integration on the
    correlator's native (approximately log-spaced) lag grid, avoiding the
    artifacts of resampling onto a uniform grid before FFT.

    The trajectory must provide velocities.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${CORRELATOR_PARAMETERS}
    ${BASE_CLASS_PARAMETERS}
    n_frequencies : int
        Number of points on the output frequency grid.
    output_prefix : str
        Prefix for the output files ``<prefix>_vacf.dat`` and
        ``<prefix>_vdos.dat``.

    Attributes
    ----------
    results.times : numpy.ndarray
        Lag times in ps, shape ``(n_lags,)``.
    results.vacf : numpy.ndarray
        Normalized velocity autocorrelation function with ``vacf[0] == 1``.
    results.frequencies : numpy.ndarray
        Angular frequencies :math:`\omega` in ps\ :sup:`-1`, shape
        ``(n_frequencies,)``.
    results.vdos : numpy.ndarray
        Vibrational density of states evaluated at ``results.frequencies``,
        in units of ps.

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        correlator_num_levels: int = 20,
        correlator_channels_per_level: int = 16,
        refgroup: mda.AtomGroup | None = None,
        unwrap: bool = False,
        pack: bool = True,
        jitter: float = 0.0,
        concfreq: int = 0,
        n_frequencies: int = 256,
        output_prefix: str = "vdos",
    ) -> None:
        self._locals = locals()
        super().__init__(
            atomgroup=atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
            wrap_compound="atoms",
            correlator_num_levels=correlator_num_levels,
            correlator_channels_per_level=correlator_channels_per_level,
        )
        if n_frequencies < 2:
            raise ValueError("`n_frequencies` must be at least 2.")
        self.n_frequencies = int(n_frequencies)
        self.output_prefix = output_prefix

    def _prepare(self) -> None:
        if not self._universe.trajectory.ts.has_velocities:
            raise NoDataError(
                "VDOSBulk requires velocities in the trajectory; the active "
                "reader does not provide them."
            )
        logger.info(
            "Computing bulk VDOS via streaming multi-tau VACF "
            f"(p={self.correlator_num_levels}, "
            f"m={self.correlator_channels_per_level})."
        )

    def _single_frame(self) -> float:
        v = self.atomgroup.velocities
        self._corr.velocity = v
        return float(np.mean(v * v))

    def _conclude(self) -> None:
        # self.correlation.velocity has shape (n_lags, n_atoms, 3). Empty upper
        # levels have already been dropped by AnalysisBase.
        summed = self.correlation.velocity.sum(axis=(-2, -1))
        vacf = summed / summed[0]

        # Derive the physical time step from the actual sampled frame times;
        # this is robust to readers that misreport `trajectory.dt`.
        if self.times.size > 1:
            dt = float(np.mean(np.diff(self.times)))
        else:
            dt = float(self._trajectory.dt)

        times = self.lags.astype(float) * dt

        self.results.times = times
        self.results.vacf = vacf

        # Frequency grid bounded by the longest resolved lag (low end) and the
        # finest sampling interval (high end / Nyquist).
        t_max = times[-1]
        omega_max = np.pi / dt
        omega_min = 2.0 * np.pi / t_max if t_max > 0 else omega_max / 64
        omega = np.linspace(omega_min, omega_max, self.n_frequencies)

        # g(omega) = (2/pi) * integral_0^inf Z(t) cos(omega t) dt, evaluated by
        # Simpson's rule on the non-uniform multi-tau lag grid.
        integrand = vacf[None, :] * np.cos(np.outer(omega, times))
        g = (2.0 / np.pi) * simpson(integrand, x=times, axis=1)

        self.results.frequencies = omega
        self.results.vdos = g

    def save(self) -> None:
        """Save VACF and VDOS to ``<output_prefix>_vacf.dat`` and ``..._vdos.dat``."""
        self.savetxt(
            f"{self.output_prefix}_vacf",
            np.column_stack([self.results.times, self.results.vacf]),
            columns=["t / ps", "VACF"],
        )
        self.savetxt(
            f"{self.output_prefix}_vdos",
            np.column_stack([self.results.frequencies, self.results.vdos]),
            columns=["omega / ps^-1", "VDOS / ps"],
        )
