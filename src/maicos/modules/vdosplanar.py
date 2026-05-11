#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing planar (slab-resolved) vibrational density of states."""

import logging

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import NoDataError
from scipy.integrate import simpson

from ..core import PlanarBase
from ..lib.util import render_docs

logger = logging.getLogger(__name__)


@render_docs
class VDOSPlanar(PlanarBase):
    r"""Slab-resolved vibrational density of states along a cartesian axis.

    For each slab :math:`k` along ``dim``, the module computes the normalized
    velocity autocorrelation function

    .. math::

        Z_k(\tau) =
            \frac{\sum_{i \in \mathcal{S}_k, d}
                  \langle v_{i,d}(0)\, v_{i,d}(\tau)\rangle_{t_0}}
                 {\sum_{i \in \mathcal{S}_k, d}
                  \langle v_{i,d}(0)^{2}\rangle_{t_0}}

    where :math:`\mathcal{S}_k` is the set of atoms assigned to slab
    :math:`k`. The vibrational density of states per slab is then

    .. math::

        g_k(\omega) = \frac{2}{\pi}
            \int_0^{\infty} Z_k(\tau)\,\cos(\omega\,\tau)\,\mathrm{d}\tau .

    The slab assignment is **static**: each atom is permanently assigned to
    the bin of its position along ``dim`` in the first analysed frame.
    Atoms that diffuse across slab boundaries continue to contribute to
    their original slab. This is the standard convention for slab-resolved
    VDOS and is well justified whenever the VACF correlation time is short
    compared to the typical residence time within a slab.

    The autocorrelation is accumulated frame by frame with the streaming
    multi-tau correlator from :class:`maicos.lib.correlator.MultiTauCorrelator`,
    so memory does not grow with trajectory length. The cosine transform is
    evaluated by Simpson integration on the correlator's native (approximately
    log-spaced) lag grid.

    The trajectory must provide velocities.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${CORRELATOR_PARAMETERS}
    ${PLANAR_CLASS_PARAMETERS}
    ${BASE_CLASS_PARAMETERS}
    n_frequencies : int
        Number of points on the output frequency grid.
    output_prefix : str
        Prefix for the output files ``<prefix>_vacf.dat`` and
        ``<prefix>_vdos.dat``.

    Attributes
    ----------
    results.bin_pos : numpy.ndarray
        Average centre of each slab (in Å) relative to the box centre, shape
        ``(n_bins,)``.
    results.bin_counts : numpy.ndarray
        Number of atoms permanently assigned to each slab, shape ``(n_bins,)``.
    results.times : numpy.ndarray
        Lag times in ps, shape ``(n_lags,)``.
    results.vacf : numpy.ndarray
        Normalized per-slab VACF with shape ``(n_lags, n_bins)``. Empty slabs
        are filled with ``NaN``.
    results.frequencies : numpy.ndarray
        Angular frequencies :math:`\omega` in ps\ :sup:`-1`, shape
        ``(n_frequencies,)``.
    results.vdos : numpy.ndarray
        Per-slab vibrational density of states, shape
        ``(n_frequencies, n_bins)``, in units of ps.

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        correlator_num_levels: int = 20,
        correlator_channels_per_level: int = 16,
        dim: int = 2,
        zmin: float | None = None,
        zmax: float | None = None,
        bin_width: float = 1.0,
        refgroup: mda.AtomGroup | None = None,
        unwrap: bool = False,
        pack: bool = True,
        jitter: float = 0.0,
        concfreq: int = 0,
        n_frequencies: int = 256,
        output_prefix: str = "vdos_planar",
    ) -> None:
        self._locals = locals()
        super().__init__(
            atomgroup=atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            wrap_compound="atoms",
            correlator_num_levels=correlator_num_levels,
            correlator_channels_per_level=correlator_channels_per_level,
        )
        if n_frequencies < 2:
            raise ValueError("`n_frequencies` must be at least 2.")
        self.n_frequencies = int(n_frequencies)
        self.output_prefix = output_prefix

    def _prepare(self) -> None:
        super()._prepare()  # PlanarBase: zmin, zmax, n_bins
        if not self._universe.trajectory.ts.has_velocities:
            raise NoDataError(
                "VDOSPlanar requires velocities in the trajectory; the active "
                "reader does not provide them."
            )
        logger.info(
            f"Computing slab-resolved VDOS in {self.n_bins} slabs along "
            f"{'xyz'[self.dim]}-axis (streaming multi-tau, "
            f"p={self.correlator_num_levels}, "
            f"m={self.correlator_channels_per_level})."
        )

    def _single_frame(self) -> float:
        super()._single_frame()  # PlanarBase: populate self._obs.bin_*

        if not hasattr(self, "_bin_index"):
            self._assign_bins()

        v = self.atomgroup.velocities
        if not self._all_included:
            # Atoms whose first-frame position fell outside [zmin, zmax) get
            # zero velocity in the streamed signal, so they contribute nothing
            # to the correlator without changing its shape.
            v = v.copy()
            v[~self._included] = 0.0

        self._corr.velocity = v
        return float(np.mean(v * v))

    def _assign_bins(self) -> None:
        """Build the static bin assignment from the first analysed frame."""
        z = self.atomgroup.positions[:, self.dim]
        edges = self._obs.bin_edges  # length n_bins+1
        bin_index = np.digitize(z, edges) - 1
        included = (bin_index >= 0) & (bin_index < self.n_bins)
        self._bin_index = bin_index
        self._included = included
        self._all_included = bool(np.all(included))
        # Per-bin atom counts, ignoring atoms outside [zmin, zmax).
        self._n_atoms_per_bin = np.bincount(
            bin_index[included], minlength=self.n_bins
        ).astype(np.int64)
        logger.info(
            "Static bin assignment built from first frame: "
            f"{self._n_atoms_per_bin.sum()} atoms in {self.n_bins} slabs "
            f"(min/max per slab: {self._n_atoms_per_bin.min()}/"
            f"{self._n_atoms_per_bin.max()})."
        )

    def _conclude(self) -> None:
        super()._conclude()  # PlanarBase: results.bin_pos

        # self.correlation.velocity has shape (n_lags, n_atoms, 3).
        # Reduce dot product, then sum per bin.
        per_atom = self.correlation.velocity.sum(axis=-1)  # (n_lags, n_atoms)
        n_lags = per_atom.shape[0]
        per_bin = np.zeros((n_lags, self.n_bins), dtype=per_atom.dtype)
        bins = self._bin_index[self._included]
        np.add.at(per_bin, (slice(None), bins), per_atom[:, self._included])

        # Normalise per bin so vacf[0, k] == 1 for populated slabs; empty
        # slabs are flagged NaN.
        z0 = per_bin[0].copy()
        empty = self._n_atoms_per_bin == 0
        z0[empty] = 1.0
        vacf = per_bin / z0[None, :]
        if np.any(empty):
            vacf[:, empty] = np.nan

        # Physical time axis from the actual frame times.
        if self.times.size > 1:
            dt = float(np.mean(np.diff(self.times)))
        else:
            dt = float(self._trajectory.dt)
        times = self.lags.astype(float) * dt

        # Frequency grid bounded by the longest resolved lag and the finest
        # sampling interval.
        t_max = times[-1]
        omega_max = np.pi / dt
        omega_min = 2.0 * np.pi / t_max if t_max > 0 else omega_max / 64
        omega = np.linspace(omega_min, omega_max, self.n_frequencies)

        # Per-slab cosine transform on the native multi-tau lag grid. The
        # integrand has shape (n_freqs, n_lags, n_bins); for the typical bin
        # counts (~10-100) this is cheap.
        cos_kernel = np.cos(np.outer(omega, times))  # (n_freqs, n_lags)
        vdos = np.empty((self.n_frequencies, self.n_bins))
        for k in range(self.n_bins):
            if empty[k]:
                vdos[:, k] = np.nan
            else:
                vdos[:, k] = (2.0 / np.pi) * simpson(
                    cos_kernel * vacf[None, :, k], x=times, axis=1
                )

        self.results.times = times
        self.results.vacf = vacf
        self.results.frequencies = omega
        self.results.vdos = vdos
        self.results.bin_counts = self._n_atoms_per_bin

    def save(self) -> None:
        """Save per-slab VACF and VDOS to two data files."""
        bin_labels = [f"bin@{z:+7.3f}Å" for z in self.results.bin_pos]

        self.savetxt(
            f"{self.output_prefix}_vacf",
            np.column_stack([self.results.times, self.results.vacf]),
            columns=["t / ps", *bin_labels],
        )
        self.savetxt(
            f"{self.output_prefix}_vdos",
            np.column_stack([self.results.frequencies, self.results.vdos]),
            columns=["omega / ps^-1", *bin_labels],
        )
