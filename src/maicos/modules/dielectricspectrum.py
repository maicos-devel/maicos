#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing dielectric spectra for bulk systems."""

import logging
from typing import Optional

import MDAnalysis as mda
import numpy as np
import scipy.constants

from ..core import AnalysisBase
from ..lib.math import FT, iFT
from ..lib.util import bin, charge_neutral, citation_reminder, get_compound, render_docs


logger = logging.getLogger(__name__)


@render_docs
@charge_neutral(filter="error")
class DielectricSpectrum(AnalysisBase):
    r"""Linear dielectric spectrum.

    This module, given a molecular dynamics trajectory, produces a `.txt` file
    containing the complex dielectric function as a function of the (linear, not radial
    - i.e., :math:`\nu` or :math:`f`, rather than :math:`\omega`) frequency, along with
    the associated standard deviations. The algorithm is based on the Fluctuation
    Dissipation Relation: :math:`\chi(f) = -1/(3 V k_B T \varepsilon_0)
    \mathcal{L}[\theta(t) \langle P(0) dP(t)/dt\rangle]`, where :math:`\mathcal{L}` is
    the Laplace transformation.

    .. note::
        The polarization time series and the average system volume are also saved.

    Please read and cite :footcite:p:`carlsonExploringAbsorptionSpectrum2020`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    temperature : float
        Reference temperature.
    output_prefix : str
        Prefix for the output files.
    segs : int
        Sets the number of segments the trajectory is broken into.
    df : float
        The desired frequency spacing in THz. This determines the minimum frequency
        about which there is data. Overrides `segs` option.
    bins : int
        Determines the number of bins used for data averaging; (this parameter sets the
        upper limit). The data are by default binned logarithmically. This helps to
        reduce noise, particularly in the high-frequency domain, and also prevents plot
        files from being too large.
    binafter : int
        The number of low-frequency data points that are left unbinned.
    nobin : bool
        Prevents the data from being binned altogether. This can result in very large
        plot files and errors.

    Attributes
    ----------
    results

    References
    ----------
    .. footbibliography::
    """

    # TODO: set up script to calc spectrum at intervals while calculating polarization
    # for very big-data trajectories
    # TODO: merge with molecular version?
    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        refgroup: Optional[mda.AtomGroup] = None,
        unwrap: bool = True,
        concfreq: int = 0,
        temperature: float = 300,
        output_prefix: str = "",
        segs: int = 20,
        df: Optional[float] = None,
        bins: int = 200,
        binafter: float = 20,
        nobin: bool = False,
        jitter: float = 0.0,
    ):
        self._locals = locals()
        wrap_compound = get_compound(atomgroup)
        super().__init__(
            atomgroup,
            unwrap=unwrap,
            refgroup=refgroup,
            concfreq=concfreq,
            wrap_compound=wrap_compound,
            jitter=jitter,
        )
        self.temperature = temperature
        self.output_prefix = output_prefix
        self.segs = segs
        self.df = df
        self.bins = bins
        self.binafter = binafter
        self.nobin = nobin

    def _prepare(self):
        # Print the Shane Carlson citation
        logger.info(citation_reminder("10.1021/acs.jpca.0c04063"))

        if len(self.output_prefix) > 0:
            self.output_prefix += "_"

        self.dt = self._trajectory.dt * self.step
        self.V = 0
        self.P = np.zeros((self.n_frames, 3))

    def _single_frame(self):
        self.V += self._ts.volume
        self.P[self._frame_index, :] = np.dot(
            self.atomgroup.charges, self.atomgroup.positions
        )

    def _conclude(self):
        self.results.t = self._trajectory.dt * self.frames
        self.results.V = self.V / self._index

        self.results.P = self.P

        # Find a suitable number of segments if it's not specified:
        if self.df is not None:
            self.segs = np.max([int(self.n_frames * self.dt * self.df), 2])

        self.seglen = int(self.n_frames / self.segs)

        # Prefactor for susceptibility: Polarization: eÅ^2 to e m^2
        pref = (scipy.constants.e) ** 2 * scipy.constants.angstrom**2
        # Volume: Å^3 to m^3
        pref /= 3 * self.results.V * scipy.constants.angstrom**3
        pref /= scipy.constants.k * self.temperature
        pref /= scipy.constants.epsilon_0

        logger.info("Calculating susceptibility and errors...")

        # if t too short to simply truncate
        if len(self.results.t) < 2 * self.seglen:
            self.results.t = np.append(
                self.results.t, self.results.t + self.results.t[-1] + self.dt
            )

        # truncate t array (it's automatically longer than 2 * seglen)
        self.results.t = self.results.t[: 2 * self.seglen]
        # get freqs
        self.results.nu = FT(
            self.results.t,
            np.append(self.results.P[: self.seglen, 0], np.zeros(self.seglen)),
        )[0]
        # susceptibility
        self.results.susc = np.zeros(self.seglen, dtype=complex)
        # std deviation of susceptibility
        self.results.dsusc = np.zeros(self.seglen, dtype=complex)
        # susceptibility for current seg
        ss = np.zeros((2 * self.seglen), dtype=complex)

        # loop over segs
        for s in range(0, self.segs):
            logger.info(f"\rSegment {s + 1} of {self.segs}")
            ss = 0 + 0j

            # loop over x, y, z
            for self._i in range(3):
                FP = FT(
                    self.results.t,
                    np.append(
                        self.results.P[
                            s * self.seglen : (s + 1) * self.seglen, self._i
                        ],
                        np.zeros(self.seglen),
                    ),
                    False,
                )
                ss += FP.real * FP.real + FP.imag * FP.imag

            ss *= self.results.nu * 1j

            # Get the real part by Kramers Kronig
            ss.real = iFT(
                self.results.t,
                1j * np.sign(self.results.nu) * FT(self.results.nu, ss, False),
                False,
            ).imag

            if s == 0:
                self.results.susc += ss[self.seglen :]

            else:
                ds = ss[self.seglen :] - (self.results.susc / s)
                self.results.susc += ss[self.seglen :]
                dif = ss[self.seglen :] - (self.results.susc / (s + 1))
                ds.real *= dif.real
                ds.imag *= dif.imag
                # variance by Welford's Method
                self.results.dsusc += ds

        self.results.dsusc.real = np.sqrt(self.results.dsusc.real)
        self.results.dsusc.imag = np.sqrt(self.results.dsusc.imag)

        # 1/2 b/c it's the full FT, not only half-domain
        self.results.susc *= pref / (2 * self.seglen * self.segs * self.dt)
        self.results.dsusc *= pref / (2 * self.seglen * self.segs * self.dt)

        # Discard negative-frequency data; contains the same information as positive
        # regime: Now nu represents positive f instead of omega
        self.results.nu = self.results.nu[self.seglen :] / (2 * np.pi)

        logger.info(
            f"Length of segments:    {self.seglen} frames,"
            f" {self.seglen * self.dt:.0f} ps"
        )
        logger.info(
            f"Frequency spacing:    "
            f"~ {self.segs / (self.n_frames * self.dt):.5f} THz"
        )

        # Bin data if there are too many points:
        if not (self.nobin or self.seglen <= self.bins):
            bins = np.logspace(
                np.log(self.binafter) / np.log(10),
                np.log(len(self.results.susc)) / np.log(10),
                self.bins - self.binafter + 1,
            ).astype(int)
            bins = np.unique(np.append(np.arange(self.binafter), bins))[:-1]

            self.results.nu_binned = bin(self.results.nu, bins)
            self.results.susc_binned = bin(self.results.susc, bins)
            self.results.dsusc_binned = bin(self.results.dsusc, bins)

            logger.info(
                f"Binning data above datapoint " f"{self.binafter} in log-spaced bins"
            )
            logger.info(
                f"Binned data consists of " f"{len(self.results.susc)} datapoints"
            )
        # data is binned
        logger.info(
            f"Not binning data: there are " f"{len(self.results.susc)} datapoints"
        )

    def save(self):
        """Save result."""
        np.save(self.output_prefix + "tseries.npy", self.results.t)

        with open(self.output_prefix + "V.txt", "w") as Vfile:
            Vfile.write(str(self.results.V))

        np.save(self.output_prefix + "P_tseries.npy", self.results.P)

        suscfilename = "{}{}".format(self.output_prefix, "susc.dat")
        self.savetxt(
            suscfilename,
            np.transpose(
                [
                    self.results.nu,
                    self.results.susc.real,
                    self.results.dsusc.real,
                    self.results.susc.imag,
                    self.results.dsusc.imag,
                ]
            ),
            columns=["ν [THz]", "real(χ)", " Δ real(χ)", "imag(χ)", "Δ imag(χ)"],
        )

        logger.info("Susceptibility data saved as " + suscfilename)

        if not (self.nobin or self.seglen <= self.bins):
            suscfilename = "{}{}".format(self.output_prefix, "susc_binned.dat")
            self.savetxt(
                suscfilename,
                np.transpose(
                    [
                        self.results.nu_binned,
                        self.results.susc_binned.real,
                        self.results.dsusc_binned.real,
                        self.results.susc_binned.imag,
                        self.results.dsusc_binned.imag,
                    ]
                ),
                columns=["ν [THz]", "real(χ)", " Δ real(χ)", "imag(χ)", "Δ imag(χ)"],
            )

            logger.info("Binned susceptibility data saved as " + suscfilename)
