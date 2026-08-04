#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bin-dependent benchmarks for profile-based analyses on synthetic data."""

from benchmarks.synthetic import make_universe
from maicos import DensityPlanar, DielectricPlanar

N_ATOMS = 30_000


class ProfileBinBenchmark:
    """ProfileBase scaling with the number of bins (via ``bin_width``).

    :func:`numpy.histogram` is linear in the atom count and nearly independent of the
    bin count, so the bin widths have to reach far below the box resolution before the
    number of bins overtakes it.
    """

    timeout = 180
    params = [1.0, 0.1, 0.01, 0.001]
    param_names = ["bin_width"]

    def setup(self, _bin_width):
        """Build the synthetic atomgroup."""
        self.atomgroup = make_universe(n_atoms=N_ATOMS)

    def time_run(self, bin_width):
        """Time DensityPlanar over a range of bin widths."""
        DensityPlanar(
            self.atomgroup,
            dens="mass",
            bin_width=bin_width,
            unwrap=False,
            pack=False,
        ).run()


class DielectricBinBenchmark:
    """Dielectric scaling with the number of bins (via ``bin_width``).

    The bin widths stop well above those of :class:`ProfileBinBenchmark` because the
    virtual cut kernel allocates ``n_bins * box / vcutwidth`` entries per frame and per
    direction parallel to the surface.
    """

    timeout = 180
    params = [1.0, 0.5, 0.1, 0.05, 0.02]
    param_names = ["bin_width"]

    def setup(self, _bin_width):
        """Build the synthetic atomgroup."""
        self.atomgroup = make_universe(n_atoms=N_ATOMS)

    def time_run(self, bin_width):
        """Time DielectricPlanar over a range of bin widths."""
        DielectricPlanar(
            self.atomgroup,
            bin_width=bin_width,
            unwrap=False,
            pack=False,
        ).run()
