#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""
.. _howto-spatial-dipole-dipole-correlations:

Calculating and interpreting dipolar pair correlation functions
===============================================================

In this examples we will calculate dipolar pair correlation functions in real and
Fourier space using the maicos modules :class:`maicos.RDFDiporder` and
:class:`maicos.DiporderStructureFactor`. We will show how these pair correlation
functions are connected to each other and electrostatic properties like the dielectric
constant :math:`\varepsilon` and the Kirkwood factor :math:`g_K`.

We start by importing the necessarary modules
"""
# %%

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import scipy
from MDAnalysis.analysis.dielectric import DielectricConstant

import maicos
from maicos.lib.math import compute_rdf_structure_factor


# %%
# Our example system is :math:`N=512` rigid SPC/E water molecules simulated in an NVT
# ensemble at :math:`300\,\mathrm{K}` in a cubic cell of :math:`L=24.635\,Å`. To follow
# this how-to guide, you should download the :download:`topology <water_nvt.tpr>` and
# the :download:`trajectory <water_nvt.xtc>` files of the system. Below we load the
# system, report and store some system properties for later usage.

u = mda.Universe("water_nvt.tpr", "water_nvt.xtc")

volume = u.trajectory.ts.volume
density = u.residues.n_residues / volume
dipole_moment = u.atoms.dipole_moment(compound="residues", unwrap=True).mean()

print(f"ρ_n = {density:.3f} Å^-3")
print(f"µ = {dipole_moment:.2f} eÅ")

# %%
# The results of our first property calculations show that the number density as well as
# the dipole moment of a single water molecule is consistent with the literature
# :footcite:p:`vega_simulating_2011`.
#
# Static dielectric constant
# --------------------------
#
# To start with the analysis we first look at the dielectric constant of the system. If
# you run a simulation using an Ewald simulation technique as usually done, the
# dielectric constant for such system with metallic boundary conditions is given
# according to :footcite:t:`neumann_dipole_1983` by
#
# .. math:: \varepsilon = 1 + \frac{\langle M^2 \rangle_\mathrm{MBE} - \langle
#     M \rangle_\mathrm{MBE}^2}{3 \varepsilon_0 V k_B T}
#
# where
#
# .. math:: \boldsymbol M = \sum_{i=1}^N \boldsymbol \mu_i
#
# is the total dipole moment of the box, :math:`V` its volume and :math:`\varepsilon_0`
# the vacuum permittivity. We use the subscript in the expectation value
# :math:`\mathrm{MBE}` indicating that the equation only holds for simulations with
# **M**\ etallic **B**\ oundary conditions in an **E**\ wald simulation style. As shown
# in the equation for :math:`\varepsilon(\mathrm{MBE})` the dielectric constant here is
# a *total cell* quantity connecting the fluctuations of the total dipole moment to the
# dielectric constant. We can calculate :math:`\varepsilon_\mathrm{MBE}` using the
# :class:`MDAnalysis.analysis.dielectric.DielectricConstant` module of MDAnalysis.

epsilon_mbe = DielectricConstant(atomgroup=u.atoms).run()
print(f"ɛ_MBE = {epsilon_mbe.results.eps_mean:.2f}")

# %%
# The value of 70 is the same as reported in the literature for the
# rigid SPC/E water model :footcite:p:`vega_simulating_2011`.
#
# Kirkwood factor
# ---------------
#
# Knowing the dielectric constant we can also calculate the Kirkwood factor :math:`g_K`
# which is a measure describing molecular correlations. I.e a Kirkwood factor greater
# than 1 indicates that neighboring molecular dipoles are more likely to align in the
# same direction, enhancing the material's polarization and, consequently, its
# dielectric constant. Based on the dielectric constant :math:`\varepsilon` Kirkwood and
# Fröhlich derived the relation for the factor :math:`g_K` according to
#
# .. math:: \frac{ N \mu^2 g_K}{\varepsilon_0 V k_B T} = \frac{(\varepsilon -
#     1)(2\varepsilon + 1)}{\varepsilon}
#
# This relation is valid for a sample in an infinity, homogenous medium of the same
# dielectric constant. Below we implement this equation and calculate the factor for our
# system.


def kirkwood_factor_KF(
    dielectric_constant: float,
    volume: float,
    n_dipoles: float,
    molecular_dipole_moment: float,
    temperature: float = 300,
) -> float:
    """Kirkwood factor in the Kirkwood-Fröhlich way.

    For the sample in an infinity, homogenous medium of the same dielectric constant.

    Parameters
    ----------
    dielectric_constant : float
        the static dielectric constant ɛ
    volume : float
        system volume in Å^3
    n_dipoles : float
        number of dipoles
    molecular_dipole_moment : float
        dipole moment of a molecule (eÅ)
    temperature : float
        temperature of the simulation K
    """
    dipole_moment_sq = (
        molecular_dipole_moment
        * scipy.constants.elementary_charge
        * scipy.constants.angstrom
    ) ** 2
    factor = (
        scipy.constants.epsilon_0
        * (volume * scipy.constants.angstrom**3)
        * scipy.constants.Boltzmann
        * temperature
    )

    return (
        factor
        / (dielectric_constant * n_dipoles * dipole_moment_sq)
        * (dielectric_constant - 1)
        * (2 * dielectric_constant + 1)
    )


kirkwood_KF = kirkwood_factor_KF(
    dielectric_constant=epsilon_mbe.results.eps_mean,
    volume=volume,
    n_dipoles=u.residues.n_residues,
    molecular_dipole_moment=dipole_moment,
)

print(f"g_K = {kirkwood_KF:.2f}")

# %%
# This value means there is a quite strong correlation between neighboring water
# molecules. The dielectric constant :math:`\varepsilon` is a material property and does
# not depend on the boundary condition. Instead, the Kirkwood factor is indicative of
# dipole-dipole correlations which instead depend on the boundary condistions in the
# simulation. This relation is described and shown below.
#
# Connecting the Kirkwood factor to real space dipolar pair-correlation functions
# -------------------------------------------------------------------------------
#
# The :math:`r`-dependent Kirkwood factor can also be calculated from real space
# dipole-dipole pair correlation function :footcite:p:`zhang_dipolar_2014`
#
# .. math:: g_\mathrm{\hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\mu}}}(r) = \frac{1}{N}
#     \left\langle \sum_i \frac{1}{n_i(r)} \sum_{j=1}^{n_i(r)} (\hat{\boldsymbol{\mu}}_i
#     \cdot \hat{\boldsymbol{\mu}}_j) \right \rangle
#
# where :math:`\hat{\boldsymbol{\mu}}` is the normalized dipole moment and
# :math:`n_i(r)` is the number of dipoles within a spherical shell of distance :math:`r`
# and :math:`r + \delta r` from dipole :math:`i`. We compute the pair correlation
# function using the :class:`maicos.RDFDiporder` module up to half of the length of
# cubic simulation box. We drop a delta like contribution in :math:`r=0` caused by
# interaction of the dipole with itself.


L_half = u.dimensions[:3].max() / 2

rdf_diporder = maicos.RDFDiporder(g1=u.atoms, rmax=L_half, bin_width=0.01)
rdf_diporder.run()

# %%
# Based on this correlation function we can calculate the radially resolved Kirkwood
# factor via :footcite:p:`zhang_computing_2016`
#
# .. math:: G_K(r) = \rho_n 4 \pi \int_0^r \mathrm{d}r^\prime {r^\prime}^2
#     g_\mathrm{\hat \mu, \hat \mu}(r^\prime) + 1
#
# where the ":math:`+ 1`" accounts for the integration of the delta function at
# :math:`r=0`. Here :math:`\rho_n = N/V` is the density of dipoles.

radial_kirkwood = 1 + (
    density
    * 4
    * np.pi
    * scipy.integrate.cumulative_trapezoid(
        x=rdf_diporder.results.bins,
        y=rdf_diporder.results.bins**2 * rdf_diporder.results.rdf,
        initial=0,
    )
)

# %%
# While, for a truly infinite system, the :math:`r`- dependent Kirkwood factor,
# :math:`G_\mathrm{K}(r)` is short range :footcite:p:`frohlich_theory_1958`
# :footcite:p:`zhang_computing_2016`, the boundary conditions on a finite system
# introduce long-range effects. In particular, within MBE,
# :footcite:t:`caillol_asymptotic_1992` has shown that :math:`G_\mathrm{K}(r)` has a
# spurious asymptotic growth proportional to :math:`r^3/V`. This effect is stil present
# at :math:`r=r_K`, where :math:`r_K` (here approximately 6 Å) indicates a distance
# after which all the physical features of
# :math:`g_\mathrm{\hat{\boldsymbol{\mu}},\hat{\boldsymbol{\mu}}}(r)` are extinct. For
# more details see the original literature. Below we show the pair correlation function
# as well as the radial and the (static) Kirkwood factor as gray dashed line.


fig, ax = plt.subplots(2)

ax[0].plot(
    rdf_diporder.results.bins,
    rdf_diporder.results.rdf,
)
ax[1].axhline(kirkwood_KF, ls="--", c="gray", label="$g_K$ (KF)")

ax[1].plot(rdf_diporder.results.bins, radial_kirkwood)

ax[0].set(
    xlim=(2, 6),
    ylim=(-0.2, 1.5),
    ylabel=r"$g_\mathrm{\hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\mu}}}(r)$",
)

ax[1].set(
    xlim=(2, 10),
    ylim=(0.95, 3.9),
    xlabel=r"$r\,/\,\mathrm{Å}$",
    ylabel=r"$G_K(r)$",
)

ax[1].legend()

fig.align_labels()
fig.tight_layout()

# %%
# Notice that the Kirkwood Fröhlich estimator for the Kirkwood factors differs from the
# value of :math:`G_K(r=r_K)` obtained from simulations in the MBE ensemble.
#
# Dipole Structure factor
# -----------------------
#
# An alternative approach to calculate the dielectric constant is via the dipole
# structure factor which is given by
#
# .. math:: S(q)_{\hat{\boldsymbol{\mu}} \hat{\boldsymbol{\mu}}} = \left \langle
#     \frac{1}{N} \sum_{i,j=1}^N \hat \mu_i \hat \mu_j \, \exp(-i\boldsymbol q\cdot
#     [\boldsymbol r_i - \boldsymbol r_j]) \right \rangle
#
# We compute the structure factor using the :class:`maicos.DiporderStructureFactor`
# module.


struct_fac = maicos.DiporderStructureFactor(atomgroup=u.atoms, dq=0.05)
struct_fac.run()

# %%
# As also shown :ref:`how to on SAXS calculations <howto-saxs>` the structure factor can
# also be obtained directly from the real space correlation functions using Fourier
# transformation via
#
# .. math:: S_{\hat{\boldsymbol{\mu}} \hat{\boldsymbol{\mu}}}^\mathrm{FT}(q) = 1 + 4 \pi
#     \rho \int_0^\infty \mathrm{d}r r \frac{\sin(qr)}{q} g_{\hat \mu\hat \mu}(r)\,,
#
# which can be obtained by the function
# :func:`maicos.lib.math.compute_rdf_structure_factor`. We have assumed an isotropic
# system so that :math:`S(\boldsymbol q) = S(q)`. Note that we added a one to the dipole
# pair correlation function due to the implementation of the Fourier transformation
# inside :func:`maicos.lib.math.compute_rdf_structure_factor`.


q_rdf, struct_fac_rdf = compute_rdf_structure_factor(
    rdf=1 + rdf_diporder.results.rdf, r=rdf_diporder.results.bins, density=density
)

# %%
# Before we plot the structure factors we first also fit the low :math:`q` limit
# according to a quadratic function as
#
# .. math:: S_\mathrm{\hat \mu\hat \mu}(q\rightarrow0) \approx S_0 + S_2q^2
#
# The fit contains no linear term because of the structure factors' symmetry around 0.

n_max = 5  # take `n_max` first data points of the structure factor for the fit

# q_max is the maximal q value corresponding to the last point taken for the fit
q_max = struct_fac.results.q[n_max]
print(f"q_max = {q_max:.2f} Å")

eps_fit = np.polynomial.Polynomial.fit(
    x=struct_fac.results.q[:n_max],
    y=struct_fac.results.struct_factor[:n_max],
    deg=(0, 2),
    domain=(-q_max, q_max),
)

print(
    f"Best fit parameters: S_0 = {eps_fit.coef[0]:.2f}, S_2 = {eps_fit.coef[2]:.2f} Å^2"
)


# %%
# Now we can finally plot the structure factor

plt.plot(
    struct_fac.results.q,
    struct_fac.results.struct_factor,
    label=r"$S_{\hat \mu\hat \mu}$",
)
plt.plot(
    q_rdf, struct_fac_rdf, ls="dashed", label=r"$S_{\hat \mu\hat \mu}^\mathrm{FT}$"
)
plt.plot(*eps_fit.linspace(50), ls="dotted", label=r"$S_0 + S_2 q^2$")

plt.axhline(1, ls=":", c="gray")
plt.ylabel(r"$S_\mathrm{\hat\mu \hat\mu}(q)$")
plt.xlabel(r"q / $Å^{-1}$")
plt.tight_layout()
plt.xlim(0, 5)
plt.legend()
plt.show()

# %%
# You see that the orange and the blue curve agree. We also add the fit as a green
# dotted line. From :math:`S_0` we can extract the dielectric constant via
# :footcite:p:`hansen_theory_2006`
#
# .. math:: \frac{\mu^2}{\varepsilon_0} S_0 =
#     \frac{(\varepsilon - 1)(2 \varepsilon + 1)}{\varepsilon}
#
# This formula can be inverted and an estimator for :math:`\varepsilon_S` can be
# obtained as we show below.


def dielectric_constant_struc_fact(S_0: float, molecular_dipole_moment: float) -> float:
    """The dielectric constant calculated from the q->0 limit of the structure factor.

    Parameters
    ----------
    q_0_limit : float
        the q -> 0 limit if the dipololar structure factor
    molecular_dipole_moment : float
        dipole moment of a molecule (eÅ)
    """
    dipole_moment_sq = (
        molecular_dipole_moment
        * scipy.constants.angstrom
        * scipy.constants.elementary_charge
    ) ** 2

    S_limit = (
        dipole_moment_sq
        * S_0
        / scipy.constants.epsilon_0
        / scipy.constants.elementary_charge
        / scipy.constants.angstrom**3
    )

    return (np.sqrt((S_limit) ** 2 + 2 * S_limit + 9) + S_limit + 1) / 4


epsilon_struct_fac = dielectric_constant_struc_fact(
    S_0=eps_fit.coef[0], molecular_dipole_moment=dipole_moment
)
print(f"ɛ_S = {epsilon_struct_fac:.2f}")


# %%
# Which is quite close the value calculated directly from the total dipole fluctuations
# of the simulations :math:`\varepsilon_\mathrm{MBE}\approx69`. This difference may
# result in the very crude fit that is performed and it could be drastically improved by
# a Bayesian fitting method as for example for fitting the Seebeck coefficient from a
# similar structure factor :footcite:p:`drigo_seebeck_2023`.
#
# References
# ----------
# .. footbibliography::
