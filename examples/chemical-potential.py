#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""
.. _howto-chemical-potential:


Ideal component of the chemical potential
=========================================

What is the chemical potential?
-------------------------------

Molecular dynamics simulations are often performed with a constant number of particles.
When modelling confined systems in molecular dynamics simulations, it is often assumed
that the confined geometry extends infinitely, while real systems have a finite size and
are connected to a reservior many times larger than the confined space.

In this case, the number of particles in the system is not constant, but changes over
time. This can be seen as a system that is exchanging particles with a reservoir. The
chemical potential describes how the free energy changes when particles are added to (or
removed from) the system. The chemical potential is therefore a very important quantity
in molecular dynamics simulations of confined systems.

If you want to know more about what the chemical potential means you can take a look at
the references below :footcite:p:`UnderstandingChemicalPotential`.

How to calculate the ideal component of the chemical potential
--------------------------------------------------------------

The chemical potential can be split up into different parts

.. math::
  \mu = \mu^0 + \mu^\text{ideal} + \mu^\text{excess},

where :math:`\mu^0` represents the standard potential of the substance,
:math:`\mu^\text{ideal}` represents the component of the potential that would also occur
for an ideal gas and :math:`\mu^\text{excess}` represents the excess contribution
generated from the interactions between the particles. In the following calculations we
are only interested in the ideal component.

For our case, we can calculate the ideal component of the potential according to

.. math::
   \mu^\text{ideal} = R T \ln \left( \rho \Lambda^3 \right),

where :math:`\Lambda = \hbar \sqrt{\frac{2\pi}{m \cdot k_\mathrm{B} \cdot T}}` is the
thermal De-Broglie wavelength, i.e. the mean De-Broglie wavelength at temperature
:math:`T`. Furthermore, :math:`m` is the mass of the particles and :math:`\rho` is the
mean density of the system. The mean density can be calculated with MAICoS by using the
Density modules. We will exemplify this in the following example using the
:class:`maicos.modules.dielectricplanar.DensityPlanar` module.

First we'll import every module we need.
"""
# %%

import MDAnalysis as mda
import numpy as np
from scipy import constants as const

import maicos


# %%
# Now we define a function that calculates :math:`\mu` according to the equation above.
# We can calculate the Volume :math:`V` with MAICoS by calculating the mean density and
# deviding it by the mass of the particles. Therefore our function takes the density as
# input instead of the Volume.


def mu(rho, T, m):
    """Calculate the chemical potential.

    The chemical potential is calculated from the density: mu = R T log(rho. / m)
    """
    # RT in KJ/mol
    RT = T * const.Boltzmann * const.Avogadro / const.kilo

    # De Broglie (converted to angstrom)
    db = (
        np.sqrt(
            const.h**2 / (2 * np.pi * m * const.atomic_mass * const.Boltzmann * T)
        )
        / const.angstrom
    )

    if np.all(rho > 0):
        return RT * np.log(rho * db**3)
    elif np.any(rho == 0):
        return np.float64("-inf") * np.ones(rho.shape)
    else:
        return np.float64("nan") * np.ones(rho.shape)


# %%
# If you're also interested in the error of the chemical potential we can calculate it
# through propagation of uncertainty from the error of the density, calculated by
# MAICoS. The error propagates according to
#
# .. math::
#   \Delta \mu &= \left| \frac{\partial \mu}{\partial \rho} \right| \cdot
#               \Delta \rho \\
#               &= \frac{RT}{\rho} \cdot \Delta \rho.
#
# The implemented function looks like this.


def dmu(rho, drho, T):
    """Calculate the error of the chemical potential.

    The error is calculated from the density using propagation of uncertainty.
    """
    RT = T * const.Boltzmann * const.Avogadro / const.kilo

    if np.all(rho > 0):
        return RT * (drho / rho)
    else:
        return np.float64("nan") * np.ones(rho.shape)


# %%
# Finally, we can use those previously defined functions to calculate the chemical
# potential and its error for an example trajectory called `water`, whose data can be
# downloaded from :download:`topology <../../static/water/water.tpr>` and
# :download:`trajectory <../../static/water/water.trr>`. To calculate the mean density
# we use the module :class:`maicos.modules.dielectricplanar.DensityPlanar` of MAICoS.
# This example uses a temperature of :math:`300 \: \rm K` and a mass of :math:`18 \: \rm
# u`.


water = mda.Universe("water.tpr", "water.trr")
ana = maicos.DensityPlanar(water.atoms)
ana.run()
print("µ_id =", mu(ana.results.profile.mean(), 300, 18))
print("Δµ_id =", dmu(ana.results.profile.mean(), ana.results.dprofile.mean(), 300))

# %%
# References
# ----------
# .. footbibliography::
