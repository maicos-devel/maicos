#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Ideal component of the chemical potential
#########################################
"""
# %%


import MDAnalysis as mda
import numpy as np
from scipy import constants as const

import maicos


def mu(rho, T, m):
    """Calculate the chemical potential.

    The chemical potential is calculated from the
    density: mu = k_B T log(rho. / m)
    """
    # kT in KJ/mol
    kT = T * const.Boltzmann * const.Avogadro / const.kilo

    # De Broglie (converted to angstrom)
    db = np.sqrt(
        const.h ** 2 / (2 * np.pi * m * const.atomic_mass * const.Boltzmann * T)
        ) / const.angstrom

    if np.all(rho > 0):
        return kT * np.log(rho * db ** 3)
    elif np.any(rho == 0):
        return np.float64("-inf") * np.ones(rho.shape)
    else:
        return np.float64("nan") * np.ones(rho.shape)


def dmu(rho, drho, T):
    """Calculate the error of the chemical potential.

    The error is calculated from the density using propagation of uncertainty.
    """
    kT = T * const.Boltzmann * const.Avogadro / const.kilo

    if np.all(rho > 0):
        return kT * (drho / rho)
    else:
        return np.float64("nan") * np.ones(rho.shape)
# %%


water = mda.Universe('water.tpr', 'water.trr')
ana = maicos.DensityPlanar(water.atoms)
ana.run()
print('µ_id =', mu(ana.results.profile.mean(), 300, 18))
print('Δµ_id =', dmu(ana.results.profile.mean(),
                     ana.results.dprofile.mean(),
                     300))

# %%
