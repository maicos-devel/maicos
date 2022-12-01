#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later
"""
SAXS
====

Small-angle X-ray scattering (SAXS) can be extracted using MAICoS. To
follow this how-to guide, you should download the
:download:`topology <../../static/water/water.tpr>`
and the
:download:`trajectory <../../static//water/water.trr>`.

First we import Matplotlib, MDAnalysis, NumPy and MAICoS:
"""

import matplotlib.pyplot as plt
import MDAnalysis as mda

import maicos


# %%
#
# The `water` system consists of 510 water molecules in the liquid state.
# The molecules are placed in a periodic cubic cell with an extension of
# :math:`25 \times 25 \times 25\,\textrm{Å}^3`.
#
# Load Simulation Data
# --------------------
#
# Create a :class:`MDAnalysis.core.universe.Universe` and define a
# group containing only the oxygen atoms, and a group containing only the
# hydrogen atoms:

u = mda.Universe('water.tpr', 'water.trr')

group_O = u.select_atoms('type O*')
group_H = u.select_atoms('type H*')

# %%
#
# Extract small angle x-ray scattering (SAXS) intensities
# -------------------------------------------------------
#
# Let us use the :class:`maicos.saxs` class of MAICoS, and apply it to all
# atoms in the systems:

saxs = maicos.Saxs(u.atoms).run(stop=30)

# %%
#
# Note: SAXS computations are extensive calculation. To get an overview of the
# scattering intensities we reduce the number of
# to be analyzed frames from `101` to the first `30` frames of the trajectory
# by adding the ``stop=30`` parameter to the `run` method. Due to the small
# number of analyzed frames, the scattering intensities
# shown in this tutorial should not be used to draw any conclusions
# from the data.
#
# Extract the $q$ values and the averaged saxs scattering intensities
# ``scat_factor`` from the ``results`` attribute:

q_vals = saxs.results.q
scat_factor = saxs.results.scat_factor

# %%
#
# The scattering factors are given as a 1D array, let us look at the 10
# first lines:

print(scat_factor[:10])

# %%
#
# By default the binwidth in q space is 0.005 1/Å
#
# Plot the structure factors profile using:

fig, ax = plt.subplots()

ax.plot(q_vals, scat_factor)

ax.set_xlabel(r"q (1/Å)")
ax.set_ylabel(r"S(q) (arb. units)")

fig.show()

# %%
#
# Computing oxygen and hydrogen contributions
# -------------------------------------------
#
# An advantage of full atomistic simulation is their ability to investigate
# atomic contributions individually. Let us calculate the oxygen and hydrogen
# contribution:

saxs_O = maicos.Saxs(group_O).run(stop=30)
saxs_H = maicos.Saxs(group_H).run(stop=30)

# %%
#
# Let us plot the results together with the full scattering intensity.
# Note that here we access the results directly from the `results` attribute
# without storing them in individual variables before:

fig, ax = plt.subplots()

ax.plot(q_vals, scat_factor, label="Water")
ax.plot(saxs_O.results.q, saxs_O.results.scat_factor, label="Oxygen")
ax.plot(saxs_H.results.q, saxs_H.results.scat_factor, label="Hydrogen")

ax.set_xlabel(r"q (1/Å)")
ax.set_ylabel(r"S(q) (arb. units)")
ax.legend()

fig.show()
