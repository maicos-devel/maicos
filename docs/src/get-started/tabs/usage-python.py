#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Basic usage - Python interpreter
================================

# .. import

To tart, let us first import ``Matplotlib``, ``MDAnalysis`` and ``MAICoS``
"""  # noqa: D415

# %%

import matplotlib.pyplot as plt
import MDAnalysis as mda

import maicos

# %%
#
# .. loading
#
# We first create an :class:`MDAnalysis.core.universe.Universe` by loading a topology
# and trajectory from disk.

u = mda.Universe("slit_flow.tpr", "slit_flow.trr")
# %%
#
# Let us print a few information about the trajectory:

print(f"Number of frames in the trajectory is {u.trajectory.n_frames}.")
timestep = round(u.trajectory.dt, 2)
print(f"Time interval between two frames is {timestep} ps.")
total_time = round(u.trajectory.totaltime, 2)
print(f"Total simulation time is {total_time} ps.")
# %%
#
# .. selection
#
# Now, we define an atom group containing the oxygen and the hydrogen atoms:
#
# .. start_basic_group_py

group_H2O = u.select_atoms("type OW HW")

# %%
#
# Let us print a few information about the groups

print(f"Number of water molecules is {group_H2O.n_atoms // 3}.")

# %%
#
# .. analysis

dplan = maicos.DensityPlanar(group_H2O).run()

# %%
#
# The warning starting with *Unwrapping* is expected and can be ignored for now.
# Let us extract the bin coordinates :math:`z`, the averaged density profile and its
# uncertainty estimated by MAICoS from the ``results`` attribute:

zcoor = dplan.results.bin_pos
dens = dplan.results.profile
uncertainity = dplan.results.dprofile

# %%
#
# The density profile is given as a 1D array, let us look at the 10 first lines:

print(dens[:10])

# %%
#
# By default the ``bin_width`` is 1 Å, and the unit is atomic mass per :math:`Å^3`
# (:math:`\text{u}/\text{Å}^3`).
#
# .. plot
#
# Using ``Matplotlib``:

fig, ax = plt.subplots()

ax.errorbar(zcoor, dens, 5 * uncertainity)

ax.set_xlabel(r"z coordinate ($\rm Å$)")
ax.set_ylabel(r"density H2O ($\rm u \cdot Å^{-3}$)")

fig.show()

# %%
#
# .. help
#
# The general help of MAICoS can be accessed using

help(maicos)
# %%
#
# Package-specific page can also be accessed

help(maicos.DensityPlanar)

# %%
#
# .. end
