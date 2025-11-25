#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Advanced - Python interpreter
=============================

The requirements to follow the tutorial are the same than for the basic example.

To start, let us first import Matplotlib, MDAnalysis and MAICoS, load the trajectory
and create our groups.
"""  # noqa: D415
# %%

import logging
import sys

import matplotlib.pyplot as plt
import MDAnalysis as mda

import maicos

u = mda.Universe("slit_flow.tpr", "slit_flow.trr")

group_H2O = u.select_atoms("type OW HW")
group_O = u.select_atoms("type OW")
group_H = u.select_atoms("type HW")
group_NaCl = u.select_atoms("type SOD CLA")

# %%
# Uncertainity estimates
# ----------------------
#
# Let us use the :class:`maicos.DensityPlanar` class to extract the density profile of
# the ``group_H2O`` along the (default) :math:`z` axis by running the analysis:

dplan = maicos.DensityPlanar(group_H2O).run()

zcoor = dplan.results.bin_pos
dens = dplan.results.profile

# MAICoS estimates the uncertainity for each profile. This uncertainity is stored inside
# the `dprofile` attribute.

uncertainity = dplan.results.dprofile

# Let us plot the results also showing the uncertainities

fig, ax = plt.subplots()

ax.errorbar(zcoor, dens, 5 * uncertainity)

ax.set_xlabel(r"z coordinate ($\rm Å$)")
ax.set_ylabel(r"density H2O ($\rm u \cdot Å^{-3}$)")


fig.show()

# %%
# For this example we scale the error by 5 to be visible in the plot.
#
# The uncertainity estimatation assumes that the trajectory data is uncorraleted. If the
# correlation time is too high or not reasonably computable a warning occurs that the
# uncertainity estimatation might be unreasonable.

maicos.DensityPlanar(group_H2O).run(start=10, stop=13, step=1)

# %%
# Improving the Results
# ---------------------
#
# By changing the value of the default parameters, one can improve the results, and
# perform more advanced operations.
#
# Let us increase the spatial resolution by reducing the ``bin_width``, and extract two
# profiles instead of one:
#
# * one for the oxygen atoms of the water molecules,
# * one from the hydrogen atoms:

dplan_smaller_bin = []
for ag in [group_O, group_H]:
    dplan_smaller_bin.append(
        maicos.DensityPlanar(ag, bin_width=0.5, unwrap=False).run()
    )

zcoor_smaller_bin_O = dplan_smaller_bin[0].results.bin_pos
dens_smaller_bin_O = dplan_smaller_bin[0].results.profile

zcoor_smaller_bin_H = dplan_smaller_bin[1].results.bin_pos
dens_smaller_bin_H = dplan_smaller_bin[1].results.profile

# %%
# Let us plot the results using two differents :math:`y`-axis:

fig, ax1 = plt.subplots()

ax1.plot(zcoor_smaller_bin_O, dens_smaller_bin_O, label=r"Oxygen")
ax2 = ax1.twinx()
ax2.plot(zcoor_smaller_bin_H, dens_smaller_bin_H, label=r"Hydrogen", color="orange")

ax1.set_xlabel(r"z coordinate ($Å$)")
ax1.set_ylabel(r"density O ($\rm u \cdot Å^{-3}$)")

ax2.set_ylabel(r"density H ($\rm u \cdot Å^{-3}$)")
ax1.legend()
ax2.legend(loc=1)

fig.show()

# %%
# AnalysisCollection example
# --------------------------
#
# When multiple analysis are done on the same trajectory, using AnalysisCollection
# can lead to a speedup compared to running the individual analyses, since the
# trajectory loop is performed only once. Let us imagine that we want to compute
# the density of both the oxygen and hydrogen group:

dplan_array = []
for ag in [group_O, group_H]:
    dplan_array.append(maicos.DensityPlanar(ag, unwrap=False))

# %%
# Now instead of running them in a loop, we will create an AnalysisCollection object:

collection = maicos.core.AnalysisCollection(*dplan_array).run()

# %%
# Now we can now access the result as before:
zcoor_smaller_bin_O = dplan_smaller_bin[0].results.bin_pos
dens_smaller_bin_O = dplan_smaller_bin[0].results.profile

zcoor_smaller_bin_H = dplan_smaller_bin[1].results.bin_pos
dens_smaller_bin_H = dplan_smaller_bin[1].results.profile

# %%
# Access to all the Module's Options
# ----------------------------------
#
# For each MAICoS module, they are several parameters similar to ``bin_width``. The
# parameter list and default options are listed in the :ref:`module's documentation
# <DielectricPlanar>`, and can be gathered by calling the help function of Python:

help(maicos.DensityPlanar)

# %%
# Here we can see that for :class:`maicos.DensityPlanar`, there are several possible
# options such as ``zmin``, ``zmax`` (the minimal and maximal coordinates to consider),
# or ``refgroup`` (to perform the binning with respect to the center of mass of a
# certain group of atoms).
#
# Knowing this, let us re-calculate the density profile of :math:`\mathrm{H_2O}`, but
# this time using the group ``group_H2O`` as a reference for the center of mass:

dplan_centered_H2O = maicos.DensityPlanar(
    group_H2O, bin_width=0.5, refgroup=group_H2O, unwrap=False
)
dplan_centered_H2O.run()
zcoor_centered_H2O = dplan_centered_H2O.results.bin_pos
dens_centered_H2O = dplan_centered_H2O.results.profile

# %%
# Let us also extract the density profile for the NaCl walls, but centered with respect
# to the center of mass of the :math:`\mathrm{H_2O}` group:

dplan_centered_NaCl = maicos.DensityPlanar(
    group_NaCl, bin_width=0.5, refgroup=group_H2O, unwrap=False
)
dplan_centered_NaCl.run()
zcoor_centered_NaCl = dplan_centered_NaCl.results.bin_pos
dens_centered_NaCl = dplan_centered_NaCl.results.profile

# %%
# An plot the two profiles with different :math:`y`-axis:

fig, ax1 = plt.subplots()

ax1.plot(zcoor_centered_H2O, dens_centered_H2O, label=r"$\rm H_2O$")
ax1.plot(zcoor_centered_NaCl, dens_centered_NaCl / 5, label=r"$\rm NaCl$")

ax1.set_xlabel(r"z coordinate ($Å$)")
ax1.set_ylabel(r"density O ($\rm u \cdot Å^{-3}$)")
ax1.legend()

ax2 = ax1.twinx()
ax2.set_ylabel(r"density NaCl ($\rm u \cdot Å^{-3}$)")

fig.show()

# %%
# Additional Options
# ------------------
#
# Use ``verbose=True`` to display extra informations and a progress bar:

dplan_verbose = maicos.DensityPlanar(group_H2O)
dplan_verbose.run(verbose=True)

# %%
# MAICoS uses Python's `standard logging library
# <https://docs.python.org/3/library/logging.html>`_ to display additional informations
# during the analysis of your trajectory. If you also want to show the `DEBUG` messages
# you can configure the logger accordingly.

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
dplan_verbose.run(verbose=True)
logging.disable(logging.WARNING)

# %%
# For additional options take a look at the `HOWTO
# <https://docs.python.org/3/howto/logging.html>`_ for the logging library.
#
# To analyse only a subpart of a trajectory file, for instance to analyse only frames 2,
# 4, 6, 8, and 10, use the ``start``, ``stop``, and ``step`` keywords as follow:

dplan = maicos.DensityPlanar(group_H2O).run(start=10, stop=20, step=2)
