#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
""".. _usage-python:

Usage - Python interpreter
##########################

To follow this tutorial, it is assumed that MAICoS has been :ref:`installed
<label_installation>` on your computer.

MAICoS heavily depends on the `MDAnalysis`_ infrastructure for trajectory loading and
atom selection. Here we will only cover a small aspects of the capabilities of
`MDAnalysis`_. If you want to learn more about the library, take a look at their
`documentation <https://docs.mdanalysis.org/stable/index.html>`_.

.. _`MDAnalysis`: https://www.mdanalysis.org

Only three MAICoS analysis modules are used in this tutorial
:class:`maicos.DensityPlanar`, :class:`maicos.VelocityPlanar` and
:class:`maicos.DiporderPlanar` but all modules follow the same structure:

1. load your simulation data into an :class:`MDAnalysis.core.universe.Universe`
2. define analysis parameters like bin width or the direction of the analysis
3. after the analysis was succesful, access all results in a
   :class:`MDAnalysis.analysis.base.Results` of the analysis object.

Note that some of the calculations may contain pitfall, such as dielectric profiles
calculation. Potential pitfalls and best practices are listed in the
:ref:`userdoc-how-to` section.

To start, let us first import Matplotlib, MDAnalysis and MAICoS
"""  # noqa: D415
# %%

import logging
import sys

import matplotlib.pyplot as plt
import MDAnalysis as mda

import maicos

# %%
# Load Simulation Data
# ====================
#
# For this tutorial we use a system consisting of a 2D slab with 1176 water molecules
# confined in a 2D slit made of NaCl atoms, where the two water/solid interfaces are
# normal to the axis :math:`z` as shown in the snapshot below:
#
# .. image:: ../../static/slit-flow-dark.png
#   :alt: Snapshot Slit Flow System
#   :class: only-dark
#
# .. image:: ../../static/slit-flow-light.png
#   :alt: Snapshot Slit Flow System
#   :class: only-light
#
# An acceleration :math:`a = 0.05\,\text{nm}\,\text{ps}^{-2}` was applied to the water
# molecules in the :math:`\boldsymbol{e}_x` direction parallel to the NaCl wall, and the
# atoms of the wall were maintained frozen along :math:`\boldsymbol{e}_x`.
#
# We first create an :class:`MDAnalysis.core.universe.Universe` by loading a topology
# and trajectory from disk. You can download the :download:`topology <slit_flow.tpr>`
# and the :download:`trajectory <slit_flow.trr>` from our website.

u = mda.Universe("slit_flow.tpr", "slit_flow.trr")

# %%
# Let us print a few information about the trajectory:

print(f"Number of frames in the trajectory is {u.trajectory.n_frames}.")
timestep = round(u.trajectory.dt, 2)
print(f"Time interval between two frames is {timestep} ps.")
total_time = round(u.trajectory.totaltime, 2)
print(f"Total simulation time is {total_time} ps.")

# %%
# Now, we define four atom groups containing repectively:
#
# 1. the oxygen and the hydrogen atoms (of the water molecules),
# 2. the oxygen atoms (of the water molecules),
# 3. the hydrogen atoms (of the water molecules),
# 4. the Na and Cl atoms (of the wall):

group_H2O = u.select_atoms("type OW HW")
group_O = u.select_atoms("type OW")
group_H = u.select_atoms("type HW")
group_NaCl = u.select_atoms("type SOD CLA")

# %%
# Let us print a few information about the groups

print(f"Number of water molecules is {group_O.n_atoms}.")
print(f"Number of NaCl atoms is {group_NaCl.n_atoms}.")

# %%
# Density Profiles
# ================
#
# Let us use the :class:`maicos.DensityPlanar` class to extract the density profile of
# the ``group_H2O`` along the (default) :math:`z` axis by running the analysis:

dplan = maicos.DensityPlanar(group_H2O).run()

# %%
# The warning starting with *Unwrapping* is perfectly normal and can be ignored for now.
#
# Let us extract the bin coordinates :math:`z` and the averaged density profile from the
# ``results`` attribute:

zcoor = dplan.results.bin_pos
dens = dplan.results.profile

# %%
# The density profile is given as a 1D array, let us look at the 10 first lines:

print(dens[:10])

# %%
# By default the ``bin_width`` is 1 Å, and the unit is atomic mass per :math:`Å^3`
# (:math:`\text{u}/\text{Å}^3`).
#
# Let us plot the density profile using Matplotlib:

fig, ax = plt.subplots()

ax.plot(zcoor, dens)

ax.set_xlabel(r"z coordinate ($\rm Å$)")
ax.set_ylabel(r"density H2O ($\rm u \cdot Å^{-3}$)")

fig.show()

# %%
# Uncertainity estimates
# ----------------------
#
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

# TODO(@PicoCentauri): Intoduce AnalysisCollection here?

zcoor_smaller_bin_O = dplan_smaller_bin[0].results.bin_pos
dens_smaller_bin_O = dplan_smaller_bin[0].results.profile

zcoor_smaller_bin_H = dplan_smaller_bin[0].results.bin_pos
dens_smaller_bin_H = dplan_smaller_bin[0].results.profile

# %%
# Let us plot the results using two differents :math:`y`-axis:

fig, ax1 = plt.subplots()

ax1.plot(zcoor_smaller_bin_O, dens_smaller_bin_O, label=r"Oxygen")
ax1.plot(zcoor_smaller_bin_H, dens_smaller_bin_H * 8, label=r"Hydrogen")

ax1.set_xlabel(r"z coordinate ($Å$)")
ax1.set_ylabel(r"density O ($\rm u \cdot Å^{-3}$)")

ax2 = ax1.twinx()
ax2.set_ylabel(r"density H ($\rm u \cdot Å^{-3}$)")
ax1.legend()

fig.show()

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

# %%
# Velocity Profile
# ================
#
# Here we use the same trajectory file, but extract the velocity profile instead of the
# density profile. Do to so, the :class:`maicos.VelocityPlanar` is used.
#
# Let us call the velocity module:

tplan = maicos.VelocityPlanar(group_H2O, bin_width=0.5, vdim=0, flux=False).run()

zcoor = tplan.results.bin_pos
vel = tplan.results.profile

# %% Here the velocity is extracted along the :math:`x` direction thanks to the ``vdim =
# 0`` option, but the binning is made along the default :math:`z` axis.
#
# And plot the velocity profile:

fig, ax = plt.subplots()

ax.axhline(0, linestyle="dotted", color="gray")
ax.plot(zcoor, vel)

ax.set_xlabel(r"z coordinate ($Å$)")
ax.set_ylabel(r"velocity H2O ($Å ps^{-1}$)")

fig.show()

# %% Water average orientation
# =========================
#
# Finally, still using the same trajectory file, we extract the average orientation of
# the water molecules.
#
# Let us call the :class:`maicos.DiporderPlanar` to extract the average orientation of
# the water molecules:

mydiporder = maicos.DiporderPlanar(
    group_H2O, refgroup=group_H2O, order_parameter="cos_theta"
).run()

# %%
# Then, let us extract the cosinus of the angle of the molecules, :math:`\cos(\theta)`:

zcoor = mydiporder.results.bin_pos
cos_theta = mydiporder.results.profile

fig, ax = plt.subplots()

ax.axhline(0, linestyle="dotted", color="gray")
ax.plot(zcoor, cos_theta)

ax.set_xlabel(r"z coordinate ($Å$)")
ax.set_ylabel(r"$\cos$($\theta$)")

plt.show()
