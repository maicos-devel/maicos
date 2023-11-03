#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""
.. _howto-saxs:

Small-angle X-ray scattering
============================

Small-angle X-ray scattering (SAXS) can be extracted using MAICoS. To follow this how-to
guide, you should download the :download:`topology <../../static/water/water.tpr>` and
the :download:`trajectory <../../static//water/water.trr>` files of the water system.

First, we import Matplotlib, MDAnalysis, NumPy and MAICoS:
"""
# %%

import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF

import maicos
from maicos.lib.math import compute_form_factor, compute_rdf_structure_factor


# %%
# The `water` system consists of 510 water molecules in the liquid state. The
# molecules are placed in a periodic cubic cell with an extent of :math:`25 \times 25
# \times 25\,\textrm{Å}^3`.
#
# Load Simulation Data
# --------------------
#
# Create a :class:`MDAnalysis.core.universe.Universe` and define a group containing only
# the oxygen atoms and a group containing only the hydrogen atoms:

u = mda.Universe("water.tpr", "water.trr")

group_O = u.select_atoms("type O*")
group_H = u.select_atoms("type H*")

# %%
# Extract small angle x-ray scattering (SAXS) intensities
# -------------------------------------------------------
#
# Let us use the :class:`maicos.Saxs` class of MAICoS and apply it to all atoms in the
# system:

saxs = maicos.Saxs(u.atoms).run(stop=30)

# %%
# .. Note::
#   SAXS computations are extensive calculations. Here, to get an overview of the
#   scattering intensities, we reduce the number of frames to be analyzed from ``101``
#   to ``30``, by adding the ``stop = 30`` parameter to the ``run`` method. Due to the
#   small number of analyzed frames, the scattering intensities shown in this tutorial
#   should not be used to draw any conclusions from the data.
#
# Extract the :math:`q` values and the averaged SAXS scattering intensities
# ``scat_factor`` from the ``results`` attribute:

q_vals = saxs.results.q
scat_factor = saxs.results.scat_factor

# %%
# The scattering factors are given as a 1D array, let us look at the 10 first lines:

print(scat_factor[:10])

# %%
# By default, the binwidth in the recipocal :math:`(q)` space is :math:`0.1 Å^{-1}`.
#
# Plot the structure factors profile using:

fig1, ax1 = plt.subplots()

ax1.plot(q_vals, scat_factor)

ax1.set_xlabel(r"q (1/Å)")
ax1.set_ylabel(r"S(q) (arb. units)")

fig1.show()


# %%
# Computing oxygen and hydrogen contributions
# -------------------------------------------
#
# An advantage of full atomistic simulations is their ability to investigate atomic
# contributions individually. Let us calculate both oxygen and hydrogen contributions,
# respectively:

saxs_O = maicos.Saxs(group_O).run(stop=30)
saxs_H = maicos.Saxs(group_H).run(stop=30)

# %%
# Let us plot the results together with the full scattering intensity. Note that here
# we access the results directly from the ``results`` attribute without storing them in
# individual variables before:

fig2, ax2 = plt.subplots()

ax2.plot(q_vals, scat_factor, label="Water")
ax2.plot(saxs_O.results.q, saxs_O.results.scat_factor, label="Oxygen")
ax2.plot(saxs_H.results.q, saxs_H.results.scat_factor, label="Hydrogen")

ax2.set_xlabel(r"q (1/Å)")
ax2.set_ylabel(r"S(q) (arb. units)")
ax2.legend()

fig2.show()

# %%
# Connection of the structure factor to the radial distribution function
# ----------------------------------------------------------------------
#
# As in details explained in :ref:`saxs-explanations`, the structure factor can be
# related to the radial distrubution function (RDF). We denote this structure factor by
# :math:`S^\mathrm{FT}(q)` since it based on Fourier transforming the RDF. The structure
# factor which can be directly obtaine from the trajectory is denoted by
# :math:`S^\mathrm{D}(q)`.
#
# To relate these two we first calculate the oxygen-oxygen RDF up to half the box length
# using :class:`MDAnalysis.analysis.rdf.InterRDF` and save the result in
# variables for an easier access.

box_lengh = u.dimensions[0]

oo_inter_rdf = InterRDF(
    g1=group_O, g2=group_O, range=(0, box_lengh / 2), exclude_same="residue"
).run()

r_oo = oo_inter_rdf.results.bins
rdf_oo = oo_inter_rdf.results.rdf

# %%
# We use ``exclude_same="residue"`` to exclude atomic self contributions resulting in a
# large peak at 0. Next, we convert the RDF into a structure factor using
# :func:`maicos.lib.math.compute_rdf_structure_factor` and the number density of the
# oxygens.

density = group_O.n_atoms / u.trajectory.ts.volume

q_rdf, struct_factor_rdf = compute_rdf_structure_factor(
    rdf=rdf_oo, r=r_oo, density=density
)

# %%
# Before we can compare we have to normalize the structure factor from the RDF by the
# form factor using :func:`maicos.lib.math.compute_form_factor`.

struct_factor_rdf *= compute_form_factor(q_rdf, "O") ** 2

# %%
# Now we can plot everything together and find that the direct evalation from above and
# the transformed RDF give the same structure factor.

fig3, ax3 = plt.subplots(2, layout="constrained")

ax3[0].axhline(1, c="gray", ls="dashed")
ax3[0].plot(r_oo, rdf_oo, label="Oxygen-Oxygen")
ax3[0].set_xlabel("r / Å")
ax3[0].set_ylabel("g(r)")
ax3[0].set_xlim(0, 10)

ax3[1].plot(q_rdf, struct_factor_rdf, label=r"$S^\mathrm{FT}$")
ax3[1].plot(
    saxs_O.results.q,
    saxs_O.results.scat_factor,
    label=r"$S^\mathrm{D}$",
    ls="dashed",
)

ax3[1].set_xlabel("q (1/Å)")
ax3[1].set_ylabel("S(q) (arb. units)")
ax3[1].set_xlim(0, 7)

ax3[1].legend()
ax3[0].legend()

fig3.show()

# %%
