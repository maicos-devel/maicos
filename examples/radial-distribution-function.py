#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Radial distribution function
============================

Basic usage
-----------

In the following example, we will show how to calculate the two-dimensional planar
radial distribution functions.

In the following, we will give an example of a trajectory of water confined by graphene
sheets simulated via GROMACS. We assume that the GROMACS topology is given by
`graphene_water.tpr` and the trajectory is given by `graphene_water.xtc`. Both can be
downloaded under :download:`topology <../../static/graphene_water/graphene_water.tpr>`
and :download:`trajectory <../../static/graphene_water/graphene_water.xtc>`,
respectively.

From these files you can create a MDAnalysis universe object.

We begin by importing the necessary modules.
"""
# %%

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np

import maicos


# %%
#
# Next, we proceed with the creation of a MDAnalysis universe object, from
# which we further select the water molecules using the `resname` selector.

u = mda.Universe("./graphene_water.tpr", "graphene_water.xtc")

# %%
# This universe object can then be passed to the planar radial distribution
# analysis object, documented in
# :class:`maicos.modules.rdfplanar.RDFPlanar`.
# It expects you to pass the atom groups you want to perform the analysis for.
# In our example, we have graphene walls and SPC/E water confined between them,
# where we are interested in the dielectric behavior of the fluid.
# Thus, we will first select the water as an MDAnalysis atom group using
# :meth:`MDAnalysis.core.groups.AtomGroup.select_atoms`. In this case we select
# the water by filtering for the residue named ``SOL``.

water = u.select_atoms("resname SOL")

ana_obj = maicos.RDFPlanar(
    water,
    water,
    dzheight=0.25,
    dim=2,
    rdf_bin_width=0.2,
    refgroup=water,
    zmin=-5.0,
    zmax=0,
)

# %%
# Next, we can run the analysis over the trajectory.
# To this end we call the member function
# :meth:`run <maicos.modules.rdfplanar.RDFPlanar.run>`.
# We may set the ``verbose`` keyword to ``True`` to get additional information
# such a a progress bar.
#
# Here you also have the chance to set ``start`` and ``stop`` keywords to
# specify which frames the analysis should start at and where to end.
# One can also specify a ``step`` keyword to only analyze every ``step``
# frames.

ana_obj.run(verbose=True, step=1)

# %%
# We also calculate the density profile of the water molecules in order to
# compare the different slabs with the layering visible in the density.

dana_obj = maicos.DensityPlanar(
    water, dim=2, refgroup=water, bin_width=0.1, sym=True, zmin=-7, zmax=7
)

dana_obj.run(verbose=True, step=10)

# %%
# The results of the analysis are stored in the ``results`` member variable.
# As per the documentation of ``RDFPlanar``, we get three different arrays:
# ``bin_pos``, ``bins``, and ``rdf``.
# Here, ``bin_pos`` is the position of the center of the slices in the
# z-direction, ``bins`` contains the bin positions of the radial distribution,
# which are shared by all slices and correspondingly ``rdf`` contains each
# profile that our code produced.
#
# In the following, we loop over all the rdf slices and plot each of them.
# Furthermore, in a separate subplot, we also show the density profile of the
# water molecules and highlight the slices that each rdf is calculated for.
# Hence, the same color in both plots corresponds to the same slice for the
# radial distribution function and the density profile.
# %%

# u per cubic angstrom to kg per cubic meter factor
u2kg = 1660.5390665999998

fig, ax = plt.subplots(1, 2)
print(ax)

tax = ax[1].twinx()
shift = 0
shift_amount = 2
for i in range(0, len(ana_obj.results.rdf[0])):
    bin_pos = ana_obj.results.bin_pos[i]

    rdf_prof = ana_obj.results.rdf[:, i]
    mean_bulk = np.mean(rdf_prof[ana_obj.results.bins > 10])

    line = ax[0].plot(
        ana_obj.results.bins, ana_obj.results.rdf[:, i] / mean_bulk + shift
    )
    tax.vlines(
        7 + bin_pos, 0, 3500, alpha=0.7, color=line[0].get_color(), linestyles="dashed"
    )

    tax.axvspan(
        7 + bin_pos - 0.25 * 2,
        7 + bin_pos + 0.25 * 2,
        color=line[0].get_color(),
        alpha=0.3,
    )
    shift += shift_amount

ax[0].set_ylabel(r"$g(r)$")
ax[0].set_xlabel(r"$r$ [$\AA$]")
ax[0].set_xlim((0, 15))
ax[0].hlines(1, 0, 15, color="black", linestyles="dashed", alpha=0.5)

tax.plot(
    7 + dana_obj.results.bin_pos,
    dana_obj.results.profile * u2kg,
    color="black",
    label="Density",
)
tax.set_xlim((1, 7))

ax[1].set_yticks(tax.get_yticks())

ax[1].set_yticklabels([])

tax.set_ylabel(r"$\rho(z)$ [kg/m$^3$]")
ax[1].set_xlabel(r"$z$ [$\AA$]")

# Set the padding between the axis to zero
plt.tight_layout()

fig.subplots_adjust(wspace=0, hspace=0)
fig.dpi = 200
