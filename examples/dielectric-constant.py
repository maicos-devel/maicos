#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Dielectric constant
===================
"""
# %%
#
# list prerequisite and Initialisation


# import warnings
# warnings.filterwarnings("ignore")

# %%
#
# and import MAICoS, NumPy, MDAnalysis, and PyPlot:

# import maicos
# import numpy as np
# import MDAnalysis as mda
# import matplotlib.pyplot as plt
# from matplotlib.ticker import AutoMinorLocator

# %%
#
# Let us set a few parameters for plotting purpose:

# fontsize = 25
# font = {'family': 'sans', 'color':  'black',
#        'weight': 'normal', 'size': fontsize}
# my_color_1 = np.array([0.090, 0.247, 0.560])
# my_color_2 = np.array([0.235, 0.682, 0.639])
# my_color_3 = np.array([1.000, 0.509, 0.333])
# my_color_4 = np.array([0.588, 0.588, 0.588])
# plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.serif": ["Palatino"],
# })

# %%
#
# Define the path to the XXX data folder of MAICoS (the path may be different,
# depending on where your jupyter notebook or python script is located):

# datapath = "mypath"


# [Describe system here]
#
# [Insert image system here]

# %%
#
# Create a MDAnalysis universe

# u = mda.Universe(datapath+'topol.tpr',
#                 datapath+'traj.trr')
# group_H2O = u.select_atoms('type OW HW')

# %%
#
# Extract XXX using MAICoS


# %%
#
# By default the bin_width is 1 Å, the unit is atomic mass per Ångstrom$^3$
# ($\text{u}/\text{Å}^3$),
# and the axis is $z$.
#
# Plot the density profile using :

# fig = plt.figure(figsize=(13,6.5))
# ax1 = plt.subplot(1, 1, 1)
# plt.xlabel(r"z coordinate (Å)", fontdict=font)
# plt.ylabel(r"density H2O (u / Å$^3$)]", fontdict=font)
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# ax1.plot(X, Y, color=my_color_1, linewidth=4)
# ax1.yaxis.offsetText.set_fontsize(20)
# ax1.minorticks_on()
# ax1.tick_params('both', length=10, width=2, which='major', direction='in')
# ax1.tick_params('both', length=6, width=1.4, which='minor', direction='in')
# ax1.xaxis.set_ticks_position('both')
# ax1.yaxis.set_ticks_position('both')
# ax1.spines["top"].set_linewidth(2)
# ax1.spines["bottom"].set_linewidth(2)
# ax1.spines["left"].set_linewidth(2)
# ax1.spines["right"].set_linewidth(2)
# ax1.yaxis.offsetText.set_fontsize(30)
# minor_locator_y = AutoMinorLocator(2)
# ax1.yaxis.set_minor_locator(minor_locator_y)
# minor_locator_x = AutoMinorLocator(2)
# ax1.xaxis.set_minor_locator(minor_locator_x)
# ax1.tick_params(axis='x', pad=10)
# plt.show()
