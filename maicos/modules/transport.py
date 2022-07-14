#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tool for computing transport properties.

The transport module of MAICoS allows for calculating mean velocity
profiles from molecular simulation trajectory files.
"""

from ..decorators import set_profile_planar_class_doc, set_verbose_doc
from .base import ProfilePlanarBase


def _velocity_weights(atomgroup, grouping, dim, vdim, flux):
    """Calculate the weights for the velocity histogram."""
    atom_vels = atomgroup.velocities[:, vdim]

    if grouping == "atoms":
        vels = atom_vels
    else:
        mass_vels = atomgroup.atoms.accumulate(
            atom_vels * atomgroup.atoms.masses, compound=grouping)
        group_mass = atomgroup.atoms.accumulate(
            atomgroup.atoms.masses, compound=grouping)
        vels = mass_vels / group_mass

    # either normalised by the number of compound (to get the velocity)
    # or do not normalise to get the flux (velocity x number of compound)
    if not flux:
        vels /= len(vels)
    return vels


@set_verbose_doc
@set_profile_planar_class_doc
class Velocity(ProfilePlanarBase):
    """Analyse mean velocity.

    Reads in coordinates and velocities from a trajectory and calculates a
    velocity profile along a given axis.

    Parameters
    ----------
    ${PLANAR_PROFILE_CLASS_PARAMETERS}
    vdim : int {0, 1, 2}
        Dimension for velocity binning (x=0, y=1, z=2)
    flux : bool
        Do not normalise the velocity to get the flux,
        i.e. the velocity multiplied by the number of compounds
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    ${PLANAR_PROFILE_CLASS_ATTRIBUTES}
    results.z : np.ndarray
        bins [nm]
    results.v : np.ndarray
        velocity profile [Å/ps]
    results.ees : np.ndarray
        velocity error estimate [Å/ps]
    results.symz : np.ndarray
        symmetrized bins [Å/ps]
    results.symvel : np.ndarray
        symmetrized velocity profile [Å/ps]
    results.symees : np.ndarray
        symmetrized velocity error estimate [Å/ps]
    """

    def __init__(self,
                 atomgroups,
                 dim=2,
                 zmin=0,
                 zmax=None,
                 binwidth=1,
                 center=False,
                 comgroup=None,
                 sym=False,
                 grouping="atoms",
                 make_whole=True,
                 binmethod="com",
                 output="velocity.dat",
                 concfreq=0,
                 vdim=2,
                 flux=False,
                 **kwargs):

        if vdim not in [0, 1, 2]:
            raise ValueError("Velocity dimension can only be x=0, y=1 or z=2.")

        super(Velocity, self).__init__(
            function=_velocity_weights,
            f_kwargs={"vdim": vdim, "flux": flux},
            normalization="volume",
            atomgroups=atomgroups,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            binwidth=binwidth,
            center=center,
            comgroup=comgroup,
            sym=sym,
            grouping=grouping,
            make_whole=make_whole,
            binmethod=binmethod,
            output=output,
            concfreq=concfreq,
            **kwargs)
