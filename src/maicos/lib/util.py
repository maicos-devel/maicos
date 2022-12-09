#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Small helper and utilities functions that don't fit anywhere else."""

import functools
import os
import sys
import warnings
from typing import Callable

import numpy as np
from scipy.signal import find_peaks


_share_path = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                            "..", "share"))


def get_compound(atomgroup, return_index=False):
    """Returns the highest order topology attribute.

    The order is "molecules", "fragments", "residues". If the topology contains
    none of those attributes, an AttributeError is raised. Optionally, the
    indices of the attribute as given by `molnums`, `fragindices` or
    `resindices` respectively are also returned.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        atomgroup taken for weight calculation
    return_index : bool, optional
        If True, also return the indices the corresponding topology attribute.

    Returns
    -------
    compound: string
        Name of the topology attribute.
    index: ndarray, optional
        The indices of the topology attribute.

    Raises
    ------
    AttributeError
        `atomgroup` is missing any connection information"
    """
    if hasattr(atomgroup, "molnums"):
        compound = "molecules"
        indices = atomgroup.atoms.molnums
    elif hasattr(atomgroup, "fragments"):
        warnings.warn("Cannot use 'molecules'. Falling back to 'fragments'")
        compound = "fragments"
        indices = atomgroup.atoms.fragindices
    elif hasattr(atomgroup, "residues"):
        warnings.warn("Cannot use 'fragments'. Falling back to 'residues'")
        compound = "residues"
        indices = atomgroup.atoms.resindices
    else:
        raise AttributeError(
            "Missing any connection information in `atomgroup`.")
    if return_index:
        return compound, indices
    else:
        return compound


def get_cli_input():
    """Return a proper formatted string of the command line input."""
    program_name = os.path.basename(sys.argv[0])
    # Add additional quotes for connected arguments.
    arguments = ['"{}"'.format(arg)
                 if " " in arg else arg for arg in sys.argv[1:]]
    return "{} {}".format(program_name, " ".join(arguments))


def atomgroup_header(AtomGroup):
    """Return a string containing infos about the AtomGroup.

    Infos include the total number of atoms, the including
    residues and the number of residues. Useful for writing
    output file headers.
    """
    if not hasattr(AtomGroup, 'types'):
        warnings.warn("AtomGroup does not contain atom types. "
                      "Not writing AtomGroup information to output.")
        return f"{len(AtomGroup.atoms)} unkown particles"
    unique, unique_counts = np.unique(AtomGroup.types,
                                      return_counts=True)
    return " & ".join(
        "{} {}".format(*i) for i in np.vstack([unique, unique_counts]).T)


def bin(a, bins):
    """Average array values in bins for easier plotting.

    Note: "bins" array should contain the INDEX (integer)
    where that bin begins
    """
    if np.iscomplex(a).any():
        avg = np.zeros(len(bins), dtype=complex)  # average of data
    else:
        avg = np.zeros(len(bins))

    count = np.zeros(len(bins), dtype=int)
    ic = -1

    for i in range(0, len(a)):
        if i in bins:
            ic += 1  # index for new average
        avg[ic] += a[i]
        count[ic] += 1

    return avg / count


doc_dict = dict(
    DENSITY_DESCRIPTION=r"""Calculations are carried out for
    ``mass`` :math:`(\rm u \cdot Å^{-3})`, ``number`` :math:`(\rm Å^{-3})` or
    ``charge`` :math:`(\rm e \cdot Å^{-3})` density profiles along a certain
    cartesian axes ``[x, y, z]`` of the simulation cell. Cell dimensions are
    allowed to fluctuate in time.

    For grouping with respect to ``molecules``, ``residues`` etc., the
    corresponding centers (i.e., center of mass) taking into account periodic
    boundary conditions are calculated.
    For these calculations molecules will be unwrapped/made whole.
    Trajectories containing already whole molecules can be run with
    ``unwrap=False`` to gain a speedup.
    For grouping with respect to atoms the `unwrap` option is always
    ignored.""",
    ATOMGROUP_PARAMETER="""atomgroup : AtomGroup
        A :class:`~MDAnalysis.core.groups.AtomGroup` for which
        the calculations are performed.""",
    ATOMGROUPS_PARAMETER="""atomgroups : list[AtomGroup]
        a list of :class:`~MDAnalysis.core.groups.AtomGroup` objects for which
        the calculations are performed.""",
    BASE_CLASS_PARAMETERS="""refgroup : AtomGroup
        Reference :class:`~MDAnalysis.core.groups.AtomGroup` used for the
        calculation.

        If refgroup is provided, the calculation is
        performed relative to the center of mass of the AtomGroup.

        If refgroup is ``None`` the calculations
        are performed to the center of the (changing) box.
    unwrap : bool
        When ``unwrap = True``, molecules that are broken due to the
        periodic boundary conditions are made whole.

        If the input contains molecules that are already whole,
        speed up the calculation by disabling unwrap. To do so,
        use the flag ``-no-unwrap`` when using MAICoS from the
        command line, or use ``unwrap = False`` when using MAICoS from
        the Python interpreter.

        Note: Molecules containing virtual sites (e.g. TIP4P water
        models) are not currently supported in MDAnalysis.
        In this case, you need to provide unwrapped trajectory files directly,
        and disable unwrap.
        Trajectories can be unwrapped, for example, using the
        ``trjconv`` command of GROMACS.
    concfreq : int
        When concfreq (for conclude frequency) is larger than 0,
        the conclude function is called and the output files are
        written every concfreq frames""",
    PROFILE_CLASS_PARAMETERS_PRIVATE="""weighting_function : callable
        The function calculating the array weights for the histogram analysis.
        It must take an `Atomgroup` as first argument and a
        grouping ('atoms', 'residues', 'segments', 'molecules', 'fragments')
        as second. Additional parameters can
        be given as `f_kwargs`. The function must return a numpy.ndarray with
        the same length as the number of group members.
    normalization : str {'None', 'number', 'volume'}
        The normalization of the profile performed in every frame.
        If `None` no normalization is performed. If `number` the histogram
        is divided by the number of occurences in each bin. If `volume` the
        profile is divided by the volume of each bin.
    f_kwargs : dict
        Additional parameters for `function`""",
    PLANAR_CLASS_PARAMETERS="""dim : int
        Dimension for binning (``x=0``, ``y=1``, ``z=2``).
    zmin : float
        Minimal coordinate for evaluation (in Å) with respect to the
        center of mass of the refgroup.

        If ``zmin=None``, all coordinates down to the lower cell boundary
        are taken into account.
    zmax : float
        Maximal coordinate for evaluation (in Å) with respect to the
        center of mass of the refgroup.

        If ``zmax = None``, all coordinates up to the upper cell boundary
        are taken into account.
    jitter : float
        If ``jitter is not None``, random numbers of the order of jitter
        (Å) are added to the atom positions.

        The appilication of a jitter is rationalized in possible aliasing
        effects when histogramming data, i.e., for spatial profiles. These
        aliasing effects can be stabilized with the application
        of a numerical jitter. The jitter value should be about the precision of
        the trajectory and will not alter the results of the histogram.

        You can estimate the precision of the positions in your trajectory
        with :func:`maicos.lib.util.trajectory_precision`. Note that if the
        precision is not the same for all frames, the smallest precision
        should be used.
        """,
    BIN_WIDTH_PARAMETER="""bin_width : float
        Width of the bins (in Å).""",
    RADIAL_CLASS_PARAMETERS="""rmin : float
        Minimal radial coordinate relative to the center of mass of the
        refgroup for evaluation (in Å).
    rmax : float
        Maximal radial coordinate relative to the center of mass of the
        refgroup for evaluation (in Å).

        If ``rmax=None``, the box extension is taken.""",
    SYM_PARAMETER="""sym : bool
        Symmetrize the profile. Only works in combinations with
        ``refgroup``.""",
    PROFILE_CLASS_PARAMETERS="""grouping : str {``'atoms'``, ``'residues'``, ``'segments'``, ``'molecules'``, ``'fragments'``}"""  # noqa
    """
          Atom grouping for the calculations of profiles.

          The possible grouping options are the atom positions (in
          the case where ``grouping='atoms'``) or the center of mass of
          the specified grouping unit (in the case where
          ``grouping='residues'``, ``'segments'``, ``'molecules'`` or
          ``'fragments'``).
    bin_method : str {``'cog'``, ``'com'``, ``'coc'``}
        Method for the position binning.

        The possible options are center of geometry (``cog``),
        center of mass (``com``), and center of charge (``coc``).
    output : str
        Output filename.""",
    PLANAR_CLASS_ATTRIBUTES="""results.bin_pos : numpy.ndarray
        Bin positions (in Å) ranging from ``zmin`` to ``zmax``.""",
    RADIAL_CLASS_ATTRIBUTES="""results.bin_pos : numpy.ndarray
        Bin positions (in Å) ranging from ``rmin`` to ``rmax``.""",
    PROFILE_CLASS_ATTRIBUTES="""results.profile : numpy.ndarray
        Calculated profile.
    results.dprofile : numpy.ndarray
        Estimated profile's uncertainity."""
    )

# Inherit docstrings
doc_dict["PLANAR_CLASS_PARAMETERS"] = \
    doc_dict["BASE_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["PLANAR_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["BIN_WIDTH_PARAMETER"]

doc_dict["CYLINDER_CLASS_PARAMETERS"] = \
    doc_dict["PLANAR_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["RADIAL_CLASS_PARAMETERS"]

doc_dict["SPHERE_CLASS_PARAMETERS"] = \
    doc_dict["BASE_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["RADIAL_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["BIN_WIDTH_PARAMETER"]

doc_dict["PROFILE_PLANAR_CLASS_PARAMETERS"] = \
    doc_dict["ATOMGROUPS_PARAMETER"] + "\n    " + \
    doc_dict["PLANAR_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["SYM_PARAMETER"] + "\n    " + \
    doc_dict["PROFILE_CLASS_PARAMETERS"]

doc_dict["PROFILE_CYLINDER_CLASS_PARAMETERS"] = \
    doc_dict["ATOMGROUPS_PARAMETER"] + "\n    " + \
    doc_dict["CYLINDER_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["PROFILE_CLASS_PARAMETERS"]

doc_dict["PROFILE_SPHERE_CLASS_PARAMETERS"] = \
    doc_dict["ATOMGROUPS_PARAMETER"] + "\n    " + \
    doc_dict["SPHERE_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["RADIAL_CLASS_PARAMETERS"] + "\n    " + \
    doc_dict["PROFILE_CLASS_PARAMETERS"]

doc_dict["CYLINDER_CLASS_ATTRIBUTES"] = doc_dict["RADIAL_CLASS_ATTRIBUTES"]
doc_dict["SPHERE_CLASS_ATTRIBUTES"] = doc_dict["RADIAL_CLASS_ATTRIBUTES"]

doc_dict["PROFILE_PLANAR_CLASS_ATTRIBUTES"] = \
    doc_dict["PLANAR_CLASS_ATTRIBUTES"] + "\n    " + \
    doc_dict["PROFILE_CLASS_ATTRIBUTES"]

doc_dict["PROFILE_CYLINDER_CLASS_ATTRIBUTES"] = \
    doc_dict["RADIAL_CLASS_ATTRIBUTES"] + "\n    " + \
    doc_dict["PROFILE_CLASS_ATTRIBUTES"]

doc_dict["PROFILE_SPHERE_CLASS_ATTRIBUTES"] = \
    doc_dict["RADIAL_CLASS_ATTRIBUTES"] + "\n    " + \
    doc_dict["PROFILE_CLASS_ATTRIBUTES"]


def _render_docs(func: Callable, doc_dict: dict = doc_dict) -> Callable:
    if func.__doc__ is not None:
        for pattern in doc_dict.keys():
            func.__doc__ = func.__doc__.replace(f"${{{pattern}}}",
                                                doc_dict[pattern])
    return func


def render_docs(func: Callable) -> Callable:
    """Replace all template phrases in the functions docstring.

    Parameters
    ----------
    func : callable
        The callable (function, class) where the phrase old should be replaced.

    Returns
    -------
    Callable
        callable with replaced phrase
    """
    return _render_docs(func, doc_dict=doc_dict)


def charge_neutral(filter):
    """Raise a Warning when AtomGroup is not charge neutral.

    Class Decorator to raise an Error/Warning when AtomGroup in an AnalysisBase
    class is not charge neutral. The behaviour of the warning can be controlled
    with the filter attribute. If the AtomGroup's corresponding universe is
    non-neutral an ValueError is raised.

    Parameters
    ----------
    filter : str
        Filter type to control warning filter Common values are: "error"
        or "default" See `warnings.simplefilter` for more options.
    """
    def inner(original_class):
        def charge_check(function):
            @functools.wraps(function)
            def wrapped(self):
                if hasattr(self, 'atomgroup'):
                    groups = [self.atomgroup]
                else:
                    groups = self.atomgroups
                for group in groups:
                    if not np.allclose(
                            group.total_charge(compound=get_compound(group)),
                            0, atol=1E-4):
                        with warnings.catch_warnings():
                            warnings.simplefilter(filter)
                            warnings.warn("At least one AtomGroup has free "
                                          "charges. Analysis for systems "
                                          "with free charges could lead to "
                                          "severe artifacts!")

                    if not np.allclose(group.universe.atoms.total_charge(), 0,
                                       atol=1E-4):
                        raise ValueError(
                            "Analysis for non-neutral systems is not supported."
                            )
                return function(self)

            return wrapped

        original_class._prepare = charge_check(original_class._prepare)

        return original_class

    return inner


def trajectory_precision(trajectory, dim=2):
    """Detect the precision of a trajectory.

    Parameters
    ----------
    trajectory : MDAnalysis trajectory
        Trajectory from which the precision is detected.
    dim : int, optional
        Dimension along which the precision is detected.

    Returns
    -------
    precision : array
        Precision of each frame of the trajectory.

        If the trajectory has a high precision, its resolution will not be
        detected, and a value of 1e-4 is returned.
    """
    # The threshold will limit the precision of the
    # detection. Using a value that is too low will end up
    # costing a lot of memory.
    # 1e-4 is enough to safely detect the resolution of
    # format like XTC
    threshold_bin_width = 1e-4
    precision = np.zeros(trajectory.n_frames)
    # to be done, add range=(0, -1, 1) parameter
    # for ts in trajectory[range[0]:range[1]:range[2]]:
    for ts in trajectory:
        n_bins = int(np.ceil(
            (np.max(trajectory.ts.positions[:, dim])
             - np.min(trajectory.ts.positions[:, dim])) / threshold_bin_width))
        hist1, z = np.histogram(trajectory.ts.positions[:, dim], bins=n_bins)
        hist2, bin_edges, = np.histogram(np.diff(z[np.where(hist1)]),
                                         bins=1000, range=(0, 0.1))
        if len(find_peaks(hist2)[0]) == 0:
            precision[ts.frame] = 1e-4
        elif bin_edges[find_peaks(hist2)[0][0]] <= 5e-4:
            precision[ts.frame] = 1e-4
        else:
            precision[ts.frame] = bin_edges[find_peaks(hist2)[0][0]]
    return precision


# All references associated with code are registered here in the format
# doi : Short Reference
DOI_LIST = {"10.1103/PhysRevLett.117.048001":
            "Schlaich, A. et al., Phys. Rev. Lett. 117, (2016).",
            "10.1021/acs.jpcb.9b09269":
            "Loche, P. et al., J. Phys. Chem. B 123, (2019).",
            "10.1021/acs.jpca.0c04063":
            "Carlson, S. et al., J. Phys. Chem. A 124, (2020).",
            "10.1103/PhysRevE.92.032718":
            "1. Schaaf, C. et al., Phys. Rev. E 92, (2015)."}


def citation_reminder(*dois):
    """Prints citations in order to remind users to give due credit.

    Parameters
    ----------
    dois : list of dois associated with the method which calls this.
        These have to be registered here.

    """
    txt = ''
    for doi in dois:
        citation = DOI_LIST[doi]
        txt += f"{citation}; doi:{doi}\n"

    txt += ("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
            + "If you use this module in your work, please read and cite:"
            + f"\n{txt}"
            + "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    return txt
