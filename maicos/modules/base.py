#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import logging
from ..utils import sort_atomgroup

import numpy as np
import MDAnalysis.analysis.base

from ..decorators import set_verbose_doc, set_planar_class_doc

logger = logging.getLogger(__name__)

@set_verbose_doc
class AnalysisBase(MDAnalysis.analysis.base.AnalysisBase):
    """Base class derived from MDAnalysis for defining multi-frame analysis.

    The class is designed as a template for creating multi-frame analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating, and it offers to show a progress meter.
    Computed results are stored inside the :attr:`results` attribute.

    To define a new analysis, `AnalysisBase` needs to be subclassed
    and :meth:`_single_frame` must be defined. It is also possible to define
    :meth:`_prepare` and :meth:`_conclude` for pre- and post-processing.
    All results should be stored as attributes of the :class:`Results`
    container.

    Parameters
    ----------
    atomgroups : Atomgroup or list[Atomgroup]
        Atomgroups taken for the Analysis
    multi_group : bool
        Analysis is able to work with list of atomgroups
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    atomgroup : mda.Atomgroup
        Atomgroup taken for the Analysis (available if `multi_group = False`)
    atomgroups : list[mda.Atomgroup]
        Atomgroups taken for the Analysis (available if `multi_group = True`)
    n_atomgroups : int
        Number of atomngroups (available if `multi_group = True`)
    _universe : mda.Universe
        The Universe the atomgroups belong to
    _trajectory : mda.trajectory
        The trajetcory the atomgroups belong to
    times : numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`AnalysisBase.run`
    frames : numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`AnalysisBase.run`
    results : :class:`Results`
        results of calculation are stored after call
        to :meth:`AnalysisBase.run`
    """
    def __init__(self,
                 atomgroups,
                 multi_group=False,
                 verbose=False,
                 **kwargs):
        if multi_group:
            if type(atomgroups) not in (list, tuple):
                atomgroups = [atomgroups]
            # Check that all atomgroups are from same universe
            if len(set([ag.universe for ag in atomgroups])) != 1:
                raise ValueError("Atomgroups belong to different Universes")

            # Sort the atomgroups,
            # such that molecules are listed one after the other
            self.atomgroups = list(map(sort_atomgroup, atomgroups))
            self.n_atomgroups = len(self.atomgroups)
            self._universe = atomgroups[0].universe
            self._allow_multiple_atomgroups = True
        else:
            self.atomgroup = sort_atomgroup(atomgroups)
            self._universe = atomgroups.universe
            self._allow_multiple_atomgroups = False

        self._trajectory = self._universe.trajectory

        super(AnalysisBase, self).__init__(trajectory=self._trajectory,
                                           verbose=verbose,
                                           **kwargs)

@set_planar_class_doc
class PlanarBase(AnalysisBase):
    """Class to provide options and attributes for analysis in planar system.

    Provied the results attribute `z`.

    Parameters
    ----------
    trajectory : MDAnalysis.coordinates.base.ReaderBase
        A trajectory Reader
    ${PLANAR_CLASS_PARAMETERS}
    kwargs : dict
        Parameters parsed to `AnalysisBase`.

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    zmin : float
        Minimal position for analysis (Å)
    zmax : float
        Maximal position for analysis (Å)
    n_bins : int
        Number of bins for analysis

    """

    def __init__(self, atomgroups, dim, binwidth, comgroup, center, **kwargs):
        super(PlanarBase, self).__init__(atomgroups, **kwargs)
        self.dim = dim
        self.binwidth = binwidth
        self.center = center
        self.comgroup = comgroup

    def _prepare(self):
        """Preparations for the planar analysis."""
        if self.dim not in [0, 1, 2]:
            raise ValueError("Dimension can only be x=0, y=1 or z=2.")

        # Workaround since currently not alle module have option
        # with zmax and zmin
        if not hasattr(self, '_zmax'):
            self._zmax = None

        if self._zmax is None:
            self.Lz = 0
            self.zmax = self._universe.dimensions[self.dim]
        else:
            self.zmax = 10 * self._zmax

        if not hasattr(self, 'zmin'):
            self.zmin = 0
        self.zmin *= 10
        self.binwidth *= 10

        self.n_bins = int(np.ceil((self.zmax - self.zmin) / self.binwidth))

        logger.info(f"Using {self.n_bins} bins")

        if self.comgroup is not None and self.comgroup.n_atoms == 0:
            raise ValueError(f"`Comgroup` does not contain any atoms.")
        if self.comgroup is not None:
            self.center = True  # always center when COM

    def get_bins(self, positions, dim=None):
        """"Calculates bins based on given positions. dim denotes
        the dimension for calculating bins. If `None` the default
        dim is taken.

        Attributes
        ----------
        positions : numpy.ndarray
            3 dimensional positions
        dim : int
            dimension for binning (x=0, y=1, z=2)

        Returns
        -------
        numpy.ndarray
            binned psoitions
        """
        dim = self.dim if dim is None else dim
        dz = self._ts.dimensions[dim] / self.n_bins
        bins = np.rint(positions[:, dim] / dz)
        bins %= self.n_bins
        return bins.astype(int)

    def _single_frame(self):
        """Single frame for the planar analysis."""
        if self._zmax is None:
            self.zmax = self._ts.dimensions[self.dim]
            self.Lz += self.zmax

        if self.comgroup is not None:
            comshift = self.comgroup.center_of_mass(pbc=True)

            if hasattr(self, 'atomgroup'):
                groups = [self.atomgroup]
            else:
                groups = self.atomgroups
            for group in groups:
                group.atoms.positions += comshift

    def _conclude(self):
        """Results calculations for the planar analysis."""
        self._index = self._frame_index + 1

        if self._zmax is None:
            zmax = self.Lz / self._index
        else:
            zmax = self.zmax

        dz = (zmax - self.zmin) / self.n_bins

        self.results.z = np.linspace(
            self.zmin+dz/2, zmax-dz/2, self.n_bins,
            endpoint=False)

        if self.center:
            self.results.z -= self.zmin + (zmax - self.zmin) / 2

        self.results.z /= 10
