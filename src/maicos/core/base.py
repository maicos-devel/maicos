#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class for building Analysis classes."""

import inspect
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union

import MDAnalysis as mda
import MDAnalysis.analysis.base
import numpy as np
import pandas as pd
from MDAnalysis.analysis.base import Results
from MDAnalysis.lib.log import ProgressBar
from tqdm.contrib.logging import logging_redirect_tqdm
from typing_extensions import Self

from .._version import get_versions
from ..lib.math import center_cluster, new_mean, new_variance
from ..lib.util import (
    atomgroup_header,
    correlation_analysis,
    get_center,
    get_cli_input,
    render_docs,
)


__version__ = get_versions()["version"]
del get_versions

logger = logging.getLogger(__name__)


@render_docs
class AnalysisBase(MDAnalysis.analysis.base.AnalysisBase):
    """Base class derived from MDAnalysis for defining multi-frame analysis.

    The class is designed as a template for creating multi-frame analyses. This class
    will automatically take care of setting up the trajectory reader for iterating, and
    it offers to show a progress meter. Computed results are stored inside the
    :attr:`results` attribute. To define a new analysis, `AnalysisBase` needs to be
    subclassed and :meth:`_single_frame` must be defined. It is also possible to define
    :meth:`_prepare` and :meth:`_conclude` for pre- and post-processing. All results
    should be stored as attributes of the :class:`MDAnalysis.analysis.base.Results`
    container.

    During the analysis, the correlation time of an observable can be estimated to
    ensure that calculated errors are reasonable. For this, the :meth:`_single_frame`
    method has to return a single :obj:`float`. For details on the computation of the
    correlation and its further analysis refer to
    :func:`maicos.lib.util.correlation_analysis`.

    Parameters
    ----------
    ${ATOMGROUPS_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    multi_group : bool
        Analysis is able to work with list of atomgroups
    wrap_compound : str
        The group which will be kept together through the wrap processes.
        Allowed values are: ``'atoms'``, ``'group'``, ``'residues'``,
        ``'segments'``, ``'molecules'``, or ``'fragments'``.

    Attributes
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        Atomgroup taken for the Analysis (available if `multi_group = False`)
    atomgroups : list[MDAnalysis.core.groups.AtomGroup]
        Atomgroups taken for the Analysis (available if `multi_group = True`)
    n_atomgroups : int
        Number of atomngroups (available if `multi_group = True`)
    _universe : MDAnalysis.core.universe.Universe
        The Universe the atomgroups belong to
    _trajectory : MDAnalysis.coordinates.base.ReaderBase
        The trajectory the atomgroups belong to
    times : numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`AnalysisBase.run`
    frames : numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`AnalysisBase.run`
    _frame_index : int
        index of the frame currently analysed
    _index : int
        Number of frames already analysed (same as _frame_index + 1)
    results : MDAnalysis.analysis.base.Results
        results of calculation are stored after call to :meth:`AnalysisBase.run`
    _obs : MDAnalysis.analysis.base.Results
        Observables of the current frame
    _obs.box_center : numpy.ndarray
        Center of the simulation cell of the current frame
    means : MDAnalysis.analysis.base.Results
        Means of the observables. Keys are the same as :attr:`_obs`.
    sems : MDAnalysis.analysis.base.Results
        Standard errors of the mean of the observables. Keys are the same as
        :attr:`_obs`
    corrtime : float
        The correlation time of the analysed data. For details on how this is
        calculated see :func:`maicos.lib.util.correlation_analysis`.

    Raises
    ------
    ValueError
        If any of the provided AtomGroups (`atomgroups` or `refgroup`) does
        not contain any atoms.
    """

    def __init__(
        self,
        atomgroups: mda.AtomGroup,
        multi_group: bool = False,
        refgroup: Optional[mda.AtomGroup] = None,
        unwrap: bool = False,
        jitter: float = 0.0,
        wrap_compound: str = "atoms",
        concfreq: int = 0,
    ):
        if multi_group:
            if type(atomgroups) not in (list, tuple):
                atomgroups = [atomgroups]
            # Check that all atomgroups are from same universe
            if len(set([ag.universe for ag in atomgroups])) != 1:
                raise ValueError("Atomgroups belong to different Universes")

            # Sort the atomgroups, such that molecules are listed one after the other
            self.atomgroups = atomgroups

            for i, ag in enumerate(self.atomgroups):
                if ag.n_atoms == 0:
                    raise ValueError(
                        f"The {i+1}. provided `atomgroup`" "does not contain any atoms."
                    )

            self.n_atomgroups = len(self.atomgroups)
            self._universe = atomgroups[0].universe
            self._allow_multiple_atomgroups = True
        else:
            self.atomgroup = atomgroups

            if self.atomgroup.n_atoms == 0:
                raise ValueError(
                    "The provided `atomgroup` does not contain " "any atoms."
                )

            self._universe = atomgroups.universe
            self._allow_multiple_atomgroups = False

        self._trajectory = self._universe.trajectory
        self.refgroup = refgroup
        self.unwrap = unwrap
        self.jitter = jitter
        self.concfreq = concfreq
        if wrap_compound not in [
            "atoms",
            "group",
            "residues",
            "segments",
            "molecules",
            "fragments",
        ]:
            raise ValueError(
                "Unrecognized `wrap_compound` definition "
                f"{wrap_compound}: \nPlease use "
                "one of 'atoms', 'group', 'residues', "
                "'segments', 'molecules', or 'fragments'."
            )
        self.wrap_compound = wrap_compound

        if self.unwrap and self.wrap_compound == "atoms":
            logger.warning(
                "Unwrapping in combination with the "
                "`wrap_compound='atoms` is superfluous. "
                "`unwrap` will be set to `False`."
            )
            self.unwrap = False

        if self.refgroup is not None and self.refgroup.n_atoms == 0:
            raise ValueError("The provided `refgroup` does not contain " "any atoms.")

        super().__init__(trajectory=self._trajectory)

    @property
    def box_center(self) -> np.ndarray:
        """Center of the simulation cell."""
        return self._universe.dimensions[:3] / 2

    def run(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> Self:
        """Iterate over the trajectory.

        Parameters
        ----------
        start : int
            start frame of analysis
        stop : int
            stop frame of analysis
        step : int
            number of frames to skip between each analysed frame
        verbose : bool
            Turn on verbosity

        Returns
        -------
        self : object
            analysis object
        """
        self._run_locals = locals()
        logger.info("Choosing frames to analyze")
        # if verbose unchanged, use class default
        verbose = getattr(self, "_verbose", False) if verbose is None else verbose

        self._setup_frames(self._trajectory, start, stop, step)
        logger.info("Starting preparation")

        if self.refgroup is not None:
            if (
                not hasattr(self.refgroup, "masses")
                or np.sum(self.refgroup.masses) == 0
            ):
                logger.warning(
                    "No masses available in refgroup, falling back "
                    "to center of geometry"
                )
                ref_weights = np.ones_like(self.refgroup.atoms)

            else:
                ref_weights = self.refgroup.masses

        compatible_types = [np.ndarray, float, int, list, np.float_, np.int_]

        self._prepare()

        # Log bin information if a spatial analysis is run.
        if hasattr(self, "n_bins"):
            logger.info(f"Using {self.n_bins} bins.")

        module_has_save = callable(getattr(self.__class__, "save", None))

        timeseries = np.zeros(self.n_frames)
        self.series = Results()
        for i, ts in enumerate(
            ProgressBar(
                self._trajectory[self.start : self.stop : self.step], verbose=verbose
            )
        ):
            self._frame_index = i
            self._index = self._frame_index + 1

            self._ts = ts
            self.frames[i] = ts.frame
            self.times[i] = ts.time

            # Before we do any coordinate transformation we first unwrap the system to
            # avoid artifacts of later wrapping.
            if self.unwrap:
                self._universe.atoms.unwrap(compound=self.wrap_compound)
            if self.refgroup is not None:
                com_refgroup = center_cluster(self.refgroup, ref_weights)
                t = self.box_center - com_refgroup
                self._universe.atoms.translate(t)

            # Wrap into the primary unit cell to use all compounds for the analysis.
            self._universe.atoms.wrap(compound=self.wrap_compound)

            if self.jitter != 0.0:
                ts.positions += (
                    np.random.random(size=(len(ts.positions), 3)) * self.jitter
                )

            self._obs = Results()

            timeseries[i] = self._single_frame()

            # This try/except block is used because it will fail only once and is
            # therefore not a performance issue like a if statement would be.
            try:
                for key in self._obs.keys():
                    if type(self._obs[key]) is list:
                        self._obs[key] = np.array(self._obs[key])
                    old_mean = self.means[key]  # type: ignore
                    old_var = self.sems[key] ** 2 * (self._index - 1)  # type: ignore
                    self.means[key] = new_mean(  # type: ignore
                        self.means[key], self._obs[key], self._index  # type: ignore
                    )  # type: ignore
                    self.sems[key] = np.sqrt(  # type: ignore
                        new_variance(
                            old_var,
                            old_mean,
                            self.means[key],  # type: ignore
                            self._obs[key],
                            self._index,
                        )
                        / self._index
                    )
                    self.series[key][ts.frame] = self._obs[key]
            except AttributeError:
                with logging_redirect_tqdm():
                    logger.info("Preparing error estimation.")
                # the means and sems are not yet defined. We initialize the means with
                # the data from the first frame and set the sems to zero (with the
                # correct shape).
                
                self.means = self._obs.copy()
                self.sems = Results()
                for key in self._obs.keys():
                    if type(self._obs[key]) not in compatible_types:
                        raise TypeError(f"Obervable {key} has uncompatible type.")
                    self.sems[key] = np.zeros(np.shape(self._obs[key]))
                    data = np.empty((self.n_frames,) + np.shape(self._obs[key]))
                    data[:] = np.nan

                    self.series[key] = pd.Series(
                        data=[i for i in data],
                        index=range(self.start, self.stop, self.step),
                        dtype=self._obs[key].dtype,
                    )
                    self.series[key][ts.frame] = self._obs[key]
            if (
                self.concfreq
                and self._index % self.concfreq == 0
                and self._frame_index > 0
            ):
                self._conclude()
                if module_has_save:
                    self.save()
        for key in self.series.keys():
            self.series[key].to_pickle(f"{key}_{self.start}_{self.stop}_{self.step}.pickle",)
        logger.info("Finishing up")

        self.corrtime = correlation_analysis(timeseries)

        self._conclude()
        if self.concfreq and module_has_save:
            self.save()
        return self

    def savetxt(
        self, fname: str, X: np.ndarray, columns: Optional[List[str]] = None
    ) -> None:
        """Save to text.

        An extension of the numpy savetxt function. Adds the command line input to the
        header and checks for a doubled defined filesuffix.

        Return a header for the text file to save the data to. This method builds a
        generic header that can be used by any MAICoS module. It is called by the save
        method of each module.

        The information it collects is:
          - timestamp of the analysis
          - name of the module
          - version of MAICoS that was used
          - command line arguments that were used to run the module
          - module call including the default arguments
          - number of frames that were analyzed
          - atomgroups that were analyzed
          - output messages from modules and base classes (if they exist)
        """
        # This method breaks if fname is a Path object. We therefore convert it to a str
        fname = str(fname)
        # Get the required information first
        current_time = datetime.now().strftime("%a, %b %d %Y at %H:%M:%S ")
        module_name = self.__class__.__name__

        # Here the specific output messages of the modules are collected. We only take
        # into account maicos modules and start at the top of the module tree.
        # Submodules without an own OUTPUT inherit from the parent class, so we want to
        # remove those duplicates.
        messages_list = []
        for cls in self.__class__.mro()[-3::-1]:
            if hasattr(cls, "OUTPUT"):
                if cls.OUTPUT not in messages_list:
                    messages_list.append(cls.OUTPUT)
        messages = "\n".join(messages_list)

        # Get information on the analyzed atomgroup
        atomgroups = ""
        if hasattr(self, "refgroup") and self.refgroup is not None:
            atomgroups += f"  (ref) {atomgroup_header(self.refgroup)}\n"
        if self._allow_multiple_atomgroups:
            for i, ag in enumerate(self.atomgroups):
                atomgroups += f"  ({i + 1}) {atomgroup_header(ag)}\n"
        else:
            atomgroups += f"  (1) {atomgroup_header(self.atomgroup)}\n"

        # We have to check this since only the modules have the _locals attribute,
        # not the base classes. Yet we still want to test output behaviour of the base
        # classes.
        if hasattr(self, "_locals") and hasattr(self, "_run_locals"):
            sig = inspect.getfullargspec(self.__class__)
            sig.args.remove("self")
            strings = []
            for param in sig.args:
                if type(self._locals[param]) is str:
                    string = f"{param}='{self._locals[param]}'"
                elif param == "atomgroup" or param == "atomgroups":
                    string = f"{param}=<AtomGroup>"
                elif param == "refgroup" and self._locals[param] is not None:
                    string = f"{param}=<AtomGroup>"
                else:
                    string = f"{param}={self._locals[param]}"
                strings.append(string)
            init_signature = ", ".join(strings)

            sig = inspect.getfullargspec(self.run)
            sig.args.remove("self")
            run_signature = ", ".join(
                [
                    f"{param}='{self._run_locals[param]}'"
                    if type(self._run_locals[param]) is str
                    else f"{param}={self._run_locals[param]}"
                    for param in sig.args
                ]
            )

            module_input = f"{module_name}({init_signature}).run({run_signature})"
        else:
            module_input = f"{module_name}(*args).run(*args)"

        header = (
            f"This file was generated by {module_name} "
            f"on {current_time}\n\n"
            f"{module_name} is part of MAICoS v{__version__}\n\n"
            f"Command line:    {get_cli_input()}\n"
            f"Module input:    {module_input}\n\n"
            f"Statistics over {self._index} frames\n\n"
            f"Considered atomgroups:\n"
            f"{atomgroups}\n"
            f"{messages}\n\n"
        )

        if columns is not None:
            header += "|".join([f"{i:^26}" for i in columns])[2:]

        fname = "{}{}".format(fname, (not fname.endswith(".dat")) * ".dat")
        np.savetxt(fname, X, header=header, fmt="% .18e ", encoding="utf8")


@render_docs
class ProfileBase:
    """Base class for computing profiles.

    Parameters
    ----------
    ${ATOMGROUPS_PARAMETER}
    ${PROFILE_CLASS_PARAMETERS}
    ${PROFILE_CLASS_PARAMETERS_PRIVATE}

    Attributes
    ----------
    ${PROFILE_CLASS_ATTRIBUTES}
    """

    def __init__(
        self,
        atomgroups: Union[mda.AtomGroup, List[mda.AtomGroup]],
        weighting_function: Callable,
        normalization: str,
        grouping: str,
        bin_method: str,
        output: str,
        f_kwargs: Optional[Dict] = None,
    ):
        self.atomgroups = atomgroups
        self.normalization = normalization.lower()
        self.grouping = grouping.lower()
        self.bin_method = bin_method.lower()
        self.output = output

        if f_kwargs is None:
            f_kwargs = {}

        self.weighting_function = lambda ag: weighting_function(
            ag, grouping, **f_kwargs
        )
        # We need to set the following dictionaries here because ProfileBase is not a
        # subclass of AnalysisBase (only needed for tests)
        self.results = Results()
        self._obs = Results()

    def _prepare(self):
        normalizations = ["none", "volume", "number"]
        if self.normalization not in normalizations:
            raise ValueError(
                f"Normalization {self.normalization!r} not supported. "
                f"Use {', '.join(normalizations)}."
            )

        groupings = ["atoms", "segments", "residues", "molecules", "fragments"]
        if self.grouping not in groupings:
            raise ValueError(
                f"{self.grouping!r} is not a valid option for "
                f"grouping. Use {', '.join(groupings)}."
            )

        # If unwrap has not been set we define it here
        if not hasattr(self, "unwrap"):
            self.unwrap = True

        if self.normalization == "number":
            self.tot_bincount = np.zeros((self.n_bins, self.n_atomgroups))

    def _compute_histogram(
        self, positions: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate histogram based on positions.

        Parameters
        ----------
        positions : numpy.ndarray
            positions
        weights : numpy.ndarray
            weights for the histogram.

        Returns
        -------
        hist : numpy.ndarray
            histogram
        """
        raise NotImplementedError("Only implemented in child classes.")

    def _single_frame(self):
        self._obs.profile = np.zeros((self.n_bins, self.n_atomgroups))
        for index, selection in enumerate(self.atomgroups):
            if self.grouping == "atoms":
                positions = selection.atoms.positions
            else:
                positions = get_center(
                    selection.atoms, bin_method=self.bin_method, compound=self.grouping
                )

            weights = self.weighting_function(selection)
            profile = self._compute_histogram(positions, weights)

            if self.normalization == "number":
                bincount = self._compute_histogram(positions, weights=None)

                self.tot_bincount[:, index] += bincount

                # If a bin does not contain any particles we divide by 0.
                with np.errstate(invalid="ignore"):
                    profile /= bincount
                profile = np.nan_to_num(profile)
            elif self.normalization == "volume":
                profile /= self._obs.bin_volume

            self._obs.profile[:, index] = profile

    def _conclude(self):
        self.results.profile = self.means.profile
        self.results.dprofile = self.sems.profile

        if self.normalization == "number":
            no_occurences_idx = self.tot_bincount == 0
            self.results.profile[no_occurences_idx] = np.nan
            self.results.dprofile[no_occurences_idx] = np.nan

    def save(self):
        """Save results of analysis to file."""
        columns = ["positions [Å]"]

        for i, _ in enumerate(self.atomgroups):
            columns.append(f"({i + 1}) profile")
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"({i + 1}) error")

        # Required attribute to use method from `AnalysisBase`
        self._allow_multiple_atomgroups = True

        AnalysisBase.savetxt(
            self,
            self.output,
            np.hstack(
                (
                    self.results.bin_pos[:, np.newaxis],
                    self.results.profile,
                    self.results.dprofile,
                )
            ),
            columns=columns,
        )
