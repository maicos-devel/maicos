#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class for building Analysis classes."""

import logging
import numbers
import warnings
from collections.abc import Callable
from datetime import datetime
from itertools import combinations
from typing import TYPE_CHECKING, Self

import MDAnalysis as mda
import MDAnalysis.analysis.base
import numpy as np
from MDAnalysis.analysis.base import Results
from MDAnalysis.lib.log import ProgressBar
from tqdm.contrib.logging import logging_redirect_tqdm

from .. import __version__
from ..lib._moments import MomentAccumulator
from ..lib.math import (
    center_cluster,
)
from ..lib.util import (
    atomgroup_header,
    check_file_extension,
    correlation_analysis,
    get_center,
    get_cli_input,
    get_module_input_str,
    maicos_banner,
    make_pair_key,
    render_docs,
    triclinic_to_orthorhombic,
)

logger = logging.getLogger(__name__)


class _Runner:
    """Private Runner class that provides a common ``run`` method.

    Class is used inside ``AnalysisBase`` as well as in ``AnalysisCollection``
    """

    def _run(
        self,
        analysis_instances: tuple["AnalysisBase", ...],
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        frames: int | None = None,
        verbose: bool | None = None,
        progressbar_kwargs: dict | None = None,
    ) -> Self:
        self._run_locals = locals()

        if frames is not None and not all(opt is None for opt in [start, stop, step]):
            raise ValueError("start/stop/step cannot be combined with frames")

        for analysis_object in analysis_instances:
            if getattr(analysis_object, "_trajectory", None) is None:
                raise RuntimeError(
                    f"{type(analysis_object).__name__} has no trajectory "
                    "attached and cannot be run. This could mean it was "
                    "restored via `load` as a read-only snapshot; build a "
                    "fresh instance from a Universe with a trajectory if you "
                    "need to run the analysis again."
                )

        # Configure the root logger if not already configured
        logging.basicConfig()
        # Redirect warnings (from the warnings library) to the logging system
        logging.captureWarnings(True)

        level = logging.INFO if verbose else logging.WARNING

        parent_logger = logging.getLogger("maicos")
        if parent_logger.level >= logging.INFO or parent_logger.level == 0:
            # User set log level manually to WARNING or INFO or not set at all
            # Overwrite based on the verbose option
            parent_logger.setLevel(level)

        logger.info(maicos_banner(frame_char="#", version=f"v{__version__}"))

        logger.debug("Choosing frames to analyze")

        for analysis_object in analysis_instances:
            analysis_object._setup_frames(
                analysis_object._trajectory,
                start=start,
                stop=stop,
                step=step,
                frames=frames,
            )
            # Reset the trajectory reader to ensure _prepare uses the first frame
            analysis_object._sliced_trajectory[0]

        for analysis_object in analysis_instances:
            analysis_object._call_prepare()

        if progressbar_kwargs is None:
            progressbar_kwargs = {}

        for i, ts in enumerate(
            ProgressBar(
                analysis_instances[0]._sliced_trajectory,
                verbose=verbose,
                **progressbar_kwargs,
            )
        ):
            ts_original = ts.copy()
            with logging_redirect_tqdm():
                for analysis_object in analysis_instances:
                    ts.positions[:] = ts_original.positions
                    if ts_original.dimensions is not None:
                        ts.dimensions[:] = ts_original.dimensions
                    else:
                        ts.dimensions = None
                    if ts.has_velocities:
                        ts.velocities[:] = ts_original.velocities
                    if ts.has_forces:
                        ts.forces[:] = ts_original.forces
                    analysis_object._call_single_frame(ts=ts, current_frame_index=i)
        logger.debug("Concluding analysis.")

        for analysis_object in analysis_instances:
            analysis_object._call_conclude()

        return self


@render_docs
class AnalysisBase(_Runner, MDAnalysis.analysis.base.AnalysisBase):
    """Base class derived from MDAnalysis for defining multi-frame analysis.

    The class is designed as a template for creating multi-frame analyses. This class
    will automatically take care of setting up the trajectory reader for iterating, and
    it offers to show a progress meter. Computed results are stored inside the
    :attr:`results` attribute. To define a new analysis, ``AnalysisBase`` needs to be
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
    ${ATOMGROUP_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    ${WRAP_COMPOUND_PARAMETER}


    Attributes
    ----------
    ${ATOMGROUP_PARAMETER}
    _universe : MDAnalysis.core.universe.Universe
        The Universe the AtomGroup belong to
    _trajectory : MDAnalysis.coordinates.base.ReaderBase
        The trajectory the AtomGroup belong to
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
    sums : MDAnalysis.analysis.base.Results
        Sum of the observables across frames. Keys are the same as :attr:`_obs`.
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
        If any of the provided AtomGroups (``atomgroup`` or ``refgroup``) does
        not contain any atoms.

    Example
    -------
    To write your own analysis module you can use the example given below. As with all
    MAICoS modules, this inherits from the :class:`AnalysisBase
    <maicos.core.base.AnalysisBase>` class.

    The example will calculate the average box volume and stores the result within the
    ``result`` object of the class.

    >>> import logging
    >>> from typing import Optional

    >>> import MDAnalysis as mda
    >>> import numpy as np

    >>> from maicos.core import AnalysisBase
    >>> from maicos.lib.util import render_docs

    >>> logger = logging.getLogger(__name__)

    Adding logging messages to your code makes debugging easier.

    Due to the similar structure of all MAICoS modules you can render the parameters
    using the :func:`maicos.lib.util.render_docs` decorator. The decorator will replace
    special keywords with a leading ``$`` with the actual docstring as defined in
    :attr:`maicos.lib.util.DOC_DICT`.

    >>> @render_docs
    ... class NewAnalysis(AnalysisBase):
    ...     '''Analysis class calcuting the average box volume.'''
    ...
    ...     def __init__(
    ...         self,
    ...         atomgroup: mda.AtomGroup,
    ...         concfreq: int = 0,
    ...         temperature: float = 300,
    ...         output: str = "outfile.dat",
    ...     ):
    ...         super().__init__(
    ...             atomgroup=atomgroup,
    ...             refgroup=None,
    ...             unwrap=False,
    ...             pack=True,
    ...             jitter=0.0,
    ...             wrap_compound="atoms",
    ...             concfreq=concfreq,
    ...         )
    ...
    ...         self.temperature = temperature
    ...         self.output = output
    ...
    ...     def _prepare(self):
    ...         '''Set things up before the analysis loop begins.'''
    ...         # self.atomgroup refers to the provided `atomgroup`
    ...         # self._universe refers to full universe of given `atomgroup`
    ...         self.volume = 0
    ...
    ...     def _single_frame(self):
    ...         '''Calculate data from a single frame of trajectory.
    ...
    ...         Don't worry about normalising, just deal with a single frame.
    ...         '''
    ...         # Current frame index: self._frame_index
    ...         # Current timestep object: self._ts
    ...
    ...         volume = self._ts.volume
    ...         self.volume += volume
    ...
    ...         # Eeach module should return a characteristic scalar which is used
    ...         # by MAICoS to estimate correlations of an Analysis.
    ...         return volume
    ...
    ...     def _conclude(self):
    ...         '''Finalise the results you've gathered.
    ...
    ...         Called at the end of the run() method to finish everything up.
    ...         '''
    ...         self.results.volume = self.volume / self.n_frames
    ...         logger.info(
    ...             f"Average volume of the simulation box {self.results.volume:.2f} Å³"
    ...         )
    ...
    ...     def save(self) -> None:
    ...         '''Save results of analysis to file specified by ``output``.
    ...
    ...         Called at the end of the run() method after _conclude.
    ...         '''
    ...         self.savetxt(
    ...             self.output,
    ...             np.array([self.results.volume]),
    ...             columns="volume / Å³",
    ...         )


    Afterwards the new analysis can be run like this

    >>> import MDAnalysis as mda
    >>> from MDAnalysisTests.datafiles import TPR, XTC

    >>> u = mda.Universe(TPR, XTC)

    >>> na = NewAnalysis(u.atoms)
    >>> _ = na.run(start=0, stop=10)
    >>> print(round(na.results.volume, 2))
    362631.65

    Results can also be accessed by key

    >>> print(round(na.results["volume"], 2))
    362631.65

    """

    #: Observable pairs to accumulate the off-diagonal covariance for. Each entry
    #: names two observable keys, e.g. ``[{"mM_r", "m_r"}, {"mM_r", "M_r"}]``.
    #: Only the requested pairs are tracked, so analyses pay only for the
    #: covariances their error estimate actually consumes. Empty (the default)
    #: disables covariance entirely. Required for :meth:`cov` /
    #: :meth:`propagate_error`; subclasses needing them declare their pairs here.
    _compute_covariance: list = []

    if TYPE_CHECKING:  # pragma: no cover
        # Type annotations for attributes set dynamically in _call_single_frame.
        means: Results
        sems: Results
        sums: Results
        pop: Results
        M2: Results
        C: Results
        _obs: Results
        _pop: Results
        _var: Results
        _cov: Results

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        unwrap: bool,
        pack: bool,
        refgroup: None | mda.AtomGroup,
        jitter: float,
        concfreq: int,
        wrap_compound: str,
    ) -> None:
        logger.debug("Debug logging activated")
        self.atomgroup = atomgroup

        if self.atomgroup.n_atoms == 0:
            raise ValueError("The provided `atomgroup` does not contain any atoms.")

        self._universe = atomgroup.universe

        self._trajectory = self._universe.trajectory
        self.refgroup = refgroup
        self.unwrap = unwrap
        self.pack = pack
        self.jitter = jitter
        self.concfreq = concfreq
        # Canonical set of requested covariance pairs (order-independent keys).
        self._requested_pairs = {
            make_pair_key(*pair) for pair in self._compute_covariance
        }
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

        if self.unwrap and self._universe.dimensions is None:
            raise ValueError(
                "Universe does not have `dimensions` and can't be unwrapped!"
            )

        if self.pack and self._universe.dimensions is None:
            raise ValueError("Universe does not have `dimensions` and can't be packed!")

        if self.unwrap and self.wrap_compound == "atoms":
            logger.debug(
                "Unwrapping in combination with the "
                "`wrap_compound='atoms` is superfluous. "
                "`unwrap` will be set to `False`."
            )
            self.unwrap = False

        if self.refgroup is not None:
            if self.refgroup.n_atoms == 0:
                raise ValueError("The provided `refgroup` does not contain any atoms.")
            if not self.pack:
                raise ValueError(
                    "Disabling `pack` with a `refgroup` is not allowed. Shifting "
                    "atoms probably outside of the primary cell withput packing them "
                    "back may lead to sever problems during the analysis!"
                )

        self.module_has_save = callable(getattr(self.__class__, "save", None))
        super().__init__(trajectory=self._trajectory)

    @property
    def box_lengths(self) -> np.ndarray:
        """Lengths of the simulation cell vectors."""
        return self._universe.dimensions[:3].astype(np.float64)

    @property
    def box_center(self) -> np.ndarray:
        """Center of the simulation cell."""
        return self.box_lengths / 2

    def _prepare(self) -> None:
        """Set things up before the analysis loop begins."""
        pass  # pylint: disable=unnecessary-pass

    def _call_prepare(self) -> None:
        """Base method wrapping all _prepare logic into a single call."""
        if self.refgroup is not None:
            if (
                not hasattr(self.refgroup, "masses")
                or np.sum(self.refgroup.masses) == 0
            ):
                logger.warning(
                    "No masses available in refgroup, falling back "
                    "to center of geometry"
                )
                self.ref_weights = np.ones_like(self.refgroup.atoms)

            else:
                self.ref_weights = self.refgroup.masses

        if hasattr(self, "_bin_width"):
            if not isinstance(self._bin_width, numbers.Real):
                raise TypeError(
                    "Binwidth must be a real number but is of type "
                    f"'{type(self._bin_width).__name__}'."
                )
            if self._bin_width <= 0:
                raise ValueError(
                    f"Binwidth must be a positive number but is {self._bin_width}."
                )
        self._warned_triclinic = False
        self._prepare()

        if self.refgroup is not None:
            logger.info(
                """Coordinates are relative to the center of mass of reference"""
                f""" atomgroup {atomgroup_header(self.refgroup)}."""
            )
        else:
            logger.info(
                """Coordinates are relative to the center """
                """of the simulation box."""
            )

        logger.info(f"Considered atomgroup {atomgroup_header(self.atomgroup)}.")

        # Log bin information if a spatial analysis is run.
        if hasattr(self, "n_bins"):
            logger.info(f"Using {self.n_bins} bins.")

        self.timeseries = np.zeros(self.n_frames)

        logger.info(f"Analysing {self.n_frames} trajectory frames.")

        logger.debug(f"Module input: {get_module_input_str(self)}")

    def _single_frame(self) -> None | float:
        """Calculate data from a single frame of trajectory.

        Don't worry about normalising, just deal with a single frame.
        """
        raise NotImplementedError("Only implemented in child classes")

    def _call_single_frame(self, ts, current_frame_index) -> None:
        """Base method wrapping all single_frame logic into a single call."""
        self._frame_index = current_frame_index
        self._index = self._frame_index + 1

        self._ts = ts
        self.frames[current_frame_index] = ts.frame
        self.times[current_frame_index] = ts.time

        # Before we do any coordinate transformation we first unwrap the system to
        # avoid artifacts of later wrapping.
        if self.unwrap:
            self._universe.atoms.unwrap(compound=self.wrap_compound)
        if self.refgroup is not None:
            com_refgroup = center_cluster(self.refgroup, self.ref_weights)
            t = self.box_center - com_refgroup
            self._universe.atoms.translate(t)

        if self._universe.dimensions is not None:
            if not self._warned_triclinic and np.any(
                ts.dimensions[-3:] != np.array([90.0, 90.0, 90.0])
            ):
                logger.warning(
                    "The trajectory contains box-dimensions that are not "
                    "orthorhombic! Continue with caution.",
                )
            self._warned_triclinic = True
            # If universe has a cell we wrap the compound into the primary unit cell to
            # use all compounds for the analysis.
            is_triclinic = np.any(ts.dimensions[-3:] != np.array([90.0, 90.0, 90.0]))
            if self.pack:
                self._universe.atoms.wrap(compound=self.wrap_compound)
                if is_triclinic:
                    ortho_box = triclinic_to_orthorhombic(ts.dimensions)
                    self._universe.atoms.wrap(
                        compound=self.wrap_compound, box=ortho_box
                    )

        if self.jitter != 0.0:
            ts.positions += np.random.random(size=(len(ts.positions), 3)) * self.jitter

        # For the current frame
        self._obs = Results()  # observable (or mean of the samples)
        self._var = Results()  # variance of the samples
        self._pop = Results()  # count of samples
        self._cov = Results()  # within-frame covariance of the samples

        self.timeseries[current_frame_index] = self._single_frame()

        # This try/except block is used because it will fail only once and is
        # therefore not a performance issue like a if statement would be.
        try:
            # Fail fast if the backend is not initialised yet.
            self._moments  # noqa B018

            # One vectorized backend updates the running means, variances and the
            # requested covariances. It accumulates the off-diagonal covariance
            # (which needs the pre-frame means) before overwriting the means.
            self._moments.update(self._obs, self._pop, self._var, self._cov)

        except AttributeError:
            with logging_redirect_tqdm():
                logger.debug("Initializing error estimation.")
            # Seed the running statistics from the first frame. The backend owns
            # the means/sems/M2/pop/sums/C containers; expose them on the analysis
            # for the modules, checkpointing and `cov`/`propagate_error`.
            self._moments = MomentAccumulator(self._requested_pairs)
            self._moments.initialize(self._obs, self._pop, self._var, self._cov)
            self.means = self._moments.means
            self.sems = self._moments.sems
            self.M2 = self._moments.M2
            self.pop = self._moments.pop
            self.sums = self._moments.sums
            self.C = self._moments.C

        if self.concfreq and self._index % self.concfreq == 0 and self._frame_index > 0:
            self._conclude()
            if self.module_has_save:
                self.save()

    def joint_pop(self, key_i: str, key_j: str) -> np.ndarray:
        """Shared sample count of two co-sampled observables.

        Parameters
        ----------
        key_i, key_j : str
            Keys of the two observables.

        Returns
        -------
        numpy.ndarray
            The (broadcast) number of samples shared by both observables.
        """
        broadcasted = np.broadcast_arrays(self.pop[key_i], self.pop[key_j])
        # find the array with the higher dimension
        if np.ndim(self.pop[key_i]) > np.ndim(self.pop[key_j]):
            return broadcasted[0]
        return broadcasted[1]

    def cov(self, key_i: str, key_j: str) -> np.ndarray:
        r"""Covariance of the means of two observables.

        The element-wise covariance :math:`\mathrm{Cov}(\bar x_i, \bar x_j)` of the
        observable means accumulated across frames. The diagonal (``key_i == key_j``)
        equals the squared standard error of the mean, :attr:`sems`.

        Parameters
        ----------
        key_i, key_j : str
            Keys of the two observables.

        Returns
        -------
        numpy.ndarray
            Covariance of the means of ``key_i`` and ``key_j``.

        Raises
        ------
        KeyError
            If the off-diagonal pair was not tracked, either because the two
            observables do not broadcast against each other or because they are
            not co-sampled (different populations).
        """
        if key_i == key_j:
            return self.sems[key_i] ** 2
        if not self._requested_pairs:
            raise RuntimeError(
                "Covariance tracking is disabled. List the observable pairs in the "
                "`_compute_covariance` class attribute to use `cov`/`propagate_error`."
            )
        pair_key = make_pair_key(key_i, key_j)
        if pair_key not in self.C:
            raise KeyError(
                f"covariance of {key_i!r} and {key_j!r} not tracked: the pair was not "
                f"requested in `_compute_covariance`, or the observables do not "
                f"broadcast or are not co-sampled (different populations), so they "
                f"cannot enter the same estimator"
            )
        return self.C[pair_key] / self.joint_pop(key_i, key_j) ** 2

    def propagate_error(self, grads: dict) -> np.ndarray:
        r"""Propagate observable errors through an estimator.

        Computes the standard error of an estimator :math:`f` from the full
        covariance of the observable means,

        .. math::

            \sigma_f^2 = \sum_{ij}
                \frac{\partial f}{\partial x_i}
                \frac{\partial f}{\partial x_j}
                \mathrm{Cov}(\bar x_i, \bar x_j),

        where ``grads[key]`` provides :math:`\partial f / \partial x_{key}`. The
        diagonal terms reproduce the independent-variable (uncorrelated) estimate;
        the off-diagonal terms add the cross-covariance contributions.

        Parameters
        ----------
        grads : dict
            Mapping of observable key to the gradient of the estimator with
            respect to that observable's mean.

        Returns
        -------
        numpy.ndarray
            Standard error of the estimator.

        Raises
        ------
        KeyError
            If two of the supplied observables have no tracked covariance (see
            :meth:`cov`).
        """
        keys = list(grads)
        # Cross terms first: an untracked pair raises before the diagonal sum,
        # which would otherwise fail to broadcast incompatible observables.
        var = 0.0
        for key_i, key_j in combinations(keys, 2):
            cov_ij = self.cov(key_i, key_j)  # raises KeyError for an untracked pair
            var = var + 2 * grads[key_i] * grads[key_j] * cov_ij

        for key in keys:
            var = var + grads[key] ** 2 * self.sems[key] ** 2

        try:
            return np.sqrt(var)
        except RuntimeWarning:
            # variance is negative (usually due to an issue with the covariance)
            var = 0.0
            for key in keys:
                var = var + grads[key] ** 2 * self.sems[key] ** 2
            return np.sqrt(var)

    def _conclude(self) -> None:
        """Finalize the results you've gathered.

        Called at the end of the :meth:`run` method to finish everything up.
        """
        pass  # pylint: disable=unnecessary-pass

    def _call_conclude(self) -> None:
        """Base method wrapping all _conclude logic into a single call."""
        self.corrtime = correlation_analysis(self.timeseries)

        self._conclude()
        if self.concfreq and self.module_has_save:
            self.save()

    @render_docs
    def run(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        frames: int | None = None,
        verbose: bool | None = None,
        progressbar_kwargs: dict | None = None,
    ) -> Self:
        """${RUN_METHOD_DESCRIPTION}"""  # noqa: D415
        return _Runner._run(
            self,
            analysis_instances=(self,),
            start=start,
            stop=stop,
            step=step,
            frames=frames,
            verbose=verbose,
            progressbar_kwargs=progressbar_kwargs,
        )

    def savetxt(
        self, fname: str, X: np.ndarray, columns: list[str] | None = None
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
          - atomgroup that was analyzed
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
            if hasattr(cls, "OUTPUT") and cls.OUTPUT not in messages_list:
                messages_list.append(cls.OUTPUT)
        messages = "\n".join(messages_list)

        # Get information on the analyzed atomgroup
        atomgroups = f"  (grp) {atomgroup_header(self.atomgroup)}\n"
        if hasattr(self, "refgroup") and self.refgroup is not None:
            atomgroups += f"  (ref) {atomgroup_header(self.refgroup)}\n"

        module_input = get_module_input_str(self)

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
            header += "|".join([f"{i:^23}" for i in columns])[3:]

        fname = check_file_extension(fname, ".dat")
        np.savetxt(fname, X, header=header, fmt="% .14e ", encoding="utf8")

    _CHECKPOINT_CONTAINERS = (
        "results",
        "_obs",
        "means",
        "sems",
        "sums",
        "pop",
        "M2",
        "C",
    )
    _CHECKPOINT_ARRAYS = ("timeseries", "frames", "times")
    _CHECKPOINT_META = ("_frame_index", "_index", "corrtime")

    _CHECKPOINT_SEP = ":::"
    _CHECKPOINT_PAIR_SEP = "|||"

    def dump(self, filename: str) -> None:
        """Save analysis state to an ``.npz`` file.

        .. warning::

            :meth:`dump` is **not** an archival format. The on-disk layout
            tracks the installed MAICoS and MDAnalysis versions and may stop
            loading after an upgrade. Use it to checkpoint or hand off an
            in-progress analysis, not to preserve results long-term. For
            archival output write summary tables with :meth:`save` or export
            :attr:`results` to a stable format.

        Persists all statistical accumulators (``means``, ``sems``, ``sums``,
        ``pop``, ``M2``), the ``results`` and ``_obs`` containers, per-frame
        arrays (``timeseries``, ``frames``, ``times``), metadata, the associated
        Universe and the analysed atomgroup. Restore the analysis with :meth:`load`.

        Parameters
        ----------
        filename : str
            Path to the output ``.npz`` file.
        """
        sep = self._CHECKPOINT_SEP
        data = {}
        for name in self._CHECKPOINT_CONTAINERS:
            container = getattr(self, name, None)
            if container is None:
                continue
            for key in container:
                if isinstance(key, tuple):
                    # Pair keys (e.g. the covariance container) are flattened to
                    # a string and reconstructed in `load`.
                    key_str = self._CHECKPOINT_PAIR_SEP.join(key)
                else:
                    key_str = key
                data[f"{name}{sep}{key_str}"] = np.asarray(container[key])

        for name in self._CHECKPOINT_ARRAYS:
            arr = getattr(self, name, None)
            if arr is not None:
                data[f"_array{sep}{name}"] = np.asarray(arr)

        for name in self._CHECKPOINT_META:
            val = getattr(self, name, None)
            if val is not None:
                data[f"_meta{sep}{name}"] = np.asarray(val)

        data.update(self._serialize_topology())
        data["_atomgroup_indices"] = np.asarray(self.atomgroup.indices)
        if self.refgroup is not None:
            data["_refgroup_indices"] = np.asarray(self.refgroup.indices)
        data["_maicos_version"] = np.asarray(__version__)

        filename = check_file_extension(filename, ".npz")
        np.savez(filename, **data)

    @classmethod
    def load(cls, filename: str) -> Self:
        """Restore an analysis instance from a file created by :meth:`dump`.

        Returns a new instance of ``cls`` with all statistical accumulators,
        per-frame arrays, metadata, and the analysed atomgroup populated from
        the file. The rebuilt :class:`MDAnalysis.Universe` carries only the
        topology — no trajectory — so calling :meth:`run` on the returned
        instance raises :class:`RuntimeError`. Use the loaded instance to
        inspect :attr:`results` or to call :meth:`save`.

        Parameters
        ----------
        filename : str
            Path to the ``.npz`` file written by :meth:`dump`.

        Returns
        -------
        Self
            A new instance of ``cls`` with state restored from ``filename``.
        """
        sep = cls._CHECKPOINT_SEP
        npz = np.load(filename, allow_pickle=False)

        if "_maicos_version" not in npz.files:
            raise ValueError(
                f"{filename!r} is missing a MAICoS version tag. It was either "
                "not produced by `dump` or written by an incompatible "
                "version."
            )
        dump_version = str(npz["_maicos_version"])
        if dump_version != __version__:
            raise ValueError(
                f"{filename!r} was written by MAICoS v{dump_version} but the "
                f"installed version is v{__version__}. `dump`/`load` is "
                f"version-locked; re-run the analysis with the current "
                f"version or install v{dump_version} to load this file."
            )

        universe = cls._deserialize_topology(npz)
        atomgroup = universe.atoms[npz["_atomgroup_indices"]]
        refgroup = None
        if "_refgroup_indices" in npz.files:
            refgroup = universe.atoms[npz["_refgroup_indices"]]

        # Bypass __init__: subclasses have varying signatures and we don't
        # know the original constructor arguments. The loaded instance is a
        # read-only snapshot, so we only restore the attributes consumers
        # (``save``, results inspection, ``AnalysisCollection``) rely on.
        instance = cls.__new__(cls)
        instance.atomgroup = atomgroup
        instance.refgroup = refgroup
        instance._universe = universe
        instance._trajectory = None
        instance.results = Results()
        instance.module_has_save = callable(getattr(cls, "save", None))

        containers: dict[str, Results] = {}
        for full_key in npz.files:
            arr = npz[full_key]
            prefix, found, key = full_key.partition(sep)
            if not found:
                continue
            if prefix == "_topology":
                continue

            if prefix in cls._CHECKPOINT_CONTAINERS:
                if prefix not in containers:
                    containers[prefix] = Results()
                # Reconstruct pair (tuple) keys flattened by `dump`.
                pair_sep = cls._CHECKPOINT_PAIR_SEP
                out_key = tuple(key.split(pair_sep)) if pair_sep in key else key
                # Convert 0-d arrays back to Python scalars
                containers[prefix][out_key] = arr.item() if arr.ndim == 0 else arr

            elif prefix == "_array":
                setattr(instance, key, arr)

            elif prefix == "_meta":
                val = arr.item() if arr.ndim == 0 else arr
                setattr(instance, key, val)

        for name, container in containers.items():
            setattr(instance, name, container)

        return instance

    _TOPOLOGY_LEVELS = ("atom", "residue", "segment")

    def _serialize_topology(self) -> dict[str, np.ndarray]:
        """Encode the universe's topology as numpy arrays (no pickle).

        Captures the atom/residue/segment counts, the atom→residue and
        residue→segment maps, and every per-atom/residue/segment
        :class:`MDAnalysis.core.topologyattrs.TopologyAttr`. String-valued
        attributes are coerced from ``object`` dtype to fixed-width unicode so
        the resulting ``.npz`` can be reopened with ``allow_pickle=False``.
        """
        sep = self._CHECKPOINT_SEP
        prefix = f"_topology{sep}"
        u = self._universe
        out: dict[str, np.ndarray] = {
            f"{prefix}atom_resindex": np.asarray(u.atoms.resindices),
            f"{prefix}residue_segindex": np.asarray(u.residues.segindices),
        }
        for attr in u._topology.attrs:
            level = getattr(attr, "per_object", None)
            if level not in self._TOPOLOGY_LEVELS:
                # Skips derived indices and connectivity (bonds, angles, ...)
                # — not needed for inspecting a loaded snapshot.
                continue
            values = np.asarray(attr.values)
            if values.dtype == object:
                values = values.astype(str)
            out[f"{prefix}attr{sep}{level}{sep}{attr.attrname}"] = values
        return out

    @classmethod
    def _deserialize_topology(cls, npz: np.lib.npyio.NpzFile) -> mda.Universe:
        """Rebuild a trajectory-free :class:`MDAnalysis.Universe` from ``npz``."""
        sep = cls._CHECKPOINT_SEP
        prefix = f"_topology{sep}"
        atom_resindex = npz[f"{prefix}atom_resindex"]
        residue_segindex = npz[f"{prefix}residue_segindex"]
        universe = mda.Universe.empty(
            n_atoms=int(atom_resindex.shape[0]),
            n_residues=int(residue_segindex.shape[0]),
            n_segments=int(residue_segindex.max() + 1) if residue_segindex.size else 1,
            atom_resindex=atom_resindex,
            residue_segindex=residue_segindex,
            trajectory=False,
        )
        attr_prefix = f"{prefix}attr{sep}"
        for full_key in npz.files:
            if not full_key.startswith(attr_prefix):
                continue
            _, _, level_and_name = full_key.partition(attr_prefix)
            level, _, attr_name = level_and_name.partition(sep)
            if level not in cls._TOPOLOGY_LEVELS:
                continue
            universe.add_TopologyAttr(attr_name, npz[full_key])
        return universe


class AnalysisCollection(_Runner):
    """Running a collection of analysis classes on the same single trajectory.

    .. warning::

        ``AnalysisCollection`` is still experimental. You should not use it for anything
        important.

    An analyses with ``AnalysisCollection`` can lead to a speedup compared to running
    the individual analyses, since the trajectory loop is performed only once. The class
    requires that each analysis is a child of :class:`AnalysisBase`. Additionally, the
    trajectory of all ``analysis_instances`` must be the same. It is ensured that all
    analysis instances use the *same original* timestep and not an altered one from a
    previous analysis instance.

    Parameters
    ----------
    *analysis_instances : AnalysisBase
        Arbitrary number of analysis instances to be run on the same trajectory.

    Raises
    ------
    AttributeError
        If the provided ``analysis_instances`` do not work on the same trajectory.
    AttributeError
        If an ``analysis_instances`` is not a child of :class:`AnalysisBase`.

    Example
    -------
    >>> import MDAnalysis as mda
    >>> from maicos import DensityPlanar
    >>> from maicos.core import AnalysisCollection
    >>> from MDAnalysisTests.datafiles import TPR, XTC
    >>> u = mda.Universe(TPR, XTC)

    Select atoms

    >>> ag_O = u.select_atoms("name O")
    >>> ag_H = u.select_atoms("name H")

    Create the individual analysis instances

    >>> dplan_O = DensityPlanar(ag_O)
    >>> dplan_H = DensityPlanar(ag_H)

    Create a collection for common trajectory

    >>> collection = AnalysisCollection(dplan_O, dplan_H)

    Run the collected analysis

    >>> _ = collection.run(start=0, stop=100, step=10)

    Results are stored in the individual instances see :class:`AnalysisBase` on how to
    access them. You can also save all results of the analysis within one call:

    >>> collection.save()

    """

    def __init__(self, *analysis_instances: AnalysisBase) -> None:
        warnings.warn(
            "`AnalysisCollection` is still experimental. You should not use it for "
            "anything important.",
            stacklevel=2,
        )
        for analysis_object in analysis_instances:
            if analysis_instances[0]._trajectory != analysis_object._trajectory:
                raise ValueError(
                    "`analysis_instances` do not have the same trajectory."
                )
            if not isinstance(analysis_object, AnalysisBase):
                raise TypeError(
                    f"Analysis object {analysis_object} is "
                    "not a child of `AnalysisBase`."
                )

        self._analysis_instances = analysis_instances

    @render_docs
    def run(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        frames: int | None = None,
        verbose: bool | None = None,
        progressbar_kwargs: dict | None = None,
    ) -> Self:
        """${RUN_METHOD_DESCRIPTION}"""  # noqa: D415
        return _Runner._run(
            self,
            analysis_instances=self._analysis_instances,
            start=start,
            stop=stop,
            step=step,
            frames=frames,
            verbose=verbose,
            progressbar_kwargs=progressbar_kwargs,
        )

    def save(self) -> None:
        """Save results of all ``analysis_instances`` to disk.

        The methods calls the :meth:`save` method of all ``analysis_instances`` if
        available. If an instance has no :meth:`save` method a warning for this instance
        is issued.
        """
        for analysis_object in self._analysis_instances:
            if analysis_object.module_has_save:
                analysis_object.save()
            else:
                warnings.warn(
                    f"`{analysis_object}` has no save() method. Analysis results of "
                    "this instance can not be written to disk.",
                    stacklevel=2,
                )


@render_docs
class ProfileBase:
    """Base class for computing profiles.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${PROFILE_CLASS_PARAMETERS}
    ${PROFILE_CLASS_PARAMETERS_PRIVATE}

    Attributes
    ----------
    ${PROFILE_CLASS_ATTRIBUTES}

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        grouping: str,
        bin_method: str,
        output: str,
        weighting_function: Callable,
        weighting_function_kwargs: None | dict,
        normalization: str,
    ) -> None:
        self.atomgroup = atomgroup
        self.grouping = grouping.lower()
        self.bin_method = bin_method.lower()
        self.output = output
        self.normalization = normalization.lower()

        if weighting_function_kwargs is None:
            weighting_function_kwargs = {}

        self.weighting_function = lambda ag: weighting_function(
            ag, grouping, **weighting_function_kwargs
        )
        # We need to set the following dictionaries here because ProfileBase is not a
        # subclass of AnalysisBase (only needed for tests)
        self.results = Results()
        self._obs = Results()
        self.n_bins: int

    def _prepare(self):
        normalizations = ["none", "volume", "number"]
        if self.normalization not in normalizations:
            raise ValueError(
                f"Normalization '{self.normalization}' not supported. "
                f"Use {', '.join(normalizations)}."
            )

        groupings = ["atoms", "segments", "residues", "molecules", "fragments"]
        if self.grouping not in groupings:
            raise ValueError(
                f"'{self.grouping}' is not a valid option for "
                f"grouping. Use {', '.join(groupings)}."
            )
        logger.info(f"Atoms grouped by {self.grouping}.")

        # If unwrap has not been set we define it here
        if not hasattr(self, "unwrap"):
            self.unwrap = True

    def _compute_histogram(
        self, positions: np.ndarray, weights: np.ndarray | None = None
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

    def _single_frame(self) -> None | float:
        self._obs.profile = np.zeros(self.n_bins)
        self._obs.bincount = np.zeros(self.n_bins)

        if self.grouping == "atoms":
            positions = self.atomgroup.positions
        else:
            positions = get_center(
                self.atomgroup, bin_method=self.bin_method, compound=self.grouping
            )

        weights = self.weighting_function(self.atomgroup)
        self._obs.profile, bin_indices = self._compute_histogram(positions, weights)

        self._obs.bincount = np.bincount(
            bin_indices[bin_indices > -1],
            minlength=self.n_bins,
        )

        if self.normalization == "volume":
            self._obs.profile /= self._obs.bin_volume
        elif self.normalization == "number":
            with np.errstate(divide="ignore", invalid="ignore"):
                self._obs.profile /= self._obs.bincount
            self._pop.profile = np.nan_to_num(self._obs.bincount, nan=0)  # type: ignore
            self._var.profile, _ = (  # type: ignore
                self._compute_histogram(  # type: ignore
                    positions,
                    weights - self._obs.profile[bin_indices],  # type: ignore
                )
            )  # type: ignore
            with np.errstate(divide="ignore", invalid="ignore"):
                self._var.profile /= self._obs.bincount  # type: ignore
        return None

    def _conclude(self) -> None:
        self.results.profile = self.means.profile  # type: ignore
        self.results.dprofile = self.sems.profile  # type: ignore

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
        columns = ["positions [Å]"]

        columns.append("profile")
        columns.append("error")

        # Required attribute to use method from `AnalysisBase`
        AnalysisBase.savetxt(
            self,  # type: ignore
            self.output,
            np.vstack(
                (
                    self.results.bin_pos,
                    self.results.profile,
                    self.results.dprofile,
                )
            ).T,
            columns=columns,
        )
