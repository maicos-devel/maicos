#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Vectorized accumulator for means, variances and covariances.

A single blocked backend for the running statistics of
:class:`maicos.core.base.AnalysisBase`, replacing a per-key variance merge and a
per-pair covariance merge with vectorized array operations.

Observables that share a result shape and per-frame/running population are
stacked into a *block* and merged with a handful of vectorized array ops instead
of one :func:`combine_subsample_variance` call per observable. ``means``,
``sems``, ``M2``, ``pop`` and ``sums`` are rebound to *views* into the per-block
stacked arrays, so the public API is unchanged.

Requested covariance pairs whose two observables live in the *same* variance
block (identical shape and population) are tracked in a single ``(N, N, *shape)``
co-moment matrix whose diagonal **is** ``M2`` -- one ``einsum`` yields both the
variance and the covariance with no redundant work, and ``M2`` / ``C`` are views
into that matrix. Pairs that broadcast across shapes (e.g. a scalar with an
array) are accumulated off-diagonal only, and scalar entries fall back to the
original per-key / per-pair merges.

The merge mirrors :func:`combine_subsample_variance` /
:func:`combine_subsample_covariance` exactly (same ``nan_to_num`` handling), so
results are identical to the per-key / per-pair loops."""

from itertools import combinations

import numpy as np
from MDAnalysis.analysis.base import Results

from .math import combine_subsample_covariance, combine_subsample_variance
from .util import make_pair_key, joint_pop, is_cosampled

__all__ = ["MomentAccumulator"]


class _WriteThruResults(Results):
    """Results subclass that writes into an existing numpy array on __setitem__.

    When the key already holds a numpy array, assignment writes into it in-place
    (preserving any views into that array) rather than rebinding the reference.
    """

    def __setitem__(self, key, value):
        if key in self and isinstance(self.data[key], np.ndarray):
            existing = self.data[key]
            if existing.ndim == 0:
                existing[()] = np.asarray(value).flat[0]
            else:
                existing[:] = np.asarray(value)
        else:
            super().__setitem__(key, value)


#: Observable types the accumulator knows how to merge.
_COMPATIBLE_TYPES = (
    np.ndarray,
    float,
    int,
    list,
    np.float32,
    np.float64,
    np.int32,
    np.int64,
)


class _Block:
    """Stacked observables sharing a shape and population.

    Always carries the per-observable means/M2/pop/sems/sums. When the block has
    embedded same-shape covariance pairs it additionally keeps the full co-moment
    matrix ``C_mat`` whose diagonal aliases ``M2``.
    """

    def __init__(self, keys, shape):
        self.keys = list(keys)
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.shape = shape
        self.single_sample = True
        self.MEANS = None  # (N, *shape)
        self.POP = None  # (N, *shape)
        self.SEMS = None  # (N, *shape)
        self.SUMS = None  # (N, *shape)
        self.M2 = None  # (N, *shape); moment of the variance
        self.C_mat = None  # (N, N, *shape) covariance matrix of pairs
        # TODO: maybe reduce C_mat to only requested pairs
        self.pairs = {}  # (i, j) index in C_mat -> pair_key
        self._cov_mat = None  # (N, N, *shape)
        self._obs_mat = None  # (N, *shape)  within-frame observables
        self._pop_mat = None  # (N, *shape)  within-frame populations
        self._var_mat = None  # (N, *shape)  within-frame variances

    @property
    def has__cov_mat(self):
        return self._cov_mat is not None



class _CovBlock:
    """Covariance of observable pairs with different shapes that broadcast."""

    # NOTE: only works if the broadcast shape is always the same
    def __init__(self, keys, shape):
        self.keys = list(keys)
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.shape = shape
        self.C_mat = np.zeros((len(self.keys), len(self.keys), *shape), dtype=float)
        self.pairs = {}  # (i, j) index in C_mat -> pair_key


class MomentAccumulator:
    """One backend producing the same means/sems/M2/pop/sums and ``C`` containers.

    Parameters
    ----------
    requested_pairs : iterable of tuple of str
        Canonical observable pairs whose
        off-diagonal covariance should be accumulated. Only the listed pairs are
        tracked. Empty (the default) disables covariance entirely.

    """

    def __init__(self, requested_pairs=()):
        # Running containers, owned by the accumulator and exposed by the
        # analysis. They are populated by :meth:`initialize` on the first frame.
        self.means = Results()  # mean of the observables across frames
        self.sems = Results()  # standard error of the mean across frames
        self.M2 = Results()  # second moment of the samples across frames
        self.pop = Results()  # count of samples across frames
        self.sums = Results()  # sum of the observables across frames
        self.C = Results()  # off-diagonal co-moments, keyed (i, j)
        self._requested_cov_pairs = set(requested_pairs)
        self._scalar_keys: set = set()
        self._blocks = []  # stacked observable blocks
        self._cov_blocks = []  # broadcast covariance blocks (off-diagonal only)
        self._cov = _WriteThruResults()
        self._obs = _WriteThruResults()
        self._pop = _WriteThruResults()
        self._var = _WriteThruResults()

    # -- first-frame initialisation -----------------------------------------

    def initialize(self, obs, _pop, _var, _cov):
        """Seed the running containers from the first frame's observables.

        Single-sample observables get a unit population and an undefined (NaN)
        within-frame variance; the requested covariance pairs are seeded once
        their observables are shown to broadcast and be co-sampled.

        Parameters
        ----------
        obs : MDAnalysis.analysis.base.Results
            The current frame's observables, keyed by observable name.
        _pop : MDAnalysis.analysis.base.Results
            The current frame's sample count for each observable. Missing entries
            default to a unit population (the observable is a single sample).
        _var : MDAnalysis.analysis.base.Results
            The current frame's within-frame variance for each observable.
        _cov : MDAnalysis.analysis.base.Results
            The current frame's within-frame covariance, keyed by canonical
            observable pair (see :func:`maicos.lib.util.make_pair_key`).

        """

        # Sanitize data
        for key in obs:
            if not isinstance(obs[key], _COMPATIBLE_TYPES):
                raise TypeError(f"Observable {key!r} has an incompatible type.")
            if isinstance(obs[key], list):
                obs[key] = np.array(obs[key])
            shape = np.shape(obs[key])
            # reshape scalar to arrays of length 1
            if np.shape(obs[key]) == ():
                shape = (1,)
                self._scalar_keys.add(key)
            if key not in _pop:
                _pop[key] = np.ones(shape, dtype=int)
                _var[key] = np.zeros(shape, dtype=float)
            obs[key] = np.reshape(obs[key], shape)
            _pop[key] = np.reshape(_pop[key], shape)
            _var[key] = np.reshape(_var[key], shape)

        # Group observables by shape
        shape_groups = {}  # shape -> [keys]
        shape_of_key = {}
        for key in obs:
            shape = np.shape(obs[key])
            if shape == ():
                shape = (1,) # make scalar into 1D array with 1 element
                obs[key] = np.reshape(obs[key], shape)
            shape_groups.setdefault(shape, []).append(key)
            shape_of_key[key] = shape

        # Group the covariances by shape
        embedded_covariances = {}
        cross_covariances = []
        for pair_key in self._requested_cov_pairs:
            key_i, key_j = pair_key
            group_i, group_j = shape_of_key[key_i], shape_of_key[key_j]
            if group_i == group_j:
                embedded_covariances.setdefault(group_i, []).append(pair_key)
            else:
                cross_covariances.append(pair_key)

        # Build stacked data blocks.
        for shape, keys in shape_groups.items():
            block = _Block(keys, shape)
            keys = block.keys
            block._obs_mat = np.stack([obs[key] for key in keys]).astype(float)
            block._pop_mat = np.stack([_pop[key] for key in keys]).astype(int)
            block._var_mat = np.stack([_var[key] for key in keys]).astype(float)
            block.MEANS = block._obs_mat.copy()
            block.POP = block._pop_mat.copy()
            block.M2 = block._var_mat.copy() * block.POP
            with np.errstate(divide="ignore", invalid="ignore"):
                block.SEMS = np.sqrt(block.M2 / block.POP**2)
            block.SUMS = block.MEANS * block.POP

            
            try:
                pair_keys = embedded_covariances[shape]
                n = len(keys)
                block.C_mat = np.zeros((n, n, *shape), dtype=float)
                # Seed diagonal with M2, off-diagonal with the requested C entries.
                for key, index in block.index.items():
                    block.C_mat[index, index] = block.M2[index]
                for pair_key in pair_keys:
                    i, j = block.index[pair_key[0]], block.index[pair_key[1]]
                    if not is_cosampled(_pop[pair_key[0]], _pop[pair_key[1]]):
                        continue
                    jpop = joint_pop(_pop[pair_key[0]], _pop[pair_key[1]])
                    # check if it's a single sample
                    if np.all(jpop == 1):
                        # Observable is a single sample, so _cov is 0
                        _cov[pair_key] = np.zeros(shape, dtype=float)
                    block.pairs[(i, j)] = pair_key
                    self.C[pair_key] = block.C_mat[i, j]
                self._create__cov_mat(block, _pop, _cov, _var)
                block.C_mat[:] = block._cov_mat * block.POP
                for key, index in block.index.items():
                    v = block.C_mat[index, index]
                    self.M2[key] = v.squeeze() if key in self._scalar_keys else v
            except KeyError:
                for key, index in block.index.items():
                    v = block.M2[index]
                    self.M2[key] = v.squeeze() if key in self._scalar_keys else v

            # Point the Results() object to the block arrays
            for key, index in block.index.items():
                sq = key in self._scalar_keys
                self.means[key] = block.MEANS[index].squeeze() if sq else block.MEANS[index]
                self.pop[key] = block.POP[index].squeeze() if sq else block.POP[index]
                self.sems[key] = block.SEMS[index].squeeze() if sq else block.SEMS[index]
                self.sums[key] = block.SUMS[index].squeeze() if sq else block.SUMS[index]
                self._obs[key] = block._obs_mat[index].squeeze() if sq else block._obs_mat[index]
                self._pop[key] = block._pop_mat[index].squeeze() if sq else block._pop_mat[index]
                self._var[key] = block._var_mat[index].squeeze() if sq else block._var_mat[index]
            self._blocks.append(block)

        # Covariances of observables with different shapes
        # first, sort them by pair_shape, so they can be blocked
        pair_shape_groups = {}
        for pair_key in cross_covariances:
            key_x, key_y = pair_key
            try:
                pair_shape = np.broadcast_shapes(
                    np.shape(obs[key_x]), np.shape(obs[key_y])
                )
            except ValueError:
                continue  # shapes do not broadcast -> covariance not tracked
            # TODO: check cosampling properly
            jpop = joint_pop(_pop[key_x], _pop[key_y])
            # check if it's a single sample
            if np.all(jpop == 1):
                # Observable is a single sample, so _cov is 0
                _cov[pair_key] = np.zeros(pair_shape, dtype=float)

            pair_shape_groups.setdefault(pair_shape, []).append(pair_key)

        # prepare blocks for cross-covariances
        for pair_shape, pair_keys in pair_shape_groups.items():
            unique_keys = list(dict.fromkeys(x for pair in pair_keys for x in pair))
            cov_block = _CovBlock(unique_keys, pair_shape)
            for pair_key in pair_keys:
                if is_cosampled(_pop[pair_key[0]], _pop[pair_key[1]]):
                    i = cov_block.index[pair_key[0]]
                    j = cov_block.index[pair_key[1]]
                    cov_block.pairs[(i, j)] = pair_key
                    self.C[pair_key] = cov_block.C_mat[i, j]
            self._create__cov_mat(cov_block, _pop, _cov, _var)
            _pop_stacked = np.stack([np.broadcast_to(_pop[key], cov_block.shape) for key in cov_block.keys])
            cov_block.C_mat[:] = cov_block._cov_mat * _pop_stacked

            self._cov_blocks.append(cov_block)

    # -- per-frame update ---------------------------------------------------
    def update(self, obs, _pop, _var, _cov):
        """Welford merge of the current frame into all running statistics.

        Covariance that broadcasts across shapes is updated first, while the
        running means still hold their pre-frame values; the variance blocks
        (which also drive embedded covariance) update the means afterwards.

        Parameters
        ----------
        obs : MDAnalysis.analysis.base.Results
            The current frame's observables, keyed by observable name.
        _pop : MDAnalysis.analysis.base.Results
            The current frame's sample count for each observable. Missing entries
            default to a unit population (the observable is a single sample).
        _var : MDAnalysis.analysis.base.Results
            The current frame's within-frame variance for each observable.
        _cov : MDAnalysis.analysis.base.Results
            The current frame's within-frame covariance, keyed by canonical
            observable pair (see :func:`maicos.lib.util.make_pair_key`).

        """
        # Sanitize: arrays for list observables, unit population / zero
        # within-frame variance for single-sample observables.
        for key in obs:
            if isinstance(obs[key], list):
                obs[key] = np.array(obs[key])
            shape = np.shape(obs[key])
            if np.shape(obs[key]) == ():
                shape = (1,)
            if key not in _pop:
                _pop[key] = np.ones(shape, dtype=int)
                _var[key] = np.zeros(shape, dtype=float)

            obs[key] = np.reshape(obs[key], shape)
            _pop[key] = np.reshape(_pop[key], shape)
            _var[key] = np.reshape(_var[key], shape)

        for block in self._cov_blocks:
            self._update_cov_block(block, obs, _pop, _cov)

        for block in self._blocks:
            self._update_block(block)

    def _update_cov_block(self, block, obs, _pop, _cov):
        shape = block.shape
        keys = block.keys

        obs_stacked = np.stack([np.broadcast_to(obs[key], shape) for key in keys])
        means_stacked = np.stack([np.broadcast_to(self.means[key], shape) for key in keys])
        pop_stacked = np.stack([np.broadcast_to(self.pop[key], shape) for key in keys])
        _pop_stacked = np.stack([np.broadcast_to(_pop[key], shape) for key in keys])

        combine_subsample_covariance(pop_stacked, _pop_stacked,
                                        means_stacked, obs_stacked,
                                        means_stacked, obs_stacked,
                                        block.C_mat, block._cov_mat * _pop_stacked, block.C_mat)

    def _update_block(self, block):
        if block.has__cov_mat:
            combine_subsample_covariance(block.POP, block._pop_mat,
                                         block.MEANS, block._obs_mat,
                                         block.MEANS, block._obs_mat,
                                         block.C_mat, block._cov_mat * block._pop_mat, block.C_mat)

        combine_subsample_variance(block.POP, block._pop_mat, block.MEANS, block._obs_mat, block.M2, block._var_mat * block._pop_mat, block.POP, block.MEANS, block.M2)

        with np.errstate(divide="ignore", invalid="ignore"):
            block.SEMS[:] = np.sqrt(block.M2 / block.POP**2)
        block.SUMS[:] = block.MEANS * block.POP

    def _create__cov_mat(self, block, _pop, _cov, _var):
        #TODO: there should be a memory matrix, that the user writes directly to via views.
        N = len(block.keys)
        block._cov_mat = np.zeros((N, N, *block.shape), dtype=float)
        for indices, pair_key in block.pairs.items():
            i = indices[0]
            j = indices[1]
            jpop = joint_pop(_pop[pair_key[0]], _pop[pair_key[1]])
            if np.all(jpop == 1):
                block._cov_mat[i,j] = np.zeros(block.shape, dtype=float)
            block._cov_mat[i, j] = np.broadcast_to(
                _cov[pair_key], block.shape
            ).astype(float)
            self._cov[pair_key] = block._cov_mat[i, j]

        if _var is not None:
            for key, i in block.index.items():
                block._cov_mat[i, i] = _var[key] * _pop[key]

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
        if not self._requested_cov_pairs:
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
        return self.C[pair_key] / joint_pop(self.pop[key_i], self.pop[key_j]) ** 2

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

