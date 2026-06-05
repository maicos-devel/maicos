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
from .util import make_pair_key

__all__ = ["MomentAccumulator"]

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
        self.pairs = []  # pair_key of index i, j in C_mat

    @property
    def has_matrix(self):
        return self.C_mat is not None



class _CovBlock:
    """Covariance of observable pairs with different shapes that broadcast."""

    # NOTE: only works if the broadcast shape is always the same
    def __init__(self, keys, shape):
        self.keys = list(keys)
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.shape = shape
        self.single_sample = True
        self.C_mat = np.zeros((len(self.keys), len(self.keys), *shape), dtype=float)
        self.pairs = []


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
        self._blocks = []  # stacked observable blocks
        self._cov_blocks = []  # broadcast covariance blocks (off-diagonal only)

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
            if key not in _pop:
                _pop[key] = np.ones(shape, dtype=int)
                _var[key] = np.empty(shape, dtype=float)
                _var[key].fill(np.nan) # TODO: maybe remove nan, do zeroes instead
            obs[key] = np.reshape(obs[key], shape)
            _pop[key] = np.reshape(obs[key], shape)
            _var[key] = np.reshape(obs[key], shape)

        # Group observables by shape
        shape_groups = {}  # shape -> [keys]
        for key in obs:
            shape = np.shape(obs[key])
            if shape == ():
                shape = (1,) # make scalar into 1D array with 1 element
                obs[key] = np.reshape(obs[key], shape)
                continue
            shape_groups[shape].append(key)
            shape_of_key[key] = shape

        # Group the covariances by shape
        embedded_covariances = {}
        cross_covariances = {}
        for pair_key in self._requested_cov_pairs:
            key_i, key_j = pair_key
            group_i, group_j = shape_of_key[key_i], shape_of_key[key_j]
            if group_i == group_j:
                embedded_covariances[group_i].append(pair_key)
            else:
                cross_covariances.append(pair_key)

        # Build stacked data blocks.
        for shape, keys in shape_groups.items():
            block = _Block(keys, shape)
            block.MEANS = np.stack([obs[key] for key in keys])
            block.POP = np.stack([_pop[key] for key in keys])
            block.M2 = np.stack([_var[key] for key in keys]) * block.POP
            with np.errstate(divide="ignore", invalid="ignore"):
                block.SEMS = block.M2 / block.POP**2
            block.SUMS = block.MEANS * block.POP

            try:
                pair_keys = embedded_covariances[shape]
                n = len(keys)
                block.C_mat = np.zeros((n, n, *shape), dtype=float)
                # Seed diagonal with M2, off-diagonal with the requested C entries.
                for key, index in block.index.items():
                    block.C_mat[index, index] = block.M2[index]
                for pair_key in pairs_key:
                    i, j = block.index[pair_key[0]], block.index[pair_key[1]]
                    block.C_mat[i, j] = _cov[pair_key]
                    block.C_mat[j, i] = _cov[pair_key]
                    block.pairs.append((i, j, pair_key))
                    self.C[pair_key] = block.C_mat[i, j]
                for key, index in block.index.items():
                    self.M2[key] = block.C_mat[index, index]

            except KeyError:
                for key, index in block.index:
                    self.M2[key] = block.M2[index]
            
            # Point the Results() object to the block arrays
            for key, index in block.index.items():
                self.means[key] = block.MEANS[index]
                self.pop[key] = block.POP[index]
                self.sems[key] = block.SEMS[index]
                self.sums[key] = block.SUMS[index]
            self._blocks.append(block)

        # Covariances of observables with different shapes
        self._organize_cross(cross, _pop)

    def _organize_cross(self, cross, _pop):
        by_shape = {}
        for pair_key in cross:
            by_shape.setdefault(np.shape(self.C[pair_key]), []).append(pair_key)

        for shape, pair_keys in by_shape.items():
            if shape == ():
                self._cov_fallback.extend(pair_keys)
                continue
            groups = {}
            for pair_key in pair_keys:
                sig = self._signature(pair_key[0], shape, self.pop, _pop)
                groups.setdefault(sig, []).append(pair_key)
            for group_pairs in groups.values():
                keys = sorted({k for pk in group_pairs for k in pk})
                if not self._cosampled(keys, shape, self.pop, _pop):
                    self._cov_fallback.extend(group_pairs)
                    continue
                block = _CovBlock(keys, shape)
                block.single_sample = bool(np.all(_pop[keys[0]] == 1))
                for pk in group_pairs:
                    i, j = block.index[pk[0]], block.index[pk[1]]
                    block.C_mat[i, j] = self.C[pk]
                    block.C_mat[j, i] = self.C[pk]
                    block.pairs.append((i, j, pk))
                    self.C[pk] = block.C_mat[i, j]
                self._cov_blocks.append(block)


    # -- shared helpers -----------------------------------------------------

    @staticmethod
    def _stack(keys, shape, container, default=None):
        """Stack `container` values for `keys`, broadcast to `shape`, as float."""
        return np.stack(
            [
                np.broadcast_to(container.get(k, default), shape).astype(float)
                for k in keys
            ]
        )

    @staticmethod
    def _signature(key, shape, pop, _pop):
        """Co-sampling fingerprint: running and frame populations on `shape`."""
        run = np.broadcast_to(pop[key], shape)
        frame = np.broadcast_to(_pop[key], shape)
        return (run.tobytes(), frame.tobytes())

    @staticmethod
    def _cosampled(keys, shape, pop, _pop):
        """True if all keys share one (broadcast) running and frame population."""
        run0 = np.broadcast_to(pop[keys[0]], shape)
        frame0 = np.broadcast_to(_pop[keys[0]], shape)
        for k in keys[1:]:
            if not np.array_equal(np.broadcast_to(pop[k], shape), run0):
                return False
            if not np.array_equal(np.broadcast_to(_pop[k], shape), frame0):
                return False
        return True

    @staticmethod
    def _joint_pop(pop, ki, kj):
        b = np.broadcast_arrays(pop[ki], pop[kj])
        return b[0] if np.ndim(pop[ki]) > np.ndim(pop[kj]) else b[1]

    # -- per-frame update ---------------------------------------------------

    def update(self, obs, _pop, _var, _cov):
        """Streaming merge of the current frame into all running statistics.

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
            if key not in _pop:
                _pop[key] = np.ones(np.shape(obs[key]), dtype=int)
                _var[key] = np.zeros(np.shape(obs[key]), dtype=float)

        if not self._organized:
            self._organize(obs, _pop)

        for block in self._cov_blocks:
            self._update_cov_block(block, obs, _pop, _cov)
        for pair_key in self._cov_fallback:
            self._update_cov_fallback(pair_key, obs, _pop, _cov)

        for block in self._blocks:
            self._update_block(block, obs, _pop, _var, _cov)
        for key in self._var_fallback:
            self._update_var_key(key, obs, _pop, _var)

    # -- queries / error propagation ----------------------------------------

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
        return self._joint_pop(self.pop, key_i, key_j)

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
        if not self._requested:
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

    def _update_block(self, block, obs, _pop, _var, _cov):
        shape = block.shape
        keys = block.keys
        x_raw = self._stack(keys, shape, obs)
        var = self._stack(keys, shape, _var, 0.0)
        n_new = self._stack(keys, shape, _pop)
        n_old = block.POP
        n_tot = n_old + n_new

        x = np.nan_to_num(x_raw)
        delta = np.nan_to_num(block.MEANS) - x  # (N, *shape)

        with np.errstate(divide="ignore", invalid="ignore"):
            if block.has_matrix:
                self._update_matrix(block, delta, var, n_new, n_old, n_tot, shape, _cov)
            else:
                block.M2[:] = (
                    np.nan_to_num(var * n_new)
                    + np.nan_to_num(block.M2)
                    + delta**2 * n_new * n_old / n_tot
                )
            block.MEANS[:] = x + delta * n_old / n_tot
            block.POP[:] = n_tot
            diag = (
                np.einsum("ii...->i...", block.C_mat) if block.has_matrix else block.M2
            )
            block.SEMS[:] = np.sqrt(diag / block.POP**2)
        block.SUMS[:] += x_raw * n_new

    def _update_matrix(self, block, delta, var, n_new, n_old, n_tot, shape, _cov):
        # One co-moment merge for the whole block: diagonal == variance M2,
        # off-diagonal == covariance. Populations are shared across the block.
        weight = n_new[0] * n_old[0] / n_tot[0]
        merged = (
            np.nan_to_num(block.C_mat)
            + np.einsum("i...,j...->ij...", delta, delta) * weight
        )
        # Within-frame term: diagonal var * n_frame, off-diagonal _cov * n_frame.
        n_frame = n_new[0]
        diag_within = np.nan_to_num(var * n_new)  # (N, *shape)
        for i in range(len(block.keys)):
            merged[i, i] += diag_within[i]
        if not block.single_sample:
            for i, j, pair_key in block.pairs:
                c = _cov.get(pair_key)
                if c is None:
                    continue
                w = np.nan_to_num(np.broadcast_to(c, shape)) * n_frame
                merged[i, j] += w
                merged[j, i] += w
        block.C_mat[:] = merged

    def _update_var_key(self, key, obs, _pop, _var):
        self.pop[key], self.means[key], self.M2[key] = combine_subsample_variance(
            _pop[key],
            self.pop[key],
            obs[key],
            self.means[key],
            _var[key] * _pop[key],
            self.M2[key],
        )
        self.sems[key] = np.sqrt(self.M2[key] / self.pop[key] ** 2)
        self.sums[key] += obs[key] * _pop[key]

    def _update_cov_block(self, block, obs, _pop, _cov):
        shape = block.shape
        keys = block.keys
        X = self._stack(keys, shape, obs)
        X = np.nan_to_num(X)
        MB = np.stack(
            [np.nan_to_num(np.broadcast_to(self.means[k], shape)) for k in keys]
        )
        n_old = np.broadcast_to(self.pop[keys[0]], shape).astype(float)
        n_new = np.broadcast_to(_pop[keys[0]], shape).astype(float)

        dx = X - MB
        n_tot = n_old + n_new
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(n_tot > 0, n_old * n_new / n_tot, 0.0)
        block.C_mat += np.einsum("i...,j...->ij...", dx, dx) * weight

        if not block.single_sample:
            for i, j, pair_key in block.pairs:
                c = _cov.get(pair_key)
                if c is None:
                    continue
                within = np.nan_to_num(np.broadcast_to(c, shape)) * n_new
                block.C_mat[i, j] += within
                block.C_mat[j, i] += within

    def _update_cov_fallback(self, pair_key, obs, _pop, _cov):
        ki, kj = pair_key
        within = _cov.get(pair_key, 0.0) * self._joint_pop(_pop, ki, kj)
        _, self.C[pair_key] = combine_subsample_covariance(
            _pop[ki],
            self.pop[ki],
            obs[ki],
            self.means[ki],
            obs[kj],
            self.means[kj],
            within,
            self.C[pair_key],
        )
