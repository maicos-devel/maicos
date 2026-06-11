#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Streaming accumulator for means, variances and covariances.

A per-observable backend for the running statistics of
:class:`maicos.core.base.AnalysisBase`. Each observable keeps its own
mean/M2/pop/sums arrays, seeded on the first frame and merged *in place* on every
later frame, so after initialisation there are no per-key writes into the
:class:`~MDAnalysis.analysis.base.Results` containers (each such write revalidates
the key with an expensive ``dir`` call). numpy vectorises the merge over each
observable's bins; there is no stacking across observables, so the code is one
Welford step per observable.

Each requested covariance pair keeps its own running co-moment
:math:`C = \sum (x - \bar x)(y - \bar y)`, merged the same way. Only pairs whose
observables broadcast against each other and are co-sampled (identical
populations) are tracked. The diagonal of the covariance of the means is the
squared standard error of the mean, so :meth:`MomentAccumulator.cov` reads it
from :attr:`~MomentAccumulator.sems` directly.

The per-frame merge delegates to :func:`maicos.lib.math.combine_subsample_variance`
and :func:`maicos.lib.math.combine_subsample_covariance` -- the parallel Welford
update is implemented once there -- so empty frames and within-frame
variances/covariances are supported. Single-sample observables take an inlined
fast path that skips the ``nan_to_num`` scrubbing those helpers do.
"""

from itertools import combinations

import numpy as np
from MDAnalysis.analysis.base import Results

from .math import combine_subsample_covariance, combine_subsample_variance
from .util import joint_pop, make_pair_key

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


class MomentAccumulator:
    """Backend producing the means/sems/M2/pop/sums and ``C`` containers.

    Parameters
    ----------
    requested_pairs : iterable of tuple of str
        Canonical observable pairs whose off-diagonal covariance should be
        accumulated. Only the listed pairs are tracked. Empty (the default)
        disables covariance entirely.

    """

    def __init__(self, requested_pairs=()):
        # Running containers, owned by the accumulator and exposed by the
        # analysis. Each value is an array seeded by :meth:`initialize` and
        # updated in place; the buffers' identities never change afterwards.
        self.means = Results()  # mean of the observables across frames
        self.sems = Results()  # standard error of the mean across frames
        self.M2 = Results()  # second moment of the samples across frames
        self.pop = Results()  # count of samples across frames
        self.sums = Results()  # sum of the observables across frames
        self.C = Results()  # off-diagonal co-moments, keyed (i, j)
        self._requested_cov_pairs = set(requested_pairs)
        self._keys = []  # observable keys, in first-seen order
        self._pairs = []  # covariance pairs actually tracked
        # Keys for which the module reports a population (multiple samples per
        # frame, possibly empty bins). Their absence marks a single-sample key,
        # whose merge skips the NaN scrubbing. Recorded on the first frame;
        # presence is assumed stable across frames.
        self._multisample_keys = set()

    # -- per-frame normalisation --------------------------------------------

    def _sanitize(self, obs, _pop, _var):
        """Return per-frame observable / population / variance arrays.

        Lists and scalars become float arrays (scalars stay 0-d), and a missing
        population / variance defaults to a single sample with undefined
        within-frame spread. The input ``Results`` containers are not mutated, so
        a module's raw observables stay visible on the analysis for ``_conclude``.
        """
        s_obs, s_pop, s_var = {}, {}, {}
        for key in obs:
            value = obs[key]
            if not isinstance(value, _COMPATIBLE_TYPES):
                raise TypeError(f"Observable {key!r} has an incompatible type.")
            s_obs[key] = np.asarray(value, dtype=float)  # 0-d for scalars
            if key in _pop:
                s_pop[key] = np.asarray(_pop[key])
                s_var[key] = np.asarray(_var[key], dtype=float)
            else:
                # Single sample: unit population, no within-frame spread. Seeding
                # the variance with 0 (rather than NaN) keeps single-sample keys
                # NaN-free so their merge can skip ``nan_to_num``.
                s_pop[key] = np.ones(s_obs[key].shape, dtype=int)
                s_var[key] = np.zeros(s_obs[key].shape)
        return s_obs, s_pop, s_var

    # -- first-frame initialisation -----------------------------------------

    def initialize(self, obs, _pop, _var, _cov):
        """Seed the running containers from the first frame's observables.

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
        # Keys the module populates explicitly are multi-sample; the rest carry a
        # single sample per frame. Recorded before sanitize fills the defaults.
        self._multisample_keys = set(_pop)

        s_obs, s_pop, s_var = self._sanitize(obs, _pop, _var)
        for key in s_obs:
            # Own writable buffers, kept as (0-d for scalars) arrays so the
            # in-place merge can write through them. Arithmetic on 0-d arrays
            # collapses to an immutable numpy scalar, so wrap each result.
            self.means[key] = np.array(s_obs[key], dtype=float)
            self.pop[key] = np.array(s_pop[key])
            self.M2[key] = np.array(s_var[key] * s_pop[key], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                self.sems[key] = np.array(
                    np.sqrt(self.M2[key] / self.pop[key] ** 2), dtype=float
                )
            self.sums[key] = np.array(self.means[key] * self.pop[key], dtype=float)
            self._keys.append(key)
        self._build_pairs(s_obs, s_pop, _cov)

    def _build_pairs(self, s_obs, s_pop, _cov):
        """Seed ``C`` for the requested pairs that broadcast and are co-sampled."""
        for pair_key in self._requested_cov_pairs:
            key_i, key_j = pair_key
            for key in pair_key:
                if key not in s_obs:
                    raise KeyError(
                        f"requested covariance pair {set(pair_key)} references "
                        f"unknown observable {key!r}; available: {list(s_obs)}"
                    )
            try:
                shape = np.broadcast_shapes(s_obs[key_i].shape, s_obs[key_j].shape)
            except ValueError:
                continue  # shapes do not broadcast -> covariance not tracked
            if not np.array_equal(
                np.broadcast_to(s_pop[key_i], shape),
                np.broadcast_to(s_pop[key_j], shape),
            ):
                continue  # not co-sampled -> covariance is undefined
            # Seed with the first frame's within-frame co-moment (zero for
            # single-sample observables, where _cov is absent).
            seed = np.nan_to_num(_cov.get(pair_key, 0.0)) * joint_pop(
                s_pop[key_i], s_pop[key_j]
            )
            self.C[pair_key] = np.broadcast_to(seed, shape).astype(float).copy()
            self._pairs.append(pair_key)

    # -- per-frame update ---------------------------------------------------

    def update(self, obs, _pop, _var, _cov):
        """Welford merge of the current frame into all running statistics.

        Covariance is merged first, while the running means still hold their
        pre-frame values; the variance merge advances the means afterwards.

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
        s_obs, s_pop, s_var = self._sanitize(obs, _pop, _var)
        for pair_key in self._pairs:  # before the means move
            self._merge_cov(pair_key, s_obs, s_pop, _cov)
        for key in self._keys:
            self._merge_var(key, s_obs, s_pop, s_var)

    def _merge_var(self, key, s_obs, s_pop, s_var):
        """Merge the current frame into one observable's mean / variance."""
        x = s_obs[key]
        mean, M2, pop = self.means[key], self.M2[key], self.pop[key]
        n_old = pop  # alias: pop is overwritten last, after mean and M2 use n_old

        if key not in self._multisample_keys:
            # Single sample: n_new == 1, no NaN, no within-frame variance.
            n_tot = n_old + 1
            delta = mean - x
            M2[...] += delta**2 * n_old / n_tot
            mean[...] = x + delta * n_old / n_tot
            pop[...] = n_tot
            self.sems[key][...] = np.sqrt(M2 / pop**2)
            self.sums[key][...] += x
            return

        n_new = s_pop[key]
        # A = this frame, B = the running statistics so far.
        n_AB, mu_AB, M_AB = combine_subsample_variance(
            n_new, n_old, x, mean, s_var[key] * n_new, M2
        )
        pop[...] = n_AB
        mean[...] = mu_AB
        M2[...] = M_AB
        with np.errstate(divide="ignore", invalid="ignore"):
            self.sems[key][...] = np.sqrt(M2 / pop**2)
        self.sums[key][...] += np.nan_to_num(x) * n_new

    def _merge_cov(self, pair_key, s_obs, s_pop, _cov):
        """Merge the current frame into one pair's running co-moment.

        The per-key inputs may have different shapes (e.g. a scalar paired with a
        profile); numpy's arithmetic in :func:`combine_subsample_covariance`
        broadcasts them, and the running co-moment ``C`` anchors the pair's result
        shape, so no explicit broadcasting is needed here.

        Single-sample observables (no NaNs, unit population, no within-frame
        co-moment) are a special case of the same merge -- ``nan_to_num`` is a
        no-op on them, the within-frame co-moment is 0, and ``n_new`` is 1 -- so
        no separate branch is needed.
        """
        key_i, key_j = pair_key
        C = self.C[pair_key]
        n_new = s_pop[key_i]  # co-sampled: either key carries the population
        within = np.nan_to_num(_cov.get(pair_key, 0.0)) * n_new

        # A = this frame, B = the running co-moment so far.
        _, C[...] = combine_subsample_covariance(
            n_new,
            self.pop[key_i],
            s_obs[key_i],
            self.means[key_i],
            s_obs[key_j],
            self.means[key_j],
            within,
            C,
        )

    # -- error propagation --------------------------------------------------

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

        with np.errstate(invalid="ignore"):
            result = np.sqrt(var)
        if np.any(np.isnan(result)):
            # Variance went negative (usually an ill-conditioned covariance);
            # fall back to the uncorrelated estimate.
            var = 0.0
            for key in keys:
                var = var + grads[key] ** 2 * self.sems[key] ** 2
            result = np.sqrt(var)
        return result
