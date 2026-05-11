#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Streaming multi-tau autocorrelator with ring buffers.

The :class:`MultiTauCorrelator` accumulates the time autocorrelation of an
array-valued signal from a frame-by-frame stream, with bounded memory and
approximately logarithmically spaced lag times. It is independent of the
MAICoS analysis machinery and reusable for any time-correlation observable
(velocity, dipole, scattering phases, stress, ...).

The algorithm follows the multi-tau scheme of Schätzel [Schaetzel1990]_ and
the ring-buffer formulation of Ramírez et al. [Ramirez2010]_.

References
----------
.. [Schaetzel1990] K. Schätzel, *Quantum Optics: Journal of the European
   Optical Society Part B* **2**, 287 (1990).
.. [Ramirez2010] J. Ramírez, S. K. Sukumaran, B. Vorselaars, A. E. Likhtman,
   *J. Chem. Phys.* **133**, 154103 (2010).
"""

import numpy as np


class MultiTauCorrelator:
    r"""Streaming multi-tau autocorrelator with ring buffers.

    Computes the time autocorrelation function

    .. math::

        C(\tau) = \left\langle A^{*}(t_0)\, A(t_0 + \tau) \right\rangle_{t_0}

    of an array-valued signal :math:`A(t)` from a frame-by-frame stream.
    The lag grid is approximately logarithmic, allowing several decades of
    dynamic range to be resolved with a memory footprint that is independent
    of the length of the input signal.

    The correlator maintains :math:`p` levels of ring buffers, each holding
    :math:`m` samples. Level :math:`k \geq 1` operates on samples that have
    been averaged over :math:`2^k` consecutive raw frames; level :math:`0`
    sees raw frames. Level :math:`0` accumulates all :math:`m` lags
    :math:`\{0, 1, \dots, m-1\}`. Each higher level :math:`k` accumulates
    only the upper-half lags :math:`\{m/2, \dots, m-1\}` at spacing
    :math:`2^k` raw frames; the lower half is already covered by the finer
    level. The total lag grid is therefore

    .. math::

        \{0, 1, \dots, m-1\}\;\cup\;\bigcup_{k=1}^{p-1}
        \{m/2,\, m/2+1,\, \dots,\, m-1\} \cdot 2^k

    in units of frame intervals.

    Parameters
    ----------
    shape : tuple of int, optional
        Shape of the per-frame signal. Use ``()`` for a scalar signal,
        ``(N,)`` for a vector, ``(N, 3)`` for vector-per-particle data,
        etc. The correlator treats every array element independently.
    num_levels : int
        Number of coarsening levels :math:`p`. Higher levels remain empty
        until enough frames have been ingested. Must be at least 1.
    channels_per_level : int
        Number of channels per level :math:`m`. Must be even and at
        least 2. Conventional choice is 16.
    dtype : numpy dtype
        Data type of the buffers and accumulators. Use a complex dtype
        (e.g. ``numpy.complex128``) for complex signals. The autocorrelation
        convention then uses the conjugate of the older sample.

    Attributes
    ----------
    shape : tuple
        Shape of the per-frame observable.
    num_levels : int
        Number of coarsening levels :math:`p`.
    channels_per_level : int
        Number of channels per level :math:`m`.
    dtype : numpy.dtype
        Buffer / accumulator dtype.

    Examples
    --------
    Autocorrelation of a scalar signal:

    >>> import numpy as np
    >>> from maicos.lib.correlator import MultiTauCorrelator
    >>> rng = np.random.default_rng(0)
    >>> c = MultiTauCorrelator(num_levels=4, channels_per_level=8)
    >>> for v in rng.normal(size=1000):
    ...     c.add(v)
    >>> lags = c.lags
    >>> corr = c.correlation
    >>> corr.shape == lags.shape
    True

    """

    def __init__(
        self,
        shape: tuple = (),
        num_levels: int = 20,
        channels_per_level: int = 16,
        dtype=np.float64,
    ) -> None:
        if channels_per_level < 2:
            raise ValueError("`channels_per_level` must be at least 2.")
        if channels_per_level % 2 != 0:
            raise ValueError("`channels_per_level` must be even.")
        if num_levels < 1:
            raise ValueError("`num_levels` must be at least 1.")

        self.shape = tuple(int(s) for s in shape) if shape else ()
        self.num_levels = int(num_levels)
        self.channels_per_level = int(channels_per_level)
        self.dtype = np.dtype(dtype)
        self._is_complex = np.issubdtype(self.dtype, np.complexfloating)

        p, m = self.num_levels, self.channels_per_level
        per_slot_shape = (p, m) + self.shape
        self._buffer = np.zeros(per_slot_shape, dtype=self.dtype)
        self._accum = np.zeros(per_slot_shape, dtype=self.dtype)
        self._count = np.zeros((p, m), dtype=np.int64)
        # Ring head per level (most recent insertion); -1 means empty.
        self._head = np.full(p, -1, dtype=np.int64)
        # Number of samples that have entered each level so far.
        self._n_inserted = np.zeros(p, dtype=np.int64)
        # Pending half-coarsened sample at each level waiting for its partner.
        self._pending = np.zeros((p,) + self.shape, dtype=self.dtype)
        self._has_pending = np.zeros(p, dtype=bool)

        self._lags = self._build_lag_schedule()

    def _build_lag_schedule(self) -> np.ndarray:
        p, m = self.num_levels, self.channels_per_level
        lags = list(range(m))
        for k in range(1, p):
            for j in range(m // 2, m):
                lags.append(j * (1 << k))
        return np.asarray(lags, dtype=np.int64)

    @property
    def lags(self) -> np.ndarray:
        """Lag grid in units of frame intervals, shape ``(n_lags,)``."""
        return self._lags

    @property
    def n_lags(self) -> int:
        """Total number of distinct lag times."""
        return self._lags.size

    @property
    def n_frames(self) -> int:
        """Number of raw frames ingested so far."""
        return int(self._n_inserted[0])

    def add(self, value) -> None:
        """Ingest one frame of the signal.

        Parameters
        ----------
        value : array_like
            Signal value for the current frame. Must broadcast to
            :attr:`shape`.
        """
        arr = np.asarray(value, dtype=self.dtype)
        if arr.shape != self.shape:
            raise ValueError(
                f"Expected signal of shape {self.shape}, got {arr.shape}."
            )
        self._ingest(0, arr)

    def _ingest(self, k: int, value: np.ndarray) -> None:
        if k >= self.num_levels:
            return
        m = self.channels_per_level

        # Advance head, write into ring buffer.
        self._head[k] = (self._head[k] + 1) % m
        head = int(self._head[k])
        self._buffer[k, head] = value
        self._n_inserted[k] += 1
        n_have = int(self._n_inserted[k])

        # Accumulate correlations for active lags at this level.
        active_start = 0 if k == 0 else m // 2
        max_j = min(m, n_have)
        if max_j > active_start:
            js = np.arange(active_start, max_j)
            idxs = (head - js) % m
            partner = self._buffer[k, idxs]
            if self._is_complex:
                partner = np.conj(partner)
            # value has shape self.shape; partner has shape (len(js),) + shape.
            # Broadcasting prepends len(js) to value.
            self._accum[k, js] += value * partner
            self._count[k, js] += 1

        # Pass a coarsened sample to the next level once we have a pair.
        if k + 1 < self.num_levels:
            if not self._has_pending[k]:
                self._pending[k] = value
                self._has_pending[k] = True
            else:
                avg = 0.5 * (self._pending[k] + value)
                self._has_pending[k] = False
                self._pending[k] = 0
                self._ingest(k + 1, avg)

    @property
    def correlation(self) -> np.ndarray:
        """Autocorrelation estimate, shape ``(n_lags, *shape)``.

        Lags with zero pair count are returned as ``NaN``.
        """
        p, m = self.num_levels, self.channels_per_level
        out = np.empty((self.n_lags,) + self.shape, dtype=self.dtype)
        idx = 0
        for j in range(m):
            c = self._count[0, j]
            if c > 0:
                out[idx] = self._accum[0, j] / c
            else:
                out[idx] = np.nan
            idx += 1
        for k in range(1, p):
            for j in range(m // 2, m):
                c = self._count[k, j]
                if c > 0:
                    out[idx] = self._accum[k, j] / c
                else:
                    out[idx] = np.nan
                idx += 1
        return out

    @property
    def counts(self) -> np.ndarray:
        """Number of pairs accumulated per lag, shape ``(n_lags,)``."""
        p, m = self.num_levels, self.channels_per_level
        counts = np.empty(self.n_lags, dtype=np.int64)
        idx = 0
        for j in range(m):
            counts[idx] = self._count[0, j]
            idx += 1
        for k in range(1, p):
            for j in range(m // 2, m):
                counts[idx] = self._count[k, j]
                idx += 1
        return counts
