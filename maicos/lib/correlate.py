#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np

from .utils import zero_pad


def _corr_pad(a):
    """Zero pads an array to the nearest power of 2, for use with Fourier transforms."""
    new_len = 2**int(np.ceil((np.log(len(a)) / np.log(2))))
    return zero_pad(a, length=new_len)


def correlate(a, b=None, subtract_mean=False):
    """
    Uses fast fourier transforms to give the correlation function
    of two arrays, or, if only one array is given, the autocorrelation.
    Setting subtract_mean = True causes the mean to be subtracted from the input data.
    """
    a_mean = int(subtract_mean) * np.mean(
        a)  # Essentially an if statement for subtracting mean
    a_pad = _corr_pad(a - a_mean)
    a_data = np.append(a_pad, np.zeros(
        len(a_pad)))  # Pad with an equal number of zeros

    fra = np.fft.fft(a_data)  # FT the data
    if b is None:
        sf = np.conj(
            fra
        ) * fra  # Take the conj and multiply pointwise if autocorrelation
    else:
        b_mean = int(subtract_mean) * np.mean(b)
        b_pad = _corr_pad(b - b_mean)
        b_data = np.append(b_pad, np.zeros(len(b_pad)))

        frb = np.fft.fft(b_data)
        sf = np.conj(fra) * frb

    corr = np.real(np.fft.ifft(sf)[:len(a)]) / np.array(range(
        len(a), 0, -1))  # Inverse FFT and normalization
    return corr


def scalar_prod_Corr(a, b=None, subtract_mean=False):
    """Gives the correlation function of the scalar product of two vector timeseries.
    Arguments should be given in the form a[t, i], where t is the time variable,
    along which the correlation is calculated, and i indexes the vector components."""
    corr = np.zeros(len(a[:, 0]))

    if b is None:
        for i in range(0, len(a[0, :])):
            corr[:] += correlation(a[:, i], None, subtract_mean)

    else:
        for i in range(0, len(a[0, :])):
            corr[:] += correlation(a[:, i], b[:, i], subtract_mean)

    return corr




def vcorrelate(a, b=None, subtract_mean=False):
    # old ScalarProdCorr
    """
    Gives the correlation function of the scalarvcorrelate_multi_tau product of two vector timeseries.
    Arguments should be given in the form a[t, i], where t is the time variable,
    along which the correlation is calculated, and i indexes the vector components.
    """
    corr = np.zeros(len(a[:, 0]))

    if b is None:
        for i in range(0, len(a[0, :])):
            corr[:] += correlate(a[:, i], None, subtract_mean)

    else:
        for i in range(0, len(a[0, :])):
            corr[:] += correlate(a[:, i], b[:, i], subtract_mean)

    return corr
