#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np

dt_dk_tolerance = 1e-8  # Max variation from the mean dt or dk that is allowed (~1e-10 suggested)


def ft(t, x, indvar=True):
    """
    Discrete fast fourier transform.
    Takes the time series and the function as arguments.
    By default, returns the FT and the frequency.
    Setting indvar = False returns only the FT values.
    """
    N = len(t)
    a = np.min(t)

    dt = (t[-1] - t[0]) / float(N - 1)  # Timestep
    if (abs((t[1:] - t[:-1] - dt)) > dt_dk_tolerance).any():
        print("Expected time step: ", dt)
        print("Maximum time step: ", np.max(abs(t[1:] - t[:-1])))
        # raise RuntimeError("Time series not equally spaced!")

    # Calculate frequency values for FT
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dt) * 2 * np.pi)
    # Calculate FT of data
    xf = np.fft.fftshift(np.fft.fft(x))
    xf *= dt * np.exp(
        -1j * k * a)  # Scales the FT by timestep and phase factor

    return (k, xf) if indvar else xf


def ift(k, xf, indvar=True):
    """
    Inverse discrete fast fourier transform.
    Takes the frequency series and the function as arguments.
    By default, returns the iFT and the time series,
    Setting indvar = False means the function returns only the inverse FT values.
    """
    N = len(k)
    dk = (k[-1] - k[0]) / float(N - 1)  # Freq spacing
    if (abs((k[1:] - k[:-1] - dk)) > dt_dk_tolerance).any():
        print("Expected wavevector step: ", dk)
        print("Maximum wavevector step: ", np.max(abs(k[1:] - k[:-1])))
        # raise RuntimeError("Time series not equally spaced!")

    x = np.fft.ifftshift(np.fft.ifft(xf))
    t = np.fft.ifftshift(np.fft.fftfreq(N, d=dk)) * 2 * np.pi

    if N % 2 == 0:
        x *= np.exp(-1j * t * N * dk / 2.) * N * dk / (2 * np.pi)
    else:
        x *= np.exp(-1j * t * (N - 1) * dk / 2.) * N * dk / (2 * np.pi)

    return (t, x) if indvar else x
