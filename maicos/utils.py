#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Utilities."""

import os
import sys
import warnings

import numpy as np


_share_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "share")


def check_compound(AtomGroup):
    """Check if compound 'molecules' exists.

    If compound molecule does not exist, it
    fallbacks to 'fragments' or 'residues'.
    """
    if hasattr(AtomGroup, "molnums"):
        return "molecules"
    elif hasattr(AtomGroup, "fragments"):
        warnings.warn("Cannot use 'molecules'. Falling back to 'fragments'")
        return "fragments"
    else:
        warnings.warn("Cannot use 'molecules'. Falling back to 'residues'")
        return "residues"


# Max variation from the mean dt or dk that is allowed (~1e-10 suggested)
dt_dk_tolerance = 1e-8


def FT(t, x, indvar=True):
    """Discrete fast fourier transform.

    Takes the time series and the function as arguments.
    By default, returns the FT and the frequency:\
    setting indvar=False means the function returns only the FT.
    """
    a, b = np.min(t), np.max(t)
    dt = (t[-1] - t[0]) / float(len(t) - 1)  # timestep
    if (abs((t[1:] - t[:-1] - dt)) > dt_dk_tolerance).any():
        raise RuntimeError("Time series not equally spaced!")
    N = len(t)
    # calculate frequency values for FT
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dt) * 2 * np.pi)
    # calculate FT of data
    xf = np.fft.fftshift(np.fft.fft(x))
    xf2 = xf * (b - a) / N * np.exp(-1j * k * a)
    if indvar:
        return k, xf2
    else:
        return xf2


def iFT(k, xf, indvar=True):
    """Inverse discrete fast fourier transform.

    Takes the frequency series and the function as arguments.
    By default, returns the iFT and the time series:\
    setting indvar=False means the function returns only the iFT.
    """
    dk = (k[-1] - k[0]) / float(len(k) - 1)  # timestep
    if (abs((k[1:] - k[:-1] - dk)) > dt_dk_tolerance).any():
        raise RuntimeError("Time series not equally spaced!")
    N = len(k)
    x = np.fft.ifftshift(np.fft.ifft(xf))
    t = np.fft.ifftshift(np.fft.fftfreq(N, d=dk)) * 2 * np.pi
    if N % 2 == 0:
        x2 = x * np.exp(-1j * t * N * dk / 2.) * N * dk / (2 * np.pi)
    else:
        x2 = x * np.exp(-1j * t * (N - 1) * dk / 2.) * N * dk / (2 * np.pi)
    if indvar:
        return t, x2
    else:
        return x2


def Correlation(a, b=None, subtract_mean=False):
    """Calculate correlation or autocorrelation.

    Uses fast fourier transforms to give the correlation function
    of two arrays, or, if only one array is given, the autocorrelation.
    Setting subtract_mean=True causes the mean to be subtracted from
    the input data.
    """
    meana = int(subtract_mean) * np.mean(
        a)  # essentially an if statement for subtracting mean
    a2 = np.append(a - meana,
                   np.zeros(2**int(np.ceil((np.log(len(a)) / np.log(2))))
                            - len(a)))  # round up to a power of 2
    data_a = np.append(a2,
                       np.zeros(len(a2)))  # pad with an equal number of zeros
    fra = np.fft.fft(data_a)  # FT the data
    if b is None:
        sf = np.conj(
            fra
            ) * fra  # take the conj and multiply pointwise if autocorrelation
    else:
        meanb = int(subtract_mean) * np.mean(b)
        b2 = np.append(
            b - meanb,
            np.zeros(2**int(np.ceil((np.log(len(b)) / np.log(2)))) - len(b)))
        data_b = np.append(b2, np.zeros(len(b2)))
        frb = np.fft.fft(data_b)
        sf = np.conj(fra) * frb
    cor = np.real(np.fft.ifft(sf)[:len(a)]) / np.array(range(
        len(a), 0, -1))  # inverse FFT and normalization
    return cor


def ScalarProdCorr(a, b=None, subtract_mean=False):
    """Give the corr. function of the scalar product of two vector timeseries.

    Arguments should be given in the form a[t, i],
    where t is the time variable along which the correlation is calculated,
    and i indexes the vector components.
    """
    corr = np.zeros(len(a[:, 0]))

    if b is None:
        for i in range(0, len(a[0, :])):
            corr[:] += Correlation(a[:, i], None, subtract_mean)

    else:
        for i in range(0, len(a[0, :])):
            corr[:] += Correlation(a[:, i], b[:, i], subtract_mean)

    return corr


def symmetrize_1D(arr, inplace=False):
    """Symmeterize a 1D-array.

    The array can have additional axes of length one.

    Paramaters
    ----------
    arr : numpy.ndarray
        array to symmetrize
    inplace : bool
        Do symmetrizations inplace. If `False` a new array is returnd.

    Returns
    -------
    np.ndarray
        the symmetrized array

    Raises
    ------
    ValueError
        If the array is not of one dimensional.
    """
    if len(arr.squeeze().shape) > 1 or arr.shape[0] == 1:
        raise ValueError("Only 1 dimensional arrays can be symmeterized")

    if inplace:
        sym_arr = arr
    else:
        sym_arr = np.copy(arr).astype(float)

    sym_arr += arr[::-1]
    sym_arr /= 2

    return sym_arr


def get_cli_input():
    """Return a proper formatted string of the command line input."""
    program_name = os.path.basename(sys.argv[0])
    # Add additional quotes for connected arguments.
    arguments = ['"{}"'.format(arg)
                 if " " in arg else arg for arg in sys.argv[1:]]
    return "Command line was: {} {}".format(program_name, " ".join(arguments))


def savetxt(fname, X, header='', fsuffix=".dat", **kwargs):
    """Save to text.

    An extension of the numpy savetxt function. Adds the command line
    input to the header and checks for a doubled defined filesuffix.
    """
    header = "{}\n{}".format(get_cli_input(), header)
    fname = "{}{}".format(fname, (not fname.endswith(fsuffix)) * fsuffix)
    np.savetxt(fname, X, header=header, **kwargs)


def atomgroup_header(AtomGroup):
    """Return a string containing infos about the AtomGroup.

    Infos include the total number of atoms, the including
    residues and the number of residues. Useful for writing
    output file headers.
    """
    unq_res, n_unq_res = np.unique(AtomGroup.residues.resnames,
                                   return_counts=True)
    return "{} atom(s): {}".format(
        AtomGroup.n_atoms, ", ".join(
            "{} {}".format(*i) for i in np.vstack([n_unq_res, unq_res]).T))


def sort_atomgroup(atomgroup):
    """Sort a atomgroup after its fragments.

    Needed in e.g. LAMMPS, as molecules are not sorted,
    but randomly distributed in atomgroup.atoms.

    atomgroup: atomgroup to sort
    """
    com = check_compound(atomgroup)
    if com == 'fragments':
        return atomgroup[np.argsort(atomgroup.fragindices)]
    elif com == 'residues':
        return atomgroup[np.argsort(atomgroup.resids)]
    elif com == 'molecules':
        return atomgroup[np.argsort(atomgroup.molnums)]
    else:
        return atomgroup
