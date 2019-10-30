#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import warnings

import numpy as np

_share_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "share")


def repairMolecules(selection):
    """Repairs molecules that are broken due to peridodic boundaries.
    To this end the center of mass is reset into the central box.
    CAVE: Only works with small (< half box) molecules."""

    with warnings.catch_warnings():
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(
            "repairMolecules is deprecated, "
            "use AtomGroup.unwrap() instead.",
            category=DeprecationWarning)

    # we repair each moleculetype individually for performance reasons
    for seg in selection.segments:
        atomsPerMolecule = seg.atoms.n_atoms // seg.atoms.n_residues

        # Make molecules whole, use first atom as reference
        distToFirst = np.empty((seg.atoms.positions.shape))
        for i in range(atomsPerMolecule):
            distToFirst[i::atomsPerMolecule] = seg.atoms.positions[i::atomsPerMolecule] - \
                seg.atoms.positions[0::atomsPerMolecule]
        seg.atoms.positions -= (
            np.abs(distToFirst) > selection.dimensions[:3] /
            2.) * selection.dimensions[:3] * np.sign(distToFirst)

        # Calculate the centers of the objects ( i.e. Molecules )
        masspos = (seg.atoms.positions *
                   seg.atoms.masses[:, np.newaxis]).reshape(
                       (seg.atoms.n_atoms // atomsPerMolecule, atomsPerMolecule,
                        3))
        # all molecules should have same mass
        centers = np.sum(masspos.T, axis=1).T / \
            seg.atoms.masses[:atomsPerMolecule].sum()

        # now shift them back into the primary simulation cell
        seg.atoms.positions += np.repeat(
            (centers % selection.dimensions[:3]) - centers,
            atomsPerMolecule,
            axis=0)


dt_dk_tolerance = 1e-8  # Max variation from the mean dt or dk that is allowed (~1e-10 suggested)


def FT(t, x, indvar=True):
    """Discrete fast fourier transform.
    Takes the time series and the function as arguments.
    By default, returns the FT and the frequency:\
    setting indvar=False means the function returns only the FT."""
    a, b = np.min(t), np.max(t)
    dt = (t[-1] - t[0]) / float(len(t) - 1)  # timestep
    if (abs((t[1:] - t[:-1] - dt)) > dt_dk_tolerance).any():
        print(np.max(abs(t[1:] - t[:-1])))
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
    setting indvar=False means the function returns only the iFT."""
    dk = (k[-1] - k[0]) / float(len(k) - 1)  # timestep
    if (abs((k[1:] - k[:-1] - dk)) > dt_dk_tolerance).any():
        print(np.max(abs(k[1:] - k[:-1])))
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
    """Uses fast fourier transforms to give the correlation function
    of two arrays, or, if only one array is given, the autocorrelation.
    Setting subtract_mean=True causes the mean to be subtracted from the input data."""
    meana = int(subtract_mean) * np.mean(
        a)  # essentially an if statement for subtracting mean
    a2 = np.append(a - meana,
                   np.zeros(2**int(np.ceil((np.log(len(a)) / np.log(2)))) -
                            len(a)))  # round up to a power of 2
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
    """Gives the correlation function of the scalar product of two vector timeseries.
    Arguments should be given in the form a[t, i], where t is the time variable,
    along which the correlation is calculated, and i indexes the vector components."""
    corr = np.zeros(len(a[:, 0]))

    if b is None:
        for i in range(0, len(a[0, :])):
            corr[:] += Correlation(a[:, i], None, subtract_mean)

    else:
        for i in range(0, len(a[0, :])):
            corr[:] += Correlation(a[:, i], b[:, i], subtract_mean)

    return corr


def get_cli_input():
    """Returns a proper fomatted string of the command line input"""
    program_name = os.path.basename(sys.argv[0])
    return "Command line was: {} {}".format(program_name,
                                            " ".join(sys.argv[1:]))


def savetxt(fname, X, header='', fsuffix=".dat", **kwargs):
    """An extension of the numpy savetxt function.
    Adds the command line input to the header and checks for a doubled defined
    filesuffix."""
    header = "{}\n{}".format(get_cli_input(), header)
    fname = "{}{}".format(fname, (not fname.endswith(fsuffix)) * fsuffix)
    np.savetxt(fname, X, header=header, **kwargs)


def atomgroup_header(AtomGroup):
    """Returns a string containing infos about the AtmGroup containing
    the total number of atoms, the including residues and the number of residues.
    Useful for writing output file headers."""

    unq_res, n_unq_res = np.unique(AtomGroup.residues.resnames,
                                   return_counts=True)
    return "{} atom(s): {}".format(
        AtomGroup.n_atoms, ", ".join(
            "{} {}".format(*i) for i in np.vstack([n_unq_res, unq_res]).T))
