#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np
from MDAnalysis.lib.util import check_box

# TODO: triclinic copies arrays, slower. Find a way to avoid...


def min_image(r, dims):
    boxtype, box = check_box(dims)
    if boxtype == "ortho":
        r = _min_image_ortho(r, box)
    else:
        hinv = np.linalg.inv(box)
        r = _min_image_triclinic(r, box, hinv)
    return r


def wrap(r, dims):
    boxtype, box = check_box(dims)
    if boxtype == "ortho":
        _wrap_ortho(r, box)
    else:
        hinv = np.linalg.inv(box)
        r = _wrap_triclinic(r, box, hinv)
    return r


def _min_image_ortho(r, box):
    nr = np.rint(r / box)
    r -= nr * box
    return r


def _wrap_ortho(r, box):
    nr = np.floor(r / box)
    r -= nr * box
    return r


def _min_image_triclinic(r, h, hinv):
    f = np.dot(r, hinv)
    f -= np.rint(f)
    r = np.dot(f, h)
    return r


def _wrap_triclinic(r, h, hinv):
    f = np.dot(r, hinv)
    f -= np.floor(f)
    r = np.dot(f, h)
    return r


def separation_array(reference, configuration, box=None):
    """
    Calculate all possible separation vectors between a reference set and another
    configuration.

    If there are ``n`` positions in `reference` and ``m`` positions in
    `configuration`, a separation array of shape ``(n, m, d)`` will be computed,
    where ``d`` is the dimensionality of each vector.

    If the optional argument `box` is supplied, the minimum image convention is
    applied when calculating separations. Either orthogonal or triclinic boxes are
    supported.
    """
    refdim = reference.shape[-1]
    confdim = configuration.shape[-1]
    if refdim != confdim:
        raise ValueError("Configuration dimension of {0} not equal to "
                         "reference dimension of {1}".format(confdim, refdim))

    # Do the whole thing by broadcasting
    separations = reference[:, np.newaxis] - configuration
    if box is not None:
        min_image(separations, box)
    return separations
