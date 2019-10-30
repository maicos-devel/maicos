#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import sys
from unittest.mock import patch
import subprocess

from MDAnalysisTests import tempdir
from maicos.__main__ import parse_args, main
from maicos import __all__ as available_modules
import pytest

from modules.datafiles import WATER_GRO


class Test_parse_args(object):

    def test_required_args(self):
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(['maicos'])

    def test_wrong_module(self):
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(['maicos', 'foo'])

    @pytest.mark.parametrize("module", tuple(available_modules))
    def test_available_modules(self, module):
        subprocess.check_call(['maicos', module, "--help"])

    @pytest.mark.parametrize('args', ("version", "help", "bash_completion"))
    def test_extra_options(self, args):
        subprocess.check_call(['maicos', '--' + args])

    def test_debug(self):
        subprocess.run(["maicos", "density_planar", "--debug"],
                       input=b'exit()\n')

    @pytest.mark.parametrize(
        'opt, dest, val',
        (('-s', "topology", "foo"), ('-top', "topology_format", "foo"),
         ('-f', "trajectory", ["foo"
                               "bar"]), ('-traj', "trajectory_format", "foo"),
         ('-atom_style', "atom_style", "foo"), ('-b', "begin", 42),
         ('-e', "end", 42), ('-dt', "dt", 42), ('-box', "box", [42, 42, 42]),
         ('-nt', "num_threads", 1)))
    def test_arguments(self, opt, dest, val):
        testargs = ["maicos", "density_planar", opt]
        if type(val) == list:
            for i in val:
                testargs.append(str(i))
        else:
            testargs.append(str(val))
        with patch.object(sys, 'argv', testargs):
            args = parse_args()
            t = type(val)
            assert t(getattr(args, dest)) == val

    @pytest.mark.parametrize(
        'args',
        (["density_planar", "-sel", "foo", "bar"], ["diporder", "-sel", "foo"]))
    def test_multiple_atomgroup(self, args):
        testargs = ["maicos"]
        for i in args:
            testargs.append(i)
        with patch.object(sys, 'argv', testargs):
            parse_args()

    def test_multi_group_on_single_module(self):
        with patch.object(sys, 'argv',
                          ["maicos", "diporder", "-sel", "foo", "bar"]):
            with pytest.raises(SystemExit):
                parse_args()


class Test_main(object):

    @pytest.fixture()
    def args(self):
        with patch.object(sys, 'argv', ["maicos", "density_planar"]):
            return parse_args()

    def test_full_run(self, args):
        args.topology = WATER_GRO
        args.trajectory = WATER_GRO
        with tempdir.in_tempdir():
            main(args)

    @pytest.mark.parametrize("boxlist", (3 * [1], 6 * [1]))
    def box(self, args, boxlist):
        args.topology = WATER_GRO
        args.trajectory = WATER_GRO
        args.box = boxlist
        with tempdir.in_tempdir():
            main(args)

    def raise_box_error(self, args):
        args.topology = WATER_GRO
        args.trajectory = WATER_GRO
        args.box = 2 * [1]
        with pytest.raises(IndexError):
            main(args)

    def raise_no_atoms(self, args):
        args.topology = WATER_GRO
        args.trajectory = WATER_GRO
        args.sel = ["resname foo"]
        args.box = 2 * [1]
        with pytest.raises(IndexError):
            main(args)
