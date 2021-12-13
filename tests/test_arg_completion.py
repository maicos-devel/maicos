#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2021 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import argparse

import pytest

from maicos.arg_completion import (_parse_docstring, _reindent,
                                   _create_doctsring_dict, _unparse_docstring,
                                   _append_to_doc, complete_parser)


def test_reindent():
    my_string = "foo \n \n bar"
    assert "foo\n\nbar" == _reindent(my_string)


def complete_docstring(p1="foo", p2=True):
    """One-line description.
Multi-
line-
description.
:param p1 (str): Param 1 description.
:param p2 (bool): Param 2
description.

:returns (str): Return value
description."""

    doc_dict = {
        'short_description': 'One-line description.',
        'long_description': 'Multi-\nline-\ndescription.',
        'params': [{
            'name': 'p1',
            'type': str,
            'doc': 'Param 1 description.\n'
        }, {
            'name': 'p2',
            'type': bool,
            'doc': 'Param 2\ndescription.\n'
        }],
        'returns': 'Return value\ndescription.',
        'returns_type': 'str'
    }
    return doc_dict


def incomplete_docstring():
    """One-line description.
:param p1 (str): Param 1 description.
:param p2 (bool): Param 2
description.
"""

    doc_dict = {
        'short_description': 'One-line description.',
        'long_description': '',
        'params': [{
            'name': 'p1',
            'type': str,
            'doc': 'Param 1 description.\n'
        }, {
            'name': 'p2',
            'type': bool,
            'doc': 'Param 2\ndescription.\n'
        }],
        'returns': '',
        'returns_type': ''
    }

    return doc_dict


def no_params_docstring():
    """One-line description.
Multi-
line-
description.
:returns (float): Return value
description."""

    doc_dict = {
        'short_description': 'One-line description.',
        'long_description': 'Multi-\nline-\ndescription.',
        'params': [],
        'returns': 'Return value\ndescription.',
        'returns_type': 'float'
    }

    return doc_dict


def short_only_docstring():
    """One-line description."""

    doc_dict = {
        'short_description': 'One-line description.',
        'long_description': '',
        'params': [],
        'returns': '',
        'returns_type': ''
    }

    return doc_dict


@pytest.mark.parametrize("func", (complete_docstring, incomplete_docstring,
                                  no_params_docstring, short_only_docstring))
def test_parse_docstring(func):
    assert func() == _parse_docstring(func.__doc__)


@pytest.mark.parametrize("func", (complete_docstring, incomplete_docstring,
                                  no_params_docstring, short_only_docstring))
def test_unparse_docstring(func):
    assert func.__doc__ == _unparse_docstring(func())


def test_create_doctsring_dict():
    doc_dict = complete_docstring()

    doc_dict["params"][0]["default"] = "foo"
    doc_dict["params"][1]["default"] = True
    assert doc_dict == _create_doctsring_dict(complete_docstring)


def test_append_to_doc_short():
    new_doc = _append_to_doc(complete_docstring.__doc__,
                             short_description="foo")
    assert "One-line description.foo" in new_doc


def test_append_to_doc_long():
    new_doc = _append_to_doc(complete_docstring.__doc__, long_description="foo")
    assert "description.foo" in new_doc


def test_append_to_doc_param():
    new_doc = _append_to_doc(complete_docstring.__doc__,
                             params=[{
                                 'name': 'p3',
                                 'type': int,
                                 'doc': 'Param 3 description.\n'
                             }])
    assert ":param p3 (int): Param 3 description.\n" in new_doc


class Test_complete_parser():

    @pytest.fixture
    def completed_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-p1", dest="p1")
        parser.add_argument("-p2", dest="p2")

        complete_parser(parser, complete_docstring)
        return parser

    def test_description(self, completed_parser):
        description = "One-line description.\nMulti-\nline-\ndescription."
        assert description == completed_parser.description

    def test_argument(self, completed_parser):
        args = completed_parser.parse_known_args()[0]

        assert args.p1 == "foo"
        assert args.p2 == True
