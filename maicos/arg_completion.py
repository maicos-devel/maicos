#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import re
import sys
import inspect

PARAM_OR_RETURNS_REGEX = re.compile(":(?:param|returns)")
RETURNS_REGEX = re.compile(r":returns \((?P<p_type>.*?)\): (?P<doc>.*)", re.S)
PARAM_REGEX = re.compile(
    r":param (?P<name>[\*\w]+) \((?P<p_type>.*?)\): (?P<doc>.*?)"
    r"(?:(?=:param)|(?=:return)|(?=:raises)|\Z)", re.S)
str_type_dict = {
    "bool": bool,
    "str": str,
    "list": list,
    "int": int,
    "float": float,
    "complex": complex
}


def _trim(docstring):
    """trim function from PEP-257"""
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    # Current code/unittests expects a line return at
    # end of multiline docstrings
    # workaround expected behavior from unittests
    if "\n" in docstring:
        trimmed.append("")

    # Return a single string:
    return "\n".join(trimmed)


def _reindent(string):
    return "\n".join(l.strip() for l in string.strip().split("\n"))


def parse_docstring(docstring):
    """Parse the docstring into its components.

    Taken from openstack rally repository.

    :returns: a dictionary of form
              {
                  "short_description": ...,
                  "long_description": ...,
                  "params": [{"name": ..., "doc": ...}, ...],
                  "returns": ...
              }
    """

    short_description = long_description = returns = ""
    params = []

    if docstring:
        docstring = _trim(docstring)

        lines = docstring.split("\n", 1)
        short_description = lines[0]

        if len(lines) > 1:
            long_description = lines[1].strip()

            params_returns_desc = None

            match = PARAM_OR_RETURNS_REGEX.search(long_description)
            if match:
                long_desc_end = match.start()
                params_returns_desc = long_description[long_desc_end:].strip()
                long_description = long_description[:long_desc_end].rstrip()

            if params_returns_desc:
                params = [{
                    "name": name,
                    "type": str_type_dict[p_type],
                    "doc": _trim(doc)
                } for name, p_type, doc in PARAM_REGEX.findall(
                    params_returns_desc)]

                match = RETURNS_REGEX.search(params_returns_desc)
                if match:
                    returns = _reindent(match.group("doc"))

    return {
        "short_description": short_description,
        "long_description": long_description,
        "params": params,
        "returns": returns
    }


def _create_doctsring_dict(func):
    """Creates a dictionary containing arguments and returns including default values from a function"""

    doctsring_dict = parse_docstring(func.__doc__)
    init_arguments = inspect.signature(func).parameters.items()

    # Find default argument
    for function_param, value in init_arguments:
        if value.default is inspect.Parameter.empty:
            continue
        for argument_dict in doctsring_dict["params"]:
            if argument_dict["name"] == function_param:
                argument_dict["default"] = value.default

    return doctsring_dict


def complete_parser(parser, module):
    """Completes options of a parser object based on the values in a docstring dictionary"""

    doctsring_dict = _create_doctsring_dict(module)
    parser.description = doctsring_dict["short_description"] + \
        "\n" + doctsring_dict["long_description"]
    for i, action in enumerate(parser._actions):
        for param_dict in doctsring_dict["params"]:
            if action.dest == param_dict["name"]:

                # Create new action if type is bool
                if param_dict["type"] is bool:

                    option_strings = [s for s in action.option_strings]
                    for s in action.option_strings:
                        parser._handle_conflict_resolve(None, [(s, action)])

                    parser.add_argument(*option_strings,
                                        dest=action.dest,
                                        action='store_' +
                                        str(not param_dict["default"]).lower(),
                                        default=param_dict["default"],
                                        help=param_dict["doc"])

                    parser._actions.insert(i, parser._actions.pop(-1))

                else:  # Add attributes if non boolean parameter
                    action.type = param_dict["type"]
                    action.default = param_dict["default"]
                    action.help = param_dict["doc"]
