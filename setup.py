#!/usr/bin/env python3
"""Setup file for MAICoS package.

Credit to MDAnalysis setup.py.
"""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import print_function

import os
import shutil
import sys
import tempfile
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler

import numpy as np
from setuptools import Extension, setup


# NOTE: keep in sync with __version__ in maicos.__init__.py
# NOTE: keep in sync with version in docs/source/conf.py
VERSION = "0.6"
is_release = 'dev' not in VERSION


def hasfunction(cc, funcname, include=None, extra_postargs=None):
    """
    Check for function.

    Credit to MDAnalysis setup.py.
    """
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            with open(fname, 'w') as f:
                if include is not None:
                    f.write('#include {0!s}\n'.format(include))
                f.write('int main(void) {\n')
                f.write('    {0!s};\n'.format(funcname))
                f.write('}\n')
            # Redirect stderr to /dev/null to hide any error messages
            # from the compiler.
            # This will have to be changed if we ever have to check
            # for a function on Windows.
            devnull = open('/dev/null', 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            objects = cc.compile([fname],
                                 output_dir=tmpdir,
                                 extra_postargs=extra_postargs)
            cc.link_executable(objects, os.path.join(tmpdir, "a.out"))
        except Exception:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)


def detect_openmp():
    # From MDAnalysis setup.py
    """
    Support for OpenMP parallelization.

    Check if this compiler support OpenMP parallelization.
    """
    print("Attempting to autodetect OpenMP support... ", end="")
    compiler = new_compiler()
    customize_compiler(compiler)
    compiler.add_library('gomp')
    include = '<omp.h>'
    extra_postargs = ['-fopenmp']
    hasopenmp = hasfunction(compiler,
                            'omp_get_num_threads()',
                            include=include,
                            extra_postargs=extra_postargs)
    if hasopenmp:
        print("Compiler supports OpenMP")
    else:
        print("Did not detect OpenMP support.")
    return hasopenmp


if __name__ == "__main__":

    # Windows automatically handles math library linking
    # and will not build if we try to specify one
    if os.name == 'nt':
        mathlib = []
    else:
        mathlib = ['m']

    has_openmp = detect_openmp()
    use_cython = not is_release or bool(os.getenv('USE_CYTHON'))
    source_suffix = '.pyx' if use_cython else '.c'

    pre_exts = [
        Extension("maicos.lib.sfactor",
                  ["src/maicos/lib/sfactor" + source_suffix],
                  include_dirs=[np.get_include()],
                  extra_compile_args=[
                      '-std=c99', '-ffast-math', '-O3', '-funroll-loops'
                      ] + has_openmp * ['-fopenmp'],
                  extra_link_args=has_openmp * ['-fopenmp'],
                  libraries=mathlib)
        ]

    if use_cython:
        from Cython.Build import cythonize
        extensions = cythonize(pre_exts, force=True)
    else:
        extensions = pre_exts
        # Let's check early for missing .c files
        for ext in extensions:
            for source in ext.sources:
                if not (os.path.isfile(source) and os.access(source, os.R_OK)):
                    raise IOError(f"Source file '{source}' not found. This "
                                  "might be caused by a missing Cython "
                                  "install, or a failed/disabled "
                                  "Cython build.")

    setup(ext_modules=extensions)
