#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import print_function
from setuptools import setup, Extension, find_packages
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
import os
import sys
import shutil
import tempfile

import numpy as np
from Cython.Build import cythonize


def hasfunction(cc, funcname, include=None, extra_postargs=None):
    # From MDAnalysis setup.py
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
    """Does this compiler support OpenMP parallelization?"""
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


VERSION = "0.1"  # NOTE: keep in sync with __version__ in maicos.__init__.py

if __name__ == "__main__":

    # Windows automatically handles math library linking
    # and will not build if we try to specify one
    if os.name == 'nt':
        mathlib = []
    else:
        mathlib = ['m']

    has_openmp = detect_openmp()

    extensions = [
        Extension("maicos.lib.sfactor", ["maicos/lib/sfactor.pyx"],
                  include_dirs=[np.get_include()],
                  extra_compile_args=[
                      '-std=c99', '-ffast-math', '-O3', '-funroll-loops'
                  ] + has_openmp * ['-fopenmp'],
                  extra_link_args=has_openmp * ['-fopenmp'],
                  libraries=mathlib)
    ]

if __name__ == "__main__":
    setup(name='maicos',
          packages=find_packages(),
          version=VERSION,
          license='MIT',
          description='Analyse molecular dynamics simulations of '
          'interfacial and confined systems.',
          author="Philip Loche et. al.",
          author_email="ploche@physik.fu-berlin.de",
          maintainer="Philip Loche",
          maintainer_email="ploche@physik.fu-berlin.de",
          package_data={'': ['share/*']},
          include_package_data=True,
          ext_modules=cythonize(extensions),
          install_requires=[
              'MDAnalysis>0.19.2',
              'matplotlib>=2.0.0',
              'numpy>=1.10.4',
              'scipy>=0.17',
              'threadpoolctl>=1.1.0',
          ],
          entry_points={
              'console_scripts': ['maicos = maicos.__main__:entry_point'],
          },
          zip_safe=False)
