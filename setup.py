#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
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

VERSION = "0.4-dev"  # NOTE: keep in sync with __version__ in maicos.__init__.py
is_release = 'dev' not in VERSION

# Handle cython modules
try:
    # cython has to be >=0.16 <0.28 to support cython.parallel
    from Cython.Build import cythonize
except ImportError:
    if not is_release:
        print("*** package: Cython not found ***")
        print("MAICoS requires cython for development builds")
        sys.exit(1)


def get_numpy_include():
    # From MDAnalysis setup.py
    # Obtain the numpy include directory. This logic works across numpy
    # versions.
    # setuptools forgets to unset numpy's setup flag and we get a crippled
    # version of it unless we do it ourselves.
    try:
        # Python 3 renamed the ``__builin__`` module into ``builtins``.
        # Here we import the python 2 or the python 3 version of the module
        # with the python 3 name. This could be done with ``six`` but that
        # module may not be installed at that point.
        import builtins
    except ImportError:
        import __builtin__ as builtins
    builtins.__NUMPY_SETUP__ = False
    try:
        import numpy as np
    except ImportError:
        print('*** package "numpy" not found ***')
        print('MAICoS requires a version of NumPy (>=1.13.3), even for setup.')
        print('Please get it from http://numpy.scipy.org/ or install it through '
              'your package manager.')
        sys.exit(-1)
    return np.get_include()


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

    pre_exts = [Extension("maicos.lib.sfactor", ["maicos/lib/sfactor" + source_suffix],
                          include_dirs=[get_numpy_include()],
                          extra_compile_args=[
                              '-std=c99', '-ffast-math', '-O3', '-funroll-loops'
                          ] + has_openmp * ['-fopenmp'],
                          extra_link_args=has_openmp * ['-fopenmp'],
                          libraries=mathlib)
                ]

    if use_cython:
        extensions = cythonize(pre_exts)
    else:
        extensions = pre_exts
        # Let's check early for missing .c files
        for ext in extensions:
            for source in ext.sources:
                if not (os.path.isfile(source) and
                        os.access(source, os.R_OK)):
                    raise IOError("Source file '{}' not found. This might be "
                                  "caused by a missing Cython install, or a "
                                  "failed/disabled Cython build.".format(source))

    with open("README.rst") as summary:
        LONG_DESCRIPTION = summary.read()

    setup(name='maicos',
          packages=find_packages(),
          version=VERSION,
          license='GPL 3',
          description='Analyse molecular dynamics simulations of '
          'interfacial and confined systems.',
          author="Philip Loche et. al.",
          author_email="ploche@physik.fu-berlin.de",
          long_description=LONG_DESCRIPTION,
          long_description_content_type='text/markdown',
          maintainer="Philip Loche",
          maintainer_email="ploche@physik.fu-berlin.de",
          include_package_data=True,
          ext_modules=extensions,
          python_requires='>=3.6',
          setup_requires=[
              'numpy>=1.16.0',
          ],
          install_requires=[
              'numpy>=1.16.0',
              'MDAnalysis>=2.0.0',
              'matplotlib>=2.0.0',
              'scipy>=1.0.0',
              'threadpoolctl>=1.1.0',
          ],
  
          entry_points={
              'console_scripts': ['maicos = maicos.__main__:entry_point'],
          },
          keywords=[
              'Science',
              'Molecular Dynamics',
              'Confined Systems',
              'MDAnalysis',
            ],
          project_urls={
                'Documentation': 'https://maicos-devel.gitlab.io/maicos',
                'Source Code': 'https://gitlab.com/maicos-devel/maicos',
            },
          classifiers=[
              'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
              'Operating System :: POSIX',
              'Operating System :: MacOS :: MacOS X',
              'Operating System :: Microsoft :: Windows ',
              'Programming Language :: Python',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
              'Programming Language :: C',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Bio-Informatics',
              'Topic :: Scientific/Engineering :: Chemistry',
              'Topic :: Scientific/Engineering :: Physics',
              'Topic :: Software Development :: Libraries :: Python Modules',
              'Topic :: System :: Shells',
          ],
          zip_safe=False)
