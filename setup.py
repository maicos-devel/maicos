#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

from setuptools import setup, find_packages
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import os
from sys import platform

from mdtools.version import __version__

# Enable openmp on LINUX:
if platform == "linux":
    extensions = Extension("sfactor", ["mdtools/ana/sfactor/sfactor.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args = ["-fopenmp"],
              extra_link_args=['-fopenmp']
              )
else:
    extensions = Extension("sfactor", ["mdtools/ana/sfactor/sfactor.pyx"],
              include_dirs=[numpy.get_include()]
              )

if __name__ == "__main__":
    setup(name='mdtools',
          packages=find_packages(),
          version=__version__,
          license='MIT',
          description='A collection of scripts to analyse and build systems '
          'for molecular dynamics simulations.',
          author="Philip Loche",
          author_email="ploche@physik.fu-berlin.de",
          package_data={'': ['share/*']},
          ext_modules = cythonize(extensions),
          include_package_data=True,
          zip_safe=False,
          requires=['numpy (>=1.10.4)', 'MDAnalysis (>=0.17.0)', 'Cython(>=0.27.3)'],
          install_requires=['numpy>=1.10.4', 'MDAnalysis>=0.17.0', 'Cython>=0.27.3'],
          entry_points={
               'console_scripts': ['mdtools=mdtools.__main__:main', ],
          },
          )
