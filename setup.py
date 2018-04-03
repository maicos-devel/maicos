#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

from setuptools import setup, find_packages
import os
from sys import platform

from mdtools.version import __version__

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
          include_package_data=True,
          requires=['numpy (>=1.10.4)', 'MDAnalysis (>=0.17.0)'],
          setup_requires=['numpy (>=1.10.4)'],
          install_requires=['numpy>=1.10.4', 'MDAnalysis>=0.17.0'],
          entry_points={
               'console_scripts': ['mdtools=mdtools.__main__:main', ],
          },
          )
