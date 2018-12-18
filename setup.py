#!/usr/bin/env python3
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import subprocess

from mdtools.version import __version__
from setuptools import find_packages, setup


def get_git_revision_hash():
    try:
        hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        return ".dev0"
    except:
        # no git repo
        return ""


if __name__ == "__main__":
    setup(
        name='mdtools',
        packages=find_packages(),
        version=__version__ + get_git_revision_hash(),
        license='MIT',
        description='A collection of scripts to analyse and build systems '
        'for molecular dynamics simulations.',
        author="Philip Loche et. al.",
        author_email="ploche@physik.fu-berlin.de",
        package_data={'': ['share/*']},
        include_package_data=True,
        install_requires=[
            'MDAnalysis>=0.19.0', 'matplotlib>=2.0.0',
            'numba>=0.38.0', 'numpy>=1.10.4', 'scipy>=0.17'
        ],
        entry_points={
            'console_scripts': [
                'mdtools=mdtools.__main__:main',
            ],
        },
        zip_safe=False)
