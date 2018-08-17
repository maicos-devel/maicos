#!/usr/bin/env python3
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import subprocess

from setuptools import find_packages, setup

from mdtools.version import __version__

def get_git_revision_hash():
    try:
        hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        return ".g"+hash.decode()[:-2]
    except:
        # no git repo
        return ""
        
print(get_git_revision_hash())

if __name__ == "__main__":
    s = setup(name='mdtools',
              packages=find_packages(),
              version=__version__ + get_git_revision_hash(),
              license='MIT',
              description='A collection of scripts to analyse and build systems '
              'for molecular dynamics simulations.',
              author="Philip Loche",
              author_email="ploche@physik.fu-berlin.de",
              package_data={'': ['share/*']},
              include_package_data=True,
              install_requires=['MDAnalysis>=0.17.0',
                                'matplotlib>=2.0.0', 'numba>=0.38.0',
                                 'numpy>=1.10.4', 'scipy>=0.17'],
              entry_points={
                  'console_scripts': ['mdtools=mdtools.__main__:main', ],
              },
              zip_safe=False)

    installation_path = s.command_obj['install'].install_lib
    # Get newest installation folder
    mdtools_path = max([os.path.join(installation_path, i) for i
                        in os.listdir(installation_path) if "mdtools" in i],
                       key=os.path.getctime)
    print("\nTo use the BASH autocompletion add")
    print("  source {}\nto your .bashr or .profile file".format(
        os.path.join(mdtools_path, "mdtools/share/mdtools-completion.bash")))
