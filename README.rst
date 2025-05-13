MAICoS
======

.. image:: https://raw.githubusercontent.com/maicos-devel/maicos/refs/heads/main/docs/static/logo_MAICOS_square.png
   :width: 200 px
   :align: left
   :alt: MAICoS logo

|tests| |codecov| |docs| |mdanalysis|

.. inclusion-readme-intro-start

**MAICoS** is the acronym for Molecular Analysis for Interfacial and Confined Systems.
It is an object-oriented python toolkit for analysing the structure and dynamics of
interfacial and confined fluids from molecular simulations. Combined with MDAnalysis_,
MAICoS can be used to extract density profiles, dielectric constants, structure factors,
or transport properties from trajectories files, including LAMMPS, GROMACS, CHARMM or
NAMD data. MAICoS is open source and is released under the GNU general public license
v3.0.

MAICoS is a tool for beginners of molecular simulations with no prior Python experience.
For these users MAICoS provides a descriptive command line interface. Also experienced
users can use the Python API for their day to day analysis or use the provided
infrastructure to build their own analysis for interfacial and confined systems.

Keep up to date with MAICoS news by following us on Twitter_. If you find an issue, you
can report it on Gitlab_. You can also join the developer team on Discord_ to discuss
possible improvements and usages of MAICoS.

.. _`MDAnalysis`: https://www.mdanalysis.org
.. _`Twitter`: https://twitter.com/maicos_analysis
.. _`Gitlab`: https://gitlab.com/maicos-devel/maicos
.. _`Discord`: https://discord.gg/mnrEQWVAed

.. inclusion-readme-intro-end

Basic example
=============

This is a simple example showing how to use MAICoS to extract the density profile from a
molecular dynamics simulation. The files ``conf.gro`` and ``traj.trr`` correspond to
simulation files from a GROMACS_ simulation package. In a Python environment, type:

.. code-block:: python

    import MDAnalysis as mda
    import maicos

    u = mda.Universe("conf.gro", "traj.trr")
    dplan = maicos.DensityPlanar(u.atoms).run()

The density profile can be accessed from ``dplan.results.profile`` and the position of
the bins from ``dplan.results.bin_pos``.

.. _`GROMACS` : https://www.gromacs.org/

Documentation
=============

For details, tutorials, and examples, please have a look at our documentation_. If you
are using an older version of MAICoS, you can access the corresponding documentation on
ReadTheDocs_.

.. _`documentation`: https://maicos-devel.gitlab.io/maicos/index.html
.. _`ReadTheDocs` : https://readthedocs.org/projects/maicos/

.. inclusion-readme-installation-start

Installation
============

Install MAICoS using `pip`_ with::

    pip install maicos

or using conda_ with::

    conda install -c conda-forge maicos

.. _`pip`: https://pip.pypa.io
.. _`conda`: https://www.anaconda.com

.. inclusion-readme-installation-end

List of analysis modules
========================

.. inclusion-marker-modules-start

Currently, MAICoS supports the following analysis modules in alphabetical order:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Module Name
     - Description

   * - DensityCylinder
     - Cylindrical partial densitiy profiles
   * - DensityPlanar
     - Cartesian partial density profiles
   * - DensitySphere
     - Spherical partial density profiles
   * - DielectricCylinder
     - Cylindrical dielectric profiles
   * - DielectricPlanar
     - Planar dielectric profiles
   * - DielectricSpectrum
     - Linear dielectric spectrum
   * - DielectricSphere
     - Spherical dielectric profiles
   * - DipoleAngle
     - Angle timeseries of dipole moments
   * - DiporderCylinder
     - Cylindrical dipolar order parameters
   * - DiporderPlanar
     - Planar dipolar order parameters
   * - RDFDiporder
     - Spherical Radial Distribution function between dipoles
   * - DiporderSphere
     - Spherical dipolar order parameters
   * - DiporderStructureFactor
     - Structure factor for dipoles
   * - KineticEnergy
     - Timeseries of energies
   * - PDFCylinder
     - Cylindrical shell-wise 1D pair distribution functions
   * - PDFPlanar
     - Slab-wise planar 2D pair distribution functions
   * - Saxs
     - Small angle X-Ray structure factors and scattering intensities (SAXS)
   * - TemperaturePlanar
     - Temperature profiles in a cartesian geometry
   * - VelocityCylinder
     - Cartesian velocity profile across a cylinder
   * - VelocityPlanar
     - Velocity profile in a cartesian geometry

.. inclusion-marker-modules-end

Contributors
============

Thanks goes to all people that make *maicos* possible:

.. image:: https://contrib.rocks/image?repo=maicos-devel/maicos
   :target: https://github.com/maicos-devel/maicos/graphs/contributors

.. |tests| image:: https://github.com/maicos-devel/maicos/workflows/Tests/badge.svg
   :alt: Github Actions Tests Job Status
   :target: https://github.com/maicos-devel/maicos/actions?query=branch%3Amain

.. |codecov| image:: https://codecov.io/gh/maicos-devel/maicos/graph/badge.svg?token=9AXPLF6CR3
   :alt: Code coverage
   :target: https://codecov.io/gh/maicos-devel/maicos

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Documentation
   :target: `documentation`_

.. |mdanalysis| image:: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
    :alt: Powered by MDAnalysis
    :target: https://www.mdanalysis.org
