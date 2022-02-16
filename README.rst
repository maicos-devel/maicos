.. image:: https://gitlab.com/maicos-devel/maicos/-/raw/develop/docs/source/images/logo_MAICOS_small.png
   :align: left
   :alt: MAICoS

.. inclusion-readme-intro-start

**MAICoS** is the acronym for Molecular Analysis for Interfacial
and Confined Systems. It is an object-oriented python toolkit for
analysing the structure and dynamics of interfacial and confined
fluids from molecular simulations. Combined with `MDAnalysis`_,
MAICoS can be used to extract density profiles, dielectric constants,
structure factors, or transport properties from trajectories files,
including LAMMPS, GROMACS, CHARMM or NAMD data. MAICoS is open source
and is released under the GNU general public license v3.0.

.. inclusion-readme-intro-end

For details, tutorials, and examples, please have a look at
our `documentation`_.

.. inclusion-readme-start

Basic example
#############

This is a simple example showing how to use MAICoS to extract the density profile
from a molecular dynamics simulation. The files ``conf.gro`` and ``traj.trr``
correspond to a water slab in vacuum that was simulated in this case using the
`GROMACS`_ simulation package. In a Python environment, type:

.. code-block:: python3

	import MDAnalysis as mda
	import maicos
	u = mda.Universe('conf.gro', 'traj.trr')
	grpH2O = u.select_atoms('type O or type H')
	dplan = maicos.DensityPlanar(grpH2O)
	dplan.run()


Results can be accessed from ``dplan.results``.

Installation
############

`Python3`_ and a C-compiler are needed to build the
underlying libraries.

Using pip
---------

If you have root access, install the package for all users by
typing in a terminal:

.. code-block:: bash

    pip3 install numpy
    pip3 install maicos

Alternatively, if you don't have special privileges, install
the package in your home directory by using the ``--user`` flag.

List of analysis modules
########################

.. inclusion-marker-modules-start

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Module Name
     - Description

   * - DensityPlanar
     - Compute partial densities/temperature profiles in the Cartesian systems.
   * - DensityCylinder
     - Compute partial densities across a cylinder.
   * - EpsilonBulk
     - Compute dipole moment fluctuations and static dielectric constant.
   * - EpsilonPlanar
     - Calculate planar dielectric profiles.
   * - EpsilonCylinder
     - Calculate cylindrical dielectric profiles.
   * - DielectricSpectrum
     - Compute the linear dielectric spectrum.
   * - Saxs
     - Compute SAXS scattering intensities.
   * - Diporder
     - Calculate dipolar order parameters.
   * - Debyer
     - Calculate scattering intensities using the debye equation. The `debyer`_
       library needs to be downloaded and build.
   * - DipoleAngle
     - Calculate angle timeseries of dipole moments with respect to an axis.
   * - KineticEnergy
     - Calculate the timeseries of energies.
   * - Velocity
     - Analyse mean velocity.

.. _`Python3`: https://www.python.org
.. _`Cython` : https://cython.org/
.. _`GROMACS` : https://www.gromacs.org/
.. _`MDAnalysis`: https://www.mdanalysis.org
.. _`documentation`: https://maicos-devel.gitlab.io/maicos/index.html
.. _`debyer`: https://github.com/wojdyr/debyer

.. inclusion-readme-end
