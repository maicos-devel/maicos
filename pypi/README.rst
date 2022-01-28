MAICoS
======

**MAICoS** is the acronym for Molecular Analysis for Interfacial
and Confined Systems. It is an object-oriented python toolkit for
analysing the structure and dynamics of interfacial and confined
fluids from molecular simulations. Combined with `MDAnalysis`_,
MAICoS can be used to extract density profiles, dielectric constants,
structure factors, or transport properties from trajectories files,
including LAMMPS, GROMACS, CHARMM or NAMD data. MAICoS is open source
and is released under the GNU general public license v3.0.

For details, tutorials, and examples, please have a look at
our `documentation`_.

List of analysis modules
########################

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Module Name
     - Description

   * - density_planar
     - Compute partial densities/temperature profiles in the Cartesian systems.
   * - density_cylinder
     - Compute partial densities across a cylinder.
   * - epsilon_bulk
     - Compute dipole moment fluctuations and static dielectric constant.
   * - epsilon_planar
     - Calculates a planar dielectric profile.
   * - epsilon_cylinder
     - Calculate cylindrical dielectric profiles.
   * - dielectric_spectrum
     - Computes the linear dielectric spectrum.
   * - saxs
     - Compute SAXS scattering intensities.
   * - diporder
     - Calculation of dipolar order parameters.
   * - debyer
     - Calculate scattering intensities using the debye equation. The `debyer`_
       library needs to be downloaded and build.
   * - dipole_angle
     - Calculate angle timeseries of dipole moments with respect to an axis.
   * - kinetic_energy
     - Calculate the timeseries of energies.
   * - velocity
     - Mean velocity analysis.

.. _`GROMACS` : https://www.gromacs.org/
.. _`MDAnalysis`: https://www.mdanalysis.org
.. _`documentation`: https://maicos-devel.gitlab.io/maicos/index.html
