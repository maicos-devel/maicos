CHANGELOG file
--------------

The rules for MAICoS' CHANGELOG file:

- entries are sorted newest-first.
- summarize sets of changes (don't reproduce every git log comment here).
- don't ever delete anything.
- keep the format consistent (79 char width, Y/M/D date format) and do not
  use tabs but use spaces for formatting

.. inclusion-marker-changelog-start

v0.4 (XXXX/XX/XX)
-----------------

Philip Loche, Simon Gravelle, Philipp Staerk, Henrik Jaeger

- Restructure docs and build docs for develop and release version
- Include README files into sphinx doc
- Add tutorial for density_cylinder module
- Add `planar_base` decorator unifying the syntax for planar analysis modules
  as `denisty_planar`, `epsilon_planar` and `diporder` (!48)
- Corrected time_series module and created a test for it
- Added support for Python 3.9
- Created sphinx documentation
- Raise error if end is to small (#40)
- Add sorting of atom groups into molecules, enabling import of LAMMPS data
- Corrected plot format selection in `dielectric_spectrum` (!66)

v0.3 (2020/03/03)
-----------------

Philip Loche, Amanuel Wolde-Kidan

- Fixed errors occurring from changes in MDAnalysis
- Increased minimal requirements
- Use new ProgressBar from MDAnalysis
- Increased total_charge to account for numerical inaccuracy

v0.2 (2020/04/03)
-----------------

Philip Loche

- Added custom module
- Less noisy DeprecationWarning
- Fixed wrong center of mass velocity in velocity module
- Fixed documentation in diporder for P0
- Fixed debug if error in parsing
- Added checks for charge neutrality in dielectric analysis
- Added test files for an air-water trajectory and the diporder module
- Performance tweaks and tests for sfactor
- Check for molecular information in modules

v0.1 (2019/10/30)
-----------------

Philip Loche

- first release out of the lab

.. inclusion-marker-changelog-end
