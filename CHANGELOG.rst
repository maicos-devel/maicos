CHANGELOG file
--------------

The rules for MAICoS' CHANGELOG file:

- entries are sorted newest-first.
- summarize sets of changes (don't reproduce every git log comment here).
- don't ever delete anything.
- keep the format consistent (88 char width, Y/M/D date format) and do not use tabs but
  use spaces for formatting

.. inclusion-marker-changelog-start

Unreleased
----------
Philip Loche, Henrik Stooß

- Add errorbars for binned q-values in SAXS module (!328)
- Fix averaging of structure factors and add informations about NpT and SAXS (!315)
- Disable tests for PMDA API (!326)
- Add additional elements for SAXS calculations (!325)
- Add more ruff linting rules (!324)
- Fix wrong mapping of sodium element in ``atomtypes.dat`` (!323)
- Switch to the ruff linter (!322)
- Make sure box dimensions are always float64, reduced output precision (#137, !321)
- Bump minimal Python version to 3.10 (!320)

v0.9 (2025/02/19)
-----------------
Philip Loche, Marc Sauter, Kira Fischer, Federico Grasselli, Henrik Stooß,
Adyant Agrawal

- Rewrite integrals in ``DielectricCylinder`` and ``DielectricSphere`` to avoid
  numerical inaccuracies from rectangle integration method (!317)
- Update to Numpy 2.0 and MDAnalysis version 2.8.0 (!319)
- Update virtualenv setup in CI (!316)
- Split document section "Reference guide" into "Analysis Modules" and "API
  Documentation" (#132, !314)
- Improve code quality by extending linting checks (!313)
- Added tests of the analytical error propagation (!292)
- Raise an error if ``pack`` is ``True`` and ``refgroup`` is not ``None`` (!311)
- Use glob for find modules in docs (!312)
- Distributing tests across multiple CPUs (!309)
- Use Python 3.11 as default in CI (!310)
- Add flag ``pack`` to turn off the system wrap at every frame (!308)
- Fix typos in ``DipoleAngle`` (!307)
- Remove handling of multiple atomgroups in favor of ``AnalysisCollection`` (!301)
- Fix openMP detection during setup (!304)
- :class:`maicos.Saxs` additionally provides structure factor. (!303)
- Remove default arguments from core classes (!302)
- Add an ``AnalaysisCollection`` class to perform multiple analyses on the same
  trajectory (!298)
- Remove custom module command line interface (!299)
- Add example for :class:`maicos.core.AnalysisBase` and rework own module section in
  developer docs (!299)
- Allow running an analysis with a universe without a cell (!297)
- Test that :class:`core.AnalysisBase` API and ``run`` method is the same as
  :class:`MDAnalysis.analysis.base.AnalysisBase` (!297)
- Add ``frames`` and ``progressbar_kwargs`` argument to
  :meth:`maicos.core.AnalysisBase.run` (!297)
- Update copyright year (!296)
- Add new diporder modules: ``RDFDiporder``, ``DiporderStructureFactor`` (!296)
- Add correlation time estimate for ``SAXS`` module (!296)
- Added tests of the analytical error propagation (!292)
- Remove symbolic links from examples (!295)

v0.8 (2024/02/05)
-----------------
Simon Gravelle, Philip Loche, Marc Sauter, Henrik Stooß, Philipp Staerk, Adyant Agrawal,
Kira Fischer

- Skip test for custom modules in case the import is not working (!294)
- Change to CHANGELOG.rst update check so that it is only executed in MRs (!198)
- Rename radial distribution function to pair distribution function (!278)
- Add RDF derivation and explain role of dz. (!278)
- Implement 1D pair distribution function in RDFCylinder (!276)
- Sort format and add more atomtypes to ``atomtypes.dat`` (!291)
- Add grouping option to `DipoleAngle` module (!290)
- Added Support for Python 3.12 (!289)
- Remove suffixes ``-linux``, ``-macos``, ``-windows`` when building wheels. Platform
  will be detected automatically. (!288)
- Use default tox error for non-exsiting enviroment (!285)
- Parse documentation metadata from ``pyproject.toml`` (!287)
- Convert ``pathlib.Path`` into ``str`` when using in ``sys.path.append`` (#123, !286)
- Update dev names (!284)
- Improvements to documentation rendering (#122, !282)
- Unify Python versions in tox environments i.e. ``py311-build-macos`` to
  ``build-macos`` (!283)
- Remove deprecated pytest tmpdir fixture (!283)
- Remove deprecated ``assert_almost_equal`` in favor of ``assert_allclose`` (!283)
- Move from ``os.path`` to ``pathlib.Path`` (!283)
- Added Support for Python 3.11 (!283)
- Update MacOS images for CI (!281)
- Removed the obsolete option for the vacuum boundary condition in the
  ``DielectricPlanar`` module and prompt users to use tin-foil boundary
  conditions instead (!280).
- Add physical integration test to test that structure factor from Saxs is the same as
  the Fourier transformed RDF. (!279)
- Add example and explenation of how to relate the radial distribution function and the
  structure factor (!279)
- Add function :func:`maicos.lib.math.rdf_structure_factor` for converting a radial
  distribution function into a structure factor. (!279)
- Change default biwnwidth (``dq``) in :class:`maicos.Saxs` to ``0.1``. (!279)
- Move ``cutils`` to ``cmath`` (!279)
- Add ``weight`` argument to :func:`maicos.lib._cmath.compute_structure_factor`
- Code cleanup of :class:`maicos.Saxs` (!279)
- Fixed markup and consistency in ``correlation`` function docs (!277)
- Add info for ``DielectricPlanar`` module for ignored combination of ``vac=True`` and
  ``is_3d=False``. (!275)
- Add description for `tox` jobs (!275)
- Cleanup coverage config and move to ``pyproject.toml`` (!275)
- Changed the way number normalization works, introduced sums dict (!274)
- Fixed typing error in RDF modules (!273)
- Update docs to reflect changes in ``mdacli`` (!271)
- Add banner to MAICoS output reporting the version (!272)
- Update UML graphic (!269)
- Show warnings if set boundaries would result in wrong results (!261)
- Small corrections to the documentation and type hinting (!268)
- Add module for calculating radial distribution functions in cylinders (!242)
- Add modules for calculating cylindrical and spherical dipolar order parameters (!259)
- Fix reproducibility information in output (!263)
- Make savetxt work with Pathlib objects (!267)
- Update versionner to 0.29 (!266)
- Use ``dipole_vector`` methods from MDAnalysis (!265)
- Bump minimum Python version to 3.9 (!264)
- Fix dipole calculation in ``DielectricCylinder`` (!258)
- Add example for RDFPlanar (!256)
- Move geometry transformations to ``lib.math`` (!257)
- Add typehints for examples (!255)
- Add typehints for modules (!253)
- Only test minimum and maximum Python version in CI (!252)
- Add typehints for core classes (!251)
- Update documentation with parameters, returns and examples for library functions
  (!248)
- Update CI to use latest MacOS (!250)
- Add tables to documentation pages (!249)
- Fix links to own classes in examples (!247)
- Update install instructions for users and devs (!246)
- Show authors on website (!245)
- Add link to developer documentation in ``CONTRIBUTING.rst`` (!244)
- Remove Python 2.x leftover of specific ``super()`` call (!243)
- Use Gitlab for showing coverage and unit test reports (!241)
- Use ``black`` formatter and `88` chars/line for the code and rst files (!240)
- Add return values for correlation analysis to all base classes (!235)
- Added more linting for rst files (!239)
- Bump minimum version of ``tqdm`` to 4.60 (!238)
- Add prompt toggle to examples (!236)
- Added description to the ideal chemical potential how-to (!232)
- Added quotation marks to command in tox.ini to account for spaces in paths (!232)
- Fixed some typos and made minor modifications to the documentation (!232)
- Cleanup .gitignore (!233)
- More consistent molecule wrapping (!230)
- Added missing AnalysisBase parameters to modules (!231)
- created dark and light images and logo (!229)
- Add explicit `stacklevel` arguments to warnings in the library (!236)
- Switch to the `build` module (!234)

v0.7.2 (2023/01/09)
-------------------
Philip Loche, Henrik Stooß

- Remove superfluous group wise wrapping (!225)
- Clarify unclear definition in Dieletric modules that could lead to wrong results
  (!228)
- Fixed windows string manipulation in test CI (!227)
- Added coverage posting on GitLab (!226)
- Corrected wrong comparison in correlation analysis and added tests
- Fixed link to changelog in pyproject.toml
- Migrated versioneer to pyproject.toml
- Added Support for Python 3.11

v0.7.1 (2023/01/01)
-------------------
Henrik Stooß

- Fix upload to PyPi. This release is identical to v0.7.

v0.7 (2022/12/27)
-----------------
Philip Loche, Simon Gravelle, Marc Sauter, Henrik Stooß, Kira Fischer, Alexander
Schlaich, Philipp Staerk

- Make sure citation are only printed once (!260)
- Added MacOS pipeline, fixed wheels (!218)
- Fix CHANGELOG testing (!220)
- Added dielectric how-to (!208)
- Raise an error if ``unwrap=False`` and ``refgroup != None`` in dielectric modules
  (!215).
- Fix velocity profiles (!211)
- Added the Theory to the Dielectric docs (!201)
- Add a logger info for citations (!205)
- Rename Diporder to DiporderPlanar (!202)
- Change default behavior of DielectricPlanar: assume slab geometry by default (removing
  the ``xy`` flag and instead introduce ``is_3d`` for 3d-periodic systems) (!202)
- Rename ``profile_mean`` to ``profile`` (!202)
- Major improvements on the documentation (!202)
- Add a check if the CHANGELOG.rst has been updated (!198)
- Fix behaviour of refgroup (!192)
- Resolve +1 is summed for epsilon for each atom group (#101, !193)
- Flatten file structure of analysis modules (#46, !196)
- Consistent mass unit in docs
- Porting examples to sphinx-gallery (!190)
- Add ``jitter`` parameter to AnalysisBase (!183)
- Test output messages (!191)
- Fixed typo in ``DielectricPlanar`` docs (!194)
- Add Sphere modules (!175)
- Add ``ProfileBase`` class (!180)
- Slight restructure of the documenation (!189)
- Fix py311 windows
- Update build requirements for py310 and py311
- Merged setup.cfg into pyproject.toml (!187)
- Use versioneer for version info (!150)
- Update project urls (!185)
- Added repository link in the documentation (!184)
- Added windows CI/CD pipeline (!182)
- Update package discovery methods in setup.cfg
- Refactor CI script (!181)
- Fix ``DielectricCylinder`` (!165)
- Unified ``n_bins`` logging (#93, !179)
- Add MAICoS UML Class Diagramm (!178)
- Changed density calculation using range in np.histogram (!77)
- Update branching model in the documentation (!177)
- remove ./ from index.rst
- Improve documentation (!174)
- Added reference for SAXS calculations (!176)
- Update type of bin_pos in docs
- Added ``VelocityCylinder`` module
- Change behavior of ``sort_atomgroup`` (#88, !152)
- ``get_compound``: option for returning indices of topology attributes
- Added Tutorial for non-spatial analysis module (!170)
- Check atomgroups if they contain any atoms (!172)
- New core attributes: ``bin_edges``, ``bin_area``, ``bin_volume``, ``bin_pos`` &
  ``bin_width`` (!167)
- Use ``frame`` dict in ``structure.py`` (!169)
- Fix box dimensions for cylindrical boundaries (!168)
- ``rmax`` for cylindrical systems now uses correct dimensions
- Transport module documentation update (!164)
- Rename frame dict (!166)
- Implement ``SphereBase`` and ``ProfileSphereBase`` (!162)
- Relative path for data (!163)
- Create Linux wheels (!160)
- Fix ``Diporder`` tests (!161)
- ``norm=number``: Declare bins with no atoms as ``nan`` (!157)
- Simplify weight functions (!158)

v0.6.1 (2022/09/26)
-------------------
Henrik Stooß

- Fix the output of the `ChemicalPotentialPlanar` module (!173)

v0.6 (2022/09/01)
-----------------
Philip Loche, Simon Gravelle, Srihas Velpuri, Henrik Stooß, Alexander Schlaich,
Maximilian Becker, Kira Fischer

- Write total epsilon as defined in paper (!155)
- Introduce generic header (!149)
- Fix error estimate in ``EpsilonPlanar`` (!153)
- Fix sym option in ``EpsilonPlanar`` (!148)
- Use standard error of the mean instead of variance for error estimate (!147)
- Make all tests that write file use temporary file directory (!151)
- Rewrite ``Velocity`` module using ``ProfilePlanarBase`` (!142)
- Add ``RDFPlanar`` (!133)
- Refactor ``EpsilonPlanar`` (!139)
- Add a correlation time estimator (!137)
- Add ``frame`` dict to ``AnalysisBase`` (!138)
- Generalize ``comgroup`` attribute to all dimensions (!132)
- Output headers do not require residue names anymore (!134)
- Remove ``Debyer`` class (!130)
- Generalize ``concfreq`` attribute in ``AnalysisBase`` (!122)
- Fix broken binning in ``EpsilonPlanar`` (!125)
- Removed ``repairMolecules`` (!119)
- Added ``grouping`` and ``bin_method`` option (!117)
- Bump minimum MDAnalysis version to 2.2.0 (!117)
- Bump minimum Python version to 3.8 (!117)
- Use base units exclusively (!115)
- Higher tolerance for non-neutral systems (1E-4 instead of 1E-5)
- ``charge``neutral decorator uses ``check_compound`` now
- Add option to symmetrize profiles using ``ProfilePlanarBase`` (!116)
- Fix ``comgroup`` parameter working only in the z direction (!116)
- Remove ``center`` option from ``ProfileBase`` (!116)
- Introduces new ``ProfilePlanarBase`` (!111)
- Split new ``DensityPlanar`` into ``ChemicalPotentialPlanar``, ``DensityPlanar``,
  ``TemperaturePlanar`` (!111)
- Convert more ``print`` statements into logger calls (!111)
- Fix wrong ``Diporder`` normalization + tests (!111)
- Add ``zmin`` and ``zmax`` to DensityPlanar and Diporder (!109)
- Fix EpsilonPlanar (!108)
- More tests for ``DensityPlanar``, ``DensityCylinder``, ``KineticEnergy`` and
  ``DipoleAngle`` (!104)
- Remove ``EpsilonBulk`` (!107)
- Add Code of Conduct (!97)
- Fix lint errors (!95)

v0.5.1 (2022/02/21)
-------------------
Henrik Stooß

- Fix pypi installation (!98)

v0.5 (2022/02/17)
-----------------
Philip Loche, Srihas Velpuri, Simon Gravelle

- Convert Tutorials into notebooks (!93)
- New docs design (!93)
- Build gitlab docs only on master branch (!94, #62)
- Removed oxygen binning from diporder (!85)
- Improved CI including tests for building and linting
- Create a consistent value of ``zmax`` in every frame (!79)
- Corrected README for pypi (!83)
- Use Results class for attributes and improved docs (!81)
- Bump minimum Python version to 3.7 (!80)
- Remove spaghetti code in ``__main__.py`` and introduce ``mdacli`` as cli server
  library. (!80)
- Remove ``SingleGroupAnalysisBase`` and ``MultiGroupAnalysisBase`` classes in favour of
  a unified ``AnalysisBase`` class (!80)
- Change ``planar_base`` decorator to a ``PlanarBase`` class (!80)
- Rename modules to be consistent with PEP8 (``density_planar`` -> ``DensityPlanar``)
  (!80)
- Use Numpy's docstyle for doc formatting (!80)
- Use Python's powerful logger library instead of bare ``print`` (!80)
- Use Python 3.6 string formatting (!80)
- Remove ``_calculate_results`` methods. This method is covered by the ``_conclude``
  method. (!80)
- Make results saving a public function (save) (!80)
- Added docstring Decorator for ``PlanarDocstring`` and ``verbose`` option (!80)
- Use ``MDAnalysis``'s' ``center_of_mass`` function for center of mass shifting (!80)

v0.4.1 (2021/12/17)
-------------------
Philip Loche

- Fixed double counting of the box length in diporder (#58, !76)

v0.4 (2021/12/13)
-----------------

Philip Loche, Simon Gravelle, Philipp Staerk, Henrik Stooß, Srihas Velpuri, Maximilian
Becker

- Restructure docs and build docs for develop and release version
- Include README files into sphinx doc
- Add tutorial for density_cylinder module
- Add ``planar_base`` decorator unifying the syntax for planar analysis modules as
  ``denisty_planar``, ``epsilon_planar`` and ``diporder`` (!48)
- Corrected time_series module and created a test for it
- Added support for Python 3.9
- Created sphinx documentation
- Raise error if end is to small (#40)
- Add sorting of atom groups into molecules, enabling import of LAMMPS data
- Corrected plot format selection in ``dielectric_spectrum`` (!66)
- Fixed box dimension not set properly (!64)
- Add docs for timeseries modulees (!72)
- Fixed diporder does not compute the right quantities (#55, !75)
- Added support of calculating the chemical potentials for multiple atomgroups.
- Changed the codes behaviour of calculating the chemical potential if atomgroups
  contain multiple residues.

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
