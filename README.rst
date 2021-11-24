
.. image:: docs/source/images/logo_MAICOS_small.png
   :align: left
   :alt: **MAICoS**

|

.. inclusion-marker-introduction-start

**MAICoS** is the acronym for Molecular Analysis for Interfacial 
and Confined Systems. It is an object-oriented python toolkit for 
analysing the structure and dynamics of interfacial and confined 
fluids from molecular simulations. Combined with `MDAnalysis`_, 
MAICoS can be used to extract density profiles, dielectric constants, 
structure factors, or transport properties from trajectories files, 
including LAMMPS, GROMACS, CHARMM or NAMD data. MAICoS is open source 
and is released under the GNU general public license v3.0.

For details, tutorials, and examples, please have a look at 
our `Sphinx documentation`_.

.. _`MDAnalysis`: https://www.mdanalysis.org

.. inclusion-marker-introduction-end

.. _`Sphinx documentation`: https://maicos-devel.gitlab.io/maicos/index.html

Basic example
#############

.. inclusion-marker-example-start

This is a simple example showing how to use MAICoS to extract the density profile from a molecular dynamics simulation. The files ``conf.gro`` and ``traj.trr`` correspond to a water slab in vacuum that was simulated in this case using the `GROMACS`_ simulation package. In a Python environment, type:

.. code-block:: python3

	import MDAnalysis as mda
	import maicos
	u = mda.Universe('conf.gro', 'traj.trr')
	grpH2O = u.select_atoms('type O or type H')	
	dplan = maicos.density_planar(grpH2O)
	dplan.run()   	

.. _`GROMACS` : https://www.gromacs.org/

Results can be accessed from ``dplan.results``.

.. inclusion-marker-example-end

Installation
############
.. inclusion-marker-installation-start

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
the package in your home directory by using the ``--user`` flag:

.. code-block:: bash

    pip3 install --user numpy
    pip3 install --user maicos
    
Bash autocompletion
*******************

You can include MAICoS to ``BASH`` suggestions by oppening your 
``.bashrc`` or ``.profile`` file with your favorite text editor 
(here vim is used):

.. code-block:: bash

    vim ~/.bashrc

and by adding

.. code-block:: bash

    source $(maicos --bash_completion)
    
Development version
-------------------

The development version of MAICoS can be compiled from source. 
`NumPy`_ and `Cython`_ are required:

.. code-block:: bash

    pip3 install numpy
    pip3 install cython
    
Then type in a terminal:

.. code-block:: bash

    git clone git@gitlab.com:maicos-devel/maicos.git
    pip3 install -e maicos/

Testing
*******

You can run the tests from the ``maicos/tests/`` directory. The tests 
rely on the `pytest`_ library, and use some work flows from NumPy 
and `MDAnalysisTests`_. In a terminal, type:

.. code-block:: bash

    pip3 install MDAnalysisTests

Then, type:

.. code-block:: bash

    cd maicos/tests
    pytest  --disable-pytest-warnings

.. _`Python3`: https://www.python.org
.. _`NumPy` : https://numpy.org/
.. _`Cython` : https://cython.org/
.. _`pytest` : https://docs.pytest.org/en/6.2.x/
.. _`MDAnalysisTests` : https://pypi.org/project/MDAnalysisTests/

.. inclusion-marker-installation-end

Usage
#####
.. inclusion-marker-usage-start

From the command line
---------------------

MAICoS can be used directly from the command line, by typing in a terminal:

.. code-block:: bash

	maicos <module> <paramaters>

You can get the general help page, 
or a package-specific page by typing, respectively:

.. code-block:: bash

	maicos -h
	
	maicos <package> -h
	
For example, to get the help page for the ``density_planar`` module, type:

.. code-block:: bash

	maicos density_planar -h

From the Python interpreter
---------------------------

MAICoS can be used within the python interpreter. In a python environment, 
create an ``analysis`` object by supplying an atom group from MDAnalysis 
as well as some (optional) parameters, then use the ``run`` method:

.. code-block:: python

	import maicos

	ana_obj = maicos.<module>(atomgroup, <paramaters>)
	ana_obj.run()

Results are available through the objects `results` dictionary. 

.. inclusion-marker-usage-end

List of modules
###############

.. inclusion-marker-modules-start

**Density**

* **density_planar**: Compute partial densities/temperature profiles in the Cartesian systems.
* **density_cylinder**: Compute partial densities across a cylinder.

**Dielectric constant**

* **epsilon_bulk**: Compute dipole moment fluctuations and static dielectric constant.
* **epsilon_planar**: Calculates a planar dielectric profile.
* **epsilon_cylinder**: Calculate cylindrical dielectric profiles.
* **dielectric_spectrum**: Computes the linear dielectric spectrum.

**Structure**

* **saxs**: Compute SAXS scattering intensities.
* **diporder**: Calculation of dipolar order parameters.
* **debyer**: Calculate scattering intensities using the debye equation. The debyer library needs to be downloaded and build, see <https://github.com/wojdyr/debyer>.

**Timeseries**

* **dipole_angle**: Calculate angle timeseries of dipole moments with respect to an axis.
* **kinetic_energy**: Calculate the timeseries of energies.

**Transport**

* **velocity**: Mean velocity analysis.

.. inclusion-marker-modules-end

Contributing
############

If you find a bug, have questions, or want to suggest an improvement,
feel free to raise an issue. Contribution via pull requests 
are always welcome. For more details see the `README`_ 
in the development section.

.. _`README` : https://gitlab.com/maicos-devel/maicos/-/tree/develop/developer
