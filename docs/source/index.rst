.. MAICoS documentation master file, created by
   sphinx-quickstart on Tue Jun 29 22:46:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MAICoS documentation
====================

.. image:: images/logo_MAICOS_square.png
   :align: right
   :width: 100

`MAICoS`_ is the acronym for Molecular Analysis for Interfacial and Confined Systems. It is an object-oriented python toolkit for analysing the structure and dynamics of interfacial and confined fluids from molecular simulations. Combined with `MDAnalysis`_, MAICos can be used to extract density profiles, dielectric constants, structure factors, or transport properties from trajectories files, including LAMMPS, GROMACS, CHARMM or NAMD data. MAICoS is open source and is released under the GNU general public license v3.0.

MAICoS uses some functions from MDAnalysis, `NumPy`_, and `SciPy`_, and can be :ref:`installed <label_installation>` using the pip3 package manager.

Basic example
*************

This is a simple example showing how to use MAICoS to extract the density profile from a molecular dynamics simulation. The files ``conf.gro`` and ``traj.trr`` correspond to a water slab in vacuum that was simulated in this case using the `GROMACS`_ simulation package. In a Python environment, type:

.. code-block:: python3

	import MDAnalysis as mda
	import maicos
	u = mda.Universe('conf.gro', 'traj.trr')
	grpH2O = u.select_atoms('type O or type H')	
	dplan = maicos.density_planar(grpH2O)
	dplan.run()   	

Results can be accessed from ``dplan.results``. An extended tutorial can be found :ref:`here <label_tutorial_density_planar>`.

.. _`MDAnalysis`: https://www.mdanalysis.org
.. _`MAICoS` : https://gitlab.com/maicos-devel/maicos/
.. _`NumPy` : https://numpy.org/
.. _`SciPy` : https://www.scipy.org/
.. _`GROMACS` : https://www.gromacs.org/

.. toctree::
   :maxdepth: 4
   :caption: MAICoS
   :hidden:
   
   ./documentation_pages/installation
   ./documentation_pages/usage
   ./documentation_pages/gettinginvolved
   
.. toctree::
   :maxdepth: 4
   :caption: Modules
   :hidden:
   
   ./documentation_pages/density
   ./documentation_pages/dielectric
   ./documentation_pages/structure
   ./documentation_pages/timeseries
   ./documentation_pages/transport


