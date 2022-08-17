.. _label_installation:

======================
Installation and usage
======================

Installation
############

`Python3`_ and a C-compiler are needed to build the
underlying libraries. Install the package using 
`pip`_ with:

.. code-block:: bash

    pip3 install maicos

or using `conda`_ with:

.. code-block:: bash

    conda install maicos

Alternatively, if you don't have special privileges, install
the package only for the current using the ``--user`` flag:

.. code-block:: bash

    pip3 install --user maicos

.. _`Python3`: https://www.python.org
.. _`pip`: http://www.pip-installer.org/en/latest/index.html
.. _`conda`: https://docs.conda.io/
.. _`MDAnalysis`: https://www.mdanalysis.org/

   
Usage
#####

To find out more, have a look at our Tutorials (for beginners) 
and How-to-guides (for advanced users). For each module, parameters
are listed in the modules section.

From the command line
---------------------

MAICoS can be used directly from the command line, by typing in a terminal:

.. code-block:: bash

	maicos densityplanar -s conf.gro -f traj.xtc

You can get the general help page,
or a package-specific page by typing, respectively:

.. code-block:: bash

	maicos -h

	maicos densityplanar -h

From the Python interpreter
---------------------------

MAICoS can be used within the python interpreter. In a Python environment,
create an ``analysis`` object by supplying an atom group from `MDAnalysis`_
as well as some (optional) parameters, then use the ``run`` method:

.. code-block:: python

	import MDAnalysis as mda
	import maicos
	u = mda.Universe('conf.gro', 'traj.trr')
	grpH2O = u.select_atoms('type O or type H')
	dplan = maicos.DensityPlanar(grpH2O)
	dplan.run()

Results are available through the objects ``results`` dictionary. Use 
``verbose=True`` to see a progress bar, and ``start``, ``stop`` and ``step`` to 
analyse only a subpart of a trajectory file:

.. code-block:: python

	dplan.run(verbose = True, stop = 50)

.. toctree::
   :maxdepth: 4
   :hidden:
   :titlesonly:
