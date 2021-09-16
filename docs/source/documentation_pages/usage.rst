Usage
=====

From the command line
*********************

MAICoS can be used directly from the command line, by typing in a terminal:

.. code-block:: bash

	maicos <module> <paramaters>

You can get the general help page, or a package-specific page by typing, respectively:

.. code-block:: bash

	maicos -h
	
	maicos <package> -h
	
For example, to get the help page for the ``density_planar`` module, type:

.. code-block:: bash

	maicos density_planar -h

From the Python interpreter
***************************

MAICoS can be used within the python interpreter. In a python environnement, create an `analysis` object by supplying an atom group from MDAnalysis as well as some (optional) parameters, then use the `run` method:

.. code-block:: python

	import maicos

	ana_obj = maicos.<module>(atomgroup, <paramaters>)
	ana_obj.run()

Results are available through the objects `results` dictionary. See this :ref:`tutorial <label_tutorial_density_planar>` for more details. 

.. toctree::
   :maxdepth: 4
   :numbered:		
   :hidden:
   :titlesonly:


