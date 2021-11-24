.. MAICoS documentation master file, created by
   sphinx-quickstart on Tue Jun 29 22:46:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MAICoS documentation
====================

.. image:: images/logo_MAICOS_square.png
   :align: right
   :width: 100

.. include:: ../../README.rst
   :start-after: inclusion-marker-introduction-start
   :end-before: inclusion-marker-introduction-end 

MAICoS uses some functions from MDAnalysis, `NumPy`_, and `SciPy`_, and can be :ref:`installed <label_installation>` using the pip3 package manager.

.. _`NumPy` : https://numpy.org/
.. _`SciPy` : https://www.scipy.org/

Basic example
*************

.. include:: ../../README.rst
   :start-after: inclusion-marker-example-start
   :end-before: inclusion-marker-example-end 
An extended tutorial can be found :ref:`here <label_tutorial_density_planar>`.

List of modules
***************

.. include:: ../../README.rst
   :start-after: inclusion-marker-modules-start
   :end-before: inclusion-marker-modules-end 


.. toctree::
   :maxdepth: 4
   :caption: MAICoS
   :hidden:
   
   ./documentation_pages/installation
   ./documentation_pages/usage
   ./documentation_pages/gettinginvolved
   ./documentation_pages/authors
   ./documentation_pages/changelog
   
.. toctree::
   :maxdepth: 4
   :caption: Modules
   :hidden:
   
   ./documentation_pages/density
   ./documentation_pages/dielectric
   ./documentation_pages/structure
   ./documentation_pages/timeseries
   ./documentation_pages/transport


