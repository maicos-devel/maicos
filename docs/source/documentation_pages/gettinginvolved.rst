Getting involved
================

Source code is available from `GitLab`_. See also the `README`_ in the development section.

Developing for MAICoS
*********************

Contribution via pull requests are always welcome. Before submitting a pull request please open an issue to discuss your changes. The main feature branch is the `develop` branch. Use this for submitting your requests. The master branch contains  all commits of the latest release. More information on the branching model we used is given in this 
`nice post blog`_.

Code formatting
---------------

We use yapf using the NumPy formatting style for our code. You can style your code from the command line or using an extension for your favorite editor. The easiest use is to install the git hook module, which will automatically format your code before committing. To install it just run the ``enable_githooks.sh`` from the command line. Currently we only format python files.

Writing your own Analysis module
--------------------------------

Example code for an analysis module can be found in the example folder. To deploy the script, follow the steps in examples/README.md.

UnitTests
---------

The tests also rely on the pytest library and use some work flows from numpy and MDAnalysisTests. In order to run the tests you need those packages. To start the test process just simply type from the root of the repository

.. code-block:: bash

	cd test
	pytest  --disable-pytest-warnings
	
Whenever you add a new feature to the code you should also add a test case. Furthermore test cases are also useful if a bug is fixed or anything you think worthwhile. Follow the philosophy - the more the better!

Contributing to the documentation
*********************************

The documentation of MAICoS is written in reStructuredText (rst) and uses `sphinx`_  documentation generator.

Creating a local version of the documentation
---------------------------------------------

In order to modify the documentation, first create a local version on your machine. Go to the `MAICoS develop project`_ page and hit the ``Fork`` button, then clone your forked branch to your machine:

.. code-block:: bash

    git clone git@gitlab.com:your-user-name/maicos.git
    
Then, build the documentation from the ``maicos/docs`` folder:

.. code-block:: bash

    cd maicos/docs/
    make html
    
Then, still from the ``maicos/docs/`` folder, you can visualise the local documentation with your favourite internet explorer (here Mozilla Firefox is used)
   
.. code-block:: bash

    firefox build/html/index.html

Modifying the .rst files
------------------------

You can modify directly the .rst files using your favourite text editor. You can find some help for writing in reStructuredText `here`_.

Modifying the doctring
----------------------

Each MAICoS module contains a documentation string, or docstring. Docstrings are processed by Sphinx and autodoc to generate the documentation. 

Adding the documentation for a new module
-----------------------------------------

If you just created a new module with a doctring, you can add it to the documentation by modifying the `toctree` in the ``index.rst`` file. 

.. _`gitlab` : https://gitlab.com/maicos-devel/maicos/
.. _`README` : https://gitlab.com/maicos-devel/maicos/-/tree/develop/developer
.. _`nice post blog` : https://nvie.com/posts/a-successful-git-branching-model/
.. _`MAICoS develop project` : https://gitlab.com/maicos-devel/maicos
.. _`sphinx` : https://www.sphinx-doc.org/en/master/
.. _`here` : http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html#restructured-text-rest-and-sphinx-cheatsheet


.. toctree::
   :maxdepth: 4
   :numbered:		
   :hidden:


