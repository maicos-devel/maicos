Getting involved
################

Contribution via pull requests are always welcome. Source code is
available from `GitLab`_. Before submitting a pull request, please
open an issue to discuss your changes. Use the main feature branch
`develop` for submitting your requests. The master branch contains
all commits of the latest release. More information on the branching
model we used is given in this `nice post blog`_.

.. _`gitlab` : https://gitlab.com/maicos-devel/maicos/
.. _`nice post blog` : https://nvie.com/posts/a-successful-git-branching-model/

Testing
-------

Continuous Integration pipeline is based on `Tox`_.
So you need to install `tox` first::

    pip install tox
    # or
    conda install tox-c conda-forge

You can run all tests by:

::

    tox

These are exactly the same tests that will be performed online in our
GitLab CI workflows.

Also, you can run individual environments if you wish to test only
specific functionalities, for example:

::

    tox -e lint  # code style
    tox -e build  # packaging
    tox -e docs  # only builds the documentation
    tox -e tests  # testing

Writing your own analysis module
--------------------------------

Example code for an analysis module can be found in the example
folder. To deploy the script, follow the steps in `examples/README.md`_.

We use yapf using the NumPy formatting style for our code.
You can style your code from the command line or using an
extension for your favorite editor. The easiest use is to
install the git hook module, which will automatically format
your code before committing. To install it just run the
``enable_githooks.sh`` from the command line. Currently,
we only format python files.

.. _`examples/README.md` : https://gitlab.com/maicos-devel/maicos/-/tree/develop/examples

MAICoS' unit testing relies on the pytest library and use some work flows
from numpy and MDAnalysisTests. In order to run the tests you need those
packages. To start the test process, simply type from the root of the
repository

.. code-block:: bash

	cd test
	pytest  --disable-pytest-warnings

Whenever you add a new feature to the code you should also add a test case.
Furthermore test cases are also useful if a bug is fixed or anything you think
worthwhile. Follow the philosophy - the more the better!

Contributing to the documentation
---------------------------------

The documentation of MAICoS is written in reStructuredText (rst)
and uses `sphinx`_ documentation generator. In order to modify the
documentation, first create a local version on your machine.
Go to the `MAICoS develop project`_ page and hit the ``Fork``
button, then clone your forked branch to your machine:

.. code-block:: bash

    git clone git@gitlab.com:your-user-name/maicos.git

Then, build the documentation from the ``maicos/docs`` folder:

.. code-block:: bash

    tox -e docs

Then, visualise the local documentation
with your favourite internet explorer (here Mozilla Firefox is used)

.. code-block:: bash

    firefox dist/docs/index.html

Each MAICoS module contains a documentation string, or docstring. Docstrings
are processed by Sphinx and autodoc to generate the documentation. If you created
a new module with a doctring, you can add it to the documentation by modifying
the `toctree` in the ``index.rst`` file.

.. _`sphinx` : https://www.sphinx-doc.org/en/master/
.. _Tox: https://tox.readthedocs.io/en/latest/
.. _`MAICoS develop project` : https://gitlab.com/maicos-devel/maicos
