Contributing to the documentation
=================================

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
.. _`MAICoS develop project` : https://gitlab.com/maicos-devel/maicos
