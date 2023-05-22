Contributing to the documentation
=================================

Local documentation
-------------------

The documentation of MAICoS is written in reStructuredText (rst) and uses the
`Sphinx`_ documentation generator. You can build the documentation from the
``maicos/docs`` folder:

.. code-block:: bash

    tox -e docs

Then, visualize the local documentation with your favorite internet explorer (here
Mozilla Firefox is used)

.. code-block:: bash

    firefox dist/docs/index.html

Structure
---------

Most of the content of the documentation is written in ``.rst`` files located within
``docs/src/``. The content in the :ref:`userdoc-references` section is directly
generated from the documentation string of the source code located in ``src/maicos``
thanks to `Sphinx`_ and `Autodoc`_.

After creating a new module, add it to the documentation by modifying the `toctree` in
the ``docs/src/references/modules/index.rst`` file, and adding a new .rst file with the
following format:

.. code-block:: rst

    .. _ModuleName:

    ModuleName
    ##########

    .. _label_module_name:

    .. autoclass:: maicos.ModuleName
        :members:
        :undoc-members:
        :show-inheritance:

Note that all files located within ``docs/src/examples`` are generated from the Python
scrips located in ``examples`` using `Sphinx-Gallery`_.

.. _`Sphinx` : https://www.sphinx-doc.org/en/master/
.. _`Sphinx-Gallery` : https://sphinx-gallery.github.io/stable/index.html
.. _`Autodoc` : https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
