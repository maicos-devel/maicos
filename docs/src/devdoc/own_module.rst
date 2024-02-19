Contributing your own analysis module
=====================================

To write your module take a look at the comprehensive example in the documentation of
:class:`maicos.core.AnalysisBase`. MAICoS also has more specific base classes for
different geometries that make developing modules much easier. You may take a look at
the source code at ``src/maicos/modules``.

After you wrote your module you can add it in a new file in ``src/maicos/modules``. On
top of that please also update the list in ``src/maicos/modules/__init__.py``
accordingly. Also, create a new ``.rst`` file with your module name in
``docs/src/references/modules`` similar to the already existing. To finally show the
documentation for the other modules add an entry in
``docs/src/references/modules/index.rst`` in alphabetical order.

All MAICoS modules are also listed in the ``README.rst`` and you should add your module
as well.

Finally, also provide meaningful tests for your module in ``test/modules``.

For further questions feel free to ask us on our Discord_ server.

.. _`Discord`: https://discord.gg/mnrEQWVAed
