Testing
=======

Whenever you add a new feature to the code you should also add a test case. Further test
cases are also useful if a bug is fixed or you consider something to be worthwhile.
Follow the philosophy - the more the better!

You can run all tests by:

.. code-block:: bash

    tox

These are exactly the same tests that will be performed online in our GitLab CI
workflows.

Also, you can run individual environments if you wish to test only specific
functionalities, for example

.. code-block:: bash

    tox -e lint  # code style
    tox -e build  # packaging
    tox -e tests  # testing
    tox -e docs  # build the documentation

You can also run only a subset of the tests with ``tox -e tests -- <tests/file.py>``,
replacing ``<tests/file.py>`` with the path to the files you want to test, e.g. ``tox -e
tests -- tests/test_main.py`` for testing only the main functions. For more details take
a look at the `usage and invocation
<https://docs.pytest.org/en/latest/usage.html#usage-and-invocations>` page of the pytest
documentation.

You can also use ``tox -e format`` to use tox to do actual formatting instead of just
testing it. Also, you may want to setup your editor to automatically apply the `black
<https://black.readthedocs.io/en/stable/>`_ code formatter when saving your files, there
are plugins to do this with `all major editors
<https://black.readthedocs.io/en/stable/editor_integration.html>`_.
