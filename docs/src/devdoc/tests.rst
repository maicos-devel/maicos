Testing
=======

Whenever you add a new feature to the code you should also add a test case. Further test
cases are also useful if a bug is fixed or you consider something to be worthwhile.
Follow the philosophy - the more the better!

Continuous Integration pipeline is based on Tox_. So you need to install `tox` first::

    pip install tox
    # or
    conda install tox-c conda-forge

You can run all tests by:

.. _Tox: https://tox.readthedocs.io/en/latest/

::

    tox

These are exactly the same tests that will be performed online in our GitLab CI
workflows. You can run only a subset of the tests with ``tox -e tests --
<test/file.py>``, replacing ``<test/file.py>`` with the path to the files you want to
test, e.g. ``tox -e tests -- tests/test_main.py`` for testing only the main functions.
For more details take a look at the `usage and invocation
<https://docs.pytest.org/en/latest/usage.html#usage-and-invocations>` page of the pytest
documentation.

Also, you can run individual environments if you wish to test only specific
functionalities, for example:

::

    tox -e lint  # code style
    tox -e py310-build-linux  # packaging
    tox -e docs  # only builds the documentation
    tox -e py310-tests  # testing

Where the commands above assume that you are using Python 3.10 and Linux. For other
Python versions i.e. 3.11 change to `py311`. For MacOs use `macos` and for Windows
`windows`.

You can also use ``tox -e format`` to use tox to do actual formatting instead of just
testing it. Also, you may want to setup your editor to automatically apply the `black
<https://black.readthedocs.io/en/stable/>`_ code formatter when saving your files, there
are plugins to do this with `all major editors
<https://black.readthedocs.io/en/stable/editor_integration.html>`_.
