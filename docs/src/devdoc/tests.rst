Testing
=======

Whenever you add a new feature to the code you should also add a test case.
Furthermore test cases are also useful if a bug is fixed or anything you think
worthwhile. Follow the philosophy - the more the better!

Continuous Integration pipeline is based on Tox_.
So you need to install `tox` first::

    pip install tox
    # or
    conda install tox-c conda-forge

You can run all tests by:

.. _Tox: https://tox.readthedocs.io/en/latest/

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
