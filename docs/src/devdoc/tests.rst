Testing
=======

Whenever you add a new feature to the code you should also add a test case. Further test
cases are also useful if a bug is fixed or you consider something to be worthwhile.
Follow the philosophy - the more the better!

You can run all tests by:

.. code-block:: bash

    tox

These are exactly the same tests that will be performed online in our GitHub CI
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

Benchmarks
----------

Performance benchmarks live in ``benchmarks/`` and are run with `airspeed velocity
<https://asv.readthedocs.io/>`_ (``asv``). To run the benchmarks on the last 5 commits
of the current branch and build the HTML results page:

.. code-block:: bash

    tox -e benchmarks

The number of commits is configurable via the ``BENCHMARK_COMMITS`` environment variable,
for example to run on the last 10 commits:

.. code-block:: bash

    BENCHMARK_COMMITS=10 tox -e benchmarks

You can also pass an explicit ``asv run`` range as posargs, e.g. ``tox -e benchmarks --
v1.0..HEAD``.

To view the results in your browser, run ``asv preview`` after the benchmark finishes.

To compare the current branch's benchmarks against ``main``:

.. code-block:: bash

    tox -e benchmarks-compare

This produces two files: ``benchmark_comparison.txt`` (informational, factor 1.1) and
``benchmark_regressions.txt`` (regression test, factor ``REGRESSION_FACTOR``,
default 1.5).

On every pull request, the ``ASV Benchmarks`` workflow runs ``tox -e benchmarks-compare``
and posts a single sticky comment summarising the results.

- Benchmarks whose ratio meets or exceeds ``REGRESSION_FACTOR`` (1.5x by default) are
  listed under a *Regressions* heading and also fail the CI job.
- Benchmarks that got slower by less than ``REGRESSION_FACTOR`` (but at least 1.1x) are
  tucked into a collapsed ``<details>`` block, since they are usually noise but
  occasionally worth a look.
- Benchmarks that stayed the same or improved are not included.
