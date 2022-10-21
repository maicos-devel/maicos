
By contributing to MAICoS, you accept and agree to the following terms and
conditions for your present and future contributions submitted to MAICoS.
Except for the license granted herein to MAICoS and recipients of software
distributed by MAICoS, you reserve all right, title, and interest in and to
your contributions.

Code of Conduct
---------------

As contributors and maintainers of MAICoS, we pledge to respect all people
who contribute through reporting issues, posting feature requests, updating
documentation, submitting merge requests or patches, and other activities.

We are committed to making participation in this project a harassment-free
experience for everyone, regardless of level of experience, gender, gender
identity and expression, sexual orientation, disability, personal appearance,
body size, race, ethnicity, age, or religion.

Examples of unacceptable behavior by participants include the use of sexual
language or imagery, derogatory comments or personal attacks, trolling, public
or private harassment, insults, or other unprofessional conduct.

Project maintainers have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct. Project maintainers who do not follow the
Code of Conduct may be removed from the project team.

This code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community.

.. Instances of abusive, harassing, or otherwise unacceptable behavior can be
.. reported by emailing contact@maicos.org.

This Code of Conduct is adapted from the `Contributor Covenant`_, version 1.1.0,
available at https://contributor-covenant.org/version/1/1/0/

.. _`Contributor Covenant` : https://contributor-covenant.org

Getting involved
----------------

Contribution via merge requests are always welcome. Source code is
available from `GitLab`_. Before submitting a merge request, please
open an issue to discuss your changes. Use only the `main` branch 
for submitting your requests.

.. _`gitlab` : https://gitlab.com/maicos-devel/maicos/

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

Some conventions
----------------

The base units of MAICoS are consistent with those of `MDAnalysis`_. Keeping inputs and outputs consistent with this set of units reduces ambiguity, so we encourage everyone to use them exclusively.

.. _`MDAnalysis` : https://docs.mdanalysis.org/stable/documentation_pages/units.html

The base units are:

.. Table:: Base units in MDAnalysis

   =========== ============== ===============================================
   quantity    unit            SI units
   =========== ============== ===============================================
   length       Å              :math:`10^{-10}` m
   time         ps             :math:`10^{-12}` s
   energy       kJ/mol         :math:`1.66053892103219 \times 10^{-21}` J
   charge       :math:`e`      :math:`1.602176565 \times 10^{-19}` As
   force        kJ/(mol·Å)     :math:`1.66053892103219 \times 10^{-11}` J/m
   speed        Å/ps           :math:`100` m/s
   =========== ============== ===============================================

Version information
-------------------

The version information in `maicos.__version__` indicates the
release of MAICoS using `semantic versioning`_.

In brief:

Given a version number MAJOR.MINOR.PATCH, we increment the

1. **MAJOR** version when we make **incompatible API changes**,
2. **MINOR** version when we **add functionality** in a
**backwards-compatible** manner, and
3. **PATCH** version when we make backwards-compatible **bug fixes**.

However, as long as the **MAJOR** number is **0** (i.e. the API has
not stabilized), even **MINOR** increases *may* introduce incompatible
API changes. As soon as we have a 1.0.0 release, the public API can
only be changed in a backward-incompatible manner with an increase in
MAJOR version.

Additional labels for pre-release and build metadata are available as
extensions to the MAJOR.MINOR.PATCH format, following :pep:`440`.


.. Note:: Development versions and pre-releases have a suffix after
        the release number, such as ``0.7.0+12.gFEED2BEEF``. If you have
        problems, try out a full release (e.g. ``0.7.0``) first.


.. _`semantic versioning`: http://semver.org/


Data
----

.. autodata:: maicos.__version__
