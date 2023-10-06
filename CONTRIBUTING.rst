Getting involved
----------------

Contribution via merge requests are always welcome. Source code is available from
`GitLab`_. Before submitting a merge request, please read the `developer documentation`_
and open an issue to discuss your changes. Use only the `main` branch for submitting
your requests.

.. _`developer documentation` : https://maicos-devel.gitlab.io/maicos/devdoc
.. _`GitLab` : https://gitlab.com/maicos-devel/maicos/

By contributing to MAICoS, you accept and agree to the following terms and conditions
for your present and future contributions submitted to MAICoS. Except for the license
granted herein to MAICoS and recipients of software distributed by MAICoS, you reserve
all right, title, and interest in and to your contributions.

Getting started
---------------

To help with developing start by installing the development dependencies. Our continuous
integration pipeline is based on Tox_. So you need to install ``tox`` first

.. code-block:: bash

    pip install tox
    # or
    conda install -c conda-forge tox

Then go to the `MAICoS develop project`_ page, hit the ``Fork`` button and clone your
forked branch to your machine. 

.. code-block:: bash

  git clone git@gitlab.com:your-user-name/maicos.git

Now you have a local version on your machine which you can install by

.. code-block:: bash

  cd maicos
  pip install -e .

This install the package in development mode, making it importable globally and allowing
you to edit the code and directly use the updated version.

.. _Tox: https://tox.readthedocs.io/en/latest/
.. _`MAICoS develop project` : https://gitlab.com/maicos-devel/maicos

Code of Conduct
---------------

As contributors and maintainers of MAICoS, we pledge to respect all people who
contribute through reporting issues, posting feature requests, updating documentation,
submitting merge requests or patches, and other activities.

We are committed to making participation in this project a harassment-free experience
for everyone, regardless of level of experience, gender, gender identity and expression,
sexual orientation, disability, personal appearance, body size, race, ethnicity, age, or
religion.

Examples of unacceptable behavior by participants include the use of sexual language or
imagery, derogatory comments or personal attacks, trolling, public or private
harassment, insults, or other unprofessional conduct.

Project maintainers have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are not
aligned to this Code of Conduct. Project maintainers who do not follow the Code of
Conduct may be removed from the project team.

This code of Conduct applies both within project spaces and in public spaces when an
individual is representing the project or its community.

.. Instances of abusive, harassing, or otherwise unacceptable behavior can be
.. reported by emailing contact@maicos.org.

This Code of Conduct is adapted from the `Contributor Covenant`_, version 1.1.0,
available at https://contributor-covenant.org/version/1/1/0/

.. _`Contributor Covenant` : https://contributor-covenant.org
