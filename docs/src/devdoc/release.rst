
Release workflow
================

Versioneer (optional)
---------------------

- Upgrade versioneer if a new `version`_ is available.

- Check the `upgrade notes`_ if additional steps are required

- Upgrade versioneer

.. code-block:: bash

    pip3 install --upgrade versioneer

- Remove the old versioneer.py file

.. code-block:: bash

    rm versioneer.py

- Install new versioneer.py file

.. code-block:: bash

    python3 -m versioneer install --vendor

    Revert the changes in `src/maicos/__init__.py`

-  Commit changes

Create release
--------------

- Make sure changelog is up to date and add release date

-  Tag commit with the new version

.. code-block:: bash

    git tag -m 'Release vX.X' vX.X

-  Test locally!!!

.. code-block:: bash

    git describe

and

.. code-block:: bash

    pip3 install .

should result in ``vX.X``

- Create release branch

.. code-block:: bash

    git branch release-X-X


-  Push branch, tag

.. code-block:: bash

    git push release-X-X
    git push --tags

- Go to the `web interface`_, add changelog as release message

After the release
-----------------

- Bump version (Create new section in CHANGELOG.rst)


.. _`version` : https://pypi.org/project/versioneer
.. _`upgrade notes` : https://github.com/python-versioneer/python-versioneer/blob/master/UPGRADING.md
.. _`web interface` : https://gitlab.com/maicos-devel/maicos/-/tags

