
Release workflow
================

Versioneer (optional)
---------------------

1. Upgrade versioneer if a new `version`_ is available.

2. Check the `upgrade notes`_ if additional steps are required

3. Upgrade versioneer

   .. code-block:: bash

    pip3 install --upgrade versioneer

4. Remove the old versioneer.py file

   .. code-block:: bash

    rm versioneer.py

5. Install new versioneer.py file

   .. code-block:: bash

    python3 -m versioneer install --vendor

   Revert the changes in ``src/maicos/__init__.py``

6. Commit changes

Create release
--------------

1. Make sure changelog is up to date and add release date and commit
   your changes

   .. code-block:: bash

    git commit -m 'Release vX.X'

2. Tag commit with the new version

   .. code-block:: bash

    git tag -m 'Release vX.X' vX.X

3. Test locally!!!

   .. code-block:: bash

    git describe

   and

   .. code-block:: bash

    pip3 install .

   should result in ``vX.X``

4. Push tag

   .. code-block:: bash

    git push --tags

5. Go to the `web interface`_, add changelog as release message

After the release
-----------------

- Bump version (Create new section in CHANGELOG.rst)

.. _`version` : https://pypi.org/project/versioneer
.. _`upgrade notes` : https://github.com/python-versioneer/python-versioneer/blob/master/UPGRADING.md
.. _`web interface` : https://github.com/maicos-devel/maicos/releases
