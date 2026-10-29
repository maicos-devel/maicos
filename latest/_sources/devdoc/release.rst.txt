
Release workflow
================

1. **Prepare a Release Pull Request**

   - Based on the main branch create branch ``release-x.y.z`` and a PR.
   - Ensure that all `CI tests
     <https://github.com/maicos-devel/maicos/actions>`_ pass.
   - Optionally, run the tests locally to double-check.

2. **Update the Changelog**

   Edit the changelog located in ``CHANGELOG`` and add the new version and the date of
   the release.

3. **Merge the PR and Create a Tag**

   - Merge the release PR.
   - Update the ``main`` branch and check that the latest commit is the release PR with
     ``git log``
   - Create a tag on directly the ``main`` branch.
   - Push the tag to GitHub. For example for a release of version ``x.y.z``:

     .. code-block:: bash

        git checkout main
        git pull
        git tag -a vx.y.z -m "Release vx.y.z"
        git push --tags

4. **Trigger release workflow**

   - Once the tag is pushed, navigate to the *Actions* tab of the repository.
   - Select the ``Build`` workflow section.
   - Click on the "Run workflow" dropdown and select the ``vx.y.z`` tag you just
     created.

5. **Finalize the GitHub Release**

   - The CI will automatically:
      - Publish the package to PyPI.
      - Create a draft release on GitHub.
   - Update the GitHub release notes by pasting the changelog for the version.

6. **Merge Conda Recipe Changes**

   - May resolve and then merge an automatically created PR on the `conda recipe
     <https://github.com/conda-forge/maicos-feedstock>`_.
   - Once thus PR is merged and the new version will be published automatically on the
     `conda-forge <https://anaconda.org/conda-forge/maicos>`_ channel.

After the release
-----------------

Add a placeholder section titled *Unreleased* for future updates.

.. _`web interface` : https://github.com/maicos-devel/maicos/releases
