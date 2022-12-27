#!/usr/bin/env python3
"""Check if the CHANGELOG has been modified with respect to the main branch."""
from os import path

import git


class ChangelogError(Exception):
    """Changelog error."""

    pass


changelog = "CHANGELOG.rst"
repo_path = path.realpath(path.join(path.dirname(__file__), ".."))

repo = git.Repo(repo_path)

file = repo.git.show(f"origin/main:{changelog}")

with open(path.join(repo_path, changelog), 'r') as f:
    workfile = f.read()

try:
    if repo.active_branch.name != "main" and file == workfile:
        raise ChangelogError("You have not updated the CHANGELOG file. Please "
                             f"add a summary of your additions to {changelog}.")
except TypeError:
    # This happens when we are (for example) checking out a tag. In that case
    # we don't care about the changelog.
    pass
