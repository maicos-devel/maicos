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

if repo.active_branch.name != "main" and file == workfile:
    raise ChangelogError("You have not updated the CHANGELOG file. Please "
                         f"add a summary of your additions to {changelog}.")
