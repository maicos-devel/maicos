#!/usr/bin/env python3
"""Check if the CHANGELOG has been modified with respect to the main branch."""
import os
import git


class ChangelogError(Exception):
    """Changelog error."""
    pass


changelog = "CHANGELOG.rst"
repo_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

repo = git.Repo(repo_path)

file = repo.git.show(f"origin/main:{changelog}")

with open(os.path.join(repo_path, changelog), 'r') as f:
    workfile = f.read()

if "CI_OPEN_MERGE_REQUESTS" in os.environ and file.strip() == workfile.strip():
    raise ChangelogError("You have not updated the CHANGELOG file. Please "
                         f"add a summary of your additions to {changelog}.")
elif "CI_OPEN_MERGE_REQUESTS" not in os.environ:
    print("Not on merge request branch. Skipping CHANGELOG check.")
else:
    print("CHANGELOG is up to date.")
