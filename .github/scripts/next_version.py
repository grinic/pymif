"""Compute the next pymif release version for a merged pull request.

Usage: python next_version.py <pr.json>

``pr.json`` is the output of
``gh pr view <n> --json title,body,labels,commits,files``. Existing tags are
read from the local git repository. Prints the next version (``X.Y.Z``,
without the leading ``v``), or nothing when the PR should not be released.

Bump rules (checked on PR title, body, labels and commit messages):

- major: ``BREAKING CHANGE``, ``[major]``, a ``type!:`` prefix or label ``major``
- minor: a ``feat:`` / ``feat(scope):`` prefix, ``[minor]`` or label ``minor``
- patch: anything else
- none:  ``[skip release]``, label ``skip-release``, or only docs/CI files changed
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

# Tags with a major at or above this are legacy calendar versions and are ignored.
CALVER_MAJOR = 2000

TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
MAJOR_RE = re.compile(r"BREAKING CHANGE|\[major\]|^\w+(\([^)]*\))?!:", re.MULTILINE)
MINOR_RE = re.compile(r"\[minor\]|^feat(\([^)]*\))?:", re.MULTILINE)
SKIP_RE = re.compile(r"\[skip release\]")
NON_RELEASE_PATHS = ("doc/", "documentation/", ".github/")


def latest_version(tags: list[str]) -> tuple[int, int, int] | None:
    versions = []
    for tag in tags:
        m = TAG_RE.match(tag.strip())
        if m:
            v = tuple(int(x) for x in m.groups())
            if v[0] < CALVER_MAJOR:
                versions.append(v)
    return max(versions) if versions else None


def bump_kind(pr: dict) -> str | None:
    labels = {label["name"].lower() for label in pr.get("labels") or []}
    texts = [pr.get("title") or "", pr.get("body") or ""]
    for commit in pr.get("commits") or []:
        texts.append(commit.get("messageHeadline") or "")
        texts.append(commit.get("messageBody") or "")
    text = "\n".join(texts)

    if "skip-release" in labels or SKIP_RE.search(text):
        return None
    files = [f["path"] for f in pr.get("files") or []]
    if files and all(p.startswith(NON_RELEASE_PATHS) or p.endswith(".md") for p in files):
        return None
    if "major" in labels or MAJOR_RE.search(text):
        return "major"
    if "minor" in labels or MINOR_RE.search(text):
        return "minor"
    return "patch"


def next_version(tags: list[str], pr: dict) -> str | None:
    kind = bump_kind(pr)
    if kind is None:
        return None
    major, minor, patch = latest_version(tags) or (0, 0, 0)
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        pr = json.load(fh)
    tags = subprocess.run(
        ["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    print(next_version(tags, pr) or "")


if __name__ == "__main__":
    main()
