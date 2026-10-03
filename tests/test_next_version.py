import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "next_version.py"
_spec = importlib.util.spec_from_file_location("next_version", _SCRIPT)
nv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nv)

LEGACY_TAGS = ["v0.2.2", "v0.3.1", "v2026.05.00", "v2026.9.1", "v2026.10.0"]


def _pr(title="Fix something", body="", labels=(), commits=(), files=("pymif/x.py",)):
    return {
        "title": title,
        "body": body,
        "labels": [{"name": n} for n in labels],
        "commits": [{"messageHeadline": c, "messageBody": ""} for c in commits],
        "files": [{"path": p} for p in files],
    }


def test_no_tags_bumps_from_zero():
    assert nv.next_version([], _pr()) == "0.0.1"
    assert nv.next_version([], _pr(title="feat: first")) == "0.1.0"


@pytest.mark.parametrize(
    "pr, expected",
    [
        (_pr(), "0.3.2"),
        (_pr(title="feat: new reader"), "0.4.0"),
        (_pr(title="feat(cli): new flag"), "0.4.0"),
        (_pr(commits=["fix typo", "feat: add widget"]), "0.4.0"),
        (_pr(body="adds stuff [minor]"), "0.4.0"),
        (_pr(labels=["minor"]), "0.4.0"),
        (_pr(title="refactor!: drop old API"), "1.0.0"),
        (_pr(body="BREAKING CHANGE: removed X"), "1.0.0"),
        (_pr(labels=["major"]), "1.0.0"),
        (_pr(title="improve feature handling"), "0.3.2"),
    ],
)
def test_bumps_from_last_semver_tag_ignoring_calver(pr, expected):
    assert nv.next_version(LEGACY_TAGS, pr) == expected


@pytest.mark.parametrize(
    "pr",
    [
        _pr(body="[skip release]"),
        _pr(labels=["skip-release"]),
        _pr(files=["README.md", "doc/conf.py", ".github/workflows/tests.yml"]),
    ],
)
def test_skip(pr):
    assert nv.next_version(["v0.10.0"], pr) is None


def test_versions_compare_numerically():
    assert nv.latest_version(["v0.9.0", "v0.10.0", "v0.2.0"]) == (0, 10, 0)
