import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "next_version.py"
_spec = importlib.util.spec_from_file_location("next_version", _SCRIPT)
nv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nv)

LEGACY_TAGS = ["v0.2.2", "v0.3.1", "v2026.05.00", "v2026.9.1", "v2026.10.0"]


def _pr(*labels):
    return {"labels": [{"name": n} for n in labels]}


def test_no_tags_bumps_from_zero():
    assert nv.next_version([], _pr("release:patch")) == "0.0.1"
    assert nv.next_version([], _pr("release:minor")) == "0.1.0"


@pytest.mark.parametrize(
    "pr, expected",
    [
        (_pr("release:patch"), "0.3.2"),
        (_pr("release:minor"), "0.4.0"),
        (_pr("release:major"), "1.0.0"),
        (_pr("release:patch", "release:minor"), "0.4.0"),
        (_pr("Release:Major", "bug"), "1.0.0"),
    ],
)
def test_bumps_from_last_semver_tag_ignoring_calver(pr, expected):
    assert nv.next_version(LEGACY_TAGS, pr) == expected


@pytest.mark.parametrize("pr", [_pr(), _pr("bug", "documentation"), _pr("release")])
def test_no_release_without_label(pr):
    assert nv.next_version(["v0.10.0"], pr) is None


def test_versions_compare_numerically():
    assert nv.latest_version(["v0.9.0", "v0.10.0", "v0.2.0"]) == (0, 10, 0)
