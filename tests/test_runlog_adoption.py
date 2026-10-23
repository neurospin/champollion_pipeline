#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adoption of the champollion_utils --log option (TASK-RUNLOG-BDRABCZUK-41F869DCB23D).

REQ-RUNLOG-BDRABCZUK-98E27E38EFEB: the champollion-utils direct reference in
pyproject.toml pins a git tag v<MAJOR>.<MINOR>.<PATCH> at version 0.3.0 or later,
the first champollion_utils release whose ScriptBuilder offers --log [PATH].
"""

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RELEASE_TAG = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")
_MINIMUM_UTILS = Version("0.3.0")


def _utils_dependency() -> Requirement:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(dep) for dep in pyproject["project"]["dependencies"]]
    utils = [req for req in requirements if canonicalize_name(req.name) == "champollion-utils"]
    assert len(utils) == 1, f"expected one champollion-utils dependency, got {utils!r}"
    return utils[0]


def test_pyproject_champollion_utils_pins_release_tag_0_3_0_or_later():
    """REQ-RUNLOG-BDRABCZUK-98E27E38EFEB."""
    url = _utils_dependency().url
    assert url is not None and url.startswith("git+"), f"{url!r} is not a PEP 508 git direct reference"
    path = url.split("://", 1)[-1]
    assert "@" in path, f"{url} does not pin a git ref"
    ref = path.rsplit("@", 1)[1]
    tag = _RELEASE_TAG.match(ref)
    assert tag is not None, f"{url} pins {ref!r}, not a v<MAJOR>.<MINOR>.<PATCH> tag"
    assert Version(tag["version"]) >= _MINIMUM_UTILS, f"{url} pins {ref}, below v{_MINIMUM_UTILS}"
