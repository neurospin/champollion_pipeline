"""The environment check: every top-level pixi PyPI dependency is present in the active environment.

Task TASK-VENV-BDRABCZUK-88A90FDCD2F3 (REQ-VENV-*). The shared verify environment twice
lost p_tqdm and pathos on 2026-11-16 (a real ``pixi install`` run against a manifest
without ``[pypi-dependencies]`` uninstalls them) and every merge run then failed with
hundreds of ``ModuleNotFoundError`` collection errors that looked like code failures.

- the check derives its distributions from the top-level ``[pypi-dependencies]`` of
  ``pixi.toml`` (nothing hardcoded) and from the distributions they require;
- it locates modules (``find_spec``), it never imports them: ``dico_toolbox`` pulls
  in ``soma``;
- one message names every missing distribution and module and the fix;
- the session stops at that message before collection (``tests/conftest.py``);
- ``pixi run check-env`` runs it alone.
"""

import inspect
import sys
import time
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib

from tests import conftest as suite_conftest
from tests import pypi_dependency_check as check

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = """
[workspace]
name = "t"

[dependencies]
python = ">=3.10"

[pypi-dependencies]
p_tqdm = ">=1.4.2,<2"
point-cloud-pattern-mining = { git = "https://example.org/pcpm", rev = "abc" }

[feature.docs.pypi-dependencies]
sphinx = ">=7"
"""


class FakeDistribution:
    """Stand-in for ``importlib.metadata.Distribution``: a name and its Requires-Dist."""

    def __init__(self, name: str, requires: list[str] | None = None):
        self.metadata = {"Name": name}
        self.requires = requires


class FakeEnvironment:
    """An environment: installed distributions, the modules each provides, the locatable modules."""

    def __init__(self, dists: dict[str, list[str] | None], modules: dict[str, list[str]], locatable: set[str]):
        self.dists = dists
        self.modules = modules
        self.locatable = locatable

    def distribution(self, name: str):
        key = check.normalize(name)
        if key not in self.dists:
            raise metadata.PackageNotFoundError(name)
        return FakeDistribution(key, self.dists[key])

    def packages_distributions(self) -> dict[str, list[str]]:
        inverted: dict[str, list[str]] = {}
        for dist, mods in self.modules.items():
            for mod in mods:
                inverted.setdefault(mod, []).append(dist)
        return inverted

    def find_spec(self, module: str):
        return object() if module in self.locatable else None

    def run(self, manifest: Path) -> check.EnvironmentReport:
        return check.find_missing(
            manifest,
            distribution=self.distribution,
            packages_distributions=self.packages_distributions,
            find_spec=self.find_spec,
        )


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "pixi.toml"
    path.write_text(MANIFEST)
    return path


def healthy_environment() -> FakeEnvironment:
    return FakeEnvironment(
        dists={
            "p-tqdm": ["tqdm", "pathos", "six"],
            "pathos": ["dill"],
            "tqdm": None,
            "six": None,
            "dill": None,
            "point-cloud-pattern-mining": ["numpy", "sklearn", 'sphinx; extra == "doc"'],
            "numpy": None,
            "sklearn": None,
        },
        modules={
            "p-tqdm": ["p_tqdm"],
            "pathos": ["pathos"],
            "tqdm": ["tqdm"],
            "six": ["six"],
            "dill": ["dill"],
            "point-cloud-pattern-mining": ["pcpm"],
            "numpy": ["numpy"],
            "sklearn": ["wheel-platform-tag-is-broken-on-empty-wheels-see-issue-141"],
        },
        locatable={"p_tqdm", "pathos", "tqdm", "six", "dill", "pcpm", "numpy"},
    )


def test_declared_distributions_come_from_the_top_level_table_only(manifest: Path) -> None:
    assert check.declared_distributions(manifest) == ["p-tqdm", "point-cloud-pattern-mining"]


def test_declared_distributions_follow_a_new_dependency(manifest: Path) -> None:
    manifest.write_text(MANIFEST.replace("[feature.docs", 'extra_dep = "*"\n\n[feature.docs'))
    assert "extra-dep" in check.declared_distributions(manifest)


def test_healthy_environment_reports_nothing(manifest: Path) -> None:
    report = healthy_environment().run(manifest)
    assert not report.missing_distributions
    assert not report.missing_modules
    assert check.render(report, ROOT) is None


def test_every_missing_item_is_named_in_one_message(manifest: Path) -> None:
    env = healthy_environment()
    del env.dists["p-tqdm"]
    del env.dists["point-cloud-pattern-mining"]
    message = check.render(env.run(manifest), ROOT)
    assert message is not None
    assert "p-tqdm" in message
    assert "point-cloud-pattern-mining" in message
    assert "pixi install" in message
    assert str(ROOT) in message
    assert "\n" not in message.strip()


def test_installed_distribution_with_a_missing_module_names_the_module(manifest: Path) -> None:
    env = healthy_environment()
    env.locatable.discard("pcpm")
    report = env.run(manifest)
    assert report.missing_modules == {"point-cloud-pattern-mining": ["pcpm"]}
    message = check.render(report, ROOT)
    assert "pcpm" in message
    assert "point-cloud-pattern-mining" in message


def test_a_required_distribution_is_checked_too(manifest: Path) -> None:
    env = healthy_environment()
    env.locatable.discard("pathos")
    message = check.render(env.run(manifest), ROOT)
    assert "pathos" in message
    assert "required by p-tqdm" in message


def test_an_extra_requirement_is_not_checked(manifest: Path) -> None:
    env = healthy_environment()
    assert "sphinx" not in env.dists
    assert env.run(manifest).missing_distributions == {}


def test_an_empty_shim_with_no_valid_module_name_is_skipped(manifest: Path) -> None:
    report = healthy_environment().run(manifest)
    assert "sklearn" not in report.missing_distributions
    assert "sklearn" not in report.missing_modules


def test_the_active_environment_provides_the_declared_pypi_dependencies() -> None:
    """The live check: a merge run in an environment without p_tqdm stops here, in one line."""
    message = check.check_environment(ROOT / "pixi.toml")
    assert message is None, message


def test_live_check_runs_in_a_blink() -> None:
    start = time.perf_counter()
    check.check_environment(ROOT / "pixi.toml")
    assert time.perf_counter() - start < 2.0


def test_session_start_stops_with_the_message_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "check_environment", lambda *a, **k: "env lacks p_tqdm; run pixi install")
    session = SimpleNamespace(config=SimpleNamespace())
    with pytest.raises(pytest.exit.Exception, match="env lacks p_tqdm"):
        suite_conftest.pytest_sessionstart(session)


def test_session_start_passes_when_the_environment_is_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "check_environment", lambda *a, **k: None)
    assert suite_conftest.pytest_sessionstart(SimpleNamespace(config=SimpleNamespace())) is None


def test_session_start_is_left_to_the_controller_under_xdist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "check_environment", lambda *a, **k: "broken")
    worker = SimpleNamespace(config=SimpleNamespace(workerinput={}))
    assert suite_conftest.pytest_sessionstart(worker) is None


def test_pixi_defines_check_env_running_this_check() -> None:
    with (ROOT / "pixi.toml").open("rb") as handle:
        tasks = tomllib.load(handle)["tasks"]
    task = tasks["check-env"]
    command = task if isinstance(task, str) else task["cmd"]
    assert "pytest" in command
    assert "tests/test_pixi_pypi_dependencies_importable.py" in command


def test_default_module_lookup_ignores_sys_modules_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that stubs a module as missing (sys.modules[name] = None) does not make it look absent."""
    monkeypatch.setitem(sys.modules, "pytest", None)
    lookup = inspect.signature(check.find_missing).parameters["find_spec"].default
    assert lookup("pytest") is not None
