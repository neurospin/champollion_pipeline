"""The env SIF Dockerfile holds the pinned pixi environment and nothing else.

``.github/Dockerfile.base`` builds the environment-only image of the e2e runs:
the pixi environment, no pipeline code (``scripts/e2e/pack_code.sh`` ships the
code, bound read-only at ``/opt/code``). The file is parsed as text: no Docker
daemon, no network.

REQ-E2ECI-BDRABCZUK-5542601F0A53: build argument PIXI_VERSION, no default.
REQ-E2ECI-BDRABCZUK-A2AAB83ABA4A: pixi binary from the versioned release URL.
REQ-E2ECI-BDRABCZUK-0F9FAA3AF354: the E2E_PIXI_VERSION value is not restated.
REQ-E2ECI-BDRABCZUK-CD7BCE411E0A: sha256sum --check against PIXI_SHA256 before
the first pixi use (supersedes REQ-E2ECI-BDRABCZUK-28B46213E246).
REQ-E2ECI-BDRABCZUK-141B8A0A4482: no curl or wget output piped into a shell.
REQ-E2ECI-BDRABCZUK-3D0E09AF0964: one ``pixi install --locked`` of default.
REQ-E2ECI-BDRABCZUK-41E0A3EAA7DD: COPY/ADD sources are the allowlisted manifests.
REQ-E2ECI-BDRABCZUK-E18709434174: no git clone.
REQ-E2ECI-BDRABCZUK-00648284B7E7: no pixi.toml task run through pixi run.
REQ-E2ECI-BDRABCZUK-F4C36401C31D: no secret-named ARG or ENV variable.
REQ-E2ECI-BDRABCZUK-96E1C270EE3F: no USER instruction, useradd or adduser.
REQ-E2ECI-BDRABCZUK-550B95460286: pixi binary and project outside /root, /home.
REQ-E2ECI-BDRABCZUK-E02EB9495FED: PYTHONPATH points at the bound code dir.
REQ-E2ECI-BDRABCZUK-BC399D8731E9: smoke import of torch, soma.aims, p_tqdm.
REQ-E2ECI-BDRABCZUK-40946B55B021: e2e.env sets E2E_PIXI_SHA256 (64 lowercase hex).
REQ-E2ECI-BDRABCZUK-D694BB5DF78B: build argument PIXI_SHA256, no default.
REQ-E2ECI-BDRABCZUK-6EC70E58B4F4: the E2E_PIXI_SHA256 value is not restated.
REQ-E2ECI-BDRABCZUK-0BE1AE6D9856: FROM ubuntu:22.04@sha256:<64 hex>.
REQ-E2ECI-BDRABCZUK-24B2C912A1FC: no instruction names nsys or nsight (bound from
the host, user decision 2026-10-28).
REQ-E2ECI-BDRABCZUK-8EE270C3616D: the default pixi.lock environment lists no
nsys or nsight package.
REQ-E2ECI-BDRABCZUK-17D933EFE643: ENV CONDA_OVERRIDE_CUDA=12.0 before pixi install
(the build sees no GPU; user decision 2026-10-28).
REQ-E2ECI-BDRABCZUK-823136777B0E: the torch import RUN asserts torch.version.cuda.
REQ-E2ECI-BDRABCZUK-61155DA65EDB: apt-get install packages pinned as name=version.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import pytest
import tomllib
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = PIPELINE_ROOT / ".github" / "Dockerfile.base"
E2E_ENV = PIPELINE_ROOT / "scripts" / "e2e" / "e2e.env"
PIXI_TOML = PIPELINE_ROOT / "pixi.toml"
PIXI_LOCK = PIPELINE_ROOT / "pixi.lock"

ALLOWED_COPY_SOURCES = {
    "pixi.toml",
    "pixi.lock",
    "external/champollion_V1/setup.py",
    "external/cortical_tiles/setup.py",
    "external/cortical_tiles/cortical_tiles/info.py",
}
EXPECTED_PYTHONPATH = (
    "/opt/code/src:/opt/code/external/champollion_V1:/opt/code/external/cortical_tiles:/opt/code/champollion_utils/src"
)
SMOKE_MODULES = ("torch", "soma.aims", "p_tqdm")
RELEASE_URL = re.compile(
    r"https://github\.com/prefix-dev/pixi/releases/download/v\$(?:\{PIXI_VERSION\}|PIXI_VERSION\b)/"
)
# A command word that runs the pixi binary: `pixi <sub>` or `<path>/pixi <sub>`.
PIXI_CALL = re.compile(r"(?:^|[\s;&|(`])(?:[\w${}./-]*/)?pixi\s+(?:-\S+\s+)*(?!-)[\w-]+")
PIXI_INSTALL = re.compile(r"(?:^|[\s;&|(])(?:[\w${}./-]*/)?pixi\s+install\b([^;&|\n]*)")
HOME_PLACES = re.compile(r"^(?:/root|/home)(?:/|$)|^\$\{?HOME\b|^~")
NSIGHT = re.compile(r"nsys|nsight", re.IGNORECASE)

pytestmark = pytest.mark.unit  # noqa: V107


@dataclass(frozen=True)
class Instruction:
    """One Dockerfile instruction, line continuations joined, comments dropped."""

    keyword: str
    args: str


def _instructions(text: str) -> list[Instruction]:
    """Split Dockerfile text into instructions (backslash continuations joined)."""
    result: list[Instruction] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not buffer and (not line or line.startswith("#")):
            continue
        if buffer and line.startswith("#"):
            continue
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        buffer += line
        keyword, _, args = buffer.strip().partition(" ")
        result.append(Instruction(keyword.upper(), args.strip()))
        buffer = ""
    if buffer.strip():
        keyword, _, args = buffer.strip().partition(" ")
        result.append(Instruction(keyword.upper(), args.strip()))
    return result


@pytest.fixture(scope="module")
def instructions() -> list[Instruction]:
    """Parsed instructions of .github/Dockerfile.base."""
    return _instructions(DOCKERFILE.read_text())


def _runs(instructions: list[Instruction]) -> list[tuple[int, str]]:
    """(index, command text) of every RUN instruction."""
    return [(i, ins.args) for i, ins in enumerate(instructions) if ins.keyword == "RUN"]


def _words(args: str) -> list[str]:
    """Arguments of a COPY/ADD/ARG/ENV instruction, exec (JSON) form included."""
    if args.startswith("["):
        return [str(word) for word in json.loads(args)]
    return shlex.split(args)


def _env_pairs(args: str) -> list[tuple[str, str]]:
    """(name, value) pairs of an ENV instruction, both `A=b C=d` and `A b` forms."""
    words = shlex.split(args)
    if words and "=" not in words[0]:
        return [(words[0], " ".join(words[1:]))]
    return [tuple(word.split("=", 1)) for word in words]  # type: ignore[misc]


def _pixi_install_index(instructions: list[Instruction]) -> int:
    """Index of the RUN instruction holding `pixi install`."""
    found = [i for i, cmd in _runs(instructions) if PIXI_INSTALL.search(cmd)]
    assert found, "no RUN instruction runs pixi install"
    return found[0]


def _pixi_tasks() -> set[str]:
    """Every task name pixi.toml defines, at top level, per feature or per target."""
    data = tomllib.loads(PIXI_TOML.read_text())
    names = set(data.get("tasks", {}))
    for feature in data.get("feature", {}).values():
        names |= set(feature.get("tasks", {}))
        for target in feature.get("target", {}).values():
            names |= set(target.get("tasks", {}))
    for target in data.get("target", {}).values():
        names |= set(target.get("tasks", {}))
    return names


def _e2e_setting(name: str) -> str | None:
    """Value of one plain KEY=value line of scripts/e2e/e2e.env, quotes stripped."""
    match = re.search(rf'^{name}="?([^"\s]+)"?\s*$', E2E_ENV.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def _assert_arg_without_default(instructions: list[Instruction], name: str) -> None:
    """ARG `name` is declared, and never with a default value."""
    declared = [word for ins in instructions if ins.keyword == "ARG" for word in _words(ins.args)]
    matching = [arg for arg in declared if arg.split("=", 1)[0] == name]
    assert matching, f"no ARG {name} declared"
    assert all("=" not in arg for arg in matching), f"{name} has a default: {matching}"


def _assert_setting_not_restated(name: str) -> None:
    """The e2e.env value of `name` does not occur in Dockerfile.base."""
    value = _e2e_setting(name)
    assert value, f"{name} missing from scripts/e2e/e2e.env"
    assert value not in DOCKERFILE.read_text(), f"Dockerfile.base restates the {name} value {value}"


def test_pixi_version_arg_has_no_default(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-5542601F0A53."""
    _assert_arg_without_default(instructions, "PIXI_VERSION")


def test_pixi_sha256_arg_has_no_default(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-D694BB5DF78B."""
    _assert_arg_without_default(instructions, "PIXI_SHA256")


def test_e2e_env_sets_pixi_sha256() -> None:
    """REQ-E2ECI-BDRABCZUK-40946B55B021 (the form only: the checksum itself needs the release file)."""
    value = _e2e_setting("E2E_PIXI_SHA256")
    assert value is not None, "E2E_PIXI_SHA256 missing from scripts/e2e/e2e.env"
    assert re.fullmatch(r"[0-9a-f]{64}", value), f"E2E_PIXI_SHA256 is not 64 lowercase hex: {value!r}"


def test_pixi_downloaded_from_versioned_release_url(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-A2AAB83ABA4A."""
    downloads = [cmd for _, cmd in _runs(instructions) if re.search(r"\b(curl|wget)\b", cmd)]
    assert any(RELEASE_URL.search(cmd) for cmd in downloads), (
        "no curl/wget download from https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/"
    )


def test_pixi_version_value_not_restated() -> None:
    """REQ-E2ECI-BDRABCZUK-0F9FAA3AF354."""
    _assert_setting_not_restated("E2E_PIXI_VERSION")


def test_pixi_sha256_value_not_restated() -> None:
    """REQ-E2ECI-BDRABCZUK-6EC70E58B4F4."""
    _assert_setting_not_restated("E2E_PIXI_SHA256")


def test_pixi_checksum_checked_before_first_use(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-CD7BCE411E0A."""
    check = re.compile(r"\bsha256sum\b[^;&|\n]*(?:\s-c\b|\s--check\b)")
    checks = [(i, m.start()) for i, cmd in _runs(instructions) for m in [check.search(cmd)] if m]
    uses = [(i, m.start()) for i, cmd in _runs(instructions) for m in [PIXI_CALL.search(cmd)] if m]
    assert checks, "no RUN instruction runs sha256sum --check"
    assert uses, "no RUN instruction runs the pixi binary"
    assert min(checks) < min(uses), "the pixi binary runs before its sha256sum --check"
    first_check = next(i for i, cmd in _runs(instructions) if check.search(cmd))
    assert "pixi" in instructions[first_check].args, "the sha256sum --check does not concern the pixi binary"
    assert re.search(r"\$\{?PIXI_SHA256\b", instructions[first_check].args), (
        "the sha256sum --check does not use the PIXI_SHA256 build argument"
    )


def test_no_download_piped_into_shell(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-141B8A0A4482."""
    piped = re.compile(r"\b(?:curl|wget)\b[^;&|]*\|\s*(?:sudo\s+)?(?:\S*/)?(?:ba|da|z|k)?sh\b")
    offenders = [cmd for _, cmd in _runs(instructions) if piped.search(cmd)]
    assert not offenders, f"download piped into a shell: {offenders}"


def test_single_locked_default_pixi_install(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-3D0E09AF0964."""
    installs = [m.group(1).split() for _, cmd in _runs(instructions) for m in PIXI_INSTALL.finditer(cmd)]
    assert len(installs) == 1, f"pixi install runs {len(installs)} times, expected exactly once"
    options = installs[0]
    assert "--locked" in options, f"pixi install without --locked: {options}"
    assert "--all" not in options and "-a" not in options, "pixi install of every environment"
    envs = [options[k + 1] for k, opt in enumerate(options[:-1]) if opt in ("-e", "--environment")]
    envs += [opt.split("=", 1)[1] for opt in options if opt.startswith("--environment=")]
    assert all(env == "default" for env in envs), f"pixi install of a non-default environment: {envs}"


def test_copy_sources_are_allowlisted_manifests(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-41E0A3EAA7DD."""
    sources: list[str] = []
    for ins in instructions:
        if ins.keyword not in ("COPY", "ADD"):
            continue
        words = [word for word in _words(ins.args) if not word.startswith("--")]
        if any(word.startswith("--from") for word in _words(ins.args)):
            continue
        sources += [re.sub(r"^\./", "", word) for word in words[:-1]]
    outside = sorted(set(sources) - ALLOWED_COPY_SOURCES)
    assert not outside, f"build-context sources outside the allowlist: {outside}"


def test_no_git_clone(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-E18709434174."""
    clone = re.compile(r"\bgit\b[^;&|\n]*?\bclone\b")
    offenders = [cmd for _, cmd in _runs(instructions) if clone.search(cmd)]
    assert not offenders, f"git clone in the env image build: {offenders}"


def test_no_pixi_task_run(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-00648284B7E7."""
    tasks = _pixi_tasks()
    assert tasks, "pixi.toml defines no task (parser broken?)"
    with_value = {"-e", "--environment", "--manifest-path", "--executable", "-x"}
    ran: list[str] = []
    for _, cmd in _runs(instructions):
        for match in re.finditer(r"\bpixi\s+run\s+(.*)", cmd):
            words = match.group(1).split()
            k = 0
            while k < len(words) and words[k].startswith("-"):
                k += 2 if words[k] in with_value else 1
            if k < len(words):
                ran.append(words[k])
    offenders = sorted(set(ran) & tasks)
    assert not offenders, f"pixi.toml tasks run at build time: {offenders}"


def test_no_secret_named_variable(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-F4C36401C31D."""
    names: list[str] = []
    for ins in instructions:
        if ins.keyword == "ARG":
            names += [word.split("=", 1)[0] for word in _words(ins.args)]
        elif ins.keyword == "ENV":
            names += [name for name, _ in _env_pairs(ins.args)]
    secret = re.compile(r"TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
    offenders = [name for name in names if secret.search(name)]
    assert not offenders, f"secret-named build variables: {offenders}"


def test_no_user_account(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-96E1C270EE3F."""
    users = [ins.args for ins in instructions if ins.keyword == "USER"]
    assert not users, f"USER instruction present: {users}"
    account = re.compile(r"\b(?:useradd|adduser)\b")
    offenders = [cmd for _, cmd in _runs(instructions) if account.search(cmd)]
    assert not offenders, f"user account created: {offenders}"


def test_pixi_outside_home_directories(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-550B95460286.

    Placement paths are the WORKDIR values, COPY/ADD destinations, PATH and
    PIXI_HOME values, and in RUN commands the targets of cd, mv, cp, the curl
    or wget output option and --manifest-path. A pixi cache cleanup under
    /root is not a placement and stays allowed.
    """
    places: list[str] = []
    for ins in instructions:
        if ins.keyword == "WORKDIR":
            places.append(ins.args)
        elif ins.keyword in ("COPY", "ADD"):
            places.append(_words(ins.args)[-1])
        elif ins.keyword == "ENV":
            for name, value in _env_pairs(ins.args):
                if name in ("PATH", "PIXI_HOME"):
                    places += value.split(":")
        elif ins.keyword == "RUN":
            cmd = ins.args
            places += re.findall(r"(?:^|[\s;&|(])cd\s+(\S+)", cmd)
            places += re.findall(r"(?:\s-o|\s-O|--output(?:-document)?)[=\s]+(\S+)", cmd)
            places += re.findall(r"--manifest-path[=\s]+(\S+)", cmd)
            places += re.findall(r"\b(?:mv|cp)\s+(?:-\S+\s+)*\S+\s+(\S+)", cmd)
    offenders = [place for place in places if HOME_PLACES.search(place.strip("\"'"))]
    assert not offenders, f"pixi placed under /root, /home or $HOME: {offenders}"


def test_pythonpath_points_at_code_dir(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-E02EB9495FED."""
    pairs = [pair for ins in instructions if ins.keyword == "ENV" for pair in _env_pairs(ins.args)]
    pythonpaths = [value for name, value in pairs if name == "PYTHONPATH"]
    assert pythonpaths, "no ENV PYTHONPATH"
    assert pythonpaths[-1] == EXPECTED_PYTHONPATH, f"PYTHONPATH is {pythonpaths[-1]!r}"


def _imported_modules(cmd: str) -> set[str]:
    """Dotted module names imported by the Python code of one command."""
    modules: set[str] = set()
    for match in re.finditer(r"\bimport\s+([\w.]+(?:\s*,\s*[\w.]+)*)", cmd):
        modules |= {name.strip() for name in match.group(1).split(",")}
    modules |= set(re.findall(r"\bfrom\s+([\w.]+)\s+import\b", cmd))
    return modules


def test_smoke_import_after_install(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-BC399D8731E9."""
    install = _pixi_install_index(instructions)
    in_default = re.compile(
        r"\bpixi\s+run\s+(?:(?:-e|--environment)[=\s]+default\s+)?(?:--\s+)?python3?\b"
        r"|\.pixi/envs/default/bin/python3?\b"
    )
    smoke = [
        _imported_modules(cmd)
        for i, cmd in _runs(instructions)
        if i > install and in_default.search(cmd) and not re.search(r"(?:-e|--environment)[=\s]+(?!default)\w", cmd)
    ]
    assert any(set(SMOKE_MODULES) <= modules for modules in smoke), (
        f"no RUN after pixi install imports {SMOKE_MODULES} in the default environment; found {smoke}"
    )


def test_base_image_pinned_by_digest(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-0BE1AE6D9856."""
    images = [
        next(word for word in ins.args.split() if not word.startswith("--"))
        for ins in instructions
        if ins.keyword == "FROM"
    ]
    assert images, "no FROM instruction"
    unpinned = [image for image in images if not re.fullmatch(r"ubuntu:22\.04@sha256:[0-9a-f]{64}", image)]
    assert not unpinned, f"FROM not pinned as ubuntu:22.04@sha256:<64 hex>: {unpinned}"


def test_no_nsight_in_dockerfile(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-24B2C912A1FC."""
    offenders = [f"{ins.keyword} {ins.args}" for ins in instructions if NSIGHT.search(f"{ins.keyword} {ins.args}")]
    assert not offenders, f"nsys/nsight named in Dockerfile.base (bound from the host instead): {offenders}"


def test_no_nsight_in_default_pixi_env() -> None:
    """REQ-E2ECI-BDRABCZUK-8EE270C3616D."""
    lock = yaml.safe_load(PIXI_LOCK.read_text())
    platforms = lock["environments"]["default"]["packages"]
    urls = [url for packages in platforms.values() for entry in packages for url in entry.values()]
    assert urls, "the default environment of pixi.lock lists no package (parser broken?)"
    names = [url.rstrip("/").rsplit("/", 1)[-1] for url in urls]
    offenders = sorted({name for name in names if NSIGHT.search(name)})
    assert not offenders, f"nsys/nsight packages in the default pixi environment: {offenders}"


def test_cuda_override_env_before_pixi_install(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-17D933EFE643."""
    install = _pixi_install_index(instructions)
    overrides = [
        (i, value.strip("\"'"))
        for i, ins in enumerate(instructions)
        if ins.keyword == "ENV"
        for name, value in _env_pairs(ins.args)
        if name == "CONDA_OVERRIDE_CUDA"
    ]
    before = [value for i, value in overrides if i < install]
    assert before, "no ENV CONDA_OVERRIDE_CUDA before the pixi install RUN"
    assert before[-1] == "12.0", f"CONDA_OVERRIDE_CUDA is {before[-1]!r} at pixi install, expected '12.0'"


def test_smoke_import_asserts_cuda_torch(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-823136777B0E.

    Accepted forms: ``assert torch.version.cuda`` (truthy, stricter) and
    ``assert torch.version.cuda is not None``.
    """
    asserts_cuda = re.compile(r"\bassert\s+torch\.version\.cuda(?:\s+is\s+not\s+None)?\s*(?:[,;\"')]|$)")
    torch_runs = [cmd for _, cmd in _runs(instructions) if "torch" in _imported_modules(cmd)]
    assert torch_runs, "no RUN instruction imports torch"
    missing = [cmd for cmd in torch_runs if not asserts_cuda.search(cmd)]
    assert not missing, f"torch import RUN without an assert on torch.version.cuda: {missing}"


def _apt_install_packages(cmd: str) -> list[str]:
    """Package words of every ``apt-get install`` in one RUN command (options dropped)."""
    packages: list[str] = []
    for match in re.finditer(r"\bapt-get\s+(?:-\S+\s+)*install\b([^;&|]*)", cmd):
        packages += [word for word in match.group(1).split() if not word.startswith("-")]
    return packages


def test_apt_packages_pinned_to_exact_versions(instructions: list[Instruction]) -> None:
    """REQ-E2ECI-BDRABCZUK-61155DA65EDB.

    A pin is ``name=version`` with a literal Debian version (no ``*`` wildcard,
    no ``$`` variable: the version must be readable in the file).
    """
    pinned = re.compile(r"[a-z0-9][a-z0-9+.-]*=[A-Za-z0-9][A-Za-z0-9.+~:-]*")
    packages = [pkg for _, cmd in _runs(instructions) for pkg in _apt_install_packages(cmd)]
    unpinned = [pkg for pkg in packages if not pinned.fullmatch(pkg)]
    assert not unpinned, f"apt packages without an exact name=version pin: {unpinned}"
