#!/usr/bin/env python3
"""
The pipeline_commit of provenance.json is the true commit of the code that ran, or null
(TASK-PROVENANCE-BDRABCZUK-C5222940695A, DEF-PROVENANCE-BDRABCZUK-C0FC651FAF1D).

The code directory is the directory holding src/champollion_pipeline:

- it is the root of a git repository whose HEAD git reports: that HEAD, whatever a COMMIT file or GIT_DIR says;
- else its COMMIT file, the one scripts/e2e/pack_code.sh writes, when that file holds one commit;
- else null, even inside an enclosing repository.

No mock of git: each test builds real repositories under tmp_path, copies the source file defining
read_pipeline_commit into the layout under test and loads it from there, so the reader finds itself
where a packed or checked out pipeline would. The source file is looked up from the function, which
keeps these tests valid when read_pipeline_commit moves out of generate_embeddings.py, as long as
generate_embeddings still exposes that name. Loaded copies do not count towards coverage.
"""

import importlib.util
import inspect
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from champollion_pipeline import generate_embeddings as ge
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
from tests.test_generate_embeddings_provenance import (
    LOCAL_MODELS,
    fake_evaluate,
    lay_out_regions,
    local_argv,
    read_provenance,
)

pytestmark = pytest.mark.unit  # noqa: V107

PACKED_COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER_COMMIT = "89abcdef0123456789abcdef0123456789abcdef"
ONE_COMMIT = re.compile(rb"[0-9a-f]{40}")
SOURCE_ROOT = Path(ge.__file__).resolve().parents[1]
READER_SOURCE = Path(inspect.getsourcefile(ge.read_pipeline_commit)).resolve()
SITE_PACKAGES = Path(".pixi/envs/default/lib/python3.12/site-packages")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def git(repo: Path, *args: str) -> str:
    """Run git in repo, away from any GIT_* of the calling process, and return its stripped stdout."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.org", "-c", "commit.gpgsign=false"]
    result = subprocess.run(
        ["git", "-C", str(repo), *identity, *args], check=True, capture_output=True, text=True, env=env
    )
    return result.stdout.strip()


def make_repo(path: Path) -> str:
    """Create a git repository at path with one commit, unique to path, and return that commit."""
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "--template=")
    git(path, "commit", "-q", "--allow-empty", "-m", f"commit of {path}")
    return git(path, "rev-parse", "HEAD")


def load_reader(package_parent: Path) -> Callable[[], str | None]:
    """Copy the file defining read_pipeline_commit under package_parent, load the copy, return its reader."""
    copy = package_parent / READER_SOURCE.relative_to(SOURCE_ROOT)
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(READER_SOURCE, copy)
    spec = importlib.util.spec_from_file_location(f"copied_reader_{uuid.uuid4().hex}", copy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_pipeline_commit


def read_commit_from(code_dir: Path) -> str | None:
    """What the pipeline reports when its source sits at <code_dir>/src/champollion_pipeline."""
    return load_reader(code_dir / "src")()


def write_commit_file(code_dir: Path, content: bytes) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "COMMIT").write_bytes(content)


def run_stage_from(code_dir: Path, workdir: Path) -> dict:
    """Run the embeddings stage with the reader of code_dir and return the provenance.json it wrote."""
    lay_out_regions(workdir / LOCAL_MODELS)
    script = GenerateEmbeddings()
    script.parse_args(local_argv(workdir))
    script.execute_command = fake_evaluate
    with patch.object(ge, "read_pipeline_commit", load_reader(code_dir / "src")):
        script.run()
    return read_provenance(workdir / "out")


@pytest.fixture(scope="module")
def shared_enclosing(tmp_path_factory) -> tuple[Path, str]:
    """One enclosing repository and its HEAD, for the property tests: their examples only add directories."""
    path = tmp_path_factory.mktemp("enclosing")
    return path, make_repo(path)


def nested_code_dir(enclosing: Path, depth: int) -> Path:
    """A new directory depth levels below enclosing."""
    return enclosing.joinpath(uuid.uuid4().hex, *["level"] * (depth - 1))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """cwd = tmp_path/work, so a Hugging Face repo id is never a local directory; keep SSL_CERT_FILE untouched."""
    path = tmp_path / "work"
    path.mkdir()
    monkeypatch.chdir(path)
    monkeypatch.setenv("SSL_CERT_FILE", str(path / "no-ca.crt"))
    return path


# --------------------------------------------------------------------------- #
# REQ-PROVENANCE-BDRABCZUK-3E7171B7F5DB: null without an own repository or a valid COMMIT file
# --------------------------------------------------------------------------- #


class TestNullWithoutOwnRepository:
    """REQ-PROVENANCE-BDRABCZUK-3E7171B7F5DB."""

    def test_null_in_a_directory_of_an_enclosing_repository(self, tmp_path):
        make_repo(tmp_path)
        assert read_commit_from(tmp_path / "packed") is None

    def test_null_for_an_installed_copy_inside_a_repository(self, tmp_path):
        make_repo(tmp_path)
        assert load_reader(tmp_path / SITE_PACKAGES)() is None

    def test_null_for_a_package_outside_a_src_folder(self, tmp_path):
        make_repo(tmp_path)
        assert load_reader(tmp_path / "vendor")() is None

    def test_null_when_a_broken_dot_git_lets_git_climb(self, tmp_path):
        make_repo(tmp_path)
        (tmp_path / "packed" / ".git").mkdir(parents=True)
        assert read_commit_from(tmp_path / "packed") is None

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(b"", id="empty"),
            pytest.param(b"\n", id="blank"),
            pytest.param(PACKED_COMMIT[:12].encode() + b"\n", id="short"),
            pytest.param(PACKED_COMMIT.upper().encode() + b"\n", id="uppercase"),
            pytest.param(b"g" * 40 + b"\n", id="not_hexadecimal"),
            pytest.param(f"{PACKED_COMMIT}\n{OTHER_COMMIT}\n".encode(), id="two_commits"),
            pytest.param(f"{PACKED_COMMIT} packed\n".encode(), id="trailing_text"),
            pytest.param(b"\xff\xfe\x00", id="not_text"),
        ],
    )
    def test_null_when_the_commit_file_holds_no_single_commit(self, tmp_path, content):
        make_repo(tmp_path)
        write_commit_file(tmp_path / "packed", content)
        assert read_commit_from(tmp_path / "packed") is None

    def test_null_when_commit_is_a_directory(self, tmp_path):
        make_repo(tmp_path)
        (tmp_path / "packed" / "COMMIT").mkdir(parents=True)
        assert read_commit_from(tmp_path / "packed") is None

    @settings(max_examples=25, deadline=None)
    @given(
        depth=st.integers(min_value=1, max_value=4),
        content=st.none() | st.binary(max_size=64).filter(lambda data: not ONE_COMMIT.search(data)),
    )
    def test_null_at_any_depth_of_an_enclosing_repository(self, shared_enclosing, depth, content):
        """Invariant candidate INV-021343AAFCD9: no commit is read from a repository rooted elsewhere."""
        enclosing, _head = shared_enclosing
        code_dir = nested_code_dir(enclosing, depth)
        if content is not None:
            write_commit_file(code_dir, content)
        assert read_commit_from(code_dir) is None


# --------------------------------------------------------------------------- #
# REQ-PROVENANCE-BDRABCZUK-F145083B698D: the COMMIT file of a packed directory
# --------------------------------------------------------------------------- #


class TestCommitFileOfPackedDirectory:
    """REQ-PROVENANCE-BDRABCZUK-F145083B698D."""

    def test_commit_file_outside_any_repository(self, tmp_path):
        write_commit_file(tmp_path / "packed", f"{PACKED_COMMIT}\n".encode())
        assert read_commit_from(tmp_path / "packed") == PACKED_COMMIT

    def test_commit_file_without_a_final_newline(self, tmp_path):
        write_commit_file(tmp_path / "packed", PACKED_COMMIT.encode())
        assert read_commit_from(tmp_path / "packed") == PACKED_COMMIT

    def test_commit_file_rather_than_the_enclosing_head(self, tmp_path):
        make_repo(tmp_path)
        write_commit_file(tmp_path / "packed", f"{PACKED_COMMIT}\n".encode())
        assert read_commit_from(tmp_path / "packed") == PACKED_COMMIT

    def test_commit_file_rather_than_a_broken_dot_git(self, tmp_path):
        make_repo(tmp_path)
        (tmp_path / "packed" / ".git").mkdir(parents=True)
        write_commit_file(tmp_path / "packed", f"{PACKED_COMMIT}\n".encode())
        assert read_commit_from(tmp_path / "packed") == PACKED_COMMIT

    @settings(max_examples=25, deadline=None)
    @given(
        depth=st.integers(min_value=1, max_value=4),
        commit=st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
    )
    def test_commit_file_at_any_depth_of_an_enclosing_repository(self, shared_enclosing, depth, commit):
        """Invariant candidate INV-021343AAFCD9: no commit is read from a repository rooted elsewhere."""
        enclosing, _head = shared_enclosing
        code_dir = nested_code_dir(enclosing, depth)
        write_commit_file(code_dir, f"{commit}\n".encode())
        assert read_commit_from(code_dir) == commit


# --------------------------------------------------------------------------- #
# REQ-PROVENANCE-BDRABCZUK-B170814D2F43: the HEAD of the repository rooted at the code directory
# --------------------------------------------------------------------------- #


class TestCommitOfOwnRepository:
    """REQ-PROVENANCE-BDRABCZUK-B170814D2F43."""

    def test_own_head_rather_than_the_enclosing_head(self, tmp_path):
        make_repo(tmp_path)
        own_head = make_repo(tmp_path / "checkout")
        assert read_commit_from(tmp_path / "checkout") == own_head

    def test_head_of_a_linked_worktree(self, tmp_path):
        main_head = make_repo(tmp_path / "checkout")
        git(tmp_path / "checkout", "worktree", "add", "-q", "-b", "task", str(tmp_path / "linked"))
        git(tmp_path / "linked", "commit", "-q", "--allow-empty", "-m", "commit of the linked worktree")
        linked_head = git(tmp_path / "linked", "rev-parse", "HEAD")
        assert linked_head != main_head
        assert read_commit_from(tmp_path / "linked") == linked_head

    def test_head_through_a_symbolic_link(self, tmp_path):
        own_head = make_repo(tmp_path / "checkout")
        (tmp_path / "link").symlink_to(tmp_path / "checkout", target_is_directory=True)
        assert read_commit_from(tmp_path / "link") == own_head

    def test_head_rather_than_a_stray_commit_file(self, tmp_path):
        own_head = make_repo(tmp_path / "checkout")
        write_commit_file(tmp_path / "checkout", f"{PACKED_COMMIT}\n".encode())
        assert read_commit_from(tmp_path / "checkout") == own_head

    def test_head_when_git_dir_names_another_repository(self, tmp_path, monkeypatch):
        own_head = make_repo(tmp_path / "checkout")
        make_repo(tmp_path / "other")
        monkeypatch.setenv("GIT_DIR", str(tmp_path / "other" / ".git"))
        assert read_commit_from(tmp_path / "checkout") == own_head


# --------------------------------------------------------------------------- #
# REQ-PROVENANCE-BDRABCZUK-1026F84A6D89: git reports no HEAD of the own repository
# --------------------------------------------------------------------------- #


class TestCommitWhenGitReportsNoHead:
    """REQ-PROVENANCE-BDRABCZUK-1026F84A6D89: one chain, HEAD then COMMIT then null."""

    @staticmethod
    def hide_git(tmp_path: Path, monkeypatch) -> None:
        """Leave PATH on an empty directory, so that no git can be started."""
        (tmp_path / "no-git").mkdir()
        monkeypatch.setenv("PATH", str(tmp_path / "no-git"))

    @staticmethod
    def make_unborn_repo(path: Path) -> None:
        """Create a git repository at path with no commit, so that its HEAD resolves to nothing."""
        path.mkdir(parents=True)
        git(path, "init", "-q", "--template=")

    def test_commit_file_when_git_is_missing(self, tmp_path, monkeypatch):
        make_repo(tmp_path / "checkout")
        write_commit_file(tmp_path / "checkout", f"{PACKED_COMMIT}\n".encode())
        self.hide_git(tmp_path, monkeypatch)
        assert read_commit_from(tmp_path / "checkout") == PACKED_COMMIT

    def test_null_when_git_is_missing_without_a_commit_file(self, tmp_path, monkeypatch):
        make_repo(tmp_path / "checkout")
        self.hide_git(tmp_path, monkeypatch)
        assert read_commit_from(tmp_path / "checkout") is None

    def test_commit_file_when_head_is_unborn(self, tmp_path):
        make_repo(tmp_path)
        self.make_unborn_repo(tmp_path / "checkout")
        write_commit_file(tmp_path / "checkout", f"{PACKED_COMMIT}\n".encode())
        assert read_commit_from(tmp_path / "checkout") == PACKED_COMMIT

    def test_null_when_head_is_unborn_without_a_commit_file(self, tmp_path):
        make_repo(tmp_path)
        self.make_unborn_repo(tmp_path / "checkout")
        assert read_commit_from(tmp_path / "checkout") is None


# --------------------------------------------------------------------------- #
# The Defect as the IXI run met it: the stage writes provenance.json
# --------------------------------------------------------------------------- #


class TestProvenanceFile:
    """DEF-PROVENANCE-BDRABCZUK-C0FC651FAF1D, through the file the embeddings stage writes."""

    def test_stage_writes_the_packed_commit_not_the_enclosing_head(self, tmp_path, workdir):
        enclosing_head = make_repo(tmp_path / "enclosing")
        write_commit_file(tmp_path / "enclosing" / "packed", f"{PACKED_COMMIT}\n".encode())
        provenance = run_stage_from(tmp_path / "enclosing" / "packed", workdir)
        assert provenance["pipeline_commit"] != enclosing_head
        assert provenance["pipeline_commit"] == PACKED_COMMIT

    def test_stage_writes_null_inside_an_enclosing_repository(self, tmp_path, workdir):
        make_repo(tmp_path / "enclosing")
        provenance = run_stage_from(tmp_path / "enclosing" / "packed", workdir)
        assert "pipeline_commit" in provenance and provenance["pipeline_commit"] is None
