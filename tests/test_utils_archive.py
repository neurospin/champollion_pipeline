"""Tests for champollion_pipeline.utils.archive, the archive helpers shared by the model fetch strategies."""

import ast
import gzip
import io
import re
import tarfile
from pathlib import Path

import pytest

from champollion_pipeline.utils import archive

GENERATE_EMBEDDINGS = Path(__file__).resolve().parents[1] / "src" / "champollion_pipeline" / "generate_embeddings.py"


def make_tar(tmp_path, name="models.tar.gz", top="models"):
    """Create a small tar archive holding top/weights.txt."""
    payload = tmp_path / "payload"
    payload.mkdir(exist_ok=True)
    (payload / "weights.txt").write_text("weights")
    path = tmp_path / name
    mode = "w:gz" if name.endswith((".tar.gz", ".tgz")) else "w:xz"
    with tarfile.open(path, mode) as tar:
        tar.add(payload / "weights.txt", arcname=f"{top}/weights.txt")
    return path


def make_tar_of(tmp_path, *members):
    """Create models.tar.gz holding a safe regular file, then *members* (TarInfo with optional bytes)."""
    path = tmp_path / "models.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for info, data in (("safe.txt", b"safe"), *members):
            if isinstance(info, str):
                info = tarfile.TarInfo(info)
            if data is not None:
                info.size = len(data)
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return path


def link_member(name, linkname, kind):
    """A TarInfo for a symbolic (SYMTYPE) or hard (LNKTYPE) link named *name* pointing at *linkname*."""
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    return info


class TestExtractArchive:
    """REQ-REFACTOR-BDRABCZUK-D67D3945822A, REQ-REFACTOR-BDRABCZUK-8F1C34F3A87E."""

    def test_tar_members_extracted_under_target(self, tmp_path):
        target = tmp_path / "out"
        target.mkdir()
        for name in ("a.tar.gz", "b.tar.xz", "c.tgz"):
            archive.extract_archive(
                str(make_tar(tmp_path, name=name, top=name.split(".")[0])), str(target), top_level=True
            )
        for top in ("a", "b", "c"):
            assert (target / top / "weights.txt").read_text() == "weights"

    def test_tar_top_level_true_returns_top_dir(self, tmp_path):
        target = tmp_path / "out"
        target.mkdir()
        result = archive.extract_archive(str(make_tar(tmp_path, top="models")), str(target), top_level=True)
        assert result == str(target / "models")

    def test_tar_top_level_false_returns_target_dir(self, tmp_path):
        target = tmp_path / "out"
        target.mkdir()
        result = archive.extract_archive(str(make_tar(tmp_path, top="models")), str(target), top_level=False)
        assert result == str(target)
        assert (target / "models" / "weights.txt").exists()

    def test_plain_gz_is_decompressed_next_to_stem(self, tmp_path):
        source = tmp_path / "weights.bin.gz"
        with gzip.open(source, "wb") as fh:
            fh.write(b"payload")
        target = tmp_path / "out"
        target.mkdir()
        result = archive.extract_archive(str(source), str(target), top_level=True)
        assert result == str(target / "weights.bin")
        assert (target / "weights.bin").read_bytes() == b"payload"

    def test_symlink_inside_target_extracted(self, tmp_path):
        target = tmp_path / "out"
        target.mkdir()
        path = make_tar_of(tmp_path, (link_member("models/latest", "../safe.txt", tarfile.SYMTYPE), None))
        archive.extract_archive(str(path), str(target), top_level=False)
        assert (target / "models" / "latest").is_symlink()
        assert (target / "models" / "latest").read_text() == "safe"


class TestExtractArchiveRefusesUnsafeMembers:
    """REQ-LINTRULES-BDRABCZUK-EC8903097CAD, REQ-LINTRULES-BDRABCZUK-62726A5939EA."""

    @staticmethod
    def _assert_refused(tmp_path, path, member_name):
        target = tmp_path / "out"
        target.mkdir()
        with pytest.raises(ValueError, match=re.escape(member_name)):
            archive.extract_archive(str(path), str(target), top_level=False)
        assert list(target.iterdir()) == [], "a member was extracted before the unsafe member was refused"

    def test_parent_traversal_member_refused(self, tmp_path):
        path = make_tar_of(tmp_path, ("../escape.txt", b"evil"))
        self._assert_refused(tmp_path, path, "../escape.txt")
        assert not (tmp_path / "escape.txt").exists()

    def test_absolute_member_refused(self, tmp_path):
        absolute = str(tmp_path / "absolute_escape.txt")
        path = make_tar_of(tmp_path, (absolute, b"evil"))
        self._assert_refused(tmp_path, path, absolute)
        assert not (tmp_path / "absolute_escape.txt").exists()

    def test_symlink_outside_target_refused(self, tmp_path):
        path = make_tar_of(tmp_path, (link_member("models/escape", "../../outside", tarfile.SYMTYPE), None))
        self._assert_refused(tmp_path, path, "models/escape")

    def test_hardlink_outside_target_refused(self, tmp_path):
        (tmp_path / "outside.txt").write_text("outside")
        path = make_tar_of(tmp_path, (link_member("models/escape", "../outside.txt", tarfile.LNKTYPE), None))
        self._assert_refused(tmp_path, path, "models/escape")


class TestReuseCachedExtraction:
    """REQ-REFACTOR-BDRABCZUK-FB87E74AAD4B."""

    def test_existing_dir_is_reused(self, tmp_path, capsys):
        cached = tmp_path / "models"
        cached.mkdir()
        (cached / "w.txt").write_text("w")
        assert archive.reuse_cached_extraction(str(cached), False) is True
        assert f"Using cached extraction: {cached}" in capsys.readouterr().out.splitlines()

    def test_no_cache_prevents_reuse(self, tmp_path, capsys):
        cached = tmp_path / "models"
        cached.mkdir()
        (cached / "w.txt").write_text("w")
        assert archive.reuse_cached_extraction(str(cached), True) is False
        assert "Using cached extraction" not in capsys.readouterr().out

    def test_missing_dir_is_not_reused(self, tmp_path):
        assert archive.reuse_cached_extraction(str(tmp_path / "absent"), False) is False
        plain_file = tmp_path / "file"
        plain_file.write_text("x")
        assert archive.reuse_cached_extraction(str(plain_file), False) is False

    def test_empty_dir_reuse_depends_on_require_content(self, tmp_path):
        cached = tmp_path / "models"
        cached.mkdir()
        assert archive.reuse_cached_extraction(str(cached), False, require_content=False) is True
        assert archive.reuse_cached_extraction(str(cached), False, require_content=True) is False


class TestRemoveStaleCache:
    """REQ-REFACTOR-BDRABCZUK-604D5A77C0D9."""

    def test_no_cache_removes_existing_dir(self, tmp_path, capsys):
        cached = tmp_path / "models"
        (cached / "sub").mkdir(parents=True)
        (cached / "sub" / "w.txt").write_text("w")
        archive.remove_stale_cache(str(cached), True)
        assert not cached.exists()
        assert f"Removing existing cache: {cached}" in capsys.readouterr().out.splitlines()

    def test_cache_kept_without_no_cache(self, tmp_path, capsys):
        cached = tmp_path / "models"
        cached.mkdir()
        archive.remove_stale_cache(str(cached), False)
        assert cached.is_dir()
        assert "Removing existing cache" not in capsys.readouterr().out


class TestGenerateEmbeddingsDelegates:
    """REQ-REFACTOR-BDRABCZUK-8013C6D5E185."""

    def test_generate_embeddings_does_not_import_tarfile(self):
        tree = ast.parse(GENERATE_EMBEDDINGS.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "tarfile" not in imported
