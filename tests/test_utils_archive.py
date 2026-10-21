"""Tests for champollion_pipeline.utils.archive, the archive helpers shared by the model fetch strategies."""

import ast
import gzip
import tarfile
from pathlib import Path

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
