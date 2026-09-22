#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the model-fetch strategies and pipeline internals of
generate_embeddings.py.

Archives are real (small) tar/gzip files built in tmp_path; every network
call (HuggingFace snapshot_download, urllib) and every subprocess invocation
of champollion/evaluate.py is mocked.  No model weights are downloaded and no
GPU is required.
"""

import gzip
import os
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from champollion_pipeline import generate_embeddings as ge
from champollion_pipeline.generate_embeddings import (
    GenerateEmbeddings,
    HuggingFaceStrategy,
    InteractiveFallbackStrategy,
    LocalPathStrategy,
    ModelFetchStrategy,
    RemoteArchiveStrategy,
    main,
)


def make_tar(tmp_path, name="models.tar.gz", top="models"):
    """Create a small tar archive containing one file under `top/`."""
    payload = tmp_path / "payload"
    payload.mkdir(exist_ok=True)
    (payload / "weights.txt").write_text("weights")
    archive = tmp_path / name
    mode = "w:gz" if name.endswith((".tar.gz", ".tgz")) else "w:xz"
    with tarfile.open(archive, mode) as tar:
        tar.add(payload / "weights.txt", arcname=f"{top}/weights.txt")
    return archive


def make_gz(tmp_path, name="weights.gz"):
    """Create a plain gzip file (not a tarball)."""
    archive = tmp_path / name
    with gzip.open(archive, "wb") as fh:
        fh.write(b"weights")
    return archive


def make_script(argv):
    """Return a GenerateEmbeddings with parsed arguments."""
    script = GenerateEmbeddings()
    script.parse_args(argv)
    return script


class TestModelFetchStrategyABC:
    """Test the abstract base class contract."""

    def test_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ModelFetchStrategy()

    def test_abstract_method_bodies_are_reachable_via_super(self):
        class Concrete(ModelFetchStrategy):
            def can_handle(self, models_path):
                return ModelFetchStrategy.can_handle(self, models_path)

            def fetch(self, models_path, extract_to, no_cache=False):
                return ModelFetchStrategy.fetch(self, models_path, extract_to, no_cache)

        strategy = Concrete()
        assert strategy.can_handle("x") is None
        assert strategy.fetch("x", "y") is None


class TestLocalPathStrategy:
    """Test LocalPathStrategy."""

    def test_can_handle_existing_path(self, tmp_path):
        assert LocalPathStrategy().can_handle(str(tmp_path)) is True

    def test_cannot_handle_missing_path(self, tmp_path):
        assert LocalPathStrategy().can_handle(str(tmp_path / "nope")) is False

    def test_directory_is_returned_as_is(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        assert LocalPathStrategy().fetch(str(models), str(tmp_path / "cache")) == str(models)

    def test_plain_file_is_returned_as_is(self, tmp_path):
        plain = tmp_path / "notes.txt"
        plain.write_text("x")
        assert LocalPathStrategy().fetch(str(plain), str(tmp_path / "cache")) == str(plain)

    def test_tar_gz_is_extracted_into_named_cache_dir(self, tmp_path):
        archive = make_tar(tmp_path)
        cache = tmp_path / "cache"
        result = LocalPathStrategy().fetch(str(archive), str(cache))
        assert result == str(cache / "models")
        assert (cache / "models" / "models" / "weights.txt").exists()

    def test_tar_xz_is_extracted(self, tmp_path):
        archive = make_tar(tmp_path, name="models.tar.xz")
        result = LocalPathStrategy().fetch(str(archive), str(tmp_path / "cache"))
        assert Path(result).is_dir()

    def test_plain_gz_is_decompressed(self, tmp_path):
        archive = make_gz(tmp_path)
        result = LocalPathStrategy().fetch(str(archive), str(tmp_path / "cache"))
        assert Path(result).read_bytes() == b"weights"

    def test_cached_extraction_is_reused(self, tmp_path, capsys):
        archive = make_tar(tmp_path)
        cache = tmp_path / "cache"
        LocalPathStrategy().fetch(str(archive), str(cache))
        capsys.readouterr()
        result = LocalPathStrategy().fetch(str(archive), str(cache))
        assert "Using cached extraction" in capsys.readouterr().out
        assert result == str(cache / "models")

    def test_empty_cache_dir_is_not_reused(self, tmp_path):
        archive = make_tar(tmp_path)
        cache = tmp_path / "cache"
        (cache / "models").mkdir(parents=True)
        result = LocalPathStrategy().fetch(str(archive), str(cache))
        assert (Path(result) / "models" / "weights.txt").exists()

    def test_no_cache_forces_re_extraction(self, tmp_path, capsys):
        archive = make_tar(tmp_path)
        cache = tmp_path / "cache"
        LocalPathStrategy().fetch(str(archive), str(cache))
        capsys.readouterr()
        LocalPathStrategy().fetch(str(archive), str(cache), no_cache=True)
        assert "Removing existing cache" in capsys.readouterr().out

    def test_extract_archive_returns_dir_for_unknown_extension(self, tmp_path):
        target = tmp_path / "out"
        result = LocalPathStrategy()._extract_archive(str(tmp_path / "x.bin"), str(target))
        assert result == str(target)


class TestHuggingFaceStrategy:
    """Test HuggingFaceStrategy."""

    @pytest.mark.parametrize(
        "path", ["neurospin/Champollion_V1", "https://huggingface.co/neurospin/Champollion_V1"]
    )
    def test_can_handle_repo_ids_and_urls(self, path):
        assert HuggingFaceStrategy().can_handle(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "https://example.com/models.tar.gz",
            "/absolute/local/path",
            "neurospin/models.tar.gz",
            "neurospin/models.zip",
        ],
    )
    def test_cannot_handle_other_paths(self, path):
        assert HuggingFaceStrategy().can_handle(path) is False

    def test_repo_id_extracted_from_url(self):
        strategy = HuggingFaceStrategy()
        assert (
            strategy._extract_repo_id("https://huggingface.co/neurospin/Champollion_V1")
            == "neurospin/Champollion_V1"
        )

    def test_repo_id_extracted_from_single_segment_url(self):
        assert HuggingFaceStrategy()._extract_repo_id("https://huggingface.co/bert") == "bert"

    def test_plain_repo_id_passed_through(self):
        assert HuggingFaceStrategy()._extract_repo_id("neurospin/X") == "neurospin/X"

    def test_fetch_calls_snapshot_download(self, tmp_path):
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path / "dl")) as dl:
            result = HuggingFaceStrategy().fetch("neurospin/X", str(tmp_path))
        assert result == str(tmp_path / "dl")
        kwargs = dl.call_args.kwargs
        assert kwargs["repo_id"] == "neurospin/X"
        assert kwargs["allow_patterns"] is None
        assert kwargs["local_dir"] == str(tmp_path / "X")

    def test_subfolder_drives_allow_patterns_and_cache_name(self, tmp_path):
        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path / "dl")) as dl:
            result = HuggingFaceStrategy(subfolder="canonical_25").fetch("neurospin/X", str(tmp_path))
        kwargs = dl.call_args.kwargs
        assert kwargs["allow_patterns"] == ["canonical_25/*"]
        assert kwargs["local_dir"] == str(tmp_path / "X_canonical_25")
        assert result == str(tmp_path / "dl" / "canonical_25")

    def test_no_cache_forces_redownload(self, tmp_path):
        with patch("huggingface_hub.snapshot_download", return_value="/dl") as dl:
            HuggingFaceStrategy().fetch("neurospin/X", str(tmp_path), no_cache=True)
        assert dl.call_args.kwargs["force_download"] is True

    def test_system_ca_bundle_is_used_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
        with (
            patch("huggingface_hub.snapshot_download", return_value="/dl"),
            patch.object(ge.os.path, "exists", return_value=True),
        ):
            HuggingFaceStrategy().fetch("neurospin/X", str(tmp_path))
        assert os.environ["SSL_CERT_FILE"] == "/etc/ssl/certs/ca-certificates.crt"

    def test_existing_ssl_env_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SSL_CERT_FILE", "/my/bundle.pem")
        with patch("huggingface_hub.snapshot_download", return_value="/dl"):
            HuggingFaceStrategy().fetch("neurospin/X", str(tmp_path))
        assert os.environ["SSL_CERT_FILE"] == "/my/bundle.pem"

    def test_missing_huggingface_hub_raises_import_error(self, tmp_path):
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub is required"):
                HuggingFaceStrategy().fetch("neurospin/X", str(tmp_path))

    def test_download_failure_is_wrapped_in_runtime_error(self, tmp_path):
        with patch("huggingface_hub.snapshot_download", side_effect=ValueError("403")):
            with pytest.raises(RuntimeError, match="Failed to download from Hugging Face"):
                HuggingFaceStrategy().fetch("neurospin/X", str(tmp_path))


class TestRemoteArchiveStrategy:
    """Test RemoteArchiveStrategy."""

    @pytest.mark.parametrize(
        "path",
        [
            "https://example.com/models.tar.gz",
            "http://example.com/models.tar.xz",
            "ftp://example.com/models.zip",
        ],
    )
    def test_can_handle_remote_archives(self, path):
        assert RemoteArchiveStrategy().can_handle(path) is True

    @pytest.mark.parametrize(
        "path", ["https://example.com/page.html", "/local/models.tar.gz", "neurospin/X"]
    )
    def test_cannot_handle_other_paths(self, path):
        assert RemoteArchiveStrategy().can_handle(path) is False

    def test_downloads_extracts_and_removes_archive(self, tmp_path):
        source = make_tar(tmp_path, name="source.tar.gz", top="models")
        cache = tmp_path / "cache"
        cache.mkdir()

        def fake_urlretrieve(url, target):
            Path(target).write_bytes(source.read_bytes())

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = RemoteArchiveStrategy().fetch(
                "https://example.com/models.tar.gz", str(cache)
            )
        assert result == str(cache / "models")
        assert not (cache / "models.tar.gz").exists()

    def test_cached_extraction_is_reused_without_download(self, tmp_path, capsys):
        cache = tmp_path / "cache"
        (cache / "models").mkdir(parents=True)
        with patch("urllib.request.urlretrieve") as retrieve:
            result = RemoteArchiveStrategy().fetch("https://example.com/models.tar.gz", str(cache))
        retrieve.assert_not_called()
        assert result == str(cache / "models")
        assert "Using cached extraction" in capsys.readouterr().out

    def test_no_cache_removes_existing_cache(self, tmp_path, capsys):
        source = make_tar(tmp_path, name="source.tar.gz")
        cache = tmp_path / "cache"
        (cache / "models").mkdir(parents=True)

        def fake_urlretrieve(url, target):
            Path(target).write_bytes(source.read_bytes())

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            RemoteArchiveStrategy().fetch(
                "https://example.com/models.tar.gz", str(cache), no_cache=True
            )
        assert "Removing existing cache" in capsys.readouterr().out

    def test_download_failure_is_wrapped_in_runtime_error(self, tmp_path):
        with patch("urllib.request.urlretrieve", side_effect=OSError("no route to host")):
            with pytest.raises(RuntimeError, match="Failed to download or extract archive"):
                RemoteArchiveStrategy().fetch(
                    "https://example.com/models.tar.gz", str(tmp_path)
                )

    def test_extract_archive_handles_plain_gz(self, tmp_path):
        archive = make_gz(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        result = RemoteArchiveStrategy()._extract_archive(str(archive), str(out))
        assert Path(result).read_bytes() == b"weights"

    def test_extract_archive_returns_dir_for_unknown_extension(self, tmp_path):
        result = RemoteArchiveStrategy()._extract_archive(str(tmp_path / "x.bin"), str(tmp_path))
        assert result == str(tmp_path)


class TestInteractiveFallbackStrategy:
    """Test InteractiveFallbackStrategy."""

    def test_can_handle_everything(self):
        assert InteractiveFallbackStrategy().can_handle("anything") is True

    def test_non_interactive_session_raises(self, tmp_path):
        with patch.object(ge.sys.stdin, "isatty", return_value=False):
            with pytest.raises(RuntimeError, match="Cannot find models at"):
                InteractiveFallbackStrategy().fetch("weights", str(tmp_path))

    def test_declining_exits_gracefully(self, tmp_path):
        with (
            patch.object(ge.sys.stdin, "isatty", return_value=True),
            patch("builtins.input", return_value="no"),
        ):
            with pytest.raises(SystemExit) as exc:
                InteractiveFallbackStrategy().fetch("weights", str(tmp_path))
        assert exc.value.code == 0

    def test_missing_answer_path_raises(self, tmp_path):
        with (
            patch.object(ge.sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["yes", str(tmp_path / "nope")]),
        ):
            with pytest.raises(FileNotFoundError, match="Path not found"):
                InteractiveFallbackStrategy().fetch("weights", str(tmp_path))

    def test_directory_answer_is_used_directly(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        with (
            patch.object(ge.sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["y", str(models)]),
        ):
            result = InteractiveFallbackStrategy().fetch("weights", str(tmp_path / "cache"))
        assert result == str(models)

    def test_archive_answer_is_extracted(self, tmp_path):
        archive = make_tar(tmp_path, name="models.tar.gz", top="models")
        cache = tmp_path / "cache"
        with (
            patch.object(ge.sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["yes", str(archive)]),
        ):
            result = InteractiveFallbackStrategy().fetch("weights", str(cache))
        assert result == str(cache / "models")

    def test_cached_extraction_is_reused(self, tmp_path, capsys):
        archive = make_tar(tmp_path, name="models.tar.gz")
        cache = tmp_path / "cache"
        (cache / "models").mkdir(parents=True)
        with (
            patch.object(ge.sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["yes", str(archive)]),
        ):
            result = InteractiveFallbackStrategy().fetch("weights", str(cache))
        assert result == str(cache / "models")
        assert "Using cached extraction" in capsys.readouterr().out

    def test_no_cache_removes_existing_cache(self, tmp_path, capsys):
        archive = make_tar(tmp_path, name="models.tar.gz")
        cache = tmp_path / "cache"
        (cache / "models").mkdir(parents=True)
        with (
            patch.object(ge.sys.stdin, "isatty", return_value=True),
            patch("builtins.input", side_effect=["yes", str(archive)]),
        ):
            InteractiveFallbackStrategy().fetch("weights", str(cache), no_cache=True)
        assert "Removing existing cache" in capsys.readouterr().out

    def test_extract_archive_handles_plain_gz(self, tmp_path):
        archive = make_gz(tmp_path)
        result = InteractiveFallbackStrategy()._extract_archive(str(archive), str(tmp_path / "o"))
        assert Path(result).read_bytes() == b"weights"

    def test_extract_archive_returns_dir_for_unknown_extension(self, tmp_path):
        target = tmp_path / "out"
        result = InteractiveFallbackStrategy()._extract_archive(str(tmp_path / "x.bin"), str(target))
        assert result == str(target)


class TestFetchModels:
    """Test GenerateEmbeddings.fetch_models strategy dispatch."""

    def _script(self, argv_extra=None):
        return make_script(["/models", "/data/TEST01"] + (argv_extra or []))

    def test_huggingface_strategy_wins_for_repo_ids(self):
        script = self._script()
        with (
            patch.object(ge.os, "makedirs"),
            patch.object(ge.HuggingFaceStrategy, "fetch", return_value="/hf/path") as fetch,
        ):
            assert script.fetch_models("neurospin/Champollion_V1") == "/hf/path"
        assert fetch.call_args.args[0] == "neurospin/Champollion_V1"

    def test_masks_version_is_passed_as_hf_subfolder(self):
        script = self._script(["--masks-version", "canonical_25"])
        with (
            patch.object(ge.os, "makedirs"),
            patch.object(ge, "HuggingFaceStrategy", wraps=ge.HuggingFaceStrategy) as cls,
        ):
            cls.return_value = MagicMock(
                can_handle=MagicMock(return_value=True), fetch=MagicMock(return_value="/hf")
            )
            script.fetch_models("neurospin/X")
        assert cls.call_args.kwargs["subfolder"] == "canonical_25"

    def test_huggingface_failure_falls_through_to_local(self, tmp_path):
        script = self._script()
        models = tmp_path / "models"
        models.mkdir()
        with (
            patch.object(ge.os, "makedirs"),
            patch.object(ge.HuggingFaceStrategy, "can_handle", return_value=True),
            patch.object(ge.HuggingFaceStrategy, "fetch", side_effect=RuntimeError("offline")),
        ):
            assert script.fetch_models(str(models)) == str(models)

    def test_remote_archive_strategy_is_used_for_urls(self):
        script = self._script()
        with (
            patch.object(ge.os, "makedirs"),
            patch.object(ge.RemoteArchiveStrategy, "fetch", return_value="/remote") as fetch,
        ):
            assert script.fetch_models("https://example.com/m.tar.gz") == "/remote"
        fetch.assert_called_once()

    def test_remote_failure_falls_through_to_interactive_fallback(self):
        script = self._script()
        with (
            patch.object(ge.os, "makedirs"),
            patch.object(ge.RemoteArchiveStrategy, "fetch", side_effect=RuntimeError("404")),
            patch.object(ge.InteractiveFallbackStrategy, "fetch", return_value="/asked") as ask,
        ):
            assert script.fetch_models("https://example.com/m.tar.gz") == "/asked"
        ask.assert_called_once()

    def test_relative_local_path_is_resolved_to_absolute(self, tmp_path, monkeypatch):
        script = self._script()
        models = tmp_path / "models"
        models.mkdir()
        monkeypatch.chdir(tmp_path)
        with patch.object(ge.os, "makedirs"):
            assert script.fetch_models("models") == str(models)

    def test_local_strategy_failure_falls_back_to_interactive(self, tmp_path):
        script = self._script()
        models = tmp_path / "models"
        models.mkdir()
        with (
            patch.object(ge.os, "makedirs"),
            patch.object(ge.LocalPathStrategy, "fetch", side_effect=OSError("permission denied")),
            patch.object(ge.InteractiveFallbackStrategy, "fetch", return_value="/asked") as ask,
        ):
            assert script.fetch_models(str(models)) == "/asked"
        ask.assert_called_once()

    def test_extract_dir_is_under_models_cache(self):
        script = self._script()
        with (
            patch.object(ge.os, "makedirs") as makedirs,
            patch.object(ge.InteractiveFallbackStrategy, "fetch", return_value="/asked"),
        ):
            script.fetch_models("/definitely/missing")
        extract_to = makedirs.call_args.args[0]
        assert extract_to.endswith("data/data/TEST01/derivatives/champollion_V1/models_cache")

    def test_no_cache_flag_is_forwarded(self):
        script = self._script(["--no-cache"])
        with (
            patch.object(ge.os, "makedirs"),
            patch.object(ge.InteractiveFallbackStrategy, "fetch", return_value="/asked") as ask,
        ):
            script.fetch_models("/definitely/missing")
        assert ask.call_args.args[2] is True


class TestRunNormalCudaHandling:
    """Test _run_normal's CUDA_VISIBLE_DEVICES save/restore contract."""

    def test_cpu_flag_sets_cuda_visible_devices_empty(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        script = make_script(["/m", "/d", "--cpu"])
        seen = {}
        script._run_pipeline = lambda local_dir: seen.setdefault(
            "cuda", os.environ["CUDA_VISIBLE_DEVICES"]
        )
        script._run_normal()
        assert seen["cuda"] == ""

    def test_cuda_env_restored_to_original_value(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        script = make_script(["/m", "/d", "--cpu"])
        script._run_pipeline = lambda local_dir: 0
        script._run_normal()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"

    def test_cuda_env_removed_again_when_absent_before(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        script = make_script(["/m", "/d", "--cpu"])
        script._run_pipeline = lambda local_dir: 0
        script._run_normal()
        assert "CUDA_VISIBLE_DEVICES" not in os.environ


class TestFindSubjectsFile:
    """Test _find_subjects_file."""

    def test_finds_participants_tsv(self, tmp_path):
        (tmp_path / "participants.tsv").touch()
        script = make_script(["/m", str(tmp_path)])
        assert script._find_subjects_file(str(tmp_path)) == str(tmp_path / "participants.tsv")

    def test_falls_back_to_participants_csv(self, tmp_path):
        (tmp_path / "participants.csv").touch()
        script = make_script(["/m", str(tmp_path)])
        assert script._find_subjects_file(str(tmp_path)) == str(tmp_path / "participants.csv")

    def test_raises_when_no_participants_file(self, tmp_path):
        script = make_script(["/m", str(tmp_path)])
        with pytest.raises(FileNotFoundError, match="No participants file found"):
            script._find_subjects_file(str(tmp_path))


class TestFindCropDir:
    """Test _find_crop_dir."""

    def test_maps_model_name_back_to_dotted_crop_dir(self, tmp_path):
        (tmp_path / "S.C.-sylv.").mkdir()
        script = make_script(["/m", "/d"])
        assert script._find_crop_dir(str(tmp_path), "SC-sylv_left") == "S.C.-sylv."

    def test_handles_right_hemisphere_suffix(self, tmp_path):
        (tmp_path / "S.Or.").mkdir()
        script = make_script(["/m", "/d"])
        assert script._find_crop_dir(str(tmp_path), "SOr_right") == "S.Or."

    def test_files_are_ignored(self, tmp_path):
        (tmp_path / "SOr").write_text("not a dir")
        script = make_script(["/m", "/d"])
        assert script._find_crop_dir(str(tmp_path), "SOr_left") == "SOr"

    def test_returns_base_name_when_crops_dir_missing(self, tmp_path):
        script = make_script(["/m", "/d"])
        assert script._find_crop_dir(str(tmp_path / "nope"), "SOr_left") == "SOr"

    def test_returns_base_name_when_no_match(self, tmp_path):
        (tmp_path / "other").mkdir()
        script = make_script(["/m", "/d"])
        assert script._find_crop_dir(str(tmp_path), "SOr_left") == "SOr"


class TestEnsureCkpt:
    """Test _ensure_ckpt (best_model_weights.pt → Lightning .ckpt)."""

    def test_noop_when_no_weights_present(self, tmp_path):
        script = make_script(["/m", "/d"])
        script._ensure_ckpt(str(tmp_path))
        assert not (tmp_path / "logs").exists()

    def test_noop_when_ckpt_already_exists(self, tmp_path):
        ckpt_dir = tmp_path / "logs" / "lightning_logs" / "version_0" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "existing.ckpt").write_text("kept")
        (tmp_path / "logs" / "best_model_weights.pt").write_text("ignored")
        script = make_script(["/m", "/d"])
        script._ensure_ckpt(str(tmp_path))
        assert (ckpt_dir / "existing.ckpt").read_text() == "kept"
        assert not (ckpt_dir / "best_model.ckpt").exists()

    def test_converts_weights_to_checkpoint(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        torch.save({"layer.weight": torch.zeros(2)}, str(logs / "best_model_weights.pt"))
        script = make_script(["/m", "/d"])
        script._ensure_ckpt(str(tmp_path))
        ckpt_path = logs / "lightning_logs" / "version_0" / "checkpoints" / "best_model.ckpt"
        assert ckpt_path.exists()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 0
        assert ckpt["global_step"] == 0
        assert "layer.weight" in ckpt["state_dict"]

    def test_conversion_is_idempotent(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        torch.save({"w": torch.zeros(1)}, str(logs / "best_model_weights.pt"))
        script = make_script(["/m", "/d"])
        script._ensure_ckpt(str(tmp_path))
        ckpt_path = logs / "lightning_logs" / "version_0" / "checkpoints" / "best_model.ckpt"
        first_mtime = ckpt_path.stat().st_mtime_ns
        script._ensure_ckpt(str(tmp_path))
        assert ckpt_path.stat().st_mtime_ns == first_mtime


class TestRunPerRegion:
    """Test _run_per_region error handling and command construction."""

    def test_missing_models_path_raises_value_error(self, tmp_path):
        script = make_script([str(tmp_path / "nope"), "/d"])
        script.args.models_path = str(tmp_path / "nope")
        with pytest.raises(ValueError, match="Models path not found"):
            script._run_per_region("evaluate.py", str(tmp_path), "subjects.tsv", str(tmp_path))

    def test_empty_models_path_raises_value_error(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        script = make_script([str(models), "/d"])
        with pytest.raises(ValueError, match="No region model directories"):
            script._run_per_region("evaluate.py", str(tmp_path), "subjects.tsv", str(tmp_path))

    def test_existing_embeddings_are_skipped(self, tmp_path, capsys):
        models = tmp_path / "models"
        (models / "SOr_left" / "logs").mkdir(parents=True)
        out = tmp_path / "out"
        (out / "SOr_left").mkdir(parents=True)
        (out / "SOr_left" / "full_embeddings.csv").touch()

        script = make_script([str(models), "/d"])
        script.execute_command = lambda cmd, shell=False: pytest.fail("must not run evaluate.py")
        script._run_per_region("evaluate.py", str(tmp_path), "subjects.tsv", str(out))
        assert "[SKIP] SOr_left" in capsys.readouterr().out

    def test_overwrite_recomputes_existing_embeddings(self, tmp_path):
        models = tmp_path / "models"
        (models / "SOr_left" / "logs").mkdir(parents=True)
        out = tmp_path / "out"
        (out / "SOr_left").mkdir(parents=True)
        (out / "SOr_left" / "full_embeddings.csv").touch()

        script = make_script([str(models), "/d", "--overwrite"])
        calls = []
        script.execute_command = lambda cmd, shell=False: calls.append(cmd) or 0
        script._run_per_region("evaluate.py", str(tmp_path), "subjects.tsv", str(out))
        assert len(calls) == 1

    def test_command_uses_left_skeleton_for_left_region(self, tmp_path):
        models = tmp_path / "models"
        (models / "SOr_left" / "logs").mkdir(parents=True)
        crops = tmp_path / "crops"
        (crops / "S.Or.").mkdir(parents=True)

        script = make_script([str(models), "/d"])
        calls = []
        script.execute_command = lambda cmd, shell=False: calls.append(cmd) or 0
        script._run_per_region("evaluate.py", str(crops), "subjects.tsv", str(tmp_path / "out"))
        cmd = calls[0]
        assert cmd[1] == "evaluate.py"
        assert cmd[cmd.index("-sk") + 1].endswith("S.Or./mask/Lskeleton.npy")
        assert cmd[cmd.index("-i") + 1] == "subjects.tsv"

    def test_command_uses_right_skeleton_for_right_region(self, tmp_path):
        models = tmp_path / "models"
        (models / "SOr_right" / "logs").mkdir(parents=True)

        script = make_script([str(models), "/d"])
        calls = []
        script.execute_command = lambda cmd, shell=False: calls.append(cmd) or 0
        script._run_per_region("evaluate.py", str(tmp_path), "subjects.tsv", str(tmp_path / "out"))
        assert calls[0][calls[0].index("-sk") + 1].endswith("Rskeleton.npy")

    def test_returns_last_command_result(self, tmp_path):
        models = tmp_path / "models"
        (models / "SOr_left" / "logs").mkdir(parents=True)
        script = make_script([str(models), "/d"])
        script.execute_command = lambda cmd, shell=False: 3
        result = script._run_per_region(
            "evaluate.py", str(tmp_path), "subjects.tsv", str(tmp_path / "out")
        )
        assert result == 3


class TestRunCkaTest:
    """Test _run_cka_test."""

    def test_invokes_coherence_test_with_output_dir(self, tmp_path):
        script = make_script(["/m", "/d"])
        with patch.object(ge, "test_models_coherence_from_directory") as cka:
            script._run_cka_test(str(tmp_path))
        kwargs = cka.call_args.kwargs
        assert kwargs["models_dir"] == str(tmp_path)
        assert kwargs["output_dir"] == str(tmp_path / "cka_results")
        assert kwargs["embedding_filename"] == "full_embeddings.csv"
        assert kwargs["subject_column"] == "Subject"

    def test_failure_is_warned_but_not_raised(self, tmp_path, capsys):
        script = make_script(["/m", "/d"])
        with patch.object(
            ge, "test_models_coherence_from_directory", side_effect=RuntimeError("no data")
        ):
            script._run_cka_test(str(tmp_path))
        assert "Warning: CKA test failed - no data" in capsys.readouterr().out

    def test_success_is_reported(self, tmp_path, capsys):
        script = make_script(["/m", "/d"])
        with patch.object(ge, "test_models_coherence_from_directory"):
            script._run_cka_test(str(tmp_path))
        assert "CKA coherence test completed." in capsys.readouterr().out


class TestRunPipelineCkaHook:
    """Test that --run-cka triggers the CKA step after embeddings."""

    def _prepare(self, script, tmp_path):
        script.fetch_models = lambda path: str(tmp_path / "models")
        script._run_per_region = lambda *args, **kwargs: 0
        script._find_subjects_file = lambda root: "subjects.tsv"

    def test_cka_not_run_by_default(self, tmp_path):
        script = make_script(["/m", str(tmp_path)])
        self._prepare(script, tmp_path)
        with patch.object(script, "_run_cka_test") as cka:
            script._run_pipeline(str(tmp_path))
        cka.assert_not_called()

    def test_cka_runs_when_requested(self, tmp_path):
        script = make_script(["/m", str(tmp_path), "--run-cka"])
        self._prepare(script, tmp_path)
        with patch.object(script, "_run_cka_test") as cka:
            script._run_pipeline(str(tmp_path))
        cka.assert_called_once()


class TestMain:
    """Test the main() entry point."""

    def test_main_builds_prints_and_runs(self, monkeypatch):
        calls = []

        class FakeScript:
            def build(self):
                calls.append("build")
                return self

            def print_args(self):
                calls.append("print_args")
                return self

            def run(self):
                calls.append("run")
                return 0

        monkeypatch.setattr(ge, "GenerateEmbeddings", FakeScript)
        assert main() == 0
        assert calls == ["build", "print_args", "run"]
