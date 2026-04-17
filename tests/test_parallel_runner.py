"""Tests for src/parallel_runner.py"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src/ to sys.path (same pattern as other test files)
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from parallel_runner import (
    _scan_embeddings_exist,
    _wait_for_scan_files,
    run_parallel_pipeline,
    run_scan_worker,
)
from file_indexer.scan_id import ScanId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_fake_config(**kwargs):
    """Return a simple namespace object acting as DatasetConfig."""
    defaults = {
        "name": "test_dataset",
        "path_to_graph": "t1mri/folds",
        "path_sk_with_hull": "t1mri/mesh",
        "sk_qc_path": "",
        "bids": False,
        "regions": [],
        "embeddings_path": "champollion_V1",
        "dataset_localization": "local",
        "datasets_root": "",
        "short_name": "eval",
        "models_path": "",
    }
    defaults.update(kwargs)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# TestWaitForScanFiles
# ---------------------------------------------------------------------------

class TestWaitForScanFiles:

    @pytest.mark.unit
    def test_returns_immediately_when_files_present(self, tmp_path):
        """When the required file exists the barrier returns without raising."""
        scan_id = ScanId("sub-001")
        # Create a file matching the pattern under sub-001/
        _touch(tmp_path / "sub-001" / "t1mri" / "folds" / "Rsub.arg")

        logger = MagicMock()
        # Should not raise
        _wait_for_scan_files(
            str(tmp_path),
            scan_id,
            patterns=["t1mri/folds/R*.arg"],
            logger=logger,
            poll_interval=1,
            timeout=5,
        )

    @pytest.mark.unit
    def test_raises_timeout_when_files_missing(self, tmp_path):
        """When files are absent the barrier raises TimeoutError after timeout."""
        scan_id = ScanId("sub-001")
        (tmp_path / "sub-001").mkdir()

        logger = MagicMock()
        with pytest.raises(TimeoutError):
            _wait_for_scan_files(
                str(tmp_path),
                scan_id,
                patterns=["t1mri/folds/R*.arg"],
                logger=logger,
                poll_interval=1,
                timeout=1,
            )


# ---------------------------------------------------------------------------
# TestRunScanWorker
# ---------------------------------------------------------------------------

class TestRunScanWorker:

    @pytest.mark.unit
    def test_dry_run_logs_and_returns(self, tmp_path):
        """dry_run=True should log and return without calling any stage."""
        scan_id = ScanId("sub-001")
        config = _make_fake_config()

        with (
            patch("parallel_runner._run_cortical_tiles") as mock_tiles,
            patch("parallel_runner._run_generate_config") as mock_cfg,
            patch("parallel_runner._run_generate_embeddings") as mock_emb,
            patch("parallel_runner._wait_for_scan_files") as mock_wait,
            patch("parallel_runner._scan_embeddings_exist", return_value=False),
        ):
            run_scan_worker(
                scan_id,
                str(tmp_path / "subjects"),
                str(tmp_path / "output"),
                config,
                str(tmp_path / "logs"),
                dry_run=True,
            )

        mock_tiles.assert_not_called()
        mock_cfg.assert_not_called()
        mock_emb.assert_not_called()
        mock_wait.assert_not_called()

    @pytest.mark.unit
    def test_skips_if_embeddings_exist(self, tmp_path):
        """When embeddings already exist the worker skips all stage runners."""
        scan_id = ScanId("sub-001")
        config = _make_fake_config()

        with (
            patch("parallel_runner._run_cortical_tiles") as mock_tiles,
            patch("parallel_runner._run_generate_config") as mock_cfg,
            patch("parallel_runner._run_generate_embeddings") as mock_emb,
            patch("parallel_runner._wait_for_scan_files") as mock_wait,
            patch("parallel_runner._scan_embeddings_exist", return_value=True),
        ):
            run_scan_worker(
                scan_id,
                str(tmp_path / "subjects"),
                str(tmp_path / "output"),
                config,
                str(tmp_path / "logs"),
            )

        mock_tiles.assert_not_called()
        mock_cfg.assert_not_called()
        mock_emb.assert_not_called()
        mock_wait.assert_not_called()

    @pytest.mark.unit
    def test_calls_stages_in_order(self, tmp_path):
        """The worker calls wait → tiles → wait → config → wait → embeddings."""
        scan_id = ScanId("sub-001")
        config = _make_fake_config()
        call_order = []

        def make_recorder(name):
            def _fn(*args, **kwargs):
                call_order.append(name)
            return _fn

        with (
            patch("parallel_runner._scan_embeddings_exist", return_value=False),
            patch("parallel_runner._wait_for_scan_files", side_effect=make_recorder("wait")),
            patch("parallel_runner._run_cortical_tiles", side_effect=make_recorder("tiles")),
            patch("parallel_runner._run_generate_config", side_effect=make_recorder("config")),
            patch("parallel_runner._run_generate_embeddings", side_effect=make_recorder("emb")),
        ):
            run_scan_worker(
                scan_id,
                str(tmp_path / "subjects"),
                str(tmp_path / "output"),
                config,
                str(tmp_path / "logs"),
            )

        assert call_order == ["wait", "tiles", "wait", "config", "wait", "emb"]

    @pytest.mark.unit
    def test_handles_stage_exception_gracefully(self, tmp_path):
        """A RuntimeError in cortical_tiles must not propagate out of the worker."""
        scan_id = ScanId("sub-001")
        config = _make_fake_config()

        with (
            patch("parallel_runner._scan_embeddings_exist", return_value=False),
            patch("parallel_runner._wait_for_scan_files"),
            patch("parallel_runner._run_cortical_tiles", side_effect=RuntimeError("boom")),
            patch("parallel_runner._run_generate_config") as mock_cfg,
            patch("parallel_runner._run_generate_embeddings") as mock_emb,
        ):
            # Must not raise
            run_scan_worker(
                scan_id,
                str(tmp_path / "subjects"),
                str(tmp_path / "output"),
                config,
                str(tmp_path / "logs"),
            )

        # Downstream stages should not have been called
        mock_cfg.assert_not_called()
        mock_emb.assert_not_called()

    @pytest.mark.unit
    def test_handles_timeout_gracefully(self, tmp_path):
        """A TimeoutError from the file barrier must not propagate out."""
        scan_id = ScanId("sub-001")
        config = _make_fake_config()

        with (
            patch("parallel_runner._scan_embeddings_exist", return_value=False),
            patch("parallel_runner._wait_for_scan_files", side_effect=TimeoutError("too slow")),
            patch("parallel_runner._run_cortical_tiles") as mock_tiles,
        ):
            # Must not raise
            run_scan_worker(
                scan_id,
                str(tmp_path / "subjects"),
                str(tmp_path / "output"),
                config,
                str(tmp_path / "logs"),
            )

        mock_tiles.assert_not_called()


# ---------------------------------------------------------------------------
# TestRunParallelPipeline
# ---------------------------------------------------------------------------

class TestRunParallelPipeline:

    def _make_checker(self, scan_ids):
        """Return a mock SubjectEligibilityChecker that yields *scan_ids*."""
        mock_checker = MagicMock()
        mock_checker._build_index.return_value = MagicMock()
        mock_checker._get_scan_ids.return_value = scan_ids
        return mock_checker

    @pytest.mark.unit
    def test_dry_run_prints_scans_and_returns_zero(self, tmp_path, capsys):
        """dry_run=True prints scan count and each scan_id, returns 0."""
        scan_ids = [ScanId("sub-001"), ScanId("sub-002")]
        config = _make_fake_config()

        mock_checker_instance = self._make_checker(scan_ids)

        with (
            patch("parallel_runner.SubjectEligibilityChecker", return_value=mock_checker_instance),
            patch("parallel_runner.run_scan_worker") as mock_worker,
            patch("parallel_runner._run_put_together_embeddings") as mock_combine,
        ):
            rc = run_parallel_pipeline(
                subjects_dir=str(tmp_path / "subjects"),
                output_dir=str(tmp_path / "output"),
                config=config,
                n_workers=2,
                log_dir=str(tmp_path / "logs"),
                dry_run=True,
            )

        assert rc == 0
        mock_worker.assert_not_called()
        mock_combine.assert_not_called()
        out = capsys.readouterr().out
        assert "sub-001" in out
        assert "sub-002" in out

    @pytest.mark.unit
    def test_zero_scans_returns_zero(self, tmp_path):
        """When no scans are found the function returns 0 immediately."""
        config = _make_fake_config()
        mock_checker_instance = self._make_checker([])

        with (
            patch("parallel_runner.SubjectEligibilityChecker", return_value=mock_checker_instance),
            patch("parallel_runner.run_scan_worker") as mock_worker,
            patch("parallel_runner._run_put_together_embeddings") as mock_combine,
        ):
            rc = run_parallel_pipeline(
                subjects_dir=str(tmp_path / "subjects"),
                output_dir=str(tmp_path / "output"),
                config=config,
                n_workers=2,
                log_dir=str(tmp_path / "logs"),
            )

        assert rc == 0
        mock_worker.assert_not_called()
        mock_combine.assert_not_called()

    @pytest.mark.unit
    def test_njobs_per_worker_calculation(self, tmp_path):
        """With n_workers=4 and cpu_count()=8, njobs_per_worker passed to workers is 2."""
        from concurrent.futures import ThreadPoolExecutor

        scan_ids = [ScanId("sub-001")]
        config = _make_fake_config()
        mock_checker_instance = self._make_checker(scan_ids)

        captured_njobs = []

        def fake_worker(scan_id, subjects_dir, output_dir, cfg, log_dir, njobs_per_worker, *args, **kwargs):
            captured_njobs.append(njobs_per_worker)

        # Use ThreadPoolExecutor to avoid pickling MagicMock objects.
        # We patch ProcessPoolExecutor inside the concurrent.futures namespace
        # that parallel_runner imports from.
        with (
            patch("parallel_runner.SubjectEligibilityChecker", return_value=mock_checker_instance),
            patch("parallel_runner.run_scan_worker", side_effect=fake_worker),
            patch("parallel_runner._run_put_together_embeddings", return_value=0),
            patch("os.cpu_count", return_value=8),
            patch("parallel_runner.ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            run_parallel_pipeline(
                subjects_dir=str(tmp_path / "subjects"),
                output_dir=str(tmp_path / "output"),
                config=config,
                n_workers=4,
                log_dir=str(tmp_path / "logs"),
            )

        assert captured_njobs == [2]


# ---------------------------------------------------------------------------
# TestScanEmbeddingsExist
# ---------------------------------------------------------------------------

class TestScanEmbeddingsExist:

    @pytest.mark.unit
    def test_returns_true_when_embeddings_found(self, tmp_path):
        """Returns True when full_embeddings.csv exists under the expected path."""
        scan_id = ScanId("sub-001", "ses-01")
        csv_path = (
            tmp_path
            / "champollion_V1"
            / "models"
            / str(scan_id)
            / "results"
            / "full_embeddings.csv"
        )
        _touch(csv_path)
        assert _scan_embeddings_exist(scan_id, str(tmp_path)) is True

    @pytest.mark.unit
    def test_returns_false_when_none(self, tmp_path):
        """Returns False when no full_embeddings.csv exists."""
        scan_id = ScanId("sub-001", "ses-01")
        assert _scan_embeddings_exist(scan_id, str(tmp_path)) is False
