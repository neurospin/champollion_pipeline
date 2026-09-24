#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guard for REQ-HFUPLOAD-01: exactly one HuggingFace-upload script exists.

``src/upload_models_to_hf.py`` is the tracked, tested script used to upload
trained Champollion model weights to HuggingFace (git history back to the
initial public release, covered by ``tests/test_upload_models_to_hf.py``).

``scripts/upload_models_to_hf.py`` is a stale, never-committed duplicate
with the same purpose (uploading model weights to HuggingFace) but a
different, hardcoded implementation (module-level ``REPO_ID``/``SOURCES``
constants pointing at ``neurospin/Champollion_V1``) and no test coverage.
Having two scripts that do the same job invites uploading with the wrong
one. This guard enforces that only the canonical script exists.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestSingleHuggingFaceUploadScript:
    def test_canonical_upload_script_exists(self) -> None:
        assert (PROJECT_ROOT / "src" / "upload_models_to_hf.py").is_file(), (
            "canonical src/upload_models_to_hf.py must exist"
        )

    def test_duplicate_scripts_upload_script_does_not_exist(self) -> None:
        duplicate = PROJECT_ROOT / "scripts" / "upload_models_to_hf.py"
        assert not duplicate.exists(), (
            f"stale duplicate {duplicate} must not exist; "
            "src/upload_models_to_hf.py is the only HuggingFace upload script"
        )
