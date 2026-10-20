#!/usr/bin/env python3
"""
generate_embeddings: the code only --use_last_checkpoint reached is gone (EPOCH80 dead-code task).

Epoch-80 harmonisation (user decision 2026-10-13): the parser rejects
--use_last_checkpoint (tests/test_generate_embeddings_checkpoint_epoch.py), so the
methods that served only that flag, or wrote best_model.ckpt into the region model
directory, have no caller left. Best weights reach evaluate.py only through the
temporary mirror (tests/test_generate_embeddings_best_weights_standin.py).

- REQ-EPOCH80-BDRABCZUK-31A4563BBC16: no _ensure_ckpt method.
- REQ-EPOCH80-BDRABCZUK-B4A46072278E: no _uses_native_ckpt method.
- REQ-EPOCH80-BDRABCZUK-CA76521D5083: no _compute_ignored_flag_suffix method.
"""

import pytest

from champollion_pipeline.generate_embeddings import GenerateEmbeddings

pytestmark = pytest.mark.unit  # noqa: V107


class TestRemovedCheckpointMethods:
    """REQ-EPOCH80-BDRABCZUK-31A4563BBC16 / -B4A46072278E / -CA76521D5083: method not declared."""

    @pytest.mark.parametrize("name", ["_ensure_ckpt", "_uses_native_ckpt", "_compute_ignored_flag_suffix"])
    def test_method_is_not_declared(self, name):
        assert not hasattr(GenerateEmbeddings, name), f"GenerateEmbeddings still declares {name}"
