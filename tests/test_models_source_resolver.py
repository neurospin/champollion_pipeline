"""resolve_models_source in champollion_pipeline.models.fetch (models_source_cascade).

TASK-MODELSRC-BDRABCZUK-EFFEE1006BFE.

An empty models_path means "not set" (no --models-path, no YAML models_path). <root_path> is the checkout;
<models_cache> is the models cache folder; <masks> is the --masks model set.

Requirements pinned here (component):

* REQ-MODELSRC-BDRABCZUK-0C79BCA65A4C - the default <root_path>/models folder first.
* REQ-MODELSRC-BDRABCZUK-7D7429F79BA7 - an existing models cache entry of the --masks model set second.
* REQ-MODELSRC-BDRABCZUK-A4F384DE609B - the Hugging Face repository last.
* REQ-MODELSRC-BDRABCZUK-6FA95D517AD7 - an explicit models_path returned unchanged.
* REQ-MODELSRC-BDRABCZUK-BCE9C3F46987 - the source logged at OUTPUT.

The helpers below are shared with tests/test_models_source_cascade.py (the stage-level tests).

No network: the Hugging Face hub module seen by champollion_pipeline.models.fetch is a fake that fails the test on
any call. Every folder lives under tmp_path.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from champollion_pipeline.models import fetch as models_fetch
from champollion_pipeline.models.fetch import resolve_models_source
from tests.log_assertions import OUTPUT

pytestmark = pytest.mark.unit  # noqa: V107

HF_REPO = "neurospin/Champollion_V1"
MASKS = "canonical_25"
OTHER_MASKS = "canonical_corrected_26_1"


def _refuse_network(*_args, **_kwargs):
    raise AssertionError("network access: a Hugging Face hub call was made")


def refuse_hub(monkeypatch) -> None:
    """Replace the Hugging Face hub module of models/fetch.py with one whose every call fails the test."""
    monkeypatch.setattr(
        models_fetch, "huggingface_hub", SimpleNamespace(snapshot_download=_refuse_network, HfApi=_refuse_network)
    )


@pytest.fixture(autouse=True)  # noqa: V103
def _no_network(monkeypatch):
    """No Hugging Face hub call reaches the network in this module."""
    refuse_hub(monkeypatch)


def make_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir(exist_ok=True)
    return checkout


def make_default_models(tmp_path: Path) -> Path:
    folder = make_checkout(tmp_path) / "models"
    folder.mkdir()
    return folder


def make_cache_entry(models_cache: Path, masks: str = MASKS) -> Path:
    """The folder HuggingFaceStrategy leaves for the model set masks: <models_cache>/Champollion_V1_<masks>/<masks>."""
    entry = models_cache / f"Champollion_V1_{masks}" / masks
    entry.mkdir(parents=True)
    return entry


class TestResolver:
    """resolve_models_source in champollion_pipeline.models.fetch (component Requirements)."""

    def test_default_models_folder(self, tmp_path):
        """REQ-MODELSRC-BDRABCZUK-0C79BCA65A4C."""
        default = make_default_models(tmp_path)
        cache = tmp_path / "models_cache"
        make_cache_entry(cache)

        assert resolve_models_source("", str(make_checkout(tmp_path)), str(cache), MASKS) == str(default)

    def test_models_cache_entry(self, tmp_path):
        """REQ-MODELSRC-BDRABCZUK-7D7429F79BA7: reused with no network call."""
        cache = tmp_path / "models_cache"
        entry = make_cache_entry(cache)

        assert resolve_models_source("", str(make_checkout(tmp_path)), str(cache), MASKS) == str(entry)

    def test_models_cache_entry_of_other_masks_ignored(self, tmp_path):
        """REQ-MODELSRC-BDRABCZUK-7D7429F79BA7: only the entry of the --masks model set counts."""
        cache = tmp_path / "models_cache"
        make_cache_entry(cache, OTHER_MASKS)

        assert resolve_models_source("", str(make_checkout(tmp_path)), str(cache), MASKS) == HF_REPO

    def test_hugging_face_repository(self, tmp_path):
        """REQ-MODELSRC-BDRABCZUK-A4F384DE609B."""
        cache = tmp_path / "models_cache"

        assert resolve_models_source("", str(make_checkout(tmp_path)), str(cache), MASKS) == HF_REPO

    def test_explicit_models_path_unchanged(self, tmp_path):
        """REQ-MODELSRC-BDRABCZUK-6FA95D517AD7."""
        explicit = tmp_path / "my_models"
        explicit.mkdir()
        make_default_models(tmp_path)

        result = resolve_models_source(
            str(explicit), str(make_checkout(tmp_path)), str(tmp_path / "models_cache"), MASKS
        )

        assert result == str(explicit)

    def test_missing_explicit_models_path_unchanged(self, tmp_path):
        """REQ-MODELSRC-BDRABCZUK-6FA95D517AD7: a missing explicit path is not replaced by a fallback."""
        typo = str(tmp_path / "no_such_models")
        cache = tmp_path / "models_cache"
        make_default_models(tmp_path)
        make_cache_entry(cache)

        assert resolve_models_source(typo, str(make_checkout(tmp_path)), str(cache), MASKS) == typo

    def test_source_logged_at_output(self, tmp_path, pipeline_caplog):
        """REQ-MODELSRC-BDRABCZUK-BCE9C3F46987."""
        cache = tmp_path / "models_cache"
        entry = make_cache_entry(cache)

        source = resolve_models_source("", str(make_checkout(tmp_path)), str(cache), MASKS)

        assert source == str(entry)
        outputs = [
            r.getMessage()
            for r in pipeline_caplog.records
            if r.name == models_fetch.logger.name and r.levelno == OUTPUT
        ]
        assert any(source in m for m in outputs), f"no OUTPUT record naming {source!r}; OUTPUT records: {outputs!r}"
