"""LOGGING convention in champollion_pipeline.config (TASK-PYTHONIC-BDRABCZUK-17F097040EE7).

REQ-LOGGING-BDRABCZUK-154038A7FC54: a non-empty dataset.hf_token left out of the saved YAML -> WARNING
    with the output file path followed by HF_TOKEN, on the champollion_pipeline.config logger.
REQ-LOGGING-BDRABCZUK-C96C8DAAF703 / -22ABEF2F2CBD (no print call, module logger) are checked by
tests/test_logging_convention.py through LOG_CONVERTED_MODULES.
"""

import logging

from champollion_pipeline.config import ConfigLoader, PipelineConfig

SECRET = "hf_SECRETtokenVALUE1234567890"


class TestSaveToYamlLogging:
    def test_unwritten_hf_token_logged_at_warning(self, tmp_path, pipeline_caplog):
        config = PipelineConfig()
        config.dataset.hf_token = SECRET  # noqa: V101
        output = str(tmp_path / "pipeline_config.yaml")

        ConfigLoader.save_to_yaml(config, output)

        record = pipeline_caplog.assert_logged(logging.WARNING, output)
        assert record.name == "champollion_pipeline.config"
        message = record.getMessage()
        assert "HF_TOKEN" in message[message.index(output) + len(output) :]
        assert SECRET not in message
