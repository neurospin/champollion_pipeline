"""Template-method base class for the pipeline stages run by main.py.

``PipelineStage.execute`` holds the one run scaffold (log start, build the argv, run the script, log the
result); each stage supplies its argv, its script and its messages through hooks.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from os.path import join
from typing import ClassVar, List, Optional

from champollion_utils.run_log import RunLogCapture
from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.config import PipelineConfig
from champollion_pipeline.derivatives_layout import RUN_LOG_TIMESTAMP_FORMAT


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""

    stage_name: str
    success: bool
    message: str
    return_code: int = 0


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    progress_message: ClassVar[str]
    success_message: ClassVar[str]
    failure_message: ClassVar[str]

    def __init__(self, name: str, config: PipelineConfig, logger: logging.Logger):
        self.name = name
        self.config = config
        self.logger = logger

    @abstractmethod
    def validate(self) -> bool:
        """Validate stage prerequisites."""
        pass

    @abstractmethod
    def build_args(self) -> List[str]:
        """Return the argv of the stage's script."""
        pass

    @abstractmethod
    def create_script(self) -> ScriptBuilder:
        """Return a fresh instance of the stage's script."""
        pass

    def precheck(self) -> Optional[StageResult]:
        """Return a result to stop the stage before any logging, or None to run it."""
        return None

    def run_log_path(self, script: ScriptBuilder) -> Optional[str]:
        """Run-log file: None without config.log, the script default for "", else DIR/<stage>_<stamp>.log."""
        log = self.config.log
        if not isinstance(log, str):
            return None
        if log == "":
            return script.default_log_path()
        return join(log, f"{self.name}_{datetime.now().astimezone().strftime(RUN_LOG_TIMESTAMP_FORMAT)}.log")

    def run_script(self, script: ScriptBuilder) -> int:
        """Run the parsed script and return its exit code."""
        return script.run()

    def execute(self) -> StageResult:
        """Execute the stage: precheck, log start, build and run the script, log the result."""
        early = self.precheck()
        if early is not None:
            return early
        self.log_start()
        try:
            self.logger.info(self.progress_message)
            args = self.build_args()
            script = self.create_script()
            script.parse_args(args)
            log_path = self.run_log_path(script)
            if log_path is None:
                return_code = self.run_script(script)
            else:
                with RunLogCapture(log_path) as capture:
                    return_code = self.run_script(script)
                    capture.exit_code = return_code if isinstance(return_code, int) else 1
            result = StageResult(
                stage_name=self.name,
                success=(return_code == 0),
                message=self.success_message if return_code == 0 else self.failure_message,
                return_code=return_code,
            )
        except Exception as e:
            self.log_failure()
            result = StageResult(stage_name=self.name, success=False, message=f"Failed: {str(e)}", return_code=1)

        self.log_end(result)
        return result

    def log_start(self):
        """Log stage start."""
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Starting stage: {self.name}")
        self.logger.info(f"{'=' * 60}")

    def log_failure(self):
        """Log the exception being handled as this stage's failure, with its traceback."""
        self.logger.fail(f"Stage {self.name} failed", exc_info=True, stacklevel=2)

    def log_end(self, result: StageResult):
        """Log stage end."""
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        self.logger.info(f"{status}: {self.name} - {result.message}")
        self.logger.info(f"{'=' * 60}\n")

    def subject_input_dir(self) -> Optional[str]:  # noqa: V105 - used by main.py
        """Return the subjects directory to index before this stage. None = skip."""
        return None

    def required_file_patterns(self) -> List[str]:  # noqa: V105 - used by main.py
        """Glob patterns (relative to subject dir) required per subject."""
        return []

    def output_dir(self) -> Optional[str]:
        """Return the output directory to index after this stage. None = skip."""
        return None
