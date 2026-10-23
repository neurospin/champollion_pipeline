#!/usr/bin/env python3
"""
Script to compare sulcal labeling between two graph annotation campaigns
(e.g. base2018_manual vs base2018b_manual) and optionally two mask sets.

For each subject found in either campaign, opens the corresponding graph file
and reports the number of voxels labeled per sulcus.  Outputs a CSV summary
with per-sulcus coverage (% subjects) and voxel density (voxels/subject), and
optionally extends the comparison with mask-level statistics.

Usage
-----
    pixi run compare-databases \\
      --labeled_subjects_dir /neurospin/.../manually_labeled/pclean/all \\
      --path_to_graph_a t1mri/t1/default_analysis/folds/3.3/base2018_manual \\
      --path_to_graph_b t1mri/t1/default_analysis/folds/3.3/base2018b_manual \\
      --label_a base2018 --label_b base2018b \\
      --masks_a external/cortical_tiles/data/mask/canonical_25/2mm \\
      --masks_b /path/to/sulci_masks_2018/2.0 \\
      --output /tmp/db_comparison.csv
"""

from datetime import datetime
from os.path import dirname

from champollion_utils.script_builder import ScriptBuilder

from champollion_pipeline.derivatives_layout import compute_run_log_path
from champollion_pipeline.process_setup import init_pipeline_process
from compare import run_subcommand

_OPTION_NAMES = (
    "labeled_subjects_dir",
    "path_to_graph_a",
    "path_to_graph_b",
    "label_a",
    "label_b",
    "side",
    "masks_a",
    "masks_b",
    "output",
    "njobs",
)


class CompareDatabases(ScriptBuilder):
    """Compare sulcal labeling between two graph annotation campaigns."""

    def __init__(self):
        super().__init__(
            script_name="compare_databases",
            description=(
                "Compare sulcal labeling coverage and voxel density between "
                "two graph annotation campaigns, with optional mask comparison."
            ),
        )
        (
            self.add_required_argument("--labeled_subjects_dir", "Root directory containing subject subdirectories.")
            .add_required_argument(
                "--path_to_graph_a",
                "Relative sub-path for campaign A (e.g. t1mri/t1/default_analysis/folds/3.3/base2018_manual).",
            )
            .add_required_argument(
                "--path_to_graph_b",
                "Relative sub-path for campaign B (e.g. t1mri/t1/default_analysis/folds/3.3/base2018b_manual).",
            )
            .add_optional_argument("--label_a", "Name for campaign A.", default="A")
            .add_optional_argument("--label_b", "Name for campaign B.", default="B")
            .add_optional_argument("--side", "Hemisphere side: L, R, or both.", default="both")
            .add_optional_argument(
                "--masks_a",
                "Mask directory for campaign A (e.g. canonical_25/2mm). Optional — skipped if not provided.",
                default=None,
            )
            .add_optional_argument("--masks_b", "Mask directory for campaign B. Optional.", default=None)
            .add_optional_argument("--output", "Output CSV file path.", default="./db_comparison.csv")
            .add_optional_argument(
                "--njobs", "Parallel workers. Default: cpu_count - 2 (max 22).", default=None, type_=int
            )
        )

    def default_log_path(self) -> str:
        """Run-log path of a bare --log: <parent of --output>/.log/compare_databases_<stamp>.log."""
        return compute_run_log_path(dirname(self.args.output), self.script_name, datetime.now().astimezone())

    def run(self) -> int:
        return run_subcommand("databases", {name: getattr(self.args, name) for name in _OPTION_NAMES})


def main():
    init_pipeline_process()
    script = CompareDatabases()
    return script.main()


if __name__ == "__main__":
    raise SystemExit(main())
