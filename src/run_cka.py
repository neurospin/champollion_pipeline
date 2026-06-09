#!/usr/bin/env python3
"""
Compare two sets of champollion embeddings using CKA coherence.

Single-pair mode (both paths are CSV/PT files):
    python3 src/run_cka.py --path_a A.csv --path_b B.csv --output_dir /tmp/cka

Cross-directory mode (both paths are directories of per-region models):
    python3 src/run_cka.py \
        --path_a /models/Champollion_V1_script/ \
        --subpath_a UKBioBank_embeddings_best_model/full_embeddings.csv \
        --path_b /models/Champollion_V1_after_ablation/ \
        --subpath_b ukb40_random_embeddings/full_embeddings.csv \
        --output_dir /tmp/cka_comparison
"""

import sys
from os import makedirs
from os.path import abspath, dirname, isdir, isfile, join

from champollion_utils.script_builder import ScriptBuilder

CKA_MODULE = "contrastive.evaluation.cka_coherence"


class RunCKA(ScriptBuilder):

    def __init__(self):
        super().__init__(
            script_name="run_cka",
            description="Compare two sets of champollion embeddings using CKA coherence.",
        )
        (
            self.add_required_argument("--path_a", "Path to first embeddings (CSV/PT file or model root directory).")
             .add_required_argument("--path_b", "Path to second embeddings (CSV/PT file or model root directory).")
             .add_required_argument("--output_dir", "Directory to write CKA results.")
             .add_optional_argument("--name_a", "Label for the first set of embeddings.", default="A")
             .add_optional_argument("--name_b", "Label for the second set of embeddings.", default="B")
             .add_optional_argument(
                "--subpath_a",
                "Sub-path appended to each region directory under --path_a when in directory mode "
                "(e.g. 'UKBioBank_embeddings_best_model/full_embeddings.csv').",
                default="full_embeddings.csv",
            )
             .add_optional_argument(
                "--subpath_b",
                "Sub-path appended to each region directory under --path_b when in directory mode "
                "(e.g. 'ukb40_random_embeddings/full_embeddings.csv').",
                default="full_embeddings.csv",
            )
             .add_optional_argument(
                "--subject_column",
                "Subject ID column name in CSV files.",
                default="ID",
            )
             .add_optional_argument(
                "--region",
                "Run CKA on a single named region only (directory mode).",
                default=None,
            )
        )

    def _cka_cmd(self, file_a: str, file_b: str, out: str) -> list:
        return [
            sys.executable, "-m", CKA_MODULE,
            f"{self.args.name_a}:{file_a}",
            f"{self.args.name_b}:{file_b}",
            "--output-dir", out,
            "--subject-column", self.args.subject_column,
        ]

    def run(self):
        path_a = self.args.path_a
        path_b = self.args.path_b
        makedirs(self.args.output_dir, exist_ok=True)

        if not isdir(path_a) and not isdir(path_b):
            # Single-pair mode
            if not isfile(path_a):
                raise FileNotFoundError(f"--path_a not found: {path_a}")
            if not isfile(path_b):
                raise FileNotFoundError(f"--path_b not found: {path_b}")
            return self.execute_command(
                self._cka_cmd(path_a, path_b, self.args.output_dir)
            )

        # Directory mode: iterate over region subdirectories
        if not isdir(path_a):
            raise NotADirectoryError(f"--path_a must be a directory in directory mode: {path_a}")
        if not isdir(path_b):
            raise NotADirectoryError(f"--path_b must be a directory in directory mode: {path_b}")

        import os
        regions = sorted(
            d for d in os.listdir(path_a)
            if isdir(join(path_a, d))
        )
        if self.args.region:
            regions = [r for r in regions if r == self.args.region]
            if not regions:
                raise ValueError(f"Region '{self.args.region}' not found under {path_a}")

        found = 0
        for region in regions:
            file_a = join(path_a, region, self.args.subpath_a)
            file_b = join(path_b, region, self.args.subpath_b)
            if not isfile(file_a):
                print(f"[skip] {region}: missing {file_a}")
                continue
            if not isfile(file_b):
                print(f"[skip] {region}: missing {file_b}")
                continue
            print(f"\n{'='*60}\nRegion: {region}\n{'='*60}")
            out = join(self.args.output_dir, region)
            rc = self.execute_command(self._cka_cmd(file_a, file_b, out))
            if rc != 0:
                print(f"[warn] CKA failed for {region} (exit code {rc})")
            found += 1

        if found == 0:
            raise RuntimeError(
                "No matching embedding pairs found. "
                "Check --subpath_a / --subpath_b point to existing files."
            )
        print(f"\nCKA completed for {found}/{len(regions)} regions. Results in {self.args.output_dir}")
        return 0


def main():
    script = RunCKA()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
