#!/usr/bin/env python3
"""
Compare two sets of champollion embeddings using CKA coherence.

Single-pair mode (both paths are CSV/PT files):
    python3 src/run_cka.py --path_a A.csv --path_b B.csv --output_dir /tmp/cka

Flat-directory mode (output of put_together_embeddings — CSVs directly in the dir):
    python3 src/run_cka.py \
        --path_a /combined/trained/ \
        --path_b /combined/reference/ \
        --output_dir /tmp/cka_comparison

Nested-directory mode (raw model directories with per-region subdirs):
    python3 src/run_cka.py \
        --path_a /models/Champollion_V1_script/ \
        --subpath_a UKBioBank_embeddings_best_model/full_embeddings.csv \
        --path_b /models/Champollion_V1_after_ablation/ \
        --subpath_b ukb40_random_embeddings/full_embeddings.csv \
        --output_dir /tmp/cka_comparison
"""

import os
import sys
from glob import glob
from os import makedirs
from os.path import basename, isdir, isfile, join

from champollion_utils.script_builder import ScriptBuilder

CKA_MODULE = "contrastive.evaluation.cka_coherence"


class RunCKA(ScriptBuilder):

    def __init__(self):
        super().__init__(
            script_name="run_cka",
            description="Compare two sets of champollion embeddings using CKA coherence.",
        )
        (
            self.add_required_argument("--path_a", "Path to first embeddings (CSV/PT file or directory).")
             .add_required_argument("--path_b", "Path to second embeddings (CSV/PT file or directory).")
             .add_required_argument("--output_dir", "Directory to write CKA results.")
             .add_optional_argument("--name_a", "Label for the first set of embeddings.", default="A")
             .add_optional_argument("--name_b", "Label for the second set of embeddings.", default="B")
             .add_optional_argument(
                "--subpath_a",
                "Sub-path appended to each region directory under --path_a in nested mode "
                "(e.g. 'UKBioBank_embeddings_best_model/full_embeddings.csv'). "
                "Ignored in flat-directory mode.",
                default="full_embeddings.csv",
            )
             .add_optional_argument(
                "--subpath_b",
                "Sub-path appended to each region directory under --path_b in nested mode "
                "(e.g. 'ukb40_random_embeddings/full_embeddings.csv'). "
                "Ignored in flat-directory mode.",
                default="full_embeddings.csv",
            )
             .add_optional_argument(
                "--subject_column",
                "Subject ID column name in CSV files.",
                default="ID",
            )
             .add_optional_argument(
                "--region",
                "Run CKA on a single region only (matched by filename stem in flat mode, "
                "or directory name in nested mode).",
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

    def _run_pairs(self, pairs: list) -> int:
        """Run CKA on a list of (label, file_a, file_b) tuples."""
        found = 0
        for label, file_a, file_b in pairs:
            print(f"\n{'='*60}\n{label}\n{'='*60}")
            out = join(self.args.output_dir, label)
            rc = self.execute_command(self._cka_cmd(file_a, file_b, out))
            if rc != 0:
                print(f"[warn] CKA failed for {label} (exit code {rc})")
            found += 1
        return found

    def run(self):
        path_a = self.args.path_a
        path_b = self.args.path_b
        makedirs(self.args.output_dir, exist_ok=True)

        # --- Single-pair mode ---
        if not isdir(path_a) and not isdir(path_b):
            if not isfile(path_a):
                raise FileNotFoundError(f"--path_a not found: {path_a}")
            if not isfile(path_b):
                raise FileNotFoundError(f"--path_b not found: {path_b}")
            return self.execute_command(
                self._cka_cmd(path_a, path_b, self.args.output_dir)
            )

        if not isdir(path_a):
            raise NotADirectoryError(f"--path_a must be a directory in directory mode: {path_a}")
        if not isdir(path_b):
            raise NotADirectoryError(f"--path_b must be a directory in directory mode: {path_b}")

        subdirs_a = [d for d in os.listdir(path_a) if isdir(join(path_a, d))]

        if subdirs_a:
            # --- Nested mode: path/{region}/{subpath} ---
            regions = sorted(subdirs_a)
            if self.args.region:
                regions = [r for r in regions if r == self.args.region]
                if not regions:
                    raise ValueError(f"Region '{self.args.region}' not found under {path_a}")

            pairs = []
            for region in regions:
                file_a = join(path_a, region, self.args.subpath_a)
                file_b = join(path_b, region, self.args.subpath_b)
                if not isfile(file_a):
                    print(f"[skip] {region}: missing {file_a}")
                    continue
                if not isfile(file_b):
                    print(f"[skip] {region}: missing {file_b}")
                    continue
                pairs.append((region, file_a, file_b))
        else:
            # --- Flat mode: put_together_embeddings output (*.csv directly in dir) ---
            csvs_a = {basename(p): p for p in glob(join(path_a, "*.csv"))}
            csvs_b = {basename(p): p for p in glob(join(path_b, "*.csv"))}
            common = sorted(set(csvs_a) & set(csvs_b))
            if self.args.region:
                common = [n for n in common if self.args.region in n]
                if not common:
                    raise ValueError(
                        f"No CSV matching region '{self.args.region}' found in both directories."
                    )
            pairs = [
                (name.removesuffix(".csv"), csvs_a[name], csvs_b[name])
                for name in common
            ]
            missing_b = sorted(set(csvs_a) - set(csvs_b))
            if missing_b:
                print(f"[info] {len(missing_b)} file(s) in path_a have no match in path_b (skipped)")

        if not pairs:
            raise RuntimeError(
                "No matching embedding pairs found. "
                "In nested mode, check --subpath_a / --subpath_b. "
                "In flat mode, both directories must contain identically named CSV files."
            )

        found = self._run_pairs(pairs)
        total = len(subdirs_a) if subdirs_a else len(csvs_a)
        print(f"\nCKA completed for {found}/{total} regions. Results in {self.args.output_dir}")
        return 0


def main():
    script = RunCKA()
    return script.build().print_args().run()


if __name__ == "__main__":
    exit(main())
