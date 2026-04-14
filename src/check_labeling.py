#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to check sulcal labeling coverage per subject in a BrainVISA database.

For each subject, opens the graph file and reports which sulci have a 'name'
vertex attribute — i.e. which sulci have been manually labeled.

Usage
-----
    pixi run check-labeling \\
      --labeled_subjects_dir /path/to/db \\
      --path_to_graph t1mri/t1/default_analysis/folds/3.3/base2018b_manual \\
      --side both \\
      --output /tmp/labeling.csv

Output
------
A CSV (rows = subjects, columns = sulci, values = 0/1) and a summary table
printed to stdout showing for each sulcus how many subjects have it labeled.
"""

import csv
import os
import sys
from os.path import abspath, dirname, join

from champollion_utils.script_builder import ScriptBuilder


def _get_subject_labels(sub, brainvisa_dir):
    """Worker: load one subject graph and return (subject_name, set_of_labels).

    Returns (sub_name, None) if no graph file is found.
    """
    import glob as _glob
    import sys as _sys

    if brainvisa_dir not in _sys.path:
        _sys.path.insert(0, brainvisa_dir)
    from soma import aims  # noqa: PLC0415

    matches = _glob.glob(_join(sub['dir'], sub['graph_file']))
    if not matches:
        return sub['subject'], None

    graph = aims.read(matches[0])
    labeled = set()
    for vertex in graph.vertices():
        name = vertex.get('name')
        if name is not None:
            labeled.add(name)
    return sub['subject'], labeled


def _join(*args):
    from os.path import join as _j
    return _j(*args)


class CheckLabeling(ScriptBuilder):
    """Check sulcal labeling coverage per subject in a BrainVISA database."""

    def __init__(self):
        super().__init__(
            script_name="check_labeling",
            description="Report sulcal labeling coverage per subject in a BrainVISA database.",
        )
        (self
         .add_required_argument(
             "--labeled_subjects_dir",
             "Root directory containing subject subdirectories "
             "(e.g. /neurospin/dico/data/bv_databases/human/manually_labeled/pclean/all).")
         .add_required_argument(
             "--path_to_graph",
             "Relative sub-path from each subject dir to the graph files "
             "(e.g. t1mri/t1/default_analysis/folds/3.3/base2018b_manual).")
         .add_optional_argument(
             "--side",
             "Hemisphere side: L, R, or both.",
             default="both")
         .add_optional_argument(
             "--output",
             "Output CSV file path.",
             default="./subject_labeling.csv")
         .add_optional_argument(
             "--njobs",
             "Parallel workers. Default: cpu_count - 2 (max 22).",
             default=None, type_=int))

    def run(self):
        from joblib import Parallel, delayed, cpu_count
        from deep_folding.brainvisa.utils.subjects import get_all_subjects_as_dictionary

        brainvisa_dir = abspath(join(
            dirname(__file__),
            '..', 'external', 'cortical_tiles', 'deep_folding', 'brainvisa'
        ))
        if brainvisa_dir not in sys.path:
            sys.path.insert(0, brainvisa_dir)

        sides = ["L", "R"] if self.args.side == "both" else [self.args.side]

        njobs = self.args.njobs
        if njobs is None:
            njobs = max(1, min(22, cpu_count() - 2))

        # subject_name -> set of labeled sulci (union across all requested sides)
        all_subjects: dict = {}

        for side in sides:
            graph_file_pattern = (
                '%(subject)s/'
                + self.args.path_to_graph
                + '/%(side)s%(subject)s*.arg'
            )
            subjects = get_all_subjects_as_dictionary(
                [self.args.labeled_subjects_dir], [graph_file_pattern], side
            )
            print(f"  [{side}] Found {len(subjects)} subjects, "
                  f"loading graphs with {njobs} worker(s)…")

            results = Parallel(n_jobs=njobs, prefer='processes')(
                delayed(_get_subject_labels)(sub, brainvisa_dir)
                for sub in subjects
            )

            n_ok = 0
            for sub_name, labeled in results:
                if labeled is None:
                    print(f"  [{side}] WARNING: no graph for {sub_name}, skipped")
                    continue
                all_subjects.setdefault(sub_name, set()).update(labeled)
                n_ok += 1
            print(f"  [{side}] Done ({n_ok}/{len(subjects)} graphs loaded).")

        if not all_subjects:
            print("No subjects found or no graphs loaded.")
            return 1

        all_sulci = sorted({s for labels in all_subjects.values() for s in labels})
        sorted_subjects = sorted(all_subjects)

        # Write CSV
        output_path = abspath(self.args.output)
        out_dir = dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject"] + all_sulci)
            for sub in sorted_subjects:
                labeled = all_subjects[sub]
                writer.writerow([sub] + [1 if s in labeled else 0 for s in all_sulci])

        # Print summary table
        print(f"\nLabeling coverage ({len(sorted_subjects)} subjects, "
              f"{len(all_sulci)} distinct sulci):\n")
        print(f"  {'Sulcus':<50}  {'N':>4}  {'%':>6}")
        print("  " + "-" * 63)
        for sulcus in all_sulci:
            count = sum(1 for sub in sorted_subjects if sulcus in all_subjects[sub])
            pct = 100.0 * count / len(sorted_subjects)
            print(f"  {sulcus:<50}  {count:>4}  {pct:>5.1f}%")

        print(f"\nCSV written to: {output_path}")
        return 0


def main():
    script = CheckLabeling()
    return script.build().print_args().run()


if __name__ == "__main__":
    raise SystemExit(main())
