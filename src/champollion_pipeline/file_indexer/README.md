# file_indexer

## Purpose

In-memory indexes of the files of a directory, used to check what a stage needs and what it wrote. A B-tree
keyed by relative path answers prefix queries. The streaming arrival barrier uses it to tell when a scan's
inputs are there; the sequential stages use it to check subjects before a stage and to count a stage's
output files after it. Patterns and paths come from the user's configuration, so they are checked, never
trusted.

## Files

| File | What it holds |
|------|---------------|
| `__init__.py` | Package marker; exports `ScanId` and `SubjectEligibilityChecker`. |
| `btree.py` | `BTree` and `BTreeNode`: a B-tree keyed by relative path strings, with prefix range queries and a JSON form that older index files still load from. |
| `pipeline_checks.py` | Per-scan eligibility checks over an index of a subjects directory, output reports of a stage, and `FileIndexStageIndexer`, the default indexer of the sequential stages. |
| `scan_id.py` | `ScanId`: the identity of one scan in Morphologist's layout (subject, acquisition folder). |

## Entry points

- `pipeline_checks.SubjectEligibilityChecker`: indexes a subjects directory, lists its scans and checks the
  required patterns of each one.
- `pipeline_checks.FileIndexStageIndexer`: writes the pre and post index files of each sequential stage.
- `pipeline_checks.build_output_report`: indexes a stage's output directory and counts its files by extension.
- `scan_id.ScanId`: names one scan.
- `btree.BTree`: the index itself.

## See also

- [Streaming mode (workflow)](../../../docs/workflow.md) - the arrival barrier that polls the checker.
- [Streaming mode (usage)](../../../docs/usage.md#streaming-mode) - the subjects layout a streaming run reads.
- [Internals](../../../docs/internals.md) - module map and path conventions.
