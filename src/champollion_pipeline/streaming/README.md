# streaming

## Purpose

The streaming mode (`champollion-pipeline --mode streaming`): it runs inference scan by scan instead of stage by
stage. Producer workers run Morphologist (when given a raw scan) and cortical_tiles for one scan each, in a
process pool. One consumer keeps a single embedding engine loaded and embeds each scan as soon as its crops
are there. When the producers are done and the consumer has drained, the per-region embeddings are combined
once.

## Files

| File | What it holds |
|------|---------------|
| `__init__.py` | Package marker; exports `StreamingSettings`. |
| `barrier.py` | The arrival barrier: yields scans as their inputs arrive, and tells whether a scan is already embedded in every region. |
| `combine.py` | Runs the combine step once, after the consumer has drained, with the same path rules as `main.py`. |
| `consumer.py` | The consumer: embeds scans through one resident embedding engine as their crops arrive; skips scans already embedded and marks a bad scan FAILED. |
| `dataset_config.py` | Generates the dataset's Champollion config once, before inference starts. |
| `runner.py` | The orchestrator of a run: starts the producer pool, the consumer and the combine step, and returns the exit code. |
| `settings.py` | `StreamingSettings`: the run-wide settings shared by the orchestrator, the workers and the consumer. |
| `stages.py` | Runs Morphologist and cortical_tiles for one scan, and computes the per-scan tiles directory and its crops root. |
| `worker.py` | The producer worker: runs a scan's stages, captures their output into a log, writes the completion marker, and never raises. |

## Entry points

- `runner.run_parallel_pipeline`: runs one streaming run and returns its exit code; the CLI calls it.
- `settings.StreamingSettings`: the settings of a run.
- `worker.run_scan_worker`: the producer work for one scan.
- `consumer.run_consumer`: embeds scans as their crops arrive.
- `combine.run_combine`: combines the per-region embeddings of a run.

## See also

- [Streaming mode (usage)](../../../docs/usage.md#streaming-mode) - options, logs, resume and exit codes.
- [Streaming mode (workflow)](../../../docs/workflow.md) - producers, barrier, consumer and combine in order.
- [Streaming profile](../../../docs/streaming_profile.md) - what is measured about a streaming run.
- [Internals](../../../docs/internals.md) - module map of the package.
