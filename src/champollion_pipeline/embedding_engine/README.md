# embedding_engine

## Purpose

An in-process embedding engine. It loads each region encoder once, keeps it in memory, and embeds scans in
batches. It reuses champollion_V1's own loading and embedding functions, so a region is embedded as
`evaluate.py` would embed it, but without one Python process and one model load per region. The streaming
consumer uses it to keep the models loaded for the whole run. One engine serves one consumer: it is not
thread-safe.

## Files

| File | What it holds |
|------|---------------|
| `__init__.py` | Package marker; exports `EmbeddingEngine`. |
| `engine.py` | `EmbeddingEngine`: finds the models under a models path, loads each encoder on first use, and embeds skeleton arrays or a crops directory into a table of `ID, dim1..dimK`. |

## Entry points

- `engine.EmbeddingEngine`: build it with a models path and a device, then call its embed method on a
  skeleton array, or its embed_crops method on a 2mm crops directory.

## See also

- [Streaming mode (workflow)](../../../docs/workflow.md) - how the streaming consumer keeps one engine resident.
- [Streaming mode (usage)](../../../docs/usage.md#streaming-mode) - options, logs and exit codes of a streaming run.
- [Internals](../../../docs/internals.md) - module map of the package.
