---
orphan: true
---

# Streaming profile: what is measured, and what is not yet

This page records the profiling of the embedding path that the streaming mode (see
[Streaming mode](usage.md)) is built on. It quotes only figures that exist in a recorded source. **No profiling run
of `scripts/profile_embedding_path.py` has been recorded yet**: no JSON report, log or ledger note holds a measured
model size, GPU memory, load time, inference time or stage cost. Every measurement below is therefore listed as an
open question, not as a number.

## What the profiling script measures

`scripts/profile_embedding_path.py MODELS CROPS_2MM --output report.json` writes one JSON report. It loads each
region model with champollion_V1's own `load_model`, runs it with `generate_embeddings`, from the same epoch-N
checkpoint the embeddings stage selects. The report holds:

| Report field | Meaning |
|---|---|
| `regions[].checkpoint_bytes` | size of the region's epoch-N checkpoint file |
| `regions[].model_load_seconds` | seconds to load the model and move it to the device |
| `regions[].inference_seconds_per_scan` | inference seconds per scan at `--batch-size` (default 32) |
| `regions[].peak_gpu_memory_bytes` | peak CUDA memory allocated; `null` on CPU |
| `regions[].input` | skeleton array path, shape, dtype and `bytes_per_scan` |
| `python_torch_startup_seconds` | seconds for a fresh interpreter to start and import torch |
| `stages` | per `--stage NAME=CMD` and `--scan`: seconds and exit code of the command |

Synthetic or OpenBHB data only; never UK Biobank data on a local machine.

## What the code and the ledger already establish

These facts come from the code and the recorded design notes, not from a measurement:

- The batch embeddings stage runs one `evaluate.py` subprocess per region and hemisphere, 56 in all (28 regions
  times 2 hemispheres). Each subprocess pays interpreter start, the torch import and a `torch.load`; the script's
  `python_torch_startup_seconds` field exists to quantify that per-process cost.
- Per scan, the input today is one `skeleton.npy` array per region: 28 region patterns per scan.
- The streaming inference engine loads each region model once per process, on first use, and keeps it resident:
  there is no eviction and no unload. Memory therefore grows with the regions used, at most one encoder per
  region. The engine logs one record per load, `<region>: encoder loaded in <seconds> s from <weights source>`.
- No residency or batching policy is settled; the engine design records "profiling numbers pending".
- Site GPU-memory thresholds are not settings or KPIs (user decision): the figures here are plain measurements with
  no threshold.

## Open questions

Each question is answered by one run of the script on a host where the models and crops are available. The run
must record the host, the date and the command, and quote the report values.

1. What is the checkpoint size, in bytes, of each of the region models, and what is their total?
2. What is the peak GPU memory, in bytes, of each region model at batch size 32, and of all resident models taken
   together?
3. What is the model load time per region, and how does it compare with the inference time per scan?
4. What is the inference time per scan per region at batch size 32, and how does it change with the batch size?
5. How many seconds does a fresh interpreter take to start and import torch (`python_torch_startup_seconds`), and
   what share of the 56-subprocess batch cost is that start-up plus model load?
6. What is the size per scan of each region's skeleton input (`bytes_per_scan`, shape and dtype)?
7. What is the Morphologist cost per scan and what is the cortical tiles cost per scan (`--stage` commands over
   `--scan` ids)?
8. Given questions 1 to 3, should the streaming engine keep all region models resident, or evict some? This
   decision is open until questions 1 to 3 are answered.
