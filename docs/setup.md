# Setup

This page maps your **location** (Local or Jean-Zay) and **use case** to the
correct `pixi` environment and install command.

> **Interactive wizard**: run `pixi run setup` (or `./install.sh`) for a
> guided prompt that selects the right command for you.

## Decision tree

```mermaid
flowchart TD
    A[Start] --> B{Where running?}
    B -->|Local| C{Use case?}
    B -->|Jean-Zay| JZ{Use case?}
    C -->|Full pipeline| E["pixi run install-all"]
    C -->|Embeddings inference| F["pixi run -e embeddings install-embeddings"]
    C -->|Training| T["pixi run -e training install-embeddings"]
    C -->|Docs/dev| G["pixi run -e docs build-docs"]
    C -->|Everything| H["install-all + build-docs"]
    JZ -->|Embeddings inference| JZE["pixi run -e embeddings install-embeddings"]
    JZ -->|Training| JZT["pixi run -e training install-embeddings"]
    JZ -->|Docs| JZD["pixi run -e docs build-docs"]
    JZE --> I[See slurm/ for SLURM scripts]
    JZT --> I
```

> **Jean-Zay note**: always pass `-e <env>`. Never use the default environment.

## Summary table

| Use case | Environment | Install command |
|---|---|---|
| Full pipeline | `default` | `pixi run install-all` |
| Embeddings inference | `embeddings` | `pixi run -e embeddings install-embeddings` |
| Training | `training` | `pixi run -e training install-embeddings` |
| Docs / development | `docs` | `pixi run -e docs build-docs` |
