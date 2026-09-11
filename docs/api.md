# Module Reference

This page documents the public Python interface of `champollion_pipeline`.
All pipeline scripts subclass `ScriptBuilder` (from `champollion_utils`),
which provides a fluent `argparse` builder API and automatic update checks on startup.

## Class diagrams

### ScriptBuilder hierarchy — pipeline scripts

```mermaid
classDiagram
    class ScriptBuilder {
        <<champollion_utils>>
        +run()
        +check_for_updates()
    }
    class GenerateMorphologistGraphs {
        +run()
    }
    class RunCorticalTiles {
        +run()
    }
    class GenerateChampollionConfig {
        +run()
    }
    class GenerateEmbeddings {
        +fetch_models(models_path)
        +run()
    }
    class PutTogetherEmbeddings {
        +run()
    }
    class GenerateSnapshots {
        +run()
    }
    class TrainChampollion {
        +run()
    }
    class GenerateMasks {
        +run()
    }
    class PruneFailedSubjects {
        +run()
    }
    class PurgeSubject {
        +run()
    }

    ScriptBuilder <|-- GenerateMorphologistGraphs
    ScriptBuilder <|-- RunCorticalTiles
    ScriptBuilder <|-- GenerateChampollionConfig
    ScriptBuilder <|-- GenerateEmbeddings
    ScriptBuilder <|-- PutTogetherEmbeddings
    ScriptBuilder <|-- GenerateSnapshots
    ScriptBuilder <|-- TrainChampollion
    ScriptBuilder <|-- GenerateMasks
    ScriptBuilder <|-- PruneFailedSubjects
    ScriptBuilder <|-- PurgeSubject
```

### Model fetch strategy (stage 4 — embeddings)

```mermaid
classDiagram
    class ModelFetchStrategy {
        <<abstract>>
        +fetch(models_path, extract_to, no_cache) str
    }
    class LocalPathStrategy {
        +fetch(models_path, extract_to, no_cache) str
    }
    class HuggingFaceStrategy {
        -subfolder str
        +fetch(models_path, extract_to, no_cache) str
    }
    class RemoteArchiveStrategy {
        +fetch(models_path, extract_to, no_cache) str
    }
    class InteractiveFallbackStrategy {
        +fetch(models_path, extract_to, no_cache) str
    }
    class GenerateEmbeddings {
        +fetch_models(models_path)
    }

    ModelFetchStrategy <|-- LocalPathStrategy
    ModelFetchStrategy <|-- HuggingFaceStrategy
    ModelFetchStrategy <|-- RemoteArchiveStrategy
    ModelFetchStrategy <|-- InteractiveFallbackStrategy
    GenerateEmbeddings --> ModelFetchStrategy : selects strategy
```

### Mask runner hierarchy (generate_masks)

```mermaid
classDiagram
    class MaskRunner {
        <<abstract>>
        +run()
    }
    class SerialRunner {
        +run()
    }
    class BufferedRunner {
        +run()
    }
    class RunConfig {
        +input_dir str
        +output_dir str
        +njobs int
    }
    class GenerateMasks {
        +run()
    }

    MaskRunner <|-- SerialRunner
    MaskRunner <|-- BufferedRunner
    GenerateMasks --> MaskRunner : delegates
    GenerateMasks --> RunConfig : reads
```

### Cortical tiles config (utils)

```mermaid
classDiagram
    class CorticalTilesConfig {
        +path_to_graph str
        +path_sk_with_hull str
    }
    class CorticalTilesConfigFactory {
        +from_args(args) CorticalTilesConfig
    }
    class RunCorticalTiles {
        +run()
    }

    CorticalTilesConfigFactory --> CorticalTilesConfig : creates
    RunCorticalTiles --> CorticalTilesConfig : uses
```

### Pipeline data flow

```mermaid
sequenceDiagram
    participant MRI as Raw T1 MRI
    participant M as Stage 1 — Morphologist
    participant CT as Stage 2 — Cortical tiles
    participant CC as Stage 3 — Config
    participant E as Stage 4 — Embeddings
    participant C as Stage 5 — Combine
    participant S as Stage 6 — Snapshots

    MRI->>M: NIfTI files
    M->>CT: .arg sulcal graph files
    CT->>CC: crops/2mm/ + pipeline_loop_2mm.json
    CC->>E: reference.yaml / local.yaml
    E->>C: 56 × full_embeddings.csv
    C->>S: combined embeddings dir
    S->>S: UMAP projection onto UKBioBank reference
```

---

## Module autodoc

### generate_champollion_config

```{eval-rst}
.. automodule:: champollion_pipeline.generate_champollion_config
   :members:
   :undoc-members:
   :show-inheritance:
```

### generate_embeddings

```{eval-rst}
.. automodule:: champollion_pipeline.generate_embeddings
   :members:
   :undoc-members:
   :show-inheritance:
```

### generate_morphologist_graphs

```{eval-rst}
.. automodule:: champollion_pipeline.generate_morphologist_graphs
   :members:
   :undoc-members:
   :show-inheritance:
```

### generate_snapshots

```{eval-rst}
.. automodule:: champollion_pipeline.generate_snapshots
   :members:
   :undoc-members:
   :show-inheritance:
```

### prune_failed_subjects

```{eval-rst}
.. automodule:: champollion_pipeline.prune_failed_subjects
   :members:
   :undoc-members:
   :show-inheritance:
```

### purge_subject

```{eval-rst}
.. automodule:: champollion_pipeline.purge_subject
   :members:
   :undoc-members:
   :show-inheritance:
```

### put_together_embeddings

```{eval-rst}
.. automodule:: champollion_pipeline.put_together_embeddings
   :members:
   :undoc-members:
   :show-inheritance:
```

### run_cortical_tiles

```{eval-rst}
.. automodule:: champollion_pipeline.run_cortical_tiles
   :members:
   :undoc-members:
   :show-inheritance:
```

### train_champollion

```{eval-rst}
.. automodule:: champollion_pipeline.train_champollion
   :members:
   :undoc-members:
   :show-inheritance:
```
