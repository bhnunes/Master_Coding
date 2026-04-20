# Master_Coding

A Python research pipeline for pathology whole-slide-image (WSI) processing. It provides a complete end-to-end workflow for extracting patches from medical images, preparing datasets, training deep learning models, and generating predictions for cancer detection in histopathology images.

## Key Capabilities

- Whole-slide image processing (supports `.svs`, `.ndpi`, `.tiff` formats)
- Annotation handling for multiple formats (XML, NDPA, JSON)
- Dataset-aware Stage 2 SVS/XML parsing, including HISEG ASAP-style XML support
- Patch extraction with tissue detection and artifact filtering
- Patient-level stratified dataset splitting on HDF5 datasets
- Train-fitted stain normalization support
- Metadata-first downstream data preparation from canonical Stage 2 patient shards
- Ensemble model training and inference
- GPU-accelerated deep learning with PyTorch

## Folder Structure

```
Master_Coding/
├── 1_artifact_detection.py          # Generate artifact GeoJSON from WSIs
├── 2_database_manager.py            # Database orchestration & case processing
├── 4_1_optimization_sampling.py     # Select samples for human-in-the-loop cleaning
├── 4_2_tune_graph_method.py         # Tune graph segmentation parameters
├── 4_3_cleaner_script.py            # Apply cleaning to remove incorrect annotations
├── 5_crossfold.py                   # HDF5-native patient-level dataset splitting
├── 6_sanity_checks.py               # HDF5-native scientific integrity checks for Stage 5 singleton splits
├── 7_smart_sampler.py               # Select informative TRAIN rows via master_manifest.sqlite
├── 8_lr_finder.py                   # Find optimal learning rates
├── 9_training_ensemble.py           # Train one approved model per execution
├── 10_optimizer_ensemble.py         # Optimize ensemble parameters
├── 11_inference_ensemble.py         # Generate predictions on test set
│
├── helpers/
│   ├── __init__.py
│   ├── logging_utils.py             # Shared logging helpers
│   ├── runtime_platform.py          # Cross-platform runtime/path utilities
│   ├── artifact/                    # Stage 1 artifact detection domain
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── model_loader.py
│   │   ├── paths.py
│   │   ├── pipeline.py
│   │   ├── processor.py
│   │   ├── repository.py
│   │   └── zip.py
│   ├── extraction/                  # Stage 2 extraction domain
│   │   ├── artifact_index.py
│   │   ├── config.py
│   │   ├── data_handlers.py
│   │   ├── image_reader_service.py
│   │   ├── patch_engine.py
│   │   └── repository.py
│   ├── optimization_sampling/       # Stage 4.1 sampling domain
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── overlay.py
│   │   ├── pipeline.py
│   │   └── sampling.py
│   ├── graph/                       # Stage 4.2/4.3 graph cleaning domain
│   │   ├── cleaning_config.py
│   │   ├── cleaning_pipeline.py
│   │   ├── contamination.py
│   │   ├── parameter_store.py
│   │   ├── tuning_config.py
│   │   └── tuning_pipeline.py
│   ├── packaging/                   # Obsolete Stage 3 packaging domain
│   │   ├── config.py
│   │   ├── discovery.py
│   │   ├── pipeline.py
│   │   └── writer.py
│   ├── crossfold/                   # Stage 5 dataset preparation domain
│   │   ├── config.py
│   │   ├── discovery.py
│   │   ├── entropy.py
│   │   ├── io.py
│   │   ├── logging.py
│   │   ├── normalization.py
│   │   ├── pipeline.py
│   │   ├── provenance.py
│   │   └── splitting.py
│   ├── sanity/                      # Stage 6 HDF5 sanity-check domain
│   │   ├── config.py
│   │   ├── contracts.py
│   │   ├── disk_checks.py
│   │   ├── manifest_checks.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── provenance.py
│   │   ├── reporting.py
│   │   └── semantic_checks.py
│   ├── smart_sampling/              # Stage 7 smart-sampling domain
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── index.py
│   │   ├── pipeline.py
│   │   ├── selection.py
│   │   ├── storage.py
│   │   └── writer.py
│   ├── patient_shard_cache.py       # Shared Drive-to-local patient-shard cache
│   ├── lr_finder/                   # Stage 8 LR-finder domain
│   │   ├── analysis.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── pipeline.py
│   │   ├── reporting.py
│   │   ├── runner.py
│   │   └── search_space.py
│   ├── training/                    # Stage 9 training domain
│   │   ├── checkpointing.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── gpu.py
│   │   ├── loop.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── registry.py
│   │   ├── reporting.py
│   │   ├── runtime.py
│   │   └── utils.py
│   ├── ensemble_optimizer/          # Stage 10 ensemble-optimizer domain
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── metadata.py
│   │   ├── models.py
│   │   ├── optimization.py
│   │   ├── pipeline.py
│   │   ├── reporting.py
│   │   └── splitting.py
│   ├── ensemble_inference/          # Stage 11 ensemble-inference domain
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── inference.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── recipe.py
│   │   └── reporting.py
│   ├── provenance.py                # Shared hashing/provenance helpers
│   ├── stage_contracts.py           # Shared Stage 5/6 contract constants
│   └── wsi/                         # Shared WSI/image-processing helpers
│       ├── colors.py
│       ├── maps.py
│       ├── process.py
│       ├── slide_info.py
│       └── tis_detect_helper_fx.py
│
├── tests/                           # Test scripts
├── databases/                       # SQLite databases
├── logs/                            # Log files
│
├── .env_example                     # Environment configuration template
├── setup_windows.ps1                # Native Windows bootstrap
├── setup_colab.sh                   # One-step Google Colab bootstrap
├── colab_setup.md                   # Google Colab usage guide
├── pyproject.toml                   # Project configuration (uv)
├── Dockerfile                       # Container definition
└── AGENTS.md                        # Developer guidelines
```

## Pipeline Architecture

The pipeline follows a script-driven architecture where each root script performs a distinct stage of the workflow. Scripts are designed to be standalone applications that can be executed independently, with defined inputs and outputs.

Helper modules are organized by domain under `helpers/<domain>/`. New domain-specific helpers should be added to the matching domain package instead of recreating a flat `helpers/*.py` layout. Only broadly shared cross-domain utilities, such as `helpers/runtime_platform.py`, remain at the top level.

### Pipeline Stages

| Stage | Script(s) | Description |
|-------|-----------|-------------|
| 1 | `1_artifact_detection.py` | Detect artifacts on whole-slide images using a `.env`-driven Stage 1 pipeline. Output: GeoJSON files with artifact annotations and SQLite processing status |
| 2 | `2_database_manager.py` | Extract canonical HDF5 patch shards from WSIs, populate `master_manifest.sqlite`, and optionally emit PNG exports when explicitly enabled |
| 4.1-4.3 | `4_1_optimization_sampling.py` → `4_2_tune_graph_method.py` → `4_3_cleaner_script.py` | HDF5-backed human-in-the-loop review plus graph-based cleaning, with PNG retained only for review/export workflows |
| 5 | `5_crossfold.py` | Create Stage 5 split assignments in `master_manifest.sqlite`, emit split/normalization sidecars, and fit stain normalization on `TRAIN` only |
| 6 | `6_sanity_checks.py` | Validate Stage 5 singleton split integrity, provenance, leakage, and mask/label semantics |
| 6.5 / 7.1 | Removed | Obsolete in the metadata-first pipeline |
| 7.2 | `7_smart_sampler.py` | Select informative TRAIN rows in `master_manifest.sqlite` and write lineage sidecars without materializing filtered shards |
| 8-11 | `8_lr_finder.py` → `9_training_ensemble.py` → `10_optimizer_ensemble.py` → `11_inference_ensemble.py` | Resolve runtime rows from `master_manifest.sqlite` once at startup, then load pixels directly from canonical Stage 2 patient shards |

## Script Documentation

### Stage 1: Artifact Detection

| Script | Purpose |
|--------|---------|
| `1_artifact_detection.py` | Thin Stage 1 orchestrator that loads `.env`, initializes logging, models, zip access, and SQLite tracking, then processes slides into GeoJSON outputs. |

Current Stage 1 behavior:

- Reads Stage 1 configuration from `.env` / `.env_example`
- Scans WSI files directly from a configured zip archive instead of extracting the full archive up front
- Extracts only one slide at a time into a temporary workspace during processing
- Stores and resumes progress through SQLite entries with fields such as `Image_Name`, `GeoJSON_Processed`, `Comments`, and operational metadata like `LastUpdate`
- Writes GeoJSON artifact annotations to the configured output folder
- Uses colorful console + file logging for long-running user-facing feedback

### Stage 2: Patch Extraction

| Script | Purpose |
|--------|---------|
| `2_database_manager.py` | Orchestrates the entire Stage 2 pipeline. Manages SQLite database, handles case ingestion, and invokes the shared extraction service for each slide. |
| `helpers/extraction/data_handlers.py` | Annotation format adapters for `.svs/.xml`, `.ndpi/.ndpa`, and JSON-based formats. |
| `helpers/extraction/patch_engine.py` | Core extraction engine: tissue detection, polygon masking, artifact coverage computation, and image/mask writing. |

Current Stage 2 artifact-aware behavior:

- `USE_ADVANCED_ARTIFACT_FILTERING=True` computes per-class artifact coverage for every saved patch instead of rejecting patches by threshold
- `USE_ADVANCED_ARTIFACT_FILTERING=False` skips artifact geometry work for speed, but still writes the same Parquet schema with zero-valued coverage columns
- Stage 2 writes compact filename-keyed artifact coverage fields into `master_manifest.sqlite`
- Stage 9 can join those SQLite-backed artifact coverage values with runtime filenames for artifact-aware loss discounting during training

Current Stage 2 multi-dataset XML behavior:

- `.svs/.xml` support is dataset-aware and selected through `TAG`
- `.svs/.xml` is supported only for `TAG=HISEG` and `TAG=Chile`
- Unsupported `.svs/.xml` tags now fail fast with an explicit error
- `TAG=HISEG` enables the HISEG-specific ASAP XML path in `helpers/extraction/data_handlers.py`
- `TAG=Chile` enables the CHILE-specific XML `LineColor` path in `helpers/extraction/data_handlers.py`
- HISEG reads `Annotation Color="#..."` plus `Coordinates/Coordinate` entries instead of the legacy `LineColor` + `Region/Vertex` layout
- Stage 2 now resolves supported SVS/XML label colors internally in code rather than through SQLite columns
- HISEG color policy is:
  - cancer: `#8B0000`, `#FF00FF`, `#800080`
  - not_cancer: `#8A2BE2`, `#0000FF`, `#4682B4`, `#00FF00`, `#008000`, `#FFFF00`
  - rejected: `#4B0082` (skipped entirely)

### Stage 4: Annotation Cleaning

| Script | Purpose |
|--------|---------|
| `4_1_optimization_sampling.py` | Selects a statistically representative subsample of cancer images for human-in-the-loop review. Uses Cochran's formula for sample size calculation. |
| `4_2_tune_graph_method.py` | Uses Bayesian optimization to find optimal graph segmentation parameters for distinguishing correct vs. incorrect annotations. |
| `4_3_cleaner_script.py` | Applies the tuned graph segmentation method to automatically remove incorrectly annotated HDF5 rows through accepted/rejected manifests. Requires the JSON parameter artifact produced by `4_2_tune_graph_method.py`. |

Current Stage 4 behavior:

- Stage 4.2 uses `skimage.segmentation.felzenszwalb` rather than the old OpenCV contrib graph-segmentation dependency
- Stage 4.2 reuses per-record contamination scores across grouped CV folds instead of rescoring the same record once per fold
- Stage 4.3 fails fast unless `GRAPH_CLEANING_PARAMS_PATH` points to the JSON artifact emitted by Stage 4.2
- Stage 4.3 now reuses HDF5 `source_signature` during candidate discovery when available instead of hashing the full source HDF5 on the hot path
- Stage 4.3 now uses contiguous batched HDF5 image/mask reads during cleaning; current best-known default behavior uses a `512`-row batch fast path in `helpers/graph/cleaning_pipeline.py`
- On the measured sample source HDF5, larger contiguous HDF5 batches beat a simple multiprocessing prototype, so the current implementation favors larger batched reads over extra parallel complexity

### Stage 5: Dataset Preparation

| Script | Purpose |
|--------|---------|
| `5_crossfold.py` | Thin Stage 5 orchestrator that loads `.env`, reads `master_manifest.sqlite`, builds patient-level TRAIN/VALIDATION/TEST split assignments, optionally fits stain normalization on TRAIN only, and writes provenance artifacts. |

Current Stage 5 behavior:

- Loads Stage 5 settings from `.env` / `.env_example` through `helpers/crossfold/config.py`
- Reads accepted canonical rows from `master_manifest.sqlite`
- Preserves patient-level split isolation and stratifies patients by `max(patch_label)`
- Uses explicit patient capacities for `TEST` and `VALIDATION`; `TRAIN` receives the remainder
- Carries a Stage 5 Optuna split-search budget via `CROSSFOLD_SPLIT_OPTUNA_TRIALS`
- Reuses cached source-HDF5 provenance inside one run instead of rehashing the same source file for every split artifact
- Uses batched HDF5-backed entropy reads when the Stage 5 source is an HDF5 dataset
- Fits stain normalization on TRAIN only when a normalization method other than `NOT_NORMALIZED` is configured
- Applies the frozen TRAIN-fitted normalizer to `TRAIN`, `VALIDATION`, and `TEST`
- Emits lightweight log-based progress for entropy and split selection
- Writes `manifest.csv`, `split_stats.csv`, `run_config.json`, optional entropy-cache CSV artifacts, and SQLite split / normalization state for traceability

### Stage 6: Quality Assurance

| Script | Purpose |
|--------|---------|
| `6_sanity_checks.py` | Thin Stage 6 orchestrator that loads `.env`, validates Stage 5 provenance sidecars and scientific split integrity, and prints a reviewer-facing PASS/WARN/FAIL report. |

Current Stage 6 behavior:

- Loads Stage 6 settings from `.env` / `.env_example` through `helpers/sanity/config.py`
- Uses `manifest.csv` as the source of truth and cross-checks `run_config.json` and `split_stats.csv`
- Fails on patient leakage, duplicate manifest rows, filename-contract violations, and manifest/disk mismatches
- Verifies HDF5 dataset parity, readable rows, image/mask shape agreement, and valid mask pixel values
- Adds scientific label checks such as empty positive masks and positive pixels inside negative masks
- Reports class balance and patches-per-patient skew to support reviewer interpretation

### Stage 6.5 / 7.1: Patient Shards

| Script | Purpose |
|--------|---------|
| Removed | Stage 6.5 / 7.1 has been removed from the active codebase. |

Current Stage 6.5 / 7.1 behavior:

- Stage 6.5 / 7.1 is no longer part of the active metadata-first pipeline
- The canonical pixel store is the Stage 2 patient shards plus `master_manifest.sqlite`

### Stage 7.2: Smart Sampling

| Script | Purpose |
|--------|---------|
| `7_smart_sampler.py` | Thin smart-sampling orchestrator that loads `.env`, filters TRAIN rows via `master_manifest.sqlite`, and writes selection/lineage sidecars. |

Current Stage 7.2 smart-sampling behavior:

- Loads smart-sampling settings from `.env` / `.env_example` through `helpers/smart_sampling/config.py`
- Builds a TRAIN patient index directly from `master_manifest.sqlite`, extracts embeddings, and selects diverse per-patient rows through `helpers/smart_sampling/*.py`
- Updates `sampling_decision` and `is_stage7_selected` in `master_manifest.sqlite` instead of materializing filtered TRAIN shards
- Supports local staging and shared patient-shard caching for Google Drive + local SSD workflows when reading canonical Stage 2 patient shards
- Writes `train_filtered_selection.csv`, `patient_filter_stats.csv`, `filter_run_config.json`, and `filter_summary.json` sidecars for lineage and review

### Stage 8+: Training & Inference

| Script | Purpose |
|--------|---------|
| `8_lr_finder.py` | Thin LR-finder orchestrator that loads `.env`, screens approved architecture/encoder pairs, and writes a LaTeX-generated PDF report plus CSV/JSON sidecars. |
| `9_training_ensemble.py` | Thin training entrypoint. Loads `.env`, validates the approved architecture/encoder pair, resolves TRAIN/VALIDATION rows from `master_manifest.sqlite`, and trains one model per execution through helper modules. |
| `10_optimizer_ensemble.py` | Thin ensemble-optimizer orchestrator that loads `.env`, resolves VALIDATION rows from `master_manifest.sqlite`, optimizes a two-stream recipe, and writes the declarative JSON consumed by Stage 11. |
| `11_inference_ensemble.py` | Thin inference orchestrator that loads the Stage 10 recipe, resolves TEST rows from `master_manifest.sqlite`, evaluates the two-stream ensemble on canonical Stage 2 patient shards, and exports JSON/CSV/LaTeX reporting artifacts. |

Current training behavior:

- `8_lr_finder.py` is now orchestration-focused; Stage 8 config loading, manifest-backed data setup, LR screening, curve analysis, and LaTeX reporting live in `helpers/lr_finder/*.py`
- Stage 8 derives the screened architecture/encoder plan from `training_model_registry.json` instead of hardcoded lists
- Stage 8 loads pretrained weights once per architecture/encoder pair, snapshots the initialized weights to CPU, and reuses that state across sampled loss configurations and repeats instead of reloading pretrained weights inside the nested screening loops
- Stage 8 accepts either `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`; the entrypoint applies the detected token to both environment variables before model creation
- Stage 8 defaults `LR_FINDER_AMP_PRECISION` to `fp32`; set it explicitly in `.env` when a different precision is desired
- Expected LR-range-test divergence now stops the active sweep early and preserves partial LR/loss history instead of treating a non-finite loss as a noisy hard failure
- Stage 8 console UX is notebook-friendly by design: one startup line, compact periodic progress snapshots, and one final summary with valid-record, completed-trial, failed-trial, and per-architecture counts
- Stage 8 writes both `report.tex` and `report.pdf`, plus `SUMMARY_ALL.csv`, per-architecture CSV summaries, `LHS_SAMPLES.json`, and `lr_finder_run_config.json`
- Stage 8 resolves TRAIN and VALIDATION rows from `master_manifest.sqlite` at startup, then loads pixels directly from canonical Stage 2 patient shards for runtime screening and provenance recording
- `9_training_ensemble.py` is now orchestration-focused; training runtime, manifest-backed data loading, model factory, losses, checkpointing, metrics, reporting, and epoch loops live in `helpers/training/*.py`
- Stage 9 resolves TRAIN and VALIDATION rows from `master_manifest.sqlite` at startup, applies shared on-the-fly stain normalization through the canonical dataset path, and loads pixels directly from canonical Stage 2 patient shards
- `10_optimizer_ensemble.py` is now orchestration-focused; Stage 10 config, metadata ranking, manifest-backed validation loading, model loading, patient holdout splitting, Optuna optimization, and JSON reporting live in `helpers/ensemble_optimizer/*.py`
- Stage 10 preserves the `two_stream_spatial_gating` JSON contract used by `11_inference_ensemble.py`, including `roi_config`, `spatial_config`, `model_registry`, and `holdout_metrics`
- `11_inference_ensemble.py` is now orchestration-focused; Stage 11 config, manifest-backed test-data loading, recipe-model loading, two-stream inference, metrics, and reporting live in `helpers/ensemble_inference/*.py`
- Architecture and encoder choices are validated against `training_model_registry.json`
- Learning-rate and weight-decay defaults are loaded from the registry instead of being hardcoded in the script
- `TRAINING_MODEL_REGISTRY_PATH` can override the default registry when a controlled experiment needs a different file
- Only the approved research architecture/encoder pairs documented in `.env_example` are supported

## Configuration

### Environment Variables (.env)

Copy `.env_example` to `.env` and configure:

```bash
# Stage 1 - Artifact detection
ARTIFACT_IMAGES_ZIP=/path/to/wsi_archive.zip
ARTIFACT_GEOJSON_OUTPUT=./artifacts/geojson
ARTIFACT_DATABASE_FOLDER=./databases
ARTIFACT_DATABASE_NAME=artifact_detection.db
ARTIFACT_TEMP_FOLDER=./temp/artifact_detection
ARTIFACT_LOG_FOLDER=./logs
ARTIFACT_DEVICE=cuda
ARTIFACT_TD_MODEL_DIR=./models/td
ARTIFACT_TD_MODEL_NAME=Tissue_Detection_MPP10.pth
ARTIFACT_QC_MODEL_DIR=./models/qc

# Optional Stage 1 runtime controls
ARTIFACT_MPP_MODEL=1.5
ARTIFACT_OVERLAY_FACTOR=10
ARTIFACT_OVERWRITE_EXISTING=false

# Stage 8 - LR Finder
LR_FINDER_MASTER_MANIFEST_PATH=./data/CAMELYON16/master_manifest.sqlite
LR_FINDER_OUTPUT_DIR=./reports/lr_finder
LR_FINDER_ARCHITECTURES=FPN,SEGFORMER
# Optional HF auth for pretrained encoders resolved from the Hugging Face Hub
HF_TOKEN=
# HUGGINGFACE_HUB_TOKEN=  # equivalent alias; Stage 8 mirrors either token to both names
LR_FINDER_AMP_PRECISION=fp32

# Stage 5 - Crossfold
CROSSFOLD_HDF5_COMPRESSION=none
CROSSFOLD_COPY_BATCH_SIZE=1024
CROSSFOLD_TEST_PATIENT_COUNT=20
CROSSFOLD_VALIDATION_PATIENT_COUNT=20
CROSSFOLD_SPLIT_OPTUNA_TRIALS=1000

# Stage 9 - Training
TRAINING_MASTER_MANIFEST_PATH=./data/CAMELYON16/master_manifest.sqlite
TRAINING_ARCHITECTURE=SEGFORMER
TRAINING_ENCODER=mit_b5
TRAINING_MODEL_REGISTRY_PATH=

# Stage 10 - Ensemble optimizer
ENSEMBLE_OPT_MASTER_MANIFEST_PATH=./data/CAMELYON16/master_manifest.sqlite
ENSEMBLE_OPT_METADATA_DIR=./metadata/CAMELYON16
ENSEMBLE_OPT_OUTPUT_DIR=./reports/ensemble_optimizer

# Patch extraction and later stages keep using their own .env variables
TAG=CAMELYON16
WINDOW_SIZE=224
STRIDE=112
TISSUE_PERCENTAGE=0.3
MATCH_PERCENTAGE=1.0
OPENSLIDE_PATH=
```

For SVS/XML datasets, use `TAG=HISEG` or `TAG=Chile`. Stage 2 resolves the supported label colors internally and does not require manual SQLite color setup.

See `.env_example` for the current commented template, including Stage 1, Stage 2 local WSI staging, Stage 4 cleaning, Stage 5/6 manifest-driven preparation, Stage 7.2 smart-sampling sidecars, Stage 8 LR-finder reporting, the Stage 9 training matrix, and Stage 10 ensemble-optimizer settings.

Stage 8 runtime notes:

- Stage 8 accepts either `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` and applies the detected token to both environment variables before model creation.
- The default `LR_FINDER_AMP_PRECISION` is `fp32`.
- Console output is intentionally compact for Colab and other notebook environments: one startup line, periodic snapshot progress lines, and one final summary. Detailed trace logging stays in `logs/lr_finder.log`.

For Stage 4 graph cleaning, the current tuning and cleaning benchmark notes live in `analysis/stage4_2_graph_tuning_performance_findings.md` and `analysis/stage4_3_graph_cleaning_performance_findings.md`.

For Stage 5 crossfold, the current optimization analysis and benchmark notes live in `analysis/stage5_crossfold_performance_findings.md`. The current best-known runtime settings on the measured 10-patient source dataset are `CROSSFOLD_HDF5_COMPRESSION=none` and `CROSSFOLD_COPY_BATCH_SIZE=1024`.

OpenSlide runtime rules:

- Native Windows requires `OPENSLIDE_PATH`, and it must point to the OpenSlide `bin` folder.
- Linux, containers, and Google Colab ignore `OPENSLIDE_PATH` and rely on the system OpenSlide install.
- When running from WSL, heavy reads and writes on `/mnt/c` or `/mnt/d` can be much slower than native Linux paths.
- If your source data and outputs live on Windows disks, native Windows remains the recommended production path.

## Installation

### Using uv (Recommended)

```bash
# Install dependencies from pyproject.toml
uv sync --python 3.12

# Or install with dev dependencies
uv sync --python 3.12 --group dev
```

### Using Google Colab

```bash
# After cloning the repository in Colab
bash setup_colab.sh

# Then run any script through uv's Python 3.12 environment
uv run --python 3.12 python 1_artifact_detection.py
```

See `colab_setup.md` for the complete Colab workflow, including Google Drive mounting and `.env` configuration.

### Using Native Windows

```powershell
./setup_windows.ps1
```

The bootstrap script validates `uv`, syncs dependencies from `pyproject.toml`, checks `OPENSLIDE_PATH`, and verifies `openslide`, `cv2`, and `torch` imports.

### Using Docker

```bash
# Build the container
docker build -t master-coding .

# Run the container
docker run --gpus all -it -v $(pwd):/workspace master-coding
```

### Container Access

To access a running container:

```bash
docker exec -it <container_name_or_id> bash
```

Example:
```bash
# Get container ID
docker ps

# Access container shell
docker exec -it container_name_or_id bash
```

## Usage Examples

### Complete Pipeline Execution

```bash
# 1. Artifact Detection (optional)
uv run --python 3.12 python 1_artifact_detection.py

# 2. Database & Patch Extraction
# Configure .env first, then:
uv run --python 3.12 python 2_database_manager.py

# HISEG example
# Set TAG=HISEG in .env, place `.svs` slides in IMAGES_HISEG,
# place matching `.xml` files in ANNOTATIONS_HISEG, then run:
uv run --python 3.12 python 2_database_manager.py

# 3. Annotation Cleaning (optional, for cancer patches)
uv run --python 3.12 python 4_1_optimization_sampling.py
# After human review, place approved/rejected in folders
uv run --python 3.12 python 4_2_tune_graph_method.py
# Stage 4.3 consumes the JSON artifact emitted by Stage 4.2 and fails fast without it
uv run --python 3.12 python 4_3_cleaner_script.py

# 4. Create HDF5 TRAIN/VALIDATION/TEST splits
uv run --python 3.12 python 5_crossfold.py

# 5. Run HDF5-native sanity checks
uv run --python 3.12 python 6_sanity_checks.py

# 6. Optional smart sampling, then training & inference (typically on a GPU machine)
uv run --python 3.12 python 7_smart_sampler.py
uv run --python 3.12 python 8_lr_finder.py
uv run --python 3.12 python 9_training_ensemble.py
uv run --python 3.12 python 10_optimizer_ensemble.py
uv run --python 3.12 python 11_inference_ensemble.py
```

### Approved Training Pairs

`9_training_ensemble.py` only supports the following approved research combinations:

| Architecture | Encoder |
|--------------|---------|
| `SWIN` | `tu-swin_large_patch4_window7_224.ms_in22k_ft_in1k` |
| `DEEPLABV3PLUS` | `tu-resnest101e` |
| `UNET++` | `efficientnet-b7` |
| `FPN` | `senet154` |
| `SEGFORMER` | `mit_b5` |
| `MANET` | `resnet152` |
| `DPT` | `tu-vit_large_patch16_224.augreg_in21k_ft_in1k` |
| `UPERNET` | `tu-hiera_large_224` |

The defaults for learning rate, weight decay, and allowed encoders live in `training_model_registry.json`.

### Running Tests

```bash
# Run all tests currently present in the repository
uv run pytest

# Run helper coverage
uv run pytest --cov=helpers --cov-report=term-missing

# Run linting and typing on the actively maintained surfaces
uv run ruff check 1_artifact_detection.py 2_database_manager.py 9_training_ensemble.py helpers tests
uv run ruff format .

# Run type checking on the same touched scope
uv run mypy 1_artifact_detection.py 2_database_manager.py 9_training_ensemble.py helpers tests
```

Recent targeted validation highlights:

- Stage 5/6/7 HDF5 migration suites passed (`49 passed` on targeted packaging/crossfold/sanity tests)
- Additional cleanup-focused targeted tests passed (`72 passed` and `9 passed` on focused subsets)
- `uv run ruff check . --select ARG001,ARG002,F401,F841` passed during the unused-code cleanup pass
- Ruff and MyPy passed on touched files during the documentation-aligned cleanup wave

## Data Integrity Rules

The pipeline enforces strict scientific integrity:

1. **Patient-level split isolation**: Patients never appear in multiple splits
2. **Image/mask pairing**: Preserved by filename
3. **Manifest schema**: Maintained for traceability
4. **Label semantics**: Cancer = positive class (1), Not-cancer = negative class (0)

## Output Folders

The pipeline produces these standardized folders and artifacts:

- `CANCER/` - Cancer patch images
- `NOT_CANCER/` - Non-cancer patch images
- `CANCER_MASK/` - Cancer segmentation masks
- `NOT_CANCER_MASK/` - Non-cancer segmentation masks
- Stage 5 source HDF5 - cleaned accepted patch pool packaged for Stage 5
- `manifest.csv`, `split_stats.csv`, `run_config.json` - Stage 5 split and lineage sidecars
- `train_filtered_selection.csv`, `patient_filter_stats.csv`, `filter_run_config.json`, `filter_summary.json` - Stage 7.2 selection and lineage sidecars

## Dependencies

Key dependencies (defined in `pyproject.toml`):

- **Deep Learning**: PyTorch, segmentation_models_pytorch, ScheduleFree
- **Image Processing**: OpenCV, OpenSlide, Pillow, albumentations
- **Scientific Computing**: NumPy, Pandas, Scikit-learn
- **Stain Normalization**: torch-staintools
- **Optimization**: Optuna

## License & Attribution

This is a research pipeline developed for medical imaging research. Ensure compliance with your institution's ethics guidelines when processing patient data.
