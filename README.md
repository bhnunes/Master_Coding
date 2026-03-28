# Master_Coding

A Python research pipeline for pathology whole-slide-image (WSI) processing. It provides a complete end-to-end workflow for extracting patches from medical images, preparing datasets, training deep learning models, and generating predictions for cancer detection in histopathology images.

## Key Capabilities

- Whole-slide image processing (supports `.svs`, `.ndpi`, `.tiff` formats)
- Annotation handling for multiple formats (XML, NDPA, JSON)
- Patch extraction with tissue detection and artifact filtering
- Patient-level stratified dataset splitting on HDF5 datasets
- Train-fitted stain normalization support
- HDF5-native downstream data preparation
- Ensemble model training and inference
- GPU-accelerated deep learning with PyTorch

## Folder Structure

```
Master_Coding/
├── 1_artifact_detection.py          # Generate artifact GeoJSON from WSIs
├── 2_database_manager.py            # Database orchestration & case processing
├── 3_1_imageReader.py               # Per-slide patch extraction worker
├── 4_1_optimization_sampling.py     # Select samples for human-in-the-loop cleaning
├── 4_2_tune_graph_method.py         # Tune graph segmentation parameters
├── 4_3_cleaner_script.py            # Apply cleaning to remove incorrect annotations
├── 5_pack_splits_to_hdf5.py         # Package cleaned patches into source HDF5
├── 6_crossfold.py                   # HDF5-native patient-level dataset splitting
├── 7_sanity_checks.py               # HDF5-native scientific integrity checks
├── 8_smart_sampler.py               # Select most informative training samples
├── 9_lr_finder.py                   # Find optimal learning rates
├── 10_training_ensemble.py          # Train one approved model per execution
├── 11_optimizer_ensemble.py         # Optimize ensemble parameters
├── 12_inference_ensemble.py         # Generate predictions on test set
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
│   ├── packaging/                   # Stage 7 source-HDF5 packaging domain
│   │   ├── config.py
│   │   ├── discovery.py
│   │   ├── pipeline.py
│   │   └── writer.py
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
│   ├── smart_sampling/              # Stage 8 smart-sampling domain
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── index.py
│   │   ├── pipeline.py
│   │   ├── selection.py
│   │   ├── storage.py
│   │   └── writer.py
│   ├── lr_finder/                   # Stage 9 LR-finder domain
│   │   ├── analysis.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── pipeline.py
│   │   ├── reporting.py
│   │   ├── runner.py
│   │   └── search_space.py
│   ├── training/                    # Stage 8 training domain
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
│   ├── ensemble_optimizer/          # Stage 11 ensemble-optimizer domain
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── metadata.py
│   │   ├── models.py
│   │   ├── optimization.py
│   │   ├── pipeline.py
│   │   ├── reporting.py
│   │   └── splitting.py
│   ├── ensemble_inference/          # Stage 12 ensemble-inference domain
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── inference.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── recipe.py
│   │   └── reporting.py
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

The pipeline follows a monolithic script-driven architecture where each script performs a distinct stage of the workflow. Scripts are designed to be standalone applications that can be executed independently, with defined inputs and outputs.

Helper modules are organized by domain under `helpers/<domain>/`. New domain-specific helpers should be added to the matching domain package instead of recreating a flat `helpers/*.py` layout. Only broadly shared cross-domain utilities, such as `helpers/runtime_platform.py`, remain at the top level.

### Pipeline Stages

| Stage | Script(s) | Description |
|-------|-----------|-------------|
| 1 | `1_artifact_detection.py` | Detect artifacts on whole-slide images using a `.env`-driven Stage 1 pipeline. Output: GeoJSON files with artifact annotations and SQLite processing status |
| 2 | `2_database_manager.py` + `3_1_imageReader.py` | Extract canonical HDF5 patch shards from WSIs based on annotations. Output: HDF5 patch data, artifact coverage Parquet metadata, and optional PNG exports when explicitly enabled |
| 4.1-4.3 | `4_1_optimization_sampling.py` → `4_2_tune_graph_method.py` → `4_3_cleaner_script.py` | HDF5-backed human-in-the-loop review plus graph-based cleaning, with PNG retained only for review/export workflows |
| 5 | `5_pack_splits_to_hdf5.py` | Finalize the cleaned HDF5 source dataset from Stage 2 HDF5 inputs and Stage 4.3 accepted manifests |
| 6 | `6_crossfold.py` | Create patient-level TRAIN/VALIDATION/TEST HDF5 splits and fit stain normalization on `TRAIN` only |
| 7 | `7_sanity_checks.py` | Validate HDF5 split integrity, provenance, leakage, and mask/label semantics |
| 8 | `8_smart_sampler.py` | Select the most informative training samples from `TRAIN.h5` into `TRAIN_FILTERED.h5` (optional) |
| 9-12 | `9_lr_finder.py` → `10_training_ensemble.py` → `11_optimizer_ensemble.py` → `12_inference_ensemble.py` | Tune LR, train one approved model per run, optimize the ensemble recipe, and generate final test predictions |

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
| `2_database_manager.py` | Orchestrates the entire processing pipeline. Manages SQLite database, handles case ingestion, and invokes the image reader for each slide. |
| `3_1_imageReader.py` | Per-slide worker that extracts patches based on annotations. Supports multiple annotation formats via handler dispatch. |
| `helpers/extraction/data_handlers.py` | Annotation format adapters for `.svs/.xml`, `.ndpi/.ndpa`, and JSON-based formats. |
| `helpers/extraction/patch_engine.py` | Core extraction engine: tissue detection, polygon masking, artifact coverage computation, and image/mask writing. |

Current Stage 2 artifact-aware behavior:

- `USE_ADVANCED_ARTIFACT_FILTERING=True` computes per-class artifact coverage for every saved patch instead of rejecting patches by threshold
- `USE_ADVANCED_ARTIFACT_FILTERING=False` skips artifact geometry work for speed, but still writes the same Parquet schema with zero-valued coverage columns
- Stage 2 writes filename-keyed artifact metadata to `PATCHES/artifact_patch_index.parquet`
- Stage 8 can join that Parquet file with HDF5 `filenames` for artifact-aware loss discounting during training

### Stage 3: Annotation Cleaning

| Script | Purpose |
|--------|---------|
| `4_1_optimization_sampling.py` | Selects a statistically representative subsample of cancer images for human-in-the-loop review. Uses Cochran's formula for sample size calculation. |
| `4_2_tune_graph_method.py` | Uses Bayesian optimization to find optimal graph segmentation parameters for distinguishing correct vs. incorrect annotations. |
| `4_3_cleaner_script.py` | Applies the tuned graph segmentation method to automatically remove incorrectly annotated patches from the cancer folder. |

### Stage 4: Dataset Preparation

| Script | Purpose |
|--------|---------|
| `6_crossfold.py` | Thin Stage 6 orchestrator that loads `.env`, reads the Stage 5 source HDF5, builds patient-level TRAIN/VALIDATION/TEST splits, optionally fits stain normalization on TRAIN only, and writes provenance artifacts. |

Current Stage 6 behavior:

- Loads Stage 6 settings from `.env` / `.env_example` through `helpers/crossfold/config.py`
- Reads one cleaned source HDF5 dataset produced after Stage 4.3 and packaged by Stage 5
- Preserves patient-level split isolation and stratifies patients by `max(patch_label)`
- Can evaluate many feasible patient-level splits and score them with the entropy objective before selecting the best candidate
- Fits stain normalization on TRAIN only when a normalization method other than `NOT_NORMALIZED` is configured
- Applies the frozen TRAIN-fitted normalizer to `TRAIN`, `VALIDATION`, and `TEST`
- Writes HDF5 split artifacts plus `manifest.csv`, `split_stats.csv`, `run_config.json`, and optional entropy-cache CSV artifacts for traceability

### Stage 5: Quality Assurance

| Script | Purpose |
|--------|---------|
| `7_sanity_checks.py` | Thin Stage 7 orchestrator that loads `.env`, validates Stage 6 provenance and HDF5 split artifacts, and prints a reviewer-facing PASS/WARN/FAIL report. |

Current Stage 7 behavior:

- Loads Stage 7 settings from `.env` / `.env_example` through `helpers/sanity/config.py`
- Uses `manifest.csv` as the source of truth and cross-checks `run_config.json` and `split_stats.csv`
- Fails on patient leakage, duplicate manifest rows, filename-contract violations, and manifest/disk mismatches
- Verifies HDF5 dataset parity, readable rows, image/mask shape agreement, and valid mask pixel values
- Adds scientific label checks such as empty positive masks and positive pixels inside negative masks
- Reports class balance and patches-per-patient skew to support reviewer interpretation

### Stage 6: Data Packaging

| Script | Purpose |
|--------|---------|
| `5_pack_splits_to_hdf5.py` | Thin Stage 5 orchestrator that loads `.env`, scans the cleaned accepted patch pool, and writes the source HDF5 dataset consumed by Stage 6. |

Current Stage 5 behavior:

- Loads Stage 5 settings from `.env` / `.env_example` through `helpers/packaging/config.py`
- Scans the cleaned accepted patch pool produced after Stage 4.3, not pre-made split folders
- Requires exact filename parity between image and mask inputs before packing
- Writes a single source HDF5 with datasets `images`, `masks`, `labels`, `patient_ids`, and `filenames`
- Preserves stable row alignment and filename identity for downstream Stage 6, Stage 8, and artifact-aware joins

### Stage 8: Smart Sampling

| Script | Purpose |
|--------|---------|
| `8_smart_sampler.py` | Thin smart-sampling orchestrator that loads `.env`, filters `TRAIN.h5` patient-by-patient, and writes `TRAIN_FILTERED.h5` for downstream LR finding and training. |

Current Stage 8 smart-sampling behavior:

- Loads smart-sampling settings from `.env` / `.env_example` through `helpers/smart_sampling/config.py`
- Builds a patient index from `TRAIN.h5`, extracts embeddings, and selects diverse per-patient samples through `helpers/smart_sampling/*.py`
- Preserves downstream HDF5 compatibility for `9_lr_finder.py` and `10_training_ensemble.py` by writing `images`, `masks`, `labels`, `patient_ids`, and `filenames`
- Accepts legacy source files with `filename` or `filenames`, but always writes `filenames` in `TRAIN_FILTERED.h5`
- Optionally writes `train_filtered_selection.csv`, `patient_filter_stats.csv`, and `filter_run_config.json` beside the filtered HDF5

### Stage 9+: Training & Inference

| Script | Purpose |
|--------|---------|
| `9_lr_finder.py` | Thin LR-finder orchestrator that loads `.env`, screens approved architecture/encoder pairs, and writes a LaTeX-generated PDF report plus CSV/JSON sidecars. |
| `10_training_ensemble.py` | Thin training entrypoint. Loads `.env`, validates the approved architecture/encoder pair, stages HDF5 data, and trains one model per execution through helper modules. |
| `11_optimizer_ensemble.py` | Thin ensemble-optimizer orchestrator that loads `.env`, stages validation HDF5 data, optimizes a two-stream recipe, and writes the declarative JSON consumed by Stage 12. |
| `12_inference_ensemble.py` | Thin inference orchestrator that loads the Stage 11 recipe, evaluates the two-stream ensemble on `TEST.h5`, and exports JSON/CSV/LaTeX reporting artifacts. |

Current training behavior:

- `9_lr_finder.py` is now orchestration-focused; Stage 9 config loading, HDF5 staging, LR screening, curve analysis, and LaTeX reporting live in `helpers/lr_finder/*.py`
- Stage 9 derives the screened architecture/encoder plan from `training_model_registry.json` instead of hardcoded lists
- Stage 9 writes both `report.tex` and `report.pdf`, plus `SUMMARY_ALL.csv`, per-architecture CSV summaries, `LHS_SAMPLES.json`, and `lr_finder_run_config.json`
- `10_training_ensemble.py` is now orchestration-focused; training runtime, data, model factory, losses, checkpointing, metrics, reporting, and epoch loops live in `helpers/training/*.py`
- `11_optimizer_ensemble.py` is now orchestration-focused; Stage 11 config, metadata ranking, validation staging, model loading, patient holdout splitting, Optuna optimization, and JSON reporting live in `helpers/ensemble_optimizer/*.py`
- Stage 11 preserves the `two_stream_spatial_gating` JSON contract used by `12_inference_ensemble.py`, including `roi_config`, `spatial_config`, `model_registry`, and `holdout_metrics`
- `12_inference_ensemble.py` is now orchestration-focused; Stage 12 config, recipe parsing, test-data staging, recipe-model loading, two-stream inference, metrics, and reporting live in `helpers/ensemble_inference/*.py`
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

# Stage 9 - LR Finder
LR_FINDER_HDF5_DRIVE_DIR=./data/CAMELYON16
LR_FINDER_OUTPUT_DIR=./reports/lr_finder
LR_FINDER_ARCHITECTURES=FPN,SEGFORMER

# Stage 10 - Training
TRAINING_ARCHITECTURE=SEGFORMER
TRAINING_ENCODER=mit_b5
TRAINING_MODEL_REGISTRY_PATH=

# Stage 11 - Ensemble optimizer
ENSEMBLE_OPT_HDF5_DRIVE_DIR=./data/CAMELYON16
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

See `.env_example` for the current commented template, including Stage 1, Stage 2 local WSI staging, Stage 5/6/7 HDF5-native preparation, Stage 8 smart sampling, Stage 9 LR-finder reporting, the Stage 10 training matrix, and Stage 11 ensemble-optimizer settings.

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

# 3. Annotation Cleaning (optional, for cancer patches)
uv run --python 3.12 python 4_1_optimization_sampling.py
# After human review, place approved/rejected in folders
uv run --python 3.12 python 4_2_tune_graph_method.py
uv run --python 3.12 python 4_3_cleaner_script.py

# 4. Package cleaned patches into one source HDF5
uv run --python 3.12 python 5_pack_splits_to_hdf5.py

# 5. Create HDF5 TRAIN/VALIDATION/TEST splits
uv run --python 3.12 python 6_crossfold.py

# 6. Run HDF5-native sanity checks
uv run --python 3.12 python 7_sanity_checks.py

# 8-12. Training & Inference (typically run on a GPU machine)
uv run --python 3.12 python 9_lr_finder.py
uv run --python 3.12 python 10_training_ensemble.py
uv run --python 3.12 python 11_optimizer_ensemble.py
uv run --python 3.12 python 12_inference_ensemble.py
```

### Approved Training Pairs

`10_training_ensemble.py` only supports the following approved research combinations:

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
uv run ruff check 1_artifact_detection.py 2_database_manager.py 3_1_imageReader.py 10_training_ensemble.py helpers tests
uv run ruff format .

# Run type checking on the same touched scope
uv run mypy 1_artifact_detection.py 2_database_manager.py 3_1_imageReader.py 10_training_ensemble.py helpers tests
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
- Stage 5 source HDF5 - cleaned accepted patch pool packaged for Stage 6
- `TRAIN.h5`, `VALIDATION.h5`, `TEST.h5` - Stage 6 HDF5 split artifacts
- `TRAIN_FILTERED.h5` - optional Stage 8 smart-sampled training set

## Dependencies

Key dependencies (defined in `pyproject.toml`):

- **Deep Learning**: PyTorch, segmentation_models_pytorch, ScheduleFree
- **Image Processing**: OpenCV, OpenSlide, Pillow, albumentations
- **Scientific Computing**: NumPy, Pandas, Scikit-learn
- **Stain Normalization**: TIAtoolbox
- **Optimization**: Optuna

## License & Attribution

This is a research pipeline developed for medical imaging research. Ensure compliance with your institution's ethics guidelines when processing patient data.
