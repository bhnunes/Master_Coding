# Master_Coding

A Python research pipeline for pathology whole-slide-image (WSI) processing. It provides a complete end-to-end workflow for extracting patches from medical images, preparing datasets, training deep learning models, and generating predictions for cancer detection in histopathology images.

## Key Capabilities

- Whole-slide image processing (supports `.svs`, `.ndpi`, `.tiff` formats)
- Annotation handling for multiple formats (XML, NDPA, JSON)
- Patch extraction with tissue detection and artifact filtering
- Patient-level stratified dataset splitting
- Stain normalization support
- HDF5 packaging for efficient training
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
├── 5_crossfold.py                   # Patient-level dataset splitting
├── 6_sanity_checks.py               # Scientific integrity checks
├── 7_pack_splits_to_hdf5.py         # Convert splits to HDF5 format
├── 8_smart_sampler.py               # Select most informative training samples
├── 9_lr_finder.py                   # Find optimal learning rates
├── 10_training_ensemble.py          # Train one approved model per execution
├── 11_optimizer_ensemble.py         # Optimize ensemble parameters
├── 12_inference_ensemble.py         # Generate predictions on test set
│
├── helpers/
│   ├── __init__.py
│   ├── artifact_config.py           # Stage 1 .env parsing and validation
│   ├── artifact_logging.py          # Colorful Stage 1 console + file logging
│   ├── artifact_model_loader.py     # Stage 1 model loading
│   ├── artifact_paths.py            # Stage 1 path management and temp workspaces
│   ├── artifact_pipeline.py         # Stage 1 orchestration flow helpers
│   ├── artifact_processor.py        # Stage 1 per-slide processing
│   ├── artifact_repository.py       # Stage 1 SQLite tracking
│   ├── artifact_zip.py              # Stage 1 zip-backed slide access
│   ├── extraction_config.py         # Stage 2 .env parsing and validation
│   ├── extraction_repository.py     # Stage 2 SQLite/project state access
│   ├── image_reader_service.py      # Stage 2 slide processing dispatch
│   ├── data_handlers.py             # Annotation format adapters
│   ├── patch_engine.py              # Core patch extraction logic
│   ├── training_checkpointing.py    # Stage 8 checkpoints and resume metadata
│   ├── training_config.py           # Stage 8 .env parsing and validation
│   ├── training_data.py             # Stage 8 HDF5 datasets and batching
│   ├── training_gpu.py              # Stage 8 GPU image utilities
│   ├── training_loop.py             # Stage 8 train/validation loops
│   ├── training_losses.py           # Stage 8 custom losses
│   ├── training_metrics.py          # Stage 8 metric tracking helpers
│   ├── training_models.py           # Stage 8 model and optimizer factories
│   ├── training_pipeline.py         # Stage 8 orchestration helpers
│   ├── training_registry.py         # Approved architecture/encoder registry loader
│   ├── training_reporting.py        # Stage 8 Aim/email reporting helpers
│   ├── training_runtime.py          # Stage 8 seeds, AMP, precision helpers
│   └── training_utils.py            # Stage 8 misc runtime utilities
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

### Pipeline Stages

| Stage | Script(s) | Description |
|-------|-----------|-------------|
| 1 | `1_artifact_detection.py` | Detect artifacts on whole-slide images using a `.env`-driven Stage 1 pipeline. Output: GeoJSON files with artifact annotations and SQLite processing status |
| 2 | `2_database_manager.py` + `3_1_imageReader.py` | Extract patches from WSIs based on annotations. Output: PNG patches, masks, and artifact coverage Parquet metadata |
| 3 | `4_1_optimization_sampling.py` → `4_2_tune_graph_method.py` → `4_3_cleaner_script` | Human-in-the-loop + graph segmentation to remove incorrect annotations |
| 4 | `5_crossfold.py` | Create patient-level stratified TRAIN/VALIDATION/TEST splits |
| 5 | `6_sanity_checks.py` | Scientific integrity checks: patient leakage, file integrity, class balance |
| 6 | `7_pack_splits_to_hdf5.py` | Convert PNG splits to HDF5 format for efficient training |
| 7 | `8_smart_sampler.py` | Select most informative training samples (optional) |
| 8 | `9_lr_finder.py` → `10_training_ensemble.py` → `11_optimizer_ensemble.py` → `12_inference_ensemble.py` | Tune LR, train one approved model per run, optimize ensemble weights, and generate predictions |

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
| `helpers/data_handlers.py` | Annotation format adapters for `.svs/.xml`, `.ndpi/.ndpa`, and JSON-based formats. |
| `helpers/patch_engine.py` | Core extraction engine: tissue detection, polygon masking, artifact coverage computation, and image/mask writing. |

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
| `5_crossfold.py` | Creates patient-level stratified TRAIN/VALIDATION/TEST splits. Supports optional stain normalization (Reinhard, Macenko, Vahadane, Ruifrok). Uses entropy-based selection for optimal splits. |

### Stage 5: Quality Assurance

| Script | Purpose |
|--------|---------|
| `6_sanity_checks.py` | Runs scientific integrity checks on prepared datasets: patient leakage detection, file integrity, class balance, checksum verification. |

### Stage 6: Data Packaging

| Script | Purpose |
|--------|---------|
| `7_pack_splits_to_hdf5.py` | Converts PNG-based split folders into HDF5 files for efficient random access during training. |

### Stage 7: Smart Sampling

| Script | Purpose |
|--------|---------|
| `8_smart_sampler.py` | Selects the most informative training samples using MiniBatch K-Means clustering. Reduces dataset size while preserving diversity. Designed for GPU execution. |

### Stage 8: Training & Inference

| Script | Purpose |
|--------|---------|
| `9_lr_finder.py` | Estimates optimal learning rate for each model using the LR Finder technique. |
| `10_training_ensemble.py` | Thin Stage 8 training entrypoint. Loads `.env`, validates the approved architecture/encoder pair, stages HDF5 data, and trains one model per execution through helper modules. |
| `11_optimizer_ensemble.py` | Optimizes ensemble weights using Optuna to maximize validation AUPRC. Organizes models by Transformers (global context) and convolutional (local context). |
| `12_inference_ensemble.py` | Generates predictions on the test set using the optimized ensemble. |

Current Stage 8 behavior:

- `10_training_ensemble.py` is now orchestration-focused; training runtime, data, model factory, losses, checkpointing, metrics, reporting, and epoch loops live in `helpers/training_*.py`
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

# Stage 8 - Training
TRAINING_ARCHITECTURE=SEGFORMER
TRAINING_ENCODER_NAME=mit_b5
TRAINING_MODEL_REGISTRY_PATH=

# Patch extraction and later stages keep using their own .env variables
TAG=CAMELYON16
WINDOW_SIZE=224
STRIDE=112
TISSUE_PERCENTAGE=0.3
MATCH_PERCENTAGE=1.0
OPENSLIDE_PATH=
```

See `.env_example` for the current commented template, including Stage 1, Stage 2, and the Stage 8 approved training matrix.

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

# 4. Create dataset splits
uv run --python 3.12 python 5_crossfold.py

# 5. Run sanity checks
uv run --python 3.12 python 6_sanity_checks.py

# 6. Pack to HDF5
uv run --python 3.12 python 7_pack_splits_to_hdf5.py

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

# Run coverage after installing pytest-cov
uv run pytest --cov=helpers --cov-report=term-missing

# Run linting and typing on the actively maintained surfaces
uv run ruff check 1_artifact_detection.py 2_database_manager.py 3_1_imageReader.py 10_training_ensemble.py helpers tests
uv run ruff format .

# Run type checking on the same touched scope
uv run mypy 1_artifact_detection.py 2_database_manager.py 3_1_imageReader.py 10_training_ensemble.py helpers tests
```

Note: scoped Ruff and MyPy checks pass on the actively maintained Stage 1, Stage 2, Stage 8, `helpers`, and `tests` surfaces. Full-repository `uv run ruff check .` and `uv run mypy .` still report issues in unrelated legacy root scripts.

## Data Integrity Rules

The pipeline enforces strict scientific integrity:

1. **Patient-level split isolation**: Patients never appear in multiple splits
2. **Image/mask pairing**: Preserved by filename
3. **Manifest schema**: Maintained for traceability
4. **Label semantics**: Cancer = positive class (1), Not-cancer = negative class (0)

## Output Folders

The pipeline produces these standardized output folders:

- `CANCER/` - Cancer patch images
- `NOT_CANCER/` - Non-cancer patch images
- `CANCER_MASK/` - Cancer segmentation masks
- `NOT_CANCER_MASK/` - Non-cancer segmentation masks
- `TRAIN/`, `VALIDATION/`, `TEST/` - Dataset splits
- `TRAIN.h5`, `VALIDATION.h5`, `TEST.h5` - HDF5 packaged data

## Dependencies

Key dependencies (defined in `pyproject.toml`):

- **Deep Learning**: PyTorch, segmentation_models_pytorch, ScheduleFree
- **Image Processing**: OpenCV, OpenSlide, Pillow, albumentations
- **Scientific Computing**: NumPy, Pandas, Scikit-learn
- **Stain Normalization**: TIAtoolbox
- **Optimization**: Optuna

## License & Attribution

This is a research pipeline developed for medical imaging research. Ensure compliance with your institution's ethics guidelines when processing patient data.
