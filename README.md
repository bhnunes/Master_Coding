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
├── 10_training_ensemble.py          # Train ensemble models
├── 11_optimizer_ensemble.py         # Optimize ensemble parameters
├── 12_inference_ensemble.py         # Generate predictions on test set
│
├── helpers/
│   ├── data_handlers.py             # Annotation format adapters
│   └── patch_engine.py              # Core patch extraction logic
│
├── tests/                           # Test scripts
├── databases/                       # SQLite databases
├── logs/                            # Log files
│
├── .env_example                     # Environment configuration template
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration (uv)
├── Dockerfile                       # Container definition
├── artifact_policy.yaml             # Artifact filtering thresholds
└── AGENTS.md                        # Developer guidelines
```

## Pipeline Architecture

The pipeline follows a monolithic script-driven architecture where each script performs a distinct stage of the workflow. Scripts are designed to be standalone applications that can be executed independently, with defined inputs and outputs.

### Pipeline Stages

| Stage | Script(s) | Description |
|-------|-----------|-------------|
| 1 | `1_artifact_detection.py` | Detect artifacts on whole-slide images using deep learning. Output: GeoJSON files with artifact annotations |
| 2 | `2_database_manager.py` + `3_1_imageReader.py` | Extract patches from WSIs based on annotations. Output: PNG patches + masks |
| 3 | `4_1_optimization_sampling.py` → `4_2_tune_graph_method.py` → `4_3_cleaner_script` | Human-in-the-loop + graph segmentation to remove incorrect annotations |
| 4 | `5_crossfold.py` | Create patient-level stratified TRAIN/VALIDATION/TEST splits |
| 5 | `6_sanity_checks.py` | Scientific integrity checks: patient leakage, file integrity, class balance |
| 6 | `7_pack_splits_to_hdf5.py` | Convert PNG splits to HDF5 format for efficient training |
| 7 | `8_smart_sampler.py` | Select most informative training samples (optional) |
| 8 | `9_lr_finder.py` → `10_training_ensemble.py` → `11_optimizer_ensemble.py` → `12_inference_ensemble.py` | Train ensemble models and generate predictions |

## Script Documentation

### Stage 1: Artifact Detection

| Script | Purpose |
|--------|---------|
| `1_artifact_detection.py` | Detects artifacts (fold, darkspot, pen markings, etc.) on whole-slide images using deep learning models. Generates GeoJSON files that mark artifact regions. |

### Stage 2: Patch Extraction

| Script | Purpose |
|--------|---------|
| `2_database_manager.py` | Orchestrates the entire processing pipeline. Manages SQLite database, handles case ingestion, and invokes the image reader for each slide. |
| `3_1_imageReader.py` | Per-slide worker that extracts patches based on annotations. Supports multiple annotation formats via handler dispatch. |
| `helpers/data_handlers.py` | Annotation format adapters for `.svs/.xml`, `.ndpi/.ndpa`, and JSON-based formats. |
| `helpers/patch_engine.py` | Core extraction engine: tissue detection, polygon masking, artifact filtering, and image/mask writing. |

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
| `10_training_ensemble.py` | Trains an ensemble of models using segmentation_models_pytorch (smp), ScheduleFree optimizer, and PyTorch. Supports multiple encoder architectures. |
| `11_optimizer_ensemble.py` | Optimizes ensemble weights using Optuna to maximize validation AUPRC. Organizes models by Transformers (global context) and convolutional (local context). |
| `12_inference_ensemble.py` | Generates predictions on the test set using the optimized ensemble. |

## Configuration

### Environment Variables (.env)

Copy `.env_example` to `.env` and configure:

```bash
# Project tag (used for folder/database naming)
TAG=CAMELYON16

# Database
SQLITE_DB_PATH=./databases/database.db

# Base paths
PROJECTS_BASE_PATH=/path/to/projects

# Patch extraction parameters
WINDOW_SIZE=224
STRIDE=112
TISSUE_PERCENTAGE=0.3
MATCH_PERCENTAGE=1.0
OPENSLIDE_PATH=/path/to/openslide/bin

# Artifact filtering (optional)
USE_ADVANCED_ARTIFACT_FILTERING=True
ARTIFACT_POLICY_PATH=./artifact_policy.yaml
ACTIVATE_SANITY_CHECK_GEOJSON=True
GEOJSON_PATH=/path/to/geojson
```

### Artifact Policy (artifact_policy.yaml)

Defines thresholds for filtering patches containing artifacts:

```yaml
DROP_THRESH:
  "Fold": 0.15
  "Darkspot & Foreign Object": 0.15
  "PenMarking": 0.12
  "Edge & Air Bubble": 0.30
  "OOF": 0.45
```

## Installation

### Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Or install with dev dependencies
uv sync --group dev
```

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
python 1_artifact_detection.py

# 2. Database & Patch Extraction
# Configure .env first, then:
python 2_database_manager.py

# 3. Annotation Cleaning (optional, for cancer patches)
python 4_1_optimization_sampling.py
# After human review, place approved/rejected in folders
python 4_2_tune_graph_method.py
python 4_3_cleaner_script.py

# 4. Create dataset splits
python 5_crossfold.py

# 5. Run sanity checks
python 6_sanity_checks.py

# 6. Pack to HDF5
python 7_pack_splits_to_hdf5.py

# 8-12. Training & Inference (typically run in Colab with GPU)
python 9_lr_finder.py
python 10_training_ensemble.py
python 11_optimizer_ensemble.py
python 12_inference_ensemble.py
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=helpers --cov-report=term-missing

# Run linting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run mypy .
```

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

Key dependencies (see `requirements.txt` and `pyproject.toml`):

- **Deep Learning**: PyTorch, segmentation_models_pytorch, ScheduleFree
- **Image Processing**: OpenCV, OpenSlide, Pillow, albumentations
- **Scientific Computing**: NumPy, Pandas, Scikit-learn
- **Stain Normalization**: TIAtoolbox
- **Optimization**: Optuna

## License & Attribution

This is a research pipeline developed for medical imaging research. Ensure compliance with your institution's ethics guidelines when processing patient data.
