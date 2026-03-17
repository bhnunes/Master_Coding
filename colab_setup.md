# Google Colab Setup

This repository uses `pyproject.toml` as the only dependency source of truth. On Google Colab, the supported path is:

1. clone the repository
2. run `setup_colab.sh`
3. configure `.env`
4. run the script you want with `uv`

The setup script installs the system packages, installs `uv`, installs Python 3.12, syncs the project dependencies, creates `.env` from `.env_example` when missing, and verifies the critical Stage 1 imports.

## 1. Prepare the Colab runtime

- In Colab, go to `Runtime` -> `Change runtime type`
- Select `T4 GPU` or a stronger GPU
- Restart the runtime if Colab asks

## 2. Clone the repository and run the setup script

Run this in the first Colab cell:

```bash
!git clone <your-repository-url>
%cd <your-repository-folder>
!bash setup_colab.sh
```

After this completes, the repository is installed through `uv` from `pyproject.toml` and is ready to run.

## 3. Mount Google Drive if your data or models live there

```python
from google.colab import drive
drive.mount('/content/drive')
```

Typical Colab paths look like:

- `/content/drive/MyDrive/path/to/slides.zip`
- `/content/drive/MyDrive/path/to/models/td`
- `/content/drive/MyDrive/path/to/models/qc`

## 4. Configure `.env`

If `.env` did not already exist, `setup_colab.sh` creates it from `.env_example`.

Update the Stage 1 variables before running `1_artifact_detection.py`:

```bash
ARTIFACT_IMAGES_ZIP=/content/drive/MyDrive/path/to/slides.zip
ARTIFACT_GEOJSON_OUTPUT=/content/artifacts/geojson
ARTIFACT_DATABASE_FOLDER=/content/artifact_databases
ARTIFACT_DATABASE_NAME=artifact_detection.db
ARTIFACT_TEMP_FOLDER=/content/artifact_temp
ARTIFACT_LOG_FOLDER=/content/artifact_logs
ARTIFACT_DEVICE=cuda
ARTIFACT_TD_MODEL_DIR=/content/drive/MyDrive/path/to/models/td
ARTIFACT_TD_MODEL_NAME=Tissue_Detection_MPP10.pth
ARTIFACT_QC_MODEL_DIR=/content/drive/MyDrive/path/to/models/qc
ARTIFACT_MPP_MODEL=1.5
ARTIFACT_OVERLAY_FACTOR=10
```

You can edit `.env` in Colab with:

```bash
!cp .env .env.bak
!sed -n '1,120p' .env
```

Or use the file browser and edit it directly.

## 5. Run Stage 1 artifact detection

```bash
!uv run --python 3.12 python 1_artifact_detection.py
```

## 6. Run any other repository script

The same environment can be used for the rest of the repository scripts:

```bash
!uv run --python 3.12 python 2_database_manager.py
!uv run --python 3.12 python 5_crossfold.py
!uv run --python 3.12 python 6_sanity_checks.py
```

## 7. What `setup_colab.sh` installs

The script mirrors the project runtime used by `Dockerfile` and the devcontainer workflow:

- system packages such as OpenSlide, OpenJPEG, GEOS, and the current container utilities
- `uv`
- Python 3.12
- all Python dependencies from `pyproject.toml`

## 8. Quick verification

`setup_colab.sh` already verifies these imports:

- `torch`
- `openslide`
- `cv2`
- `segmentation_models_pytorch`

If you want to rerun the check manually:

```bash
!uv run --python 3.12 python - <<'PY'
import cv2
import openslide
import segmentation_models_pytorch as smp
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(openslide.__library_version__)
print(cv2.__version__)
print(smp.__version__)
PY
```

## 9. Common blockers

- `ARTIFACT_IMAGES_ZIP` points to a missing file: update `.env` with the real zip path
- model weights are missing: place the tissue detector and QC weights in the configured model folders
- `ARTIFACT_DEVICE=cuda` but no GPU is enabled: switch the Colab runtime to GPU or set `ARTIFACT_DEVICE=cpu`
- outputs go to Drive and feel slow: prefer `/content/...` for temporary folders and logs

## 10. Re-running setup after updates

If `pyproject.toml` changes, rerun:

```bash
!bash setup_colab.sh
```

That keeps Colab aligned with the repository configuration.
