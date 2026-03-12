# 🚀 GrandQC: Google Colab Setup Guide

This document explains how to mirror your local **VS Code Dev
Container** environment inside a **Google Colab notebook**. By using
**uv** and our **pyproject.toml**, we ensure that versions (like **NumPy
1.26.4** and **PyTorch 2.8**) stay consistent across environments.

------------------------------------------------------------------------

# 1. Prerequisites

-   A Google account with access to **Google Colab**.

### Change Runtime Type

    Runtime → Change runtime type → Hardware accelerator → T4 GPU (or higher)

------------------------------------------------------------------------

# 2. One-Cell Environment Setup

Copy and paste the following code into the **very first cell** of your
Colab notebook.

``` python
# --- 1. Install uv (Fastest Python Package Manager) ---
!curl -fsSL https://astral.sh/uv/install.sh | sh
import os
os.environ['PATH'] = f"{os.environ['PATH']}:/root/.cargo/bin"

# --- 2. Clone the Repository ---
# Replace with your actual repository URL
!git clone https://github.com/your-username/grandqc-project.git
%cd grandqc-project

# --- 3. Install System Dependencies (OpenSlide, LaTeX) ---
print("Installing system libraries...")
!apt-get update -qq
!apt-get install -y -qq libopenjp2-7-dev libopenjp2-tools openslide-tools \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-fonts-extra > /dev/null

# --- 4. Sync Python Environment via uv ---
# This pulls the exact versions from your pyproject.toml
# and ensures the CUDA 12.6 wheels are used for PyTorch.
print("Syncing Python dependencies (this is fast!)...")
!uv pip install --system --no-cache-dir -U \
    --index-url https://download.pytorch.org/whl/cu126 .

# --- 5. Verify the Installation ---
import torch
import numpy as np
import openslide

print(f"✅ GPU Available: {torch.cuda.is_available()}")
print(f"✅ Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"✅ NumPy Version: {np.__version__} (Should be 1.26.4)")
print(f"✅ OpenSlide Version: {openslide.__library_version__}")
```

------------------------------------------------------------------------

# 3. Handling Large Datasets

Since you are dealing with **massive datasets**, **do not upload files
directly to Colab**. Use one of the following methods.

------------------------------------------------------------------------

## Method A: Google Drive (Recommended)

Mount your drive to access datasets stored there.

``` python
from google.colab import drive
drive.mount('/content/drive')

# Update your script paths to point to /content/drive/MyDrive/...
```

------------------------------------------------------------------------

## Method B: GCS / S3

If your data is stored in the cloud, use **gsutil** or **boto3** to
stream data directly to the `/content` local disk (SSD) for maximum
training speed.

------------------------------------------------------------------------

# 4. Development Workflow

### Code Locally

Use **VS Code** and your **Dev Container**.

Run:

    ruff
    mypy

to catch linting and typing errors.

------------------------------------------------------------------------

### Commit & Push

Push your changes to GitHub.

------------------------------------------------------------------------

### Pull on Colab

``` bash
!git pull origin main
!uv pip install --system .  # Refresh dependencies if pyproject.toml changed
```

------------------------------------------------------------------------

### Run Experiments

Execute your **high-memory / GPU training loops**.

------------------------------------------------------------------------

# 5. Troubleshooting

### NumPy Conflicts

If Colab warns about a **NumPy restart**, it is because Colab
pre-installs **NumPy 2.x**.

The command:

    uv pip install --system

overrides this version.

You may need to click **Restart Session** once if prompted, but the
files will remain.

------------------------------------------------------------------------

### LaTeX Errors

If `pdflatex` fails, verify that the package below was installed
correctly in the setup cell:

    texlive-latex-extra
