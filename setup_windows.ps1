param(
    [string]$PythonVersion = "3.12",
    [string]$OpenSlidePath = "C:\MASTER\openslide\bin"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Command {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Step "Checking uv"
if (-not (Test-Command "uv")) {
    Write-Host "uv is not installed." -ForegroundColor Yellow
    Write-Host "Install it from https://docs.astral.sh/uv/getting-started/installation/ and rerun this script."
    exit 1
}

$OpenSlidePath = $OpenSlidePath.Trim()

Write-Step "Validating OPENSLIDE_PATH"
if ([string]::IsNullOrWhiteSpace($OpenSlidePath)) {
    Write-Host "OPENSLIDE_PATH is required on native Windows." -ForegroundColor Yellow
    Write-Host "Install the OpenSlide Windows binaries and set OPENSLIDE_PATH to the OpenSlide 'bin' folder."
    Write-Host "Example: OPENSLIDE_PATH=C:\MASTER\openslide\bin"
    exit 1
}

Write-Step "Validating OPENSLIDE_PATH Exists..."
if (-not (Test-Path $openSlidePath -PathType Container)) {
    Write-Host "OPENSLIDE_PATH does not exist or is not a directory: $openSlidePath" -ForegroundColor Red
    exit 1
}

Write-Step "Syncing Python dependencies..."
uv sync --python $PythonVersion --group dev
if ($LASTEXITCODE -ne 0) {
    Write-Host "uv sync failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

$env:OPENSLIDE_PATH = $OpenSlidePath

Write-Step "Verifying Python imports"

$pythonCheck = @'
import os
from pathlib import Path

openslide_path = os.environ.get("OPENSLIDE_PATH", "").strip()
if not openslide_path:
    raise SystemExit("OPENSLIDE_PATH is required on Windows.")
if not Path(openslide_path).is_dir():
    raise SystemExit(f"OPENSLIDE_PATH is not a valid directory: {openslide_path}")

with os.add_dll_directory(openslide_path):
    import openslide

import cv2
import torch

print("openslide", getattr(openslide, "__file__", "loaded"))
print("cv2", cv2.__version__)
print("torch", torch.__version__)
'@

$tempFile = [System.IO.Path]::GetTempFileName() + ".py"
Set-Content -Path $tempFile -Value $pythonCheck -Encoding UTF8

uv run --python $PythonVersion python $tempFile
$exitCode = $LASTEXITCODE

Remove-Item $tempFile -Force

if ($exitCode -ne 0) {
    Write-Host "Python import verification failed." -ForegroundColor Red
    exit $exitCode
}

Write-Host "Windows bootstrap complete." -ForegroundColor Green
