from __future__ import annotations

import tomllib
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _normalized_dependency_names(pyproject_path: Path) -> set[str]:
    data = tomllib.loads(pyproject_path.read_text())
    dependencies: list[str] = data["project"]["dependencies"]
    names = set()
    for dependency in dependencies:
        normalized = dependency.split(";", 1)[0].strip()
        for separator in ("==", ">=", "<=", "~=", "!=", "<", ">"):
            normalized = normalized.split(separator, 1)[0].strip()
        names.add(normalized.lower())
    return names


def test_pyproject_declares_repo_runtime_dependencies() -> None:
    dependency_names = _normalized_dependency_names(WORKSPACE_ROOT / "pyproject.toml")

    expected_dependencies = {
        "aim",
        "albumentations",
        "h5py",
        "joblib",
        "matplotlib",
        "numba",
        "numpy",
        "opencv-contrib-python",
        "openslide-python",
        "optuna",
        "pandas",
        "pillow",
        "psutil",
        "python-dotenv",
        "pyyaml",
        "schedulefree",
        "scikit-image",
        "scikit-learn",
        "scikit-optimize",
        "seaborn",
        "segmentation-models-pytorch",
        "shapely",
        "tiatoolbox",
        "torch",
        "torch-lr-finder",
        "torchaudio",
        "torchmetrics",
        "torchvision",
        "tqdm",
    }

    assert expected_dependencies.issubset(dependency_names)
