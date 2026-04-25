from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from helpers.crossfold.entropy import (
    compute_patient_entropy_median,
)
from helpers.cv2_compat import ensure_cv2_compat

cv2 = ensure_cv2_compat(cv2)
FITTED_NORMALIZATION_METHODS = ("REINHARD", "RUIFROK", "MACENKO", "VAHADANE")
_RUIFROK_HE_STAIN_MATRIX = torch.tensor(
    [
        [0.644211, 0.716556, 0.266844],
        [0.092789, 0.954111, 0.283111],
    ],
    dtype=torch.float32,
)
_EPSILON = 1e-6


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def make_aggregate_target(image_paths: list[str]) -> npt.NDArray[np.uint8]:
    """Build the median RGB target image used to fit the stain normalizer."""

    images_rgb: list[npt.NDArray[np.uint8]] = []
    for image_path in image_paths:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue
        images_rgb.append(cast(npt.NDArray[np.uint8], cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
    if not images_rgb:
        raise ValueError("No valid images found to build aggregate target.")
    stack = np.stack(images_rgb, axis=0)
    return cast(npt.NDArray[np.uint8], np.median(stack, axis=0).astype(np.uint8))


def _make_aggregate_target_from_images(
    images_rgb: list[npt.NDArray[np.uint8]],
) -> npt.NDArray[np.uint8]:
    if not images_rgb:
        raise ValueError("No valid images found to build aggregate target.")
    stack = np.stack(images_rgb, axis=0)
    return cast(npt.NDArray[np.uint8], np.median(stack, axis=0).astype(np.uint8))


def _load_rgb_images_from_hdf5_rows(selected_rows: pd.DataFrame) -> list[npt.NDArray[np.uint8]]:
    images_rgb: list[npt.NDArray[np.uint8]] = []
    grouped = selected_rows.groupby("source_hdf5_path", sort=False)
    for source_hdf5_path, group_df in grouped:
        with h5py.File(Path(str(source_hdf5_path)), "r") as handle:
            for row_index in group_df["source_row_index"].tolist():
                image = np.asarray(handle["images"][int(row_index)], dtype=np.uint8)
                images_rgb.append(image)
    return images_rgb


def _can_use_hdf5_rows(frame: pd.DataFrame) -> bool:
    return {"source_hdf5_path", "source_row_index"}.issubset(frame.columns)


def _parse_hdf5_image_ref(template_ref: str) -> tuple[Path, int] | None:
    if "::images[" not in template_ref or not template_ref.endswith("]"):
        return None
    source_hdf5_path, row_index_text = template_ref.split("::images[", maxsplit=1)
    return Path(source_hdf5_path), int(row_index_text[:-1])


def _export_hdf5_template_image(template_ref: str, destination: Path) -> None:
    parsed = _parse_hdf5_image_ref(template_ref)
    if parsed is None:
        raise ValueError(f"Invalid HDF5 template reference: {template_ref}")
    source_hdf5_path, row_index = parsed
    with h5py.File(source_hdf5_path, "r") as handle:
        image_rgb = np.asarray(handle["images"][row_index], dtype=np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(destination), image_bgr):
        raise ValueError(f"Could not export HDF5 template image: {template_ref}")


def select_template_rows_from_entropy(
    train_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
) -> pd.DataFrame:
    """Pick one highest-entropy TRAIN patch per patient."""

    if train_df.empty:
        raise ValueError("TRAIN dataframe is empty; cannot select template rows.")
    if entropy_df.empty:
        raise ValueError("Entropy dataframe is empty; cannot select template rows.")

    merged = train_df.merge(entropy_df, on="image_path", how="left")
    merged["entropy"] = merged["entropy"].fillna(0.0)
    idx = merged.groupby("patient_id")["entropy"].idxmax()
    selected_rows = merged.loc[idx].dropna(subset=["image_path"]).reset_index(drop=True)
    if selected_rows.empty:
        raise ValueError("No template rows were selected from TRAIN entropy values.")
    return selected_rows


def build_aggregate_target_from_template_rows(
    selected_rows: pd.DataFrame,
) -> npt.NDArray[np.uint8]:
    """Build the shared aggregate target from selected TRAIN template rows."""

    if selected_rows.empty:
        raise ValueError("Template row selection is empty; cannot build aggregate target.")
    template_paths = selected_rows["image_path"].astype(str).tolist()
    if _can_use_hdf5_rows(selected_rows):
        return _make_aggregate_target_from_images(_load_rgb_images_from_hdf5_rows(selected_rows))
    return make_aggregate_target(template_paths)


def _export_template_images(template_paths: list[str], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(template_paths):
        parsed = _parse_hdf5_image_ref(source)
        if parsed is not None:
            source_hdf5_path, row_index = parsed
            destination = destination_dir / (
                f"template_{index:03d}_{source_hdf5_path.stem}_row_{row_index:06d}.png"
            )
            _export_hdf5_template_image(source, destination)
            continue

        source_path = Path(source)
        destination = destination_dir / f"template_{index:03d}_{source_path.name}"
        shutil.copy(source_path, destination)


def save_template_selection_artifacts(
    output_dir: Path,
    *,
    train_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    selected_rows: pd.DataFrame,
    aggregate_target_rgb: npt.NDArray[np.uint8],
    save_entropy_cache_csv: bool,
) -> dict[str, Any]:
    """Persist shared Stage 4 entropy and template-selection provenance once."""

    output_dir.mkdir(parents=True, exist_ok=True)
    template_paths = selected_rows["image_path"].astype(str).tolist()
    template_dir = output_dir / "template_selection"
    _export_template_images(template_paths, template_dir)

    aggregate_target_path = output_dir / "aggregate_target.png"
    aggregate_target_bgr = cv2.cvtColor(aggregate_target_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(aggregate_target_path), aggregate_target_bgr):
        raise ValueError(f"Could not save aggregate target image: {aggregate_target_path}")

    patient_entropy_df = compute_patient_entropy_median(train_df, entropy_df)
    if save_entropy_cache_csv:
        entropy_df.to_csv(output_dir / "entropy_cache.csv", index=False)
        patient_entropy_df.to_csv(output_dir / "patient_entropy_median.csv", index=False)

    template_selection_records: list[dict[str, Any]] = []
    for row in selected_rows.to_dict("records"):
        template_selection_records.append(
            {
                "patient_id": int(row["patient_id"]),
                "image_path": str(row["image_path"]),
                "entropy": float(row["entropy"]),
                "source_hdf5_path": (
                    str(row["source_hdf5_path"])
                    if row.get("source_hdf5_path") is not None
                    else None
                ),
                "source_row_index": (
                    int(row["source_row_index"])
                    if row.get("source_row_index") is not None
                    else None
                ),
            }
        )
    metadata = {
        "selection_method": "highest_entropy_per_train_patient",
        "template_count": len(template_selection_records),
        "aggregate_target_path": str(aggregate_target_path.name),
        "template_dir": str(template_dir.name),
        "templates": template_selection_records,
    }
    metadata_path = output_dir / "template_selection.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, cls=NumpyEncoder), encoding="utf-8")
    return metadata


def generate_all_normalization_artifacts(
    output_dir: Path,
    *,
    aggregate_target_rgb: npt.NDArray[np.uint8],
    shared_template_dir: Path,
) -> list[dict[str, object]]:
    """Generate all Stage 4 runtime normalization artifacts from one target."""

    artifacts_root = output_dir / "runtime_normalization_artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    artifact_records: list[dict[str, object]] = []
    for method_name in FITTED_NORMALIZATION_METHODS:
        logging.info(
            "Fitting runtime '%s' normalization state from aggregate target...", method_name
        )
        stats = fit_runtime_normalization_state(method_name, aggregate_target_rgb)
        method_dir = artifacts_root / method_name.lower()
        method_dir.mkdir(parents=True, exist_ok=True)
        state_path = method_dir / "normalization_stats.json"
        state_path.write_text(json.dumps(stats, indent=2, cls=NumpyEncoder), encoding="utf-8")
        artifact_records.append(
            {
                "method": method_name,
                "state_path": state_path,
                "template_path": shared_template_dir,
                "fit_scope": "TRAIN",
            }
        )
    return artifact_records


def fit_runtime_normalization_state(
    method_name: str,
    target_rgb: npt.NDArray[np.uint8],
) -> dict[str, Any]:
    """Fit the downstream runtime-normalizer state without requiring TIAToolbox."""

    normalized_method = method_name.strip().upper()
    target_tensor = _rgb_uint8_to_torch_tensor(target_rgb)
    if normalized_method == "RUIFROK":
        return _fit_ruifrok_runtime_state(target_tensor)

    builder = _load_torch_staintools_builder()
    runtime_module = builder.build(
        normalized_method.lower(),
        concentration_solver="qr",
        use_cache=False,
        device=torch.device("cpu"),
    )
    runtime_module.fit(target_tensor)
    return _collect_torch_runtime_stats(runtime_module, normalized_method)


def _load_torch_staintools_builder() -> Any:
    try:
        from torch_staintools.constants import CONFIG
        from torch_staintools.normalizer import NormalizerBuilder
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Stage 4 runtime normalization artifact generation requires the "
            "`torch-staintools` package. Install project dependencies with "
            "`uv sync --python 3.12`."
        ) from error
    CONFIG.ENABLE_COMPILE = False
    return NormalizerBuilder


def _collect_torch_runtime_stats(runtime_module: Any, method_name: str) -> dict[str, Any]:
    if method_name == "REINHARD":
        return {
            "method": method_name,
            "target_means": _squeeze_json_tensor(runtime_module.target_means),
            "target_stds": _squeeze_json_tensor(runtime_module.target_stds),
        }
    return {
        "method": method_name,
        "stain_matrix_target": _squeeze_json_tensor(runtime_module.stain_matrix_target),
        "maxC_target": _squeeze_json_tensor(runtime_module.maxC_target),
    }


def _fit_ruifrok_runtime_state(target_tensor: torch.Tensor) -> dict[str, Any]:
    stain_matrix = _RUIFROK_HE_STAIN_MATRIX.unsqueeze(0)
    flattened_od = (
        _rgb_to_od(target_tensor).permute(0, 2, 3, 1).reshape(target_tensor.shape[0], -1, 3)
    )
    target_concentration = _solve_concentration(flattened_od, stain_matrix)
    max_c_target = torch.quantile(target_concentration, q=0.99, dim=1).clamp_min(_EPSILON)
    return {
        "method": "RUIFROK",
        "stain_matrix_source": _RUIFROK_HE_STAIN_MATRIX.tolist(),
        "stain_matrix_target": _RUIFROK_HE_STAIN_MATRIX.tolist(),
        "maxC_target": _squeeze_json_tensor(max_c_target),
    }


def _rgb_uint8_to_torch_tensor(image: npt.NDArray[np.uint8]) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(image, dtype=np.uint8))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def _squeeze_json_tensor(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().squeeze().tolist()


def _rgb_to_od(x: torch.Tensor) -> torch.Tensor:
    return -torch.log(x.clamp_min(_EPSILON))


def _solve_concentration(flattened_od: torch.Tensor, stain_matrix: torch.Tensor) -> torch.Tensor:
    solved = cast(
        torch.Tensor,
        torch.linalg.lstsq(stain_matrix.transpose(1, 2), flattened_od.transpose(1, 2)).solution,
    )
    return solved.transpose(1, 2).clamp_min(0.0)
