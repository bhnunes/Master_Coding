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

from helpers.crossfold.entropy import (
    calculate_image_entropy_from_hdf5_row,
    calculate_image_entropy_from_path,
    compute_patient_entropy_median,
)
from helpers.cv2_compat import ensure_cv2_compat
from helpers.runtime_platform import load_openslide_module

cv2 = ensure_cv2_compat(cv2)
FITTED_NORMALIZATION_METHODS = ("REINHARD", "RUIFROK", "MACENKO", "VAHADANE")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def load_stain_normalizer_backend() -> Any:
    """Load the TIAToolbox stain normalizer backend with OpenSlide initialized."""

    load_openslide_module()
    try:
        from tiatoolbox.tools import stainnorm
    except ModuleNotFoundError as error:
        message = (
            "Stage 5 normalization requires a working TIAToolbox stain normalization backend. "
            "Install a Python 3.12-compatible TIAToolbox release."
        )
        if error.name == "pkg_resources":
            message = (
                "Stage 5 normalization could not import TIAToolbox because `pkg_resources` is "
                "missing. This usually means an older TIAToolbox release is installed alongside "
                "a newer setuptools version. Upgrade TIAToolbox to a Python 3.12-compatible 2.x "
                "release, or temporarily pin `setuptools<82`."
            )
        raise RuntimeError(message) from error

    return stainnorm


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


def _row_has_hdf5_columns(row: pd.Series) -> bool:
    return {"source_hdf5_path", "source_row_index"}.issubset(row.index)


def _parse_hdf5_image_ref(template_ref: str) -> tuple[Path, int] | None:
    if "::images[" not in template_ref or not template_ref.endswith("]"):
        return None
    source_hdf5_path, row_index_text = template_ref.split("::images[", maxsplit=1)
    return Path(source_hdf5_path), int(row_index_text[:-1])


def _build_image_path_lookup(train_df: pd.DataFrame) -> dict[str, pd.Series]:
    if train_df.empty:
        return {}
    deduplicated = train_df.drop_duplicates(subset=["image_path"], keep="first")
    return {
        str(image_path): row
        for image_path, row in deduplicated.set_index("image_path", drop=False).iterrows()
    }


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


def fit_normalizer_to_target(
    method_name: str,
    target_rgb: npt.NDArray[np.uint8],
) -> Any:
    """Fit one stain normalizer to a shared aggregate target image."""

    logging.info("Fitting '%s' normalizer from shared aggregate target...", method_name)
    stainnorm_module = load_stain_normalizer_backend()
    normalizer = stainnorm_module.get_normalizer(method_name)
    normalizer.fit(target_rgb)
    logging.info("Normalizer '%s' fitted from shared aggregate target.", method_name)
    return normalizer


def _collect_normalizer_stats(
    normalizer: Any,
    method_name: str,
    *,
    stainnorm_module: Any | None = None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {"method": method_name}
    stainnorm = stainnorm_module or load_stain_normalizer_backend()
    try:
        if isinstance(normalizer, stainnorm.StainNormalizer):
            if hasattr(normalizer, "stain_matrix_target"):
                stats["stain_matrix_target"] = normalizer.stain_matrix_target.tolist()
            if hasattr(normalizer, "maxC_target"):
                stats["maxC_target"] = normalizer.maxC_target.tolist()
            if method_name == "MACENKO" and hasattr(normalizer.extractor, "stains"):
                stats["stain_vectors_source_estimate"] = normalizer.extractor.stains.tolist()
        elif isinstance(normalizer, stainnorm.ReinhardNormalizer):
            stats["target_means"] = list(normalizer.target_means)
            stats["target_stds"] = list(normalizer.target_stds)
        else:
            stats["info"] = "Unknown or unsupported normalizer type."
            logging.warning("Unrecognized normalizer type for '%s'.", method_name)
    except Exception as error:
        stats["error"] = str(error)
        logging.error("Failed extracting normalizer stats: %s", error, exc_info=True)
    return stats


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
    stainnorm_module = load_stain_normalizer_backend()
    for method_name in FITTED_NORMALIZATION_METHODS:
        normalizer = fit_normalizer_to_target(method_name, aggregate_target_rgb)
        stats = _collect_normalizer_stats(
            normalizer,
            method_name,
            stainnorm_module=stainnorm_module,
        )
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


def fit_normalizer_on_train_set(
    train_df: pd.DataFrame,
    method_name: str,
    entropy_df: pd.DataFrame | None = None,
) -> tuple[Any, list[str]]:
    """Fit one stain normalizer from TRAIN-only template images."""

    logging.info("Fitting '%s' normalizer using TRAIN only...", method_name)
    if train_df.empty:
        raise ValueError("TRAIN dataframe is empty; cannot fit normalizer.")

    template_paths: list[str]
    if entropy_df is not None and not entropy_df.empty:
        logging.info("Selecting templates using cached entropy_df (no disk rereads).")
        selected_rows = select_template_rows_from_entropy(train_df, entropy_df)
        template_paths = selected_rows["image_path"].tolist()
    else:
        logging.warning("entropy_df not provided; selecting templates by rereading images (slow).")
        patient_files = train_df.groupby("patient_id")["image_path"].apply(list).to_dict()
        image_path_lookup = _build_image_path_lookup(train_df)
        template_paths = []
        for files in patient_files.values():
            best_path: str | None = None
            best_entropy = -1.0
            for image_path in files:
                row = image_path_lookup.get(image_path)
                if row is not None and _row_has_hdf5_columns(row):
                    _, entropy = calculate_image_entropy_from_hdf5_row(
                        (str(row["source_hdf5_path"]), int(row["source_row_index"]), image_path)
                    )
                else:
                    _, entropy = calculate_image_entropy_from_path(image_path)
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_path = image_path
            if best_path is not None:
                template_paths.append(best_path)
        selected_rows = train_df[train_df["image_path"].isin(template_paths)]

    if not template_paths:
        raise ValueError("No templates found to fit normalizer.")

    target_rgb = build_aggregate_target_from_template_rows(selected_rows)
    stainnorm_module = load_stain_normalizer_backend()
    normalizer = stainnorm_module.get_normalizer(method_name)
    normalizer.fit(target_rgb)
    logging.info("Normalizer '%s' fitted.", method_name)
    return normalizer, template_paths


def save_normalizer_stats(
    normalizer: Any,
    method_name: str,
    output_dir: Path,
    template_paths: list[str],
    stainnorm_module: Any | None = None,
) -> None:
    """Persist normalizer metadata and template images for provenance."""

    stats = _collect_normalizer_stats(
        normalizer,
        method_name,
        stainnorm_module=stainnorm_module,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "normalization_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, cls=NumpyEncoder), encoding="utf-8")
    logging.info("Normalization stats saved: %s", stats_path)

    template_dir = output_dir / "normalization_templates"
    _export_template_images(template_paths, template_dir)
    logging.info("Saved %s template images: %s", len(template_paths), template_dir)
