from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from helpers.crossfold.entropy import calculate_image_entropy_from_path
from helpers.runtime_platform import load_openslide_module


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
    from tiatoolbox.tools import stainnorm

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
        merged = train_df.merge(entropy_df, on="image_path", how="left")
        merged["entropy"] = merged["entropy"].fillna(0.0)
        idx = merged.groupby("patient_id")["entropy"].idxmax()
        template_paths = merged.loc[idx, "image_path"].dropna().tolist()
    else:
        logging.warning("entropy_df not provided; selecting templates by rereading images (slow).")
        patient_files = train_df.groupby("patient_id")["image_path"].apply(list).to_dict()
        template_paths = []
        for files in patient_files.values():
            best_path: str | None = None
            best_entropy = -1.0
            for image_path in files:
                _, entropy = calculate_image_entropy_from_path(image_path)
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_path = image_path
            if best_path is not None:
                template_paths.append(best_path)

    if not template_paths:
        raise ValueError("No templates found to fit normalizer.")

    target_rgb = make_aggregate_target(template_paths)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "normalization_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, cls=NumpyEncoder), encoding="utf-8")
    logging.info("Normalization stats saved: %s", stats_path)

    template_dir = output_dir / "normalization_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(template_paths):
        source_path = Path(source)
        destination = template_dir / f"template_{index:03d}_{source_path.name}"
        shutil.copy(source_path, destination)
    logging.info("Saved %s template images: %s", len(template_paths), template_dir)
