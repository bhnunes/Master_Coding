from __future__ import annotations

from pathlib import Path

from helpers.provenance import hash_file_sha256, hash_json_payload


def resolve_geojson_for_slide(geojson_dir: Path, image_path: Path) -> Path | None:
    """Resolve the artifact GeoJSON for one slide, failing closed on ambiguity."""

    legacy_candidate = geojson_dir / f"{image_path.stem}.geojson"
    hashed_candidates = sorted(geojson_dir.glob(f"{image_path.stem}__*.geojson"))
    if legacy_candidate.exists() and hashed_candidates:
        raise ValueError(
            f"Ambiguous artifact GeoJSON mapping for '{image_path.name}'. "
            "Remove duplicate legacy/collision-safe files before processing."
        )
    if legacy_candidate.exists():
        return legacy_candidate
    if len(hashed_candidates) > 1:
        raise ValueError(
            f"Ambiguous artifact GeoJSON mapping for '{image_path.name}'. "
            "Multiple collision-safe GeoJSON files share this slide stem."
        )
    if len(hashed_candidates) == 1:
        return hashed_candidates[0]
    return None


def build_processing_signature(
    *,
    image_path: Path,
    annotation_path: Path,
    artifacts_geojson_path: Path | None,
    window_size: int,
    stride: int,
    match_percentage: float,
    tissue_percentage: float,
    target_level: int,
    use_advanced_artifact_filtering: bool,
) -> str:
    """Build a fail-closed signature for Stage 2 processing inputs and settings."""

    return hash_json_payload(
        {
            "image_path": str(image_path),
            "image_sha256": hash_file_sha256(image_path),
            "annotation_path": str(annotation_path),
            "annotation_sha256": hash_file_sha256(annotation_path),
            "artifacts_geojson_path": (
                str(artifacts_geojson_path) if artifacts_geojson_path is not None else None
            ),
            "artifacts_geojson_sha256": (
                hash_file_sha256(artifacts_geojson_path)
                if artifacts_geojson_path is not None and artifacts_geojson_path.exists()
                else None
            ),
            "window_size": int(window_size),
            "stride": int(stride),
            "match_percentage": float(match_percentage),
            "tissue_percentage": float(tissue_percentage),
            "target_level": int(target_level),
            "use_advanced_artifact_filtering": bool(use_advanced_artifact_filtering),
        }
    )
