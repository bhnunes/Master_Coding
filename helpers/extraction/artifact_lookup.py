from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from helpers.provenance import hash_file_sha256, hash_json_payload


@dataclass(frozen=True)
class GeoJsonLookup:
    """Single-pass index for resolving artifact GeoJSON files by slide stem."""

    legacy_by_stem: dict[str, Path]
    hashed_by_stem: dict[str, tuple[Path, ...]]

    @classmethod
    def from_directory(cls, geojson_dir: Path) -> GeoJsonLookup:
        legacy_by_stem: dict[str, Path] = {}
        hashed_lists_by_stem: dict[str, list[Path]] = {}

        for candidate in geojson_dir.iterdir():
            if not candidate.is_file() or candidate.suffix.lower() != ".geojson":
                continue

            stem = candidate.stem
            stem_prefix, separator, _suffix = stem.partition("__")
            if separator:
                hashed_lists_by_stem.setdefault(stem_prefix, []).append(candidate)
                continue

            legacy_by_stem[stem] = candidate

        return cls(
            legacy_by_stem=legacy_by_stem,
            hashed_by_stem={
                stem: tuple(sorted(paths)) for stem, paths in hashed_lists_by_stem.items()
            },
        )

    def resolve_for_slide(self, image_path: Path) -> Path | None:
        """Resolve the artifact GeoJSON for one slide, failing closed on ambiguity."""

        legacy_candidate = self.legacy_by_stem.get(image_path.stem)
        hashed_candidates = self.hashed_by_stem.get(image_path.stem, ())
        if legacy_candidate is not None and hashed_candidates:
            raise ValueError(
                f"Ambiguous artifact GeoJSON mapping for '{image_path.name}'. "
                "Remove duplicate legacy/collision-safe files before processing."
            )
        if legacy_candidate is not None:
            return legacy_candidate
        if len(hashed_candidates) > 1:
            raise ValueError(
                f"Ambiguous artifact GeoJSON mapping for '{image_path.name}'. "
                "Multiple collision-safe GeoJSON files share this slide stem."
            )
        if len(hashed_candidates) == 1:
            return hashed_candidates[0]
        return None


def resolve_geojson_for_slide(
    geojson_dir: Path,
    image_path: Path,
    *,
    lookup: GeoJsonLookup | None = None,
) -> Path | None:
    """Resolve the artifact GeoJSON for one slide, failing closed on ambiguity."""

    resolver = lookup if lookup is not None else GeoJsonLookup.from_directory(geojson_dir)
    return resolver.resolve_for_slide(image_path)


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
    hiseg_xml_coord_level: int,
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
            "hiseg_xml_coord_level": int(hiseg_xml_coord_level),
        }
    )
