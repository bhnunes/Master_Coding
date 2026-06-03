# mypy: ignore-errors

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from helpers.runtime_platform import load_openslide_module


@dataclass(frozen=True)
class HiesdLabelColors:
    """Hardcoded HIESD XML color policy."""

    cancer: frozenset[str] = frozenset({"#8B0000", "#FF00FF", "#800080"})
    not_cancer: frozenset[str] = frozenset(
        {"#8A2BE2", "#0000FF", "#4682B4", "#00FF00", "#008000", "#FFFF00"}
    )
    rejected: frozenset[str] = frozenset({"#4B0082"})


HIESD_LABEL_COLORS = HiesdLabelColors()


@dataclass(frozen=True)
class ChileLabelColors:
    """Hardcoded CHILE XML line-color policy."""

    cancer: str = "255"
    not_cancer: frozenset[str] = frozenset({"65280", "65408"})


CHILE_LABEL_COLORS = ChileLabelColors()

MIN_POLYGON_POINTS = 3
MIN_FLAT_COORD_VALUES = 6
NDPI_CANCER_LABELS = frozenset({"R1", "R2", "R3", "R4", "R5"})
NDPI_NON_CANCER_LABELS = frozenset({"BG", "T", "N", "A"})


@dataclass(frozen=True)
class NdpiSlideCalibration:
    offset_x_nm: int
    offset_y_nm: int
    nm_per_pixel_x: float
    nm_per_pixel_y: float
    slide_width_level0: int
    slide_height_level0: int


def _iter_polygon_parts(geometry: BaseGeometry):
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    if isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
        return
    if isinstance(geometry, GeometryCollection):
        for subgeometry in geometry.geoms:
            yield from _iter_polygon_parts(subgeometry)


def _to_coord_list(geometry: BaseGeometry) -> list[list[tuple[float, float]]]:
    """Convert a Shapely polygon geometry into plain coordinate lists."""
    if geometry.is_empty:
        return []
    coord_lists: list[list[tuple[float, float]]] = []
    for polygon in _iter_polygon_parts(geometry):
        if polygon.is_empty:
            continue
        coords: list[tuple[float, float]] = [
            (float(x_coord), float(y_coord)) for x_coord, y_coord in polygon.exterior.coords
        ]
        coord_lists.append(coords)
    return coord_lists


def _coords_to_shapely_polygons(coord_lists: list[Any]) -> list[Polygon]:
    """Convert raw coordinate collections into valid Shapely polygons."""
    polygons: list[Polygon] = []
    for sublist in coord_lists:
        points_raw = sublist[0] if len(sublist) == 1 and isinstance(sublist[0], list) else sublist
        if not points_raw:
            continue

        if isinstance(points_raw[0], (int, float)):
            if len(points_raw) < MIN_FLAT_COORD_VALUES:
                continue
            points = [(points_raw[i], points_raw[i + 1]) for i in range(0, len(points_raw), 2)]
        else:
            if len(points_raw) < MIN_POLYGON_POINTS:
                continue
            points = points_raw

        try:
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty and isinstance(polygon, Polygon):
                polygons.append(polygon)
        except Exception as error:
            logging.warning(
                "Could not convert points to a valid polygon. Error: %s. Skipping.", error
            )
    return polygons


def _has_positive_area_overlap(first_polygon: Polygon, second_polygon: Polygon) -> bool:
    if not first_polygon.intersects(second_polygon):
        return False
    return first_polygon.intersection(second_polygon).area > 0


def _union_polygons(polygons: list[Polygon]) -> BaseGeometry:
    return unary_union(polygons) if polygons else Polygon()


def _remove_ambiguous_regions(
    cancer_polygons: list[Polygon], non_cancer_polygons: list[Polygon]
) -> tuple[BaseGeometry, BaseGeometry]:
    """Drop whole annotations with positive-area cross-label overlap."""
    cancer_to_remove: set[int] = set()
    non_cancer_to_remove: set[int] = set()

    for cancer_index, cancer_polygon in enumerate(cancer_polygons):
        for non_cancer_index, non_cancer_polygon in enumerate(non_cancer_polygons):
            if _has_positive_area_overlap(cancer_polygon, non_cancer_polygon):
                cancer_to_remove.add(cancer_index)
                non_cancer_to_remove.add(non_cancer_index)

    if cancer_to_remove or non_cancer_to_remove:
        logging.warning(
            "Discarding %s cancer annotation polygon(s) and %s not-cancer annotation "
            "polygon(s) because they have positive-area cross-label overlap.",
            len(cancer_to_remove),
            len(non_cancer_to_remove),
        )
    else:
        logging.info("No positive-area cross-label annotation overlaps found.")

    clean_cancer_polygons = [
        polygon for index, polygon in enumerate(cancer_polygons) if index not in cancer_to_remove
    ]
    clean_non_cancer_polygons = [
        polygon
        for index, polygon in enumerate(non_cancer_polygons)
        if index not in non_cancer_to_remove
    ]
    return _union_polygons(clean_cancer_polygons), _union_polygons(clean_non_cancer_polygons)


class BaseHandler(ABC):
    """Template for all annotation data handlers."""

    @abstractmethod
    def _load_raw_annotations(self, slide: Any, **kwargs: Any) -> dict[str, list[Any]]:
        """Load and normalize annotations for one slide."""

    def load_annotations(self, slide: Any, **kwargs: Any) -> dict[str, list[Any]]:
        """Load annotations and validate the output contract."""
        raw_data = self._load_raw_annotations(slide, **kwargs)
        if not isinstance(raw_data, dict):
            raise TypeError(
                f"{self.__class__.__name__} failed to return a dictionary. "
                f"Got {type(raw_data)} instead."
            )

        required_keys = {"cancer_polygons", "not_cancer_polygons"}
        actual_keys = set(raw_data.keys())
        if not required_keys.issubset(actual_keys):
            missing_keys = required_keys - actual_keys
            raise ValueError(
                f"{self.__class__.__name__} failed to return required keys. Missing: {missing_keys}"
            )
        return raw_data


class SVS_XML_Handler(BaseHandler):
    """Handle `.svs` slides with `.xml` annotations."""

    def _load_raw_annotations(self, slide: Any, **kwargs: Any) -> dict[str, list[Any]]:
        logging.info("Using SVS_XML_Handler with explicit overlap cleaning.")
        annotation_path = _require_annotation_path(kwargs)
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        root = ET.parse(annotation_path).getroot()
        if _is_hiesd_tag(kwargs.get("dataset_tag")):
            return self._load_hiesd_annotations(
                slide,
                root,
                annotation_level=_required_int(kwargs.get("hiesd_xml_coord_level")),
            )
        if _is_chile_tag(kwargs.get("dataset_tag")):
            return self._load_chile_annotations(root)
        raise ValueError(
            "Unsupported .svs/.xml dataset tag. Supported tags are 'HIESD' and 'CHILE'."
        )

    def _load_chile_annotations(self, root: ET.Element) -> dict[str, list[Any]]:
        raw_cancer_coords: list[list[tuple[float, float]]] = []
        raw_not_cancer_coords: list[list[tuple[float, float]]] = []

        for annotation in root.findall(".//Annotation"):
            line_color = str(annotation.get("LineColor"))
            is_cancer = line_color == CHILE_LABEL_COLORS.cancer
            is_non_cancer = line_color in CHILE_LABEL_COLORS.not_cancer
            if not (is_cancer or is_non_cancer):
                continue

            for region in annotation.findall(".//Region"):
                vertices = [
                    (_required_float(vertex.get("X")), _required_float(vertex.get("Y")))
                    for vertex in region.findall(".//Vertex")
                    if vertex.get("X") is not None and vertex.get("Y") is not None
                ]
                if len(vertices) < MIN_POLYGON_POINTS:
                    continue
                if is_cancer:
                    raw_cancer_coords.append(vertices)
                else:
                    raw_not_cancer_coords.append(vertices)

        return _finalize_polygons(raw_cancer_coords, raw_not_cancer_coords)

    def _load_hiesd_annotations(
        self,
        slide: Any,
        root: ET.Element,
        *,
        annotation_level: int,
    ) -> dict[str, list[Any]]:
        if slide is None:
            raise ValueError("slide is required for HIESD annotation scaling")
        if annotation_level < 0:
            raise ValueError("hiesd_xml_coord_level must be non-negative")
        if annotation_level >= len(slide.level_downsamples):
            raise ValueError(
                "HIESD annotation level "
                f"{annotation_level} is out of range for slide with "
                f"{len(slide.level_downsamples)} levels."
            )

        annotation_scale = float(slide.level_downsamples[annotation_level])
        logging.info(
            "Scaling HIESD XML annotations from level %s to level 0 with downsample %.6f.",
            annotation_level,
            annotation_scale,
        )
        raw_cancer_coords: list[list[tuple[float, float]]] = []
        raw_not_cancer_coords: list[list[tuple[float, float]]] = []

        for annotation in root.findall(".//Annotation"):
            normalized_color = _normalize_hex_color(annotation.get("Color"))
            if normalized_color in HIESD_LABEL_COLORS.rejected:
                continue

            is_cancer = normalized_color in HIESD_LABEL_COLORS.cancer
            is_non_cancer = normalized_color in HIESD_LABEL_COLORS.not_cancer
            if not (is_cancer or is_non_cancer):
                continue

            vertices = [
                (
                    _required_float(node.get("X")) * annotation_scale,
                    _required_float(node.get("Y")) * annotation_scale,
                )
                for node in annotation.findall("./Coordinates/Coordinate")
                if node.get("X") is not None and node.get("Y") is not None
            ]
            if len(vertices) < MIN_POLYGON_POINTS:
                continue
            if is_cancer:
                raw_cancer_coords.append(vertices)
            else:
                raw_not_cancer_coords.append(vertices)

        return _finalize_polygons(raw_cancer_coords, raw_not_cancer_coords)


class NDPI_NDPA_Handler(BaseHandler):
    """Handle `.ndpi` slides with `.ndpa` annotations."""

    def _load_raw_annotations(self, slide: Any, **kwargs: Any) -> dict[str, list[Any]]:
        logging.info("Using NDPI_NDPA_Handler to load annotations.")
        annotation_path = _require_annotation_path(kwargs)
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        raw_cancer_polygons: list[Polygon] = []
        raw_not_cancer_polygons: list[Polygon] = []
        calibration = _build_ndpi_slide_calibration(slide)

        root = ET.parse(annotation_path).getroot()
        for view in root.findall("ndpviewstate"):
            title_element = view.find("title")
            if title_element is None or title_element.text is None:
                continue

            label = title_element.text.strip()
            is_cancer, is_non_cancer = _classify_ndpi_label(label)
            if not (is_cancer or is_non_cancer):
                continue

            polygon = _build_ndpi_polygon(view, calibration)
            if polygon is None:
                continue
            if is_cancer:
                raw_cancer_polygons.append(polygon)
            else:
                raw_not_cancer_polygons.append(polygon)

        return _finalize_polygon_objects(raw_cancer_polygons, raw_not_cancer_polygons)


class JSON_Handler(BaseHandler):
    """Handle JSON annotation files with overlap cleaning."""

    def _load_raw_annotations(self, slide: Any, **kwargs: Any) -> dict[str, list[Any]]:
        del slide
        logging.info("Using JSON_Handler to load and clean annotations.")
        annotation_path = _require_annotation_path(kwargs)
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        with open(annotation_path, encoding="utf-8") as annotation_file:
            data = json.load(annotation_file)

        raw_cancer_coords = data.get("cancer_polygons", [])
        raw_not_cancer_coords = data.get("not_cancer_polygons", [])
        logging.info(
            "Found %s raw 'cancer' annotation(s) and %s raw 'not_cancer' "
            "annotation(s) in JSON file.",
            len(raw_cancer_coords),
            len(raw_not_cancer_coords),
        )
        return _finalize_polygons(raw_cancer_coords, raw_not_cancer_coords)


def _finalize_polygons(
    raw_cancer_coords: list[Any], raw_not_cancer_coords: list[Any]
) -> dict[str, list[Any]]:
    raw_cancer_polygons = _coords_to_shapely_polygons(raw_cancer_coords)
    raw_not_cancer_polygons = _coords_to_shapely_polygons(raw_not_cancer_coords)
    return _finalize_polygon_objects(raw_cancer_polygons, raw_not_cancer_polygons)


def _finalize_polygon_objects(
    raw_cancer_polygons: list[Polygon], raw_not_cancer_polygons: list[Polygon]
) -> dict[str, list[Any]]:
    logging.info(
        "Successfully converted to %s 'cancer' and %s 'not_cancer' Shapely polygons.",
        len(raw_cancer_polygons),
        len(raw_not_cancer_polygons),
    )
    if not raw_cancer_polygons and not raw_not_cancer_polygons:
        logging.warning(
            "No valid polygons were parsed from the annotation file. "
            "Please check the structure and content."
        )
        return {"cancer_polygons": [], "not_cancer_polygons": []}

    clean_cancer_area, clean_non_cancer_area = _remove_ambiguous_regions(
        raw_cancer_polygons, raw_not_cancer_polygons
    )
    final_cancer_list = _to_coord_list(clean_cancer_area)
    final_not_cancer_list = _to_coord_list(clean_non_cancer_area)
    logging.info(
        "After cleaning overlaps, returning %s final 'cancer' polygons and %s "
        "final 'not_cancer' polygons.",
        len(final_cancer_list),
        len(final_not_cancer_list),
    )
    return {
        "cancer_polygons": final_cancer_list,
        "not_cancer_polygons": final_not_cancer_list,
    }


def _require_annotation_path(kwargs: dict[str, Any]) -> str:
    annotation_path = kwargs.get("annotation_path")
    if not isinstance(annotation_path, str) or not annotation_path:
        raise ValueError("annotation_path is required")
    return annotation_path


def _required_float(value: str | None) -> float:
    if value is None:
        raise ValueError("Missing numeric value")
    return float(value)


def _required_int(value: object) -> int:
    if value is None:
        raise ValueError("Missing integer value")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Missing integer value")
    integer_value = cast(int | str, value)
    return int(integer_value)


def _node_text(node: Any) -> str | None:
    if node is None:
        return None
    return cast(str | None, node.text)


def _normalize_hex_color(value: str | None) -> str:
    if value is None:
        return ""
    normalized = value.strip().upper()
    if not normalized:
        return ""
    return normalized if normalized.startswith("#") else f"#{normalized}"


def _is_hiesd_tag(value: object) -> bool:
    return isinstance(value, str) and value.strip().upper() == "HIESD"


def _is_chile_tag(value: object) -> bool:
    return isinstance(value, str) and value.strip() == "CHILE"


def _build_ndpi_slide_calibration(slide: Any) -> NdpiSlideCalibration:
    openslide_module = load_openslide_module()
    mpp_x = float(slide.properties.get(openslide_module.PROPERTY_NAME_MPP_X, 0.25))
    mpp_y = float(slide.properties.get(openslide_module.PROPERTY_NAME_MPP_Y, 0.25))
    slide_width_level0, slide_height_level0 = slide.level_dimensions[0]
    return NdpiSlideCalibration(
        offset_x_nm=int(slide.properties.get("hamamatsu.XOffsetFromSlideCentre", 0)),
        offset_y_nm=int(slide.properties.get("hamamatsu.YOffsetFromSlideCentre", 0)),
        nm_per_pixel_x=mpp_x * 1000,
        nm_per_pixel_y=mpp_y * 1000,
        slide_width_level0=slide_width_level0,
        slide_height_level0=slide_height_level0,
    )


def _classify_ndpi_label(label: str) -> tuple[bool, bool]:
    return label in NDPI_CANCER_LABELS, label in NDPI_NON_CANCER_LABELS


def _build_ndpi_polygon(
    view: ET.Element,
    calibration: NdpiSlideCalibration,
) -> Polygon | None:
    pointlist = view.find("annotation/pointlist")
    if pointlist is None:
        return None

    temp_poly_coords: list[tuple[float, float]] = []
    for point in pointlist.findall("point"):
        pixel_coords = _ndpi_point_to_pixel_coords(point, calibration)
        if pixel_coords is not None:
            temp_poly_coords.append(pixel_coords)

    if len(temp_poly_coords) < MIN_POLYGON_POINTS:
        return None

    polygon = Polygon(temp_poly_coords)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if not polygon.is_valid or polygon.is_empty or not isinstance(polygon, Polygon):
        return None
    return polygon


def _ndpi_point_to_pixel_coords(
    point: ET.Element,
    calibration: NdpiSlideCalibration,
) -> tuple[float, float] | None:
    try:
        x_nm = _required_float(_node_text(point.find("x")))
        y_nm = _required_float(_node_text(point.find("y")))
    except (ValueError, TypeError, AttributeError):
        return None

    x_pixel = ((x_nm - calibration.offset_x_nm) / calibration.nm_per_pixel_x) + (
        calibration.slide_width_level0 / 2
    )
    y_pixel = ((y_nm - calibration.offset_y_nm) / calibration.nm_per_pixel_y) + (
        calibration.slide_height_level0 / 2
    )
    return x_pixel, y_pixel
