# mypy: ignore-errors

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from helpers.runtime_platform import load_openslide_module


@dataclass(frozen=True)
class HisegLabelColors:
    """Hardcoded HISEG XML color policy."""

    cancer: frozenset[str] = frozenset({"#8B0000", "#FF00FF", "#800080"})
    not_cancer: frozenset[str] = frozenset(
        {"#8A2BE2", "#0000FF", "#4682B4", "#00FF00", "#008000", "#FFFF00"}
    )
    rejected: frozenset[str] = frozenset({"#4B0082"})


HISEG_LABEL_COLORS = HisegLabelColors()


def _to_coord_list(geometry: Polygon | MultiPolygon) -> list[list[tuple[float, float]]]:
    """Convert a Shapely polygon geometry into plain coordinate lists."""
    if geometry.is_empty:
        return []
    geoms = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
    return [list(polygon.exterior.coords) for polygon in geoms if not polygon.is_empty]


def _coords_to_shapely_polygons(coord_lists: list[Any]) -> list[Polygon]:
    """Convert raw coordinate collections into valid Shapely polygons."""
    polygons: list[Polygon] = []
    for sublist in coord_lists:
        points_raw = sublist[0] if len(sublist) == 1 and isinstance(sublist[0], list) else sublist
        if not points_raw:
            continue

        if isinstance(points_raw[0], (int, float)):
            if len(points_raw) < 6:
                continue
            points = [(points_raw[i], points_raw[i + 1]) for i in range(0, len(points_raw), 2)]
        else:
            if len(points_raw) < 3:
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


def _remove_ambiguous_regions(
    cancer_polygons: list[Polygon], non_cancer_polygons: list[Polygon]
) -> tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """Remove overlapping regions between cancer and non-cancer polygons."""
    cancer_area = unary_union(cancer_polygons) if cancer_polygons else Polygon()
    non_cancer_area = unary_union(non_cancer_polygons) if non_cancer_polygons else Polygon()
    ambiguous_area = cancer_area.intersection(non_cancer_area)
    return cancer_area.difference(ambiguous_area), non_cancer_area.difference(ambiguous_area)


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
        del slide
        logging.info("Using SVS_XML_Handler with explicit overlap cleaning.")
        annotation_path = _require_annotation_path(kwargs)
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        root = ET.parse(annotation_path).getroot()
        if _is_hiseg_tag(kwargs.get("dataset_tag")):
            return self._load_hiseg_annotations(root)

        return self._load_line_color_annotations(
            root,
            cancer_color=kwargs.get("cancer_color"),
            not_cancer_color=kwargs.get("not_cancer_color"),
        )

    def _load_line_color_annotations(
        self,
        root: ET.Element,
        *,
        cancer_color: object,
        not_cancer_color: object,
    ) -> dict[str, list[Any]]:
        raw_cancer_coords: list[list[tuple[float, float]]] = []
        raw_not_cancer_coords: list[list[tuple[float, float]]] = []

        for annotation in root.findall(".//Annotation"):
            is_cancer = str(annotation.get("LineColor")) == str(cancer_color)
            is_non_cancer = str(annotation.get("LineColor")) == str(not_cancer_color)
            if not (is_cancer or is_non_cancer):
                continue

            for region in annotation.findall(".//Region"):
                vertices = [
                    (_required_float(vertex.get("X")), _required_float(vertex.get("Y")))
                    for vertex in region.findall(".//Vertex")
                    if vertex.get("X") is not None and vertex.get("Y") is not None
                ]
                if len(vertices) < 3:
                    continue
                if is_cancer:
                    raw_cancer_coords.append(vertices)
                else:
                    raw_not_cancer_coords.append(vertices)

        return _finalize_polygons(raw_cancer_coords, raw_not_cancer_coords)

    def _load_hiseg_annotations(self, root: ET.Element) -> dict[str, list[Any]]:
        raw_cancer_coords: list[list[tuple[float, float]]] = []
        raw_not_cancer_coords: list[list[tuple[float, float]]] = []

        for annotation in root.findall(".//Annotation"):
            normalized_color = _normalize_hex_color(annotation.get("Color"))
            if normalized_color in HISEG_LABEL_COLORS.rejected:
                continue

            is_cancer = normalized_color in HISEG_LABEL_COLORS.cancer
            is_non_cancer = normalized_color in HISEG_LABEL_COLORS.not_cancer
            if not (is_cancer or is_non_cancer):
                continue

            vertices = [
                (_required_float(node.get("X")), _required_float(node.get("Y")))
                for node in annotation.findall("./Coordinates/Coordinate")
                if node.get("X") is not None and node.get("Y") is not None
            ]
            if len(vertices) < 3:
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

        cancer_labels = {"R1", "R2", "R3", "R4", "R5"}
        non_cancer_labels = {"BG", "T", "N", "A"}
        raw_cancer_polygons: list[Polygon] = []
        raw_not_cancer_polygons: list[Polygon] = []

        openslide_module = load_openslide_module()

        offset_x_nm = int(slide.properties.get("hamamatsu.XOffsetFromSlideCentre", 0))
        offset_y_nm = int(slide.properties.get("hamamatsu.YOffsetFromSlideCentre", 0))
        mpp_x = float(slide.properties.get(openslide_module.PROPERTY_NAME_MPP_X, 0.25))
        mpp_y = float(slide.properties.get(openslide_module.PROPERTY_NAME_MPP_Y, 0.25))
        nm_per_pixel_x = mpp_x * 1000
        nm_per_pixel_y = mpp_y * 1000
        slide_width_level0, slide_height_level0 = slide.level_dimensions[0]

        root = ET.parse(annotation_path).getroot()
        for view in root.findall("ndpviewstate"):
            title_element = view.find("title")
            if title_element is None or title_element.text is None:
                continue

            label = title_element.text.strip()
            is_cancer = label in cancer_labels
            is_non_cancer = label in non_cancer_labels
            if not (is_cancer or is_non_cancer):
                continue

            pointlist = view.find("annotation/pointlist")
            if pointlist is None:
                continue

            temp_poly_coords: list[tuple[float, float]] = []
            for point in pointlist.findall("point"):
                try:
                    x_nm = _required_float(_node_text(point.find("x")))
                    y_nm = _required_float(_node_text(point.find("y")))
                    x_pixel = ((x_nm - offset_x_nm) / nm_per_pixel_x) + (slide_width_level0 / 2)
                    y_pixel = ((y_nm - offset_y_nm) / nm_per_pixel_y) + (slide_height_level0 / 2)
                    temp_poly_coords.append((x_pixel, y_pixel))
                except (ValueError, TypeError, AttributeError):
                    continue

            if len(temp_poly_coords) < 3:
                continue

            polygon = Polygon(temp_poly_coords)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_valid or polygon.is_empty or not isinstance(polygon, Polygon):
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


def _is_hiseg_tag(value: object) -> bool:
    return isinstance(value, str) and value.strip().upper() == "HISEG"
