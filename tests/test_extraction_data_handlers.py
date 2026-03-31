# mypy: disable-error-code=no-untyped-call

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from helpers.extraction.data_handlers import (
    BaseHandler,
    JSON_Handler,
    NDPI_NDPA_Handler,
    SVS_XML_Handler,
    _coords_to_shapely_polygons,
    _node_text,
    _remove_ambiguous_regions,
    _require_annotation_path,
    _required_float,
    _to_coord_list,
)


class DummyHandler(BaseHandler):
    def __init__(self, result: object) -> None:
        self.result = result

    def _load_raw_annotations(self, slide: object, **kwargs: object) -> dict[str, list[object]]:
        del slide, kwargs
        return cast(dict[str, list[object]], self.result)


class FakeSlide:
    def __init__(self) -> None:
        self.properties = {
            "hamamatsu.XOffsetFromSlideCentre": "1000",
            "hamamatsu.YOffsetFromSlideCentre": "2000",
            "openslide.mpp-x": "0.5",
            "openslide.mpp-y": "0.25",
        }
        self.level_dimensions = [(1000, 800)]
        self.level_downsamples = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def test_to_coord_list_handles_empty_polygon() -> None:
    from shapely.geometry import Polygon

    assert _to_coord_list(Polygon()) == []


def test_coords_to_shapely_polygons_accepts_flat_and_nested_coordinate_formats() -> None:
    polygons = _coords_to_shapely_polygons(
        [
            [[0, 0, 4, 0, 4, 4, 0, 4]],
            [[(10.0, 10.0), (14.0, 10.0), (14.0, 14.0), (10.0, 14.0)]],
            [[1, 2, 3, 4]],
        ]
    )

    assert len(polygons) == 2
    assert round(polygons[0].area, 2) == 16.0
    assert round(polygons[1].area, 2) == 16.0


def test_remove_ambiguous_regions_removes_overlap_from_both_classes() -> None:
    from shapely.geometry import Polygon

    cancer = [Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])]
    not_cancer = [Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])]

    clean_cancer, clean_non_cancer = _remove_ambiguous_regions(cancer, not_cancer)

    assert round(clean_cancer.area, 2) == 12.0
    assert round(clean_non_cancer.area, 2) == 12.0


def test_base_handler_rejects_non_dict_results() -> None:
    with pytest.raises(TypeError, match="failed to return a dictionary"):
        DummyHandler([]).load_annotations(slide=None)


def test_base_handler_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="failed to return required keys"):
        DummyHandler({"cancer_polygons": []}).load_annotations(slide=None)


def test_json_handler_loads_and_cleans_overlapping_annotations(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "cancer_polygons": [[[(0, 0), (4, 0), (4, 4), (0, 4)]]],
                "not_cancer_polygons": [[[(2, 2), (6, 2), (6, 6), (2, 6)]]],
            }
        ),
        encoding="utf-8",
    )

    result = JSON_Handler().load_annotations(None, annotation_path=str(annotation_path))

    assert len(result["cancer_polygons"]) == 1
    assert len(result["not_cancer_polygons"]) == 1


def test_svs_xml_handler_rejects_unsupported_dataset_tag(tmp_path: Path) -> None:
    annotation_path = tmp_path / "slide.xml"
    annotation_path.write_text(
        """
        <Annotations>
          <Annotation LineColor="255">
            <Region>
              <Vertex X="0" Y="0" />
              <Vertex X="4" Y="0" />
              <Vertex X="4" Y="4" />
            </Region>
          </Annotation>
          <Annotation LineColor="65280">
            <Region>
              <Vertex X="10" Y="10" />
              <Vertex X="14" Y="10" />
              <Vertex X="14" Y="14" />
            </Region>
          </Annotation>
          <Annotation LineColor="123">
            <Region>
              <Vertex X="20" Y="20" />
              <Vertex X="24" Y="20" />
              <Vertex X="24" Y="24" />
            </Region>
          </Annotation>
        </Annotations>
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Supported tags are 'HISEG' and 'CHILE'"):
        SVS_XML_Handler().load_annotations(
            None,
            annotation_path=str(annotation_path),
            dataset_tag="TCGA",
        )


def test_svs_xml_handler_supports_hiseg_asap_color_mapping(tmp_path: Path) -> None:
    annotation_path = tmp_path / "slide.xml"
    annotation_path.write_text(
        """
        <ASAP_Annotations>
          <Annotations>
            <Annotation Color="#8B0000">
              <Coordinates>
                <Coordinate Order="0" X="0" Y="0" />
                <Coordinate Order="1" X="4" Y="0" />
                <Coordinate Order="2" X="4" Y="4" />
              </Coordinates>
            </Annotation>
            <Annotation Color="#4682B4">
              <Coordinates>
                <Coordinate Order="0" X="10" Y="10" />
                <Coordinate Order="1" X="14" Y="10" />
                <Coordinate Order="2" X="14" Y="14" />
              </Coordinates>
            </Annotation>
            <Annotation Color="#4B0082">
              <Coordinates>
                <Coordinate Order="0" X="20" Y="20" />
                <Coordinate Order="1" X="24" Y="20" />
                <Coordinate Order="2" X="24" Y="24" />
              </Coordinates>
            </Annotation>
          </Annotations>
        </ASAP_Annotations>
        """,
        encoding="utf-8",
    )

    result = SVS_XML_Handler().load_annotations(
        FakeSlide(),
        annotation_path=str(annotation_path),
        dataset_tag="HISEG",
        hiseg_xml_coord_level=0,
    )

    assert len(result["cancer_polygons"]) == 1
    assert len(result["not_cancer_polygons"]) == 1


def test_svs_xml_handler_uses_hardcoded_chile_colors_with_65280(tmp_path: Path) -> None:
    annotation_path = tmp_path / "slide.xml"
    annotation_path.write_text(
        """
        <Annotations>
          <Annotation LineColor="255">
            <Region>
              <Vertex X="0" Y="0" />
              <Vertex X="4" Y="0" />
              <Vertex X="4" Y="4" />
            </Region>
          </Annotation>
          <Annotation LineColor="65280">
            <Region>
              <Vertex X="10" Y="10" />
              <Vertex X="14" Y="10" />
              <Vertex X="14" Y="14" />
            </Region>
          </Annotation>
        </Annotations>
        """,
        encoding="utf-8",
    )

    result = SVS_XML_Handler().load_annotations(
        None,
        annotation_path=str(annotation_path),
        dataset_tag="CHILE",
    )

    assert len(result["cancer_polygons"]) == 1
    assert len(result["not_cancer_polygons"]) == 1


def test_svs_xml_handler_uses_hardcoded_chile_colors_with_65408(tmp_path: Path) -> None:
    annotation_path = tmp_path / "slide.xml"
    annotation_path.write_text(
        """
        <Annotations>
          <Annotation LineColor="255">
            <Region>
              <Vertex X="0" Y="0" />
              <Vertex X="4" Y="0" />
              <Vertex X="4" Y="4" />
            </Region>
          </Annotation>
          <Annotation LineColor="65408">
            <Region>
              <Vertex X="10" Y="10" />
              <Vertex X="14" Y="10" />
              <Vertex X="14" Y="14" />
            </Region>
          </Annotation>
        </Annotations>
        """,
        encoding="utf-8",
    )

    result = SVS_XML_Handler().load_annotations(
        None,
        annotation_path=str(annotation_path),
        dataset_tag="CHILE",
    )

    assert len(result["cancer_polygons"]) == 1
    assert len(result["not_cancer_polygons"]) == 1


def test_svs_xml_handler_merges_multiple_hiseg_cancer_colors(tmp_path: Path) -> None:
    annotation_path = tmp_path / "slide.xml"
    annotation_path.write_text(
        """
        <ASAP_Annotations>
          <Annotations>
            <Annotation Color="#8B0000">
              <Coordinates>
                <Coordinate Order="0" X="0" Y="0" />
                <Coordinate Order="1" X="4" Y="0" />
                <Coordinate Order="2" X="4" Y="4" />
              </Coordinates>
            </Annotation>
            <Annotation Color="#FF00FF">
              <Coordinates>
                <Coordinate Order="0" X="10" Y="10" />
                <Coordinate Order="1" X="14" Y="10" />
                <Coordinate Order="2" X="14" Y="14" />
              </Coordinates>
            </Annotation>
            <Annotation Color="#800080">
              <Coordinates>
                <Coordinate Order="0" X="20" Y="20" />
                <Coordinate Order="1" X="24" Y="20" />
                <Coordinate Order="2" X="24" Y="24" />
              </Coordinates>
            </Annotation>
          </Annotations>
        </ASAP_Annotations>
        """,
        encoding="utf-8",
    )

    result = SVS_XML_Handler().load_annotations(
        FakeSlide(),
        annotation_path=str(annotation_path),
        dataset_tag="HISEG",
        hiseg_xml_coord_level=0,
    )

    assert len(result["cancer_polygons"]) == 3
    assert result["not_cancer_polygons"] == []


def test_svs_xml_handler_scales_hiseg_coordinates_from_configured_level(tmp_path: Path) -> None:
    annotation_path = tmp_path / "slide.xml"
    annotation_path.write_text(
        """
        <ASAP_Annotations>
          <Annotations>
            <Annotation Color="#8B0000">
              <Coordinates>
                <Coordinate Order="0" X="1" Y="2" />
                <Coordinate Order="1" X="3" Y="2" />
                <Coordinate Order="2" X="3" Y="4" />
              </Coordinates>
            </Annotation>
          </Annotations>
        </ASAP_Annotations>
        """,
        encoding="utf-8",
    )

    result = SVS_XML_Handler().load_annotations(
        FakeSlide(),
        annotation_path=str(annotation_path),
        dataset_tag="HISEG",
        hiseg_xml_coord_level=6,
    )

    assert result["not_cancer_polygons"] == []
    assert len(result["cancer_polygons"]) == 1
    assert result["cancer_polygons"][0][0] == result["cancer_polygons"][0][-1]
    x_coords = [x_coord for x_coord, _ in result["cancer_polygons"][0]]
    y_coords = [y_coord for _, y_coord in result["cancer_polygons"][0]]
    assert min(x_coords) == 64.0
    assert max(x_coords) == 192.0
    assert min(y_coords) == 128.0
    assert max(y_coords) == 256.0


def test_ndpi_ndpa_handler_converts_points_and_filters_titles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    annotation_path = tmp_path / "slide.ndpa"
    annotation_path.write_text(
        """
        <root>
          <ndpviewstate>
            <title>R1</title>
            <annotation>
              <pointlist>
                <point><x>1000</x><y>2000</y></point>
                <point><x>2000</x><y>2000</y></point>
                <point><x>2000</x><y>3000</y></point>
              </pointlist>
            </annotation>
          </ndpviewstate>
          <ndpviewstate>
            <title>BG</title>
            <annotation>
              <pointlist>
                <point><x>3000</x><y>4000</y></point>
                <point><x>4000</x><y>4000</y></point>
                <point><x>4000</x><y>5000</y></point>
              </pointlist>
            </annotation>
          </ndpviewstate>
          <ndpviewstate>
            <title>IGNORE</title>
            <annotation>
              <pointlist>
                <point><x>0</x><y>0</y></point>
                <point><x>1</x><y>0</y></point>
                <point><x>1</x><y>1</y></point>
              </pointlist>
            </annotation>
          </ndpviewstate>
        </root>
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "helpers.extraction.data_handlers.load_openslide_module",
        lambda: type(
            "FakeOpenSlideModule",
            (),
            {"PROPERTY_NAME_MPP_X": "openslide.mpp-x", "PROPERTY_NAME_MPP_Y": "openslide.mpp-y"},
        )(),
    )

    result = NDPI_NDPA_Handler().load_annotations(
        FakeSlide(),
        annotation_path=str(annotation_path),
    )

    assert len(result["cancer_polygons"]) == 1
    assert len(result["not_cancer_polygons"]) == 1


def test_finalize_helpers_return_empty_lists_when_no_polygons(tmp_path: Path) -> None:
    annotation_path = tmp_path / "empty.json"
    annotation_path.write_text(
        json.dumps({"cancer_polygons": [], "not_cancer_polygons": []}),
        encoding="utf-8",
    )

    result = JSON_Handler().load_annotations(None, annotation_path=str(annotation_path))

    assert result == {"cancer_polygons": [], "not_cancer_polygons": []}


def test_annotation_helpers_validate_required_values() -> None:
    assert _require_annotation_path({"annotation_path": "file.xml"}) == "file.xml"
    assert _required_float("1.5") == 1.5
    assert _node_text(type("Node", (), {"text": "abc"})()) == "abc"

    with pytest.raises(ValueError, match="annotation_path is required"):
        _require_annotation_path({})
    with pytest.raises(ValueError, match="Missing numeric value"):
        _required_float(None)
    assert _node_text(None) is None
