import openslide
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import logging
import os
import json
from abc import ABC, abstractmethod


# --- Module-Level Helper Functions ---

def _to_coord_list(geometry):
    """Converts a Shapely Polygon or MultiPolygon into a list of coordinate lists."""
    if geometry.is_empty: return []
    geoms = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
    return [list(p.exterior.coords) for p in geoms if p.geom_type == 'Polygon' and not p.is_empty]


def _coords_to_shapely_polygons(coord_lists):
    """Converts lists of vertex coordinates into a list of Shapely Polygon objects."""
    polygons = []
    for sublist in coord_lists:
        # Handles potential extra list wrappers, e.g., in CATCH JSON format
        points = sublist[0] if len(sublist) == 1 and isinstance(sublist[0], list) else sublist
        if len(points) >= 3:
            try:
                poly = Polygon(points)
                if not poly.is_valid: poly = poly.buffer(0)
                if not poly.is_empty: polygons.append(poly)
            except Exception:
                continue
    return polygons


def _remove_ambiguous_regions(cancer_polygons, non_cancer_polygons):
    """Removes overlapping regions between cancer and non-cancer polygons."""
    cancer_area = unary_union(cancer_polygons) if cancer_polygons else Polygon()
    non_cancer_area = unary_union(non_cancer_polygons) if non_cancer_polygons else Polygon()
    ambiguous_area = cancer_area.intersection(non_cancer_area)
    
    clean_cancer_area = cancer_area.difference(ambiguous_area)
    clean_non_cancer_area = non_cancer_area.difference(ambiguous_area)
    
    return clean_cancer_area, clean_non_cancer_area


# --- REFACTORED BaseHandler with Template Method Pattern ---
class BaseHandler(ABC):
    """A template for all data handlers, enforcing an output contract."""

    @abstractmethod
    def _load_raw_annotations(self, slide, **kwargs):
        """
        Protected abstract method for subclasses to implement.
        
        This method must perform the file-specific parsing and return a dictionary.
        The dictionary should ideally contain 'cancer_polygons' and 'not_cancer_polygons'.
        """
        raise NotImplementedError

    def load_annotations(self, slide, **kwargs):
        """
        Public final method that defines the template for loading and validating annotations.
        This method should NOT be overridden by subclasses.
        """
        # Step 1: Call the subclass implementation
        raw_data = self._load_raw_annotations(slide, **kwargs)

        # Step 2: Validate the output contract
        if not isinstance(raw_data, dict):
            raise TypeError(f"{self.__class__.__name__} failed to return a dictionary. Got {type(raw_data)} instead.")
        
        required_keys = {"cancer_polygons", "not_cancer_polygons"}
        actual_keys = set(raw_data.keys())
        
        if not required_keys.issubset(actual_keys):
            missing_keys = required_keys - actual_keys
            raise ValueError(f"{self.__class__.__name__} failed to return required keys. Missing: {missing_keys}")

        # Step 3: Return the validated data
        return raw_data


class SVS_XML_Handler(BaseHandler):
    """Handles .svs slides with .xml annotations."""
    def _load_raw_annotations(self, slide, **kwargs):
        logging.info("Using SVS_XML_Handler with explicit overlap cleaning.")
        annotation_path = kwargs.get('annotation_path')
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        cancer_color = kwargs.get('cancer_color')
        not_cancer_color = kwargs.get('not_cancer_color')
        
        raw_cancer_coords = []
        raw_not_cancer_coords = []

        root = ET.parse(annotation_path).getroot()
        for ann in root.findall('.//Annotation'):
            is_cancer = str(ann.get('LineColor')) == str(cancer_color)
            is_non_cancer = str(ann.get('LineColor')) == str(not_cancer_color)
            if is_cancer or is_non_cancer:
                for reg in ann.findall('.//Region'):
                    verts = [(float(v.get("X")), float(v.get("Y"))) for v in reg.findall('.//Vertex')]
                    if len(verts) >= 3:
                        if is_cancer:
                            raw_cancer_coords.append(verts)
                        else:
                            raw_not_cancer_coords.append(verts)

        raw_cancer_polygons = _coords_to_shapely_polygons(raw_cancer_coords)
        raw_not_cancer_polygons = _coords_to_shapely_polygons(raw_not_cancer_coords)

        clean_cancer_area, clean_non_cancer_area = _remove_ambiguous_regions(raw_cancer_polygons, raw_not_cancer_polygons)
        
        return {
            "cancer_polygons": _to_coord_list(clean_cancer_area),
            "not_cancer_polygons": _to_coord_list(clean_non_cancer_area)
        }

class NDPI_NDPA_Handler(BaseHandler):
    """Handles .ndpi slides with .ndpa annotations."""
    def _load_raw_annotations(self, slide, **kwargs):
        logging.info("Using NDPI_NDPA_Handler to load annotations.")
        annotation_path = kwargs.get('annotation_path')
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        cancer_labels = {'R1', 'R2', 'R3', 'R4', 'R5'}
        non_cancer_labels = {'BG', 'T', 'N', 'A'}
        
        raw_cancer_polygons = []
        raw_not_cancer_polygons = []
        
        offset_x_nm = int(slide.properties.get('hamamatsu.XOffsetFromSlideCentre', 0))
        offset_y_nm = int(slide.properties.get('hamamatsu.YOffsetFromSlideCentre', 0))
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0.25))
        nm_per_pixel_x = mpp_x * 1000
        nm_per_pixel_y = mpp_y * 1000
        slide_width_level0, slide_height_level0 = slide.level_dimensions[0]
        
        root = ET.parse(annotation_path).getroot()
        for view in root.findall('ndpviewstate'):
            title_element = view.find('title')
            if title_element is None or title_element.text is None: continue
            label = title_element.text.strip()
            
            is_cancer = label in cancer_labels
            is_non_cancer = label in non_cancer_labels
            
            if not is_cancer and not is_non_cancer: continue
            
            pointlist = view.find('annotation/pointlist')
            if pointlist is None: continue

            temp_poly_coords = []
            for point in pointlist.findall('point'):
                try:
                    x_nm = float(point.find('x').text)
                    y_nm = float(point.find('y').text)
                    x_pixel = ((x_nm - offset_x_nm) / nm_per_pixel_x) + (slide_width_level0 / 2)
                    y_pixel = ((y_nm - offset_y_nm) / nm_per_pixel_y) + (slide_height_level0 / 2)
                    temp_poly_coords.append((x_pixel, y_pixel))
                except (ValueError, TypeError, AttributeError): continue
            
            if len(temp_poly_coords) >= 3:
                poly = Polygon(temp_poly_coords)
                if not poly.is_valid: poly = poly.buffer(0)
                if poly.is_valid and not poly.is_empty:
                    if is_cancer: raw_cancer_polygons.append(poly)
                    else: raw_not_cancer_polygons.append(poly)

        clean_cancer_area, clean_non_cancer_area = _remove_ambiguous_regions(raw_cancer_polygons, raw_not_cancer_polygons)
        
        return {
            "cancer_polygons": _to_coord_list(clean_cancer_area),
            "not_cancer_polygons": _to_coord_list(clean_non_cancer_area)
        }

class JSON_Handler(BaseHandler):
    """Handles annotations from JSON files and enforces overlap cleaning."""
    def _load_raw_annotations(self, slide, **kwargs):
        logging.info("Using JSON_Handler to load and clean annotations.")
        annotation_path = kwargs.get('annotation_path')
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        raw_cancer_coords = data.get("cancer_polygons", [])
        raw_not_cancer_coords = data.get("not_cancer_polygons", [])
            
        raw_cancer_polygons = _coords_to_shapely_polygons(raw_cancer_coords)
        raw_not_cancer_polygons = _coords_to_shapely_polygons(raw_not_cancer_coords)

        clean_cancer_area, clean_non_cancer_area = _remove_ambiguous_regions(raw_cancer_polygons, raw_not_cancer_polygons)
                
        return {
            "cancer_polygons": _to_coord_list(clean_cancer_area),
            "not_cancer_polygons": _to_coord_list(clean_non_cancer_area)
        }