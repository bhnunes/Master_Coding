import openslide
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from shapely.ops import unary_union
import logging
import os


class BaseHandler:
    """A template for all data handlers."""
    def load_annotations(self, slide, **kwargs):
        """
        Loads annotations and returns them in a standardized dictionary format.
        MUST return: {'cancer': [poly1, poly2], 'not_cancer': [poly3, ...]}
        """
        raise NotImplementedError

class SVS_XML_Handler(BaseHandler):
    """Handles .svs slides with .xml annotations."""
    def load_annotations(self, slide, **kwargs):
        logging.info("Using SVS_XML_Handler to load annotations.")
        annotation_path = kwargs.get('annotation_path')
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        cancer_color = kwargs.get('cancer_color')
        not_cancer_color = kwargs.get('not_cancer_color')
        
        root = ET.parse(annotation_path).getroot()
        annotations_cancer_level0 = []
        annotations_not_cancer_level0 = []

        for ann in root.findall('.//Annotation'):
            is_cancer = str(ann.get('LineColor')) == str(cancer_color)
            is_non_cancer = str(ann.get('LineColor')) == str(not_cancer_color)
            if is_cancer or is_non_cancer:
                for reg in ann.findall('.//Region'):
                    verts = [(float(v.get("X")), float(v.get("Y"))) for v in reg.findall('.//Vertex')]
                    if len(verts) >= 3:
                        if is_cancer:
                            annotations_cancer_level0.append(verts)
                        else:
                            annotations_not_cancer_level0.append(verts)
        
        return {
            "cancer_polygons": annotations_cancer_level0,
            "not_cancer_polygons": annotations_not_cancer_level0
        }

class NDPI_NDPA_Handler(BaseHandler):
    """Handles .ndpi slides with .ndpa annotations."""
    def load_annotations(self, slide, **kwargs):
        logging.info("Using NDPI_NDPA_Handler to load annotations.")
        annotation_path = kwargs.get('annotation_path')
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        cancer_labels = {'R1', 'R2', 'R3', 'R4', 'R5'}
        non_cancer_labels = {'BG', 'T', 'N', 'A'}
        
        raw_cancer_polygons = []
        raw_non_cancer_polygons = []
        
        offset_x_nm = int(slide.properties.get('hamamatsu.XOffsetFromSlideCentre'))
        offset_y_nm = int(slide.properties.get('hamamatsu.YOffsetFromSlideCentre'))
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
        mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y))
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

            temp_poly = []
            for point in pointlist.findall('point'):
                try:
                    x_nm = float(point.find('x').text)
                    y_nm = float(point.find('y').text)
                    x_pixel = ((x_nm - offset_x_nm) / nm_per_pixel_x) + (slide_width_level0 / 2)
                    y_pixel = ((y_nm - offset_y_nm) / nm_per_pixel_y) + (slide_height_level0 / 2)
                    temp_poly.append((x_pixel, y_pixel))
                except Exception: continue
            
            if len(temp_poly) >= 3:
                poly = Polygon(temp_poly)
                if not poly.is_valid: poly = poly.buffer(0)
                if poly.is_valid and not poly.is_empty:
                    if is_cancer: raw_cancer_polygons.append(poly)
                    else: raw_non_cancer_polygons.append(poly)

        cancer_area = unary_union(raw_cancer_polygons) if raw_cancer_polygons else Polygon()
        non_cancer_area = unary_union(raw_non_cancer_polygons) if raw_non_cancer_polygons else Polygon()
        ambiguous_area = cancer_area.intersection(non_cancer_area)
        
        clean_cancer_area = cancer_area.difference(ambiguous_area)
        clean_non_cancer_area = non_cancer_area.difference(ambiguous_area)
        
        final_cancer_polys = [list(p.exterior.coords) for p in (clean_cancer_area.geoms if clean_cancer_area.geom_type == 'MultiPolygon' else [clean_cancer_area]) if p.geom_type == 'Polygon']
        final_non_cancer_polys = [list(p.exterior.coords) for p in (clean_non_cancer_area.geoms if clean_non_cancer_area.geom_type == 'MultiPolygon' else [clean_non_cancer_area]) if p.geom_type == 'Polygon']

        return {
            "cancer_polygons": final_cancer_polys,
            "not_cancer_polygons": final_non_cancer_polys
        }