import numpy as np
import cv2
import os
from PIL import Image
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.prepared import prep
from datetime import datetime
from multiprocessing import Pool
import random
import argparse
from dotenv import load_dotenv
import json
import warnings
import xml.etree.ElementTree as ET
import sys

load_dotenv(override=True)

# --- Configuration from .env ---
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', 224))
STRIDE = int(os.getenv('STRIDE', WINDOW_SIZE // 2))
MATCH_PERCENTAGE = float(os.getenv('MATCH_PERCENTAGE', 0.9))
TISSUE_PERCENTAGE = float(os.getenv('TISSUE_PERCENTAGE', 0.9))
OPENSLIDE_PATH = os.getenv('OPENSLIDE_PATH')
TARGET_LEVEL = int(os.getenv('TARGET_LEVEL', 0))
NUM_WORKERS = os.cpu_count()

# --- OpenSlide Initialization ---
try:
    if hasattr(os, 'add_dll_directory') and OPENSLIDE_PATH and os.path.isdir(OPENSLIDE_PATH):
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide
except ImportError as e:
    print(f"Error importing OpenSlide: {e}", file=sys.stderr)
    print("Ensure OpenSlide C library is installed and OPENSLIDE_PATH is correctly set in .env if on Windows.", file=sys.stderr)
    exit(1)
from openslide import OpenSlide

# --- Constants ---
KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
PATCH_AREA = WINDOW_SIZE * WINDOW_SIZE
HALF_WINDOW = WINDOW_SIZE // 2

# ==========================================================================================
# AS FUNÇÕES ABAIXO (updateMax, check_tissue_percentage_robust, polygons_to_mask, process_window)
# NÃO PRECISAM DE ALTERAÇÃO, pois são genéricas e operam sobre listas de polígonos
# e dados de imagem, que serão corretamente fornecidos pela função principal modificada.
# ISSO DEMONSTRA A ROBUSTEZ DO FRAMEWORK.
# ==========================================================================================
def updateMax(Xmax, Xmin, Ymax, Ymin, x, y):
    """Updates the bounding box coordinates."""
    Xmax = max(Xmax, x)
    Xmin = min(Xmin, x)
    Ymax = max(Ymax, y)
    Ymin = min(Ymin, y)
    return Xmax, Xmin, Ymax, Ymin

def check_tissue_percentage_robust(patch_np, required_percentage):
    """
    Checks if the tissue content in a patch meets the required percentage
    using a robust HSV + Otsu thresholding method.
    """
    if patch_np is None or patch_np.size == 0:
        return False
    patch_hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    saturation_channel = patch_hsv[:, :, 1]
    _, tissue_mask = cv2.threshold(saturation_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    tissue_pixels = np.count_nonzero(tissue_mask)
    tissue_ratio = tissue_pixels / PATCH_AREA
    return tissue_ratio >= required_percentage

def polygons_to_mask(mask_shape, polygons_level0, scale_factor, patch_coords_target_level):
    """Creates a binary mask (0/1) for a patch based on annotation polygons intersecting it."""
    mask = np.zeros(mask_shape, dtype=np.uint8)
    patch_x_level0 = patch_coords_target_level[0] * scale_factor
    patch_y_level0 = patch_coords_target_level[1] * scale_factor
    window_width_level0 = mask_shape[1] * scale_factor
    window_height_level0 = mask_shape[0] * scale_factor
    window_poly_level0 = Polygon([
        (patch_x_level0, patch_y_level0),
        (patch_x_level0 + window_width_level0, patch_y_level0),
        (patch_x_level0 + window_width_level0, patch_y_level0 + window_height_level0),
        (patch_x_level0, patch_y_level0 + window_height_level0)
    ])
    prepared_window = prep(window_poly_level0)
    for poly_coords_level0 in polygons_level0:
        if len(poly_coords_level0) < 3: continue
        try:
            annotation_polygon_level0 = Polygon(poly_coords_level0)
        except ValueError:
            continue
        if not prepared_window.intersects(annotation_polygon_level0):
            continue
        intersection = window_poly_level0.intersection(annotation_polygon_level0)
        if intersection.is_empty: continue
        geoms = intersection.geoms if isinstance(intersection, MultiPolygon) else [intersection]
        for geom in geoms:
            if geom.geom_type == 'Polygon' and not geom.is_empty:
                coords = np.array(geom.exterior.coords)
                coords[:, 0] = (coords[:, 0] - patch_x_level0) / scale_factor
                coords[:, 1] = (coords[:, 1] - patch_y_level0) / scale_factor
                coords = np.round(np.clip(coords, 0, mask_shape[1] - 1)).astype(np.int32)
                coords[:, 1] = np.round(np.clip(coords[:, 1], 0, mask_shape[0] - 1)).astype(np.int32)
                if coords.shape[0] >= 3:
                    cv2.fillPoly(mask, [coords], 1)
    return mask

def process_window(args):
    """
    Processes a single *pre-filtered* window (potential patch location).
    This function remains unchanged as its logic is generic.
    """
    (path_Image, target_level, window_size, stride, tissue_percentage_req, match_percentage_req,
     path_cancer_folder, path_not_cancer_folder, path_cancer_mask_folder, path_not_cancer_mask_folder,
     patient, x, y, annotations_cancer_level0, annotations_not_cancer_level0) = args
    slide = None
    try:
        slide = OpenSlide(path_Image)
        scale_factor = slide.level_downsamples[target_level]
        x_int, y_int = int(x), int(y)
        patch_pil = slide.read_region((x_int, y_int), target_level, (window_size, window_size)).convert("RGB")
        patch_np = np.array(patch_pil)

        if not check_tissue_percentage_robust(patch_np, tissue_percentage_req):
            slide.close()
            return True, None

        patch_coords_target_level = (x_int, y_int)
        mask_shape = (window_size, window_size)
        cancer_mask_patch = polygons_to_mask(mask_shape, annotations_cancer_level0, scale_factor, patch_coords_target_level)
        non_cancer_mask_patch = polygons_to_mask(mask_shape, annotations_not_cancer_level0, scale_factor, patch_coords_target_level)

        cancer_pixels = np.count_nonzero(cancer_mask_patch)
        non_cancer_pixels = np.count_nonzero(non_cancer_mask_patch)
        cancer_overlap_ratio = cancer_pixels / PATCH_AREA
        non_cancer_overlap_ratio = non_cancer_pixels / PATCH_AREA

        patch_saved = False
        save_path_img = None
        save_path_mask = None
        final_mask_to_save = None

        meets_cancer_threshold = cancer_overlap_ratio >= match_percentage_req
        meets_non_cancer_threshold = non_cancer_overlap_ratio >= match_percentage_req

        if meets_cancer_threshold and meets_non_cancer_threshold:
            patch_saved = False
        elif meets_cancer_threshold:
            save_folder_img = path_cancer_folder
            save_folder_mask = path_cancer_mask_folder
            label = "CANCER"
            final_mask_to_save = cancer_mask_patch
            patch_saved = True
        elif meets_non_cancer_threshold:
            save_folder_img = path_not_cancer_folder
            save_folder_mask = path_not_cancer_mask_folder
            label = "NOT_CANCER"
            final_mask_to_save = np.zeros(mask_shape, dtype=np.uint8)
            patch_saved = True

        if patch_saved:
            now = datetime.now()
            timestamp = now.strftime('%Y%m%d_%H%M%S_%f')
            random_number = random.randint(1000, 9999)
            file_basename = f"{label}_PATIENT_{patient}_{x_int}_{y_int}_{random_number}_{timestamp}.png"
            save_path_img = os.path.join(save_folder_img, file_basename)
            patch_pil.save(save_path_img)
            save_path_mask = os.path.join(save_folder_mask, file_basename)
            mask_image = Image.fromarray(final_mask_to_save * 255)
            mask_image.save(save_path_mask)

        slide.close()
        return True, None
    except Exception as e:
        if slide:
            try: slide.close()
            except: pass
        error_message = f"Error processing window at ({x},{y}): {e.__class__.__name__}: {e}"
        return False, error_message

# --- Main Processing Function ---
def extract_patches_for_slide(path_Image, target_level, window_size, stride,
                             tissue_percentage_req, match_percentage_req,
                             path_cancer_folder, path_not_cancer_folder,
                             path_cancer_mask_folder, path_not_cancer_mask_folder,
                             cancer_color, not_cancer_color, patient):
    """
    MODIFICADO: Carrega anotações .ndpa, converte coordenadas de nm para pixels,
    classifica polígonos com base no <title> e então executa o pipeline de patching.
    """
    slide = None
    annotations_cancer_level0 = []
    annotations_not_cancer_level0 = []
    
    # MODIFICAÇÃO: Define as labels de câncer para o DiagSet-A
    cancer_labels = {'R1', 'R2', 'R3', 'R4', 'R5'}

    try:
        # MODIFICAÇÃO: O caminho da anotação agora é .ndpi.ndpa
        path_Annotation = path_Image + '.ndpa'
        path_Annotation = path_Annotation.replace('/IMAGES/', '/ANNOTATIONS/')
        if not os.path.exists(path_Annotation):
             raise FileNotFoundError(f"Annotation file not found: {path_Annotation}")

        slide = OpenSlide(path_Image)
        target_width, target_height = slide.level_dimensions[target_level]

        # MODIFICAÇÃO CRÍTICA: Obter metadados para conversão de coordenadas
        # OpenSlide armazena o offset em nanômetros.
        try:
            offset_x_nm = int(slide.properties.get('hamamatsu.XOffsetFromSlideCentre'))
            offset_y_nm = int(slide.properties.get('hamamatsu.YOffsetFromSlideCentre'))
        except (TypeError, ValueError):
            raise ValueError("Could not read Hamamatsu offset properties from .ndpi file.")

        # OpenSlide armazena Microns Per Pixel (MPP). Precisamos de nanômetros por pixel.
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
        mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y))
        nm_per_pixel_x = mpp_x * 1000
        nm_per_pixel_y = mpp_y * 1000

        # As coordenadas no .ndpa são relativas ao centro, enquanto em OpenSlide são relativas ao canto superior esquerdo.
        # A conversão é: pixel = (coord_nm - offset_centro_nm) / nm_por_pixel + (dimensão_total_pixels / 2)
        slide_width_level0, slide_height_level0 = slide.level_dimensions[0]
        
        # --- MODIFICAÇÃO: Parsing da estrutura XML do .ndpa ---
        annotations_tree = ET.parse(path_Annotation)
        root = annotations_tree.getroot()
        all_polygons_level0 = []

        for view in root.findall('ndpviewstate'):
            title_element = view.find('title')
            if title_element is None:
                continue
            
            # MODIFICAÇÃO: Lógica de Classificação para DiagSet-A baseada no <title>
            label = title_element.text
            is_cancer = label in cancer_labels
            
            # Encontra a anotação dentro do ndpviewstate
            annotation = view.find('annotation')
            if annotation is None:
                continue

            pointlist = annotation.find('pointlist')
            if pointlist is None:
                continue

            temp_poly_level0 = []
            for point in pointlist.findall('point'):
                try:
                    x_nm = float(point.find('x').text)
                    y_nm = float(point.find('y').text)

                    # MODIFICAÇÃO CRÍTICA: Conversão de Coordenadas (nm para pixels no nível 0)
                    x_pixel = ((x_nm - offset_x_nm) / nm_per_pixel_x) + (slide_width_level0 / 2)
                    y_pixel = ((y_nm - offset_y_nm) / nm_per_pixel_y) + (slide_height_level0 / 2)
                    
                    temp_poly_level0.append((x_pixel, y_pixel))
                except (ValueError, TypeError, AttributeError):
                    continue
            
            if len(temp_poly_level0) >= 3:
                # =================== NOVA LÓGICA DE CORREÇÃO AQUI ===================
                try:
                    polygon = Polygon(temp_poly_level0)
                    # Se o polígono não for válido, tente corrigi-lo
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    
                    # Se após a correção ele se tornou inválido, vazio ou não é mais um polígono, descarte-o
                    if not polygon.is_valid or polygon.is_empty or polygon.geom_type != 'Polygon':
                        print(f"Warning: Skipping invalid or empty geometry in {os.path.basename(path_Annotation)} after buffer(0) fix.", file=sys.stderr)
                        continue # Pula para a próxima anotação

                    # Extrai as coordenadas do polígono corrigido
                    corrected_coords = list(polygon.exterior.coords)

                    all_polygons_level0.append(corrected_coords)
                    if is_cancer:
                        annotations_cancer_level0.append(corrected_coords)
                    else:
                        annotations_not_cancer_level0.append(corrected_coords)

                except Exception as e:
                    # Se a criação do polígono falhar mesmo antes da verificação, ignore-o
                    print(f"Warning: Could not create polygon from annotation in {os.path.basename(path_Annotation)}. Skipping. Error: {e}", file=sys.stderr)
                    continue
                # =================== FIM DA NOVA LÓGICA DE CORREÇÃO ===================
        
        # --- A LÓGICA A SEGUIR PERMANECE A MESMA DO FRAMEWORK ORIGINAL ---
        # Ela é robusta o suficiente para funcionar com as listas de polígonos que acabamos de gerar.

        if not all_polygons_level0:
             print(f"Warning: No valid annotations found for slide {path_Image}.", file=sys.stderr)
             slide.close()
             return

        print(f"Pre-filtering candidate windows for {path_Image}...", file=sys.stderr)
        scale_factor = slide.level_downsamples[target_level]
        shapely_polygons_target_level = []
        for poly_level0 in all_polygons_level0:
            try:
                scaled_coords = [(x / scale_factor, y / scale_factor) for x, y in poly_level0]
                if len(scaled_coords) >= 3:
                     shapely_polygons_target_level.append(Polygon(scaled_coords))
            except Exception as e:
                continue
        if not shapely_polygons_target_level:
            slide.close()
            return

        combined_annotations = MultiPolygon(shapely_polygons_target_level)
        prepared_annotations = prep(combined_annotations)
        x_coords = np.arange(0, target_width - window_size + 1, stride)
        y_coords = np.arange(0, target_height - window_size + 1, stride)
        filtered_windows_coords = []
        for x in x_coords:
            for y in y_coords:
                if prepared_annotations.contains(Point(x + HALF_WINDOW, y + HALF_WINDOW)):
                    filtered_windows_coords.append((int(x), int(y)))
        
        num_candidates = len(filtered_windows_coords)
        if num_candidates == 0:
            slide.close()
            return
        
        args_list = [(path_Image, target_level, window_size, stride,
                      tissue_percentage_req, match_percentage_req,
                      path_cancer_folder, path_not_cancer_folder,
                      path_cancer_mask_folder, path_not_cancer_mask_folder,
                      patient, x, y,
                      annotations_cancer_level0, annotations_not_cancer_level0)
                     for x, y in filtered_windows_coords]

        print(f"Processing {num_candidates} filtered windows for {path_Image} using {NUM_WORKERS} workers...", file=sys.stderr)
        with Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(process_window, args_list)

        errors = [msg for success, msg in results if not success]
        if errors:
            error_summary = f"Errors occurred during parallel processing for {path_Image}: " + "; ".join(set(errors))
            raise Exception(error_summary)

        print(f"Finished processing {path_Image}.", file=sys.stderr)
    except Exception as e:
         raise Exception(f"Failed processing slide {path_Image}: {e.__class__.__name__}: {e}")
    finally:
        if slide:
            try: slide.close()
            except Exception as close_exc: pass

# --- Main Execution Block (sem alterações) ---
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    status = 'UNKNOWN'
    comments = ''
    parser = argparse.ArgumentParser(description='Extract patches and masks from WSI based on annotations.')
    parser.add_argument('--path_Image', type=str, required=True, help='Path to the WSI file (.ndpi)')
    parser.add_argument('--path_cancer_folder', type=str, required=True, help='Output folder for cancer patches')
    parser.add_argument('--path_not_cancer_folder', type=str, required=True, help='Output folder for non-cancer patches')
    parser.add_argument('--path_cancer_mask_folder', type=str, required=True, help='Output folder for cancer masks')
    parser.add_argument('--path_not_cancer_mask_folder', type=str, required=True, help='Output folder for non-cancer masks')
    parser.add_argument('--cancer_color', type=str, required=True, help='Placeholder, not used for DiagSet')
    parser.add_argument('--not_cancer_color', type=str, required=True, help='Placeholder, not used for DiagSet')
    parser.add_argument('--patient', type=str, required=True, help='Patient identifier')

    try:
        args = parser.parse_args()
        os.makedirs(args.path_cancer_folder, exist_ok=True)
        os.makedirs(args.path_not_cancer_folder, exist_ok=True)
        os.makedirs(args.path_cancer_mask_folder, exist_ok=True)
        os.makedirs(args.path_not_cancer_mask_folder, exist_ok=True)
        extract_patches_for_slide(
            path_Image=args.path_Image,
            target_level=TARGET_LEVEL,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            tissue_percentage_req=TISSUE_PERCENTAGE,
            match_percentage_req=MATCH_PERCENTAGE,
            path_cancer_folder=args.path_cancer_folder,
            path_not_cancer_folder=args.path_not_cancer_folder,
            path_cancer_mask_folder=args.path_cancer_mask_folder,
            path_not_cancer_mask_folder=args.path_not_cancer_mask_folder,
            cancer_color=args.cancer_color,
            not_cancer_color=args.not_cancer_color,
            patient=args.patient
        )
        status = 'COMPLETED'
        comments = f'Successfully processed {os.path.basename(args.path_Image)}.'
    except Exception as e:
        status = 'FAILED'
        comments = f"An unexpected error occurred: {e.__class__.__name__}: {e}"
    finally:
        output_data = { "status": status, "comments": comments }
        print(json.dumps(output_data))