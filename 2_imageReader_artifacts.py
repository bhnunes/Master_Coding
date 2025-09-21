import numpy as np
import cv2
import os
from PIL import Image
from shapely.geometry import Polygon, Point, MultiPolygon # Added Point, MultiPolygon
from shapely.prepared import prep # For optimization
import shapely.geos # For handling exceptions during polygon creation
from datetime import datetime
from multiprocessing import Pool
import random
import argparse
from dotenv import load_dotenv
import json
import warnings
import xml.etree.ElementTree as ET
import sys # Add this import at the top

load_dotenv(override=True)

# --- Configuration from .env ---
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', 224)) # Default 224
STRIDE = int(os.getenv('STRIDE', WINDOW_SIZE // 2)) # Default 50% overlap
MATCH_PERCENTAGE = float(os.getenv('MATCH_PERCENTAGE', 0.9)) # Default 90%
TISSUE_PERCENTAGE = float(os.getenv('TISSUE_PERCENTAGE', 0.9)) # Default 90%
OPENSLIDE_PATH = os.getenv('OPENSLIDE_PATH')
TARGET_LEVEL = int(os.getenv('TARGET_LEVEL', 0)) # Level to extract patches from (usually 0)
NUM_WORKERS = os.cpu_count() # Use all available CPU cores

# Define artifact classes to reject
REJECT_CLASSES = {"Fold", "Darkspot & Foreign Object", "PenMarking", "Edge & Air Bubble", "OOF"}

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
KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # For noise removal
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # For hole filling
PATCH_AREA = WINDOW_SIZE * WINDOW_SIZE
HALF_WINDOW = WINDOW_SIZE // 2

# --- Helper Functions ---

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

    # Apply Otsu's thresholding
    _, tissue_mask = cv2.threshold(saturation_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations for refinement
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)

    tissue_pixels = np.count_nonzero(tissue_mask)
    tissue_ratio = tissue_pixels / PATCH_AREA

    return tissue_ratio >= required_percentage

def polygons_to_mask(mask_shape, polygons_level0, scale_factor, patch_coords_target_level):
    """
    Creates a binary mask (0/1) for a patch based on annotation polygons intersecting it.

    Args:
        mask_shape (tuple): (height, width) of the mask (e.g., WINDOW_SIZE, WINDOW_SIZE).
        polygons_level0 (list): List of polygons, where each polygon is a list of (x, y) tuples at level 0.
        scale_factor (float): Downsample factor from level 0 to the target patch level.
        patch_coords_target_level (tuple): (patch_x, patch_y) top-left coordinates of the patch at the target level.

    Returns:
        np.ndarray: Binary mask (0 or 1).
    """
    mask = np.zeros(mask_shape, dtype=np.uint8)
    patch_x_level0 = patch_coords_target_level[0] * scale_factor
    patch_y_level0 = patch_coords_target_level[1] * scale_factor
    window_width_level0 = mask_shape[1] * scale_factor
    window_height_level0 = mask_shape[0] * scale_factor

    # Define the patch boundary as a Polygon at level 0
    window_poly_level0 = Polygon([
        (patch_x_level0, patch_y_level0),
        (patch_x_level0 + window_width_level0, patch_y_level0),
        (patch_x_level0 + window_width_level0, patch_y_level0 + window_height_level0),
        (patch_x_level0, patch_y_level0 + window_height_level0)
    ])
    # Prepare the window polygon for faster intersection checks
    prepared_window = prep(window_poly_level0)

    for poly_coords_level0 in polygons_level0:
        if len(poly_coords_level0) < 3: continue # Need at least 3 points for a valid polygon
        try:
            annotation_polygon_level0 = Polygon(poly_coords_level0)
        except ValueError:
            print(f"Warning: Skipping invalid polygon geometry: {poly_coords_level0}", file=sys.stderr) # Optional warning
            continue # Skip malformed polygons

        # Quick check if the polygon intersects the prepared window boundary
        if not prepared_window.intersects(annotation_polygon_level0):
            continue

        # Calculate the actual intersection
        intersection = window_poly_level0.intersection(annotation_polygon_level0)
        if intersection.is_empty: continue

        # Handle both single Polygon and MultiPolygon intersections
        geoms = intersection.geoms if isinstance(intersection, MultiPolygon) else [intersection]

        for geom in geoms:
            if geom.geom_type == 'Polygon' and not geom.is_empty:
                coords = np.array(geom.exterior.coords)
                # Convert level 0 intersection coords to local patch coords at target level
                coords[:, 0] = (coords[:, 0] - patch_x_level0) / scale_factor
                coords[:, 1] = (coords[:, 1] - patch_y_level0) / scale_factor
                # Clip and convert to integer coordinates for fillPoly
                coords = np.round(np.clip(coords, 0, mask_shape[1] - 1)).astype(np.int32) # width index
                coords[:, 1] = np.round(np.clip(coords[:, 1], 0, mask_shape[0] - 1)).astype(np.int32) # height index

                if coords.shape[0] >= 3: # Need at least 3 points for fillPoly
                    cv2.fillPoly(mask, [coords], 1) # Fill with 1

    return mask


def process_window(args):
    """
    Processes a single *pre-filtered* window (potential patch location).
    Reads the patch, CHECKS FOR ARTIFACTS, checks tissue %, checks annotation overlap %, saves patch and mask if criteria met.
    """
    # Unpack new 'prepared_artifacts_target_level' argument
    (path_Image, target_level, window_size, stride, tissue_percentage_req, match_percentage_req,
     path_cancer_folder, path_not_cancer_folder, path_cancer_mask_folder, path_not_cancer_mask_folder,
     patient, x, y, annotations_cancer_level0, annotations_not_cancer_level0, prepared_artifacts_target_level) = args

    slide = None
    try:
        # Define the patch boundary at the target level for artifact checking
        patch_polygon_target_level = Polygon([
            (x, y), (x + window_size, y),
            (x + window_size, y + window_size), (x, y + window_size)
        ])

        # 1. NEW ARTIFACT CHECK (Perform this first to fail fast)
        if prepared_artifacts_target_level and prepared_artifacts_target_level.intersects(patch_polygon_target_level):
            # If there is any intersection with the artifact geometry, reject the patch immediately.
            return True, None # Criteria not met, but process completed successfully

        # Open slide only if the patch is not in an artifact zone
        slide = OpenSlide(path_Image)
        scale_factor = slide.level_downsamples[target_level]
        x_int, y_int = int(x), int(y)

        # Read patch data at the target level
        patch_pil = slide.read_region((x_int, y_int), target_level, (window_size, window_size)).convert("RGB")
        patch_np = np.array(patch_pil)

        # 1. Check Tissue Percentage (Robust Method)
        if not check_tissue_percentage_robust(patch_np, tissue_percentage_req):
            slide.close()
            return True, None # Criteria not met, but process completed successfully

        # 2. Create Masks based on Annotations for THIS patch location
        patch_coords_target_level = (x_int, y_int)
        mask_shape = (window_size, window_size) # Assuming width=height

        cancer_mask_patch = polygons_to_mask(mask_shape, annotations_cancer_level0, scale_factor, patch_coords_target_level)
        non_cancer_mask_patch = polygons_to_mask(mask_shape, annotations_not_cancer_level0, scale_factor, patch_coords_target_level)

        # 3. Calculate Annotation Overlap Ratios
        cancer_pixels = np.count_nonzero(cancer_mask_patch)
        non_cancer_pixels = np.count_nonzero(non_cancer_mask_patch)

        cancer_overlap_ratio = cancer_pixels / PATCH_AREA
        non_cancer_overlap_ratio = non_cancer_pixels / PATCH_AREA

        # --- Decision Logic ---
        patch_coords_target_level = (x_int, y_int)
        mask_shape = (window_size, window_size) # Assuming width=height
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
            print(f"Info: Patch at ({x_int}, {y_int}) discarded due to overlapping Cancer ({cancer_overlap_ratio:.2f}) and Non-Cancer ({non_cancer_overlap_ratio:.2f}) annotations.", file=sys.stderr)
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
        return True, None # Process completed successfully
        
    except Exception as e:
        if slide:
            try: slide.close()
            except: pass
        error_message = f"Error processing window at ({x},{y}): {e.__class__.__name__}: {e}"
        print(error_message, file=sys.stderr)
        return False, error_message

# --- Main Processing Function ---

def extract_patches_for_slide(path_Image, target_level, window_size, stride,
                             tissue_percentage_req, match_percentage_req,
                             path_cancer_folder, path_not_cancer_folder,
                             path_cancer_mask_folder, path_not_cancer_mask_folder,
                             cancer_color, not_cancer_color, patient,
                             path_artifacts_geojson): # New argument
    """
    Loads annotations AND ARTIFACTS, generates candidate windows, filters them, and processes them in parallel.
    """
    slide = None
    annotations_cancer_level0 = []
    annotations_not_cancer_level0 = []
    prepared_artifacts_target_level = None # Initialize as None

    try:
        slide = OpenSlide(path_Image)
        scale_factor = slide.level_downsamples[target_level]
        target_width, target_height = slide.level_dimensions[target_level]

        # --- NEW: Load Artifact GeoJSON and Prepare Polygons ---
        artifact_polygons_level0 = []
        if path_artifacts_geojson and os.path.exists(path_artifacts_geojson):
            with open(path_artifacts_geojson, 'r') as f:
                artifact_data = json.load(f)

            for feature in artifact_data.get('features', []):
                if feature.get('properties', {}).get('classification') in REJECT_CLASSES:
                    coords = feature.get('geometry', {}).get('coordinates', [])
                    if coords:
                        # GeoJSON format is often [[(x, y), ...]], we take the first list.
                        artifact_polygons_level0.append(coords[0])

            if artifact_polygons_level0:
                print(f"Loaded {len(artifact_polygons_level0)} artifact polygons to avoid.", file=sys.stderr)
                # Scale polygons to target level and prepare them for fast intersection checks
                shapely_artifact_polygons_target_level = []
                for poly_level0 in artifact_polygons_level0:
                    try:
                        scaled_coords = [(x / scale_factor, y / scale_factor) for x, y in poly_level0]
                        if len(scaled_coords) >= 3:
                            shapely_artifact_polygons_target_level.append(Polygon(scaled_coords))
                    except (ValueError, shapely.geos.errors.TopologicalError) as e:
                        print(f"Warning: Skipping invalid artifact polygon after scaling: {e}", file=sys.stderr)
                        continue

                if shapely_artifact_polygons_target_level:
                    # Combine into a single geometry and prepare it
                    combined_artifacts = MultiPolygon(shapely_artifact_polygons_target_level)
                    prepared_artifacts_target_level = prep(combined_artifacts)
            else:
                print("No rejectable artifacts found in GeoJSON.", file=sys.stderr)
        else:
            print("Warning: Artifact GeoJSON not found or not provided. Proceeding without artifact check.", file=sys.stderr)

        # --- Load Slide and Annotations (Level 0 assumed) ---
        path_Annotation = path_Image.replace('.svs', '.xml', 1)
        if not os.path.exists(path_Annotation):
            raise FileNotFoundError(f"Annotation file not found: {path_Annotation}")

        slide = OpenSlide(path_Image)
        target_width, target_height = slide.level_dimensions[target_level]
        # scale_factor is needed later by workers, calculated inside process_window or polygons_to_mask
        # It's defined here just to explain the relationship between levels if needed, but not passed directly
        # scale_factor = slide.level_downsamples[target_level] # Factor from level 0 to target_level

        annotations_tree = ET.parse(path_Annotation)
        annotations_root = annotations_tree.getroot()

        all_polygons_level0 = [] # Store all polygons for bounds calculation
        for annotation in annotations_root.findall('.//Annotation'):
            line_color = annotation.get('LineColor')
            is_cancer = line_color == str(cancer_color)
            is_non_cancer = line_color == str(not_cancer_color)

            if is_cancer or is_non_cancer:
                for region in annotation.findall('.//Region'):
                    temp_poly_level0 = []
                    vertices = region.findall('.//Vertex')
                    for vertex in vertices:
                        try:
                            x_level0 = float(vertex.get("X"))
                            y_level0 = float(vertex.get("Y"))
                            temp_poly_level0.append((x_level0, y_level0))
                        except (ValueError, TypeError):
                            print(f"Warning: Skipping invalid vertex data in {path_Annotation}", file=sys.stderr)
                            continue # Skip vertex if coordinates are invalid

                    if len(temp_poly_level0) >= 3: # Need at least 3 points for a valid polygon
                        all_polygons_level0.append(temp_poly_level0)
                        if is_cancer:
                            annotations_cancer_level0.append(temp_poly_level0)
                        elif is_non_cancer:
                            annotations_not_cancer_level0.append(temp_poly_level0)

        if not all_polygons_level0:
            print(f"Warning: No valid annotations found for slide {path_Image} with specified colors.", file=sys.stderr)
            slide.close()
            return # Nothing to process

        # --- OPTIMIZATION: Pre-filter candidate windows ---
        print(f"Pre-filtering candidate windows for {path_Image}...", file=sys.stderr)

        # 1. Create scaled Shapely polygons at target_level for filtering check
        scale_factor = slide.level_downsamples[target_level]
        shapely_polygons_target_level = []
        for poly_level0 in all_polygons_level0:
            try:
                scaled_coords = [(x / scale_factor, y / scale_factor) for x, y in poly_level0]
                if len(scaled_coords) >= 3:
                    shapely_polygons_target_level.append(Polygon(scaled_coords))
            except Exception as e:
                print(f"Warning: Skipping polygon during scaling/creation: {e}", file=sys.stderr)
                continue # Skip if polygon becomes invalid after scaling

        if not shapely_polygons_target_level:
            print(f"Warning: No valid polygons after scaling for {path_Image}.", file=sys.stderr)
            slide.close()
            return

        # 2. Prepare the combined geometry for faster checking
        combined_annotations = MultiPolygon(shapely_polygons_target_level)
        prepared_annotations = prep(combined_annotations) # Prepare for faster 'contains' check

        # 3. Generate potential grid coordinates at target_level
        x_coords = np.arange(0, target_width - window_size + 1, stride)
        y_coords = np.arange(0, target_height - window_size + 1, stride)

        # 4. Filter coordinates: Keep if patch CENTER is within ANY annotation polygon
        filtered_windows_coords = []
        for x in x_coords:
            for y in y_coords:
                center_x = x + HALF_WINDOW
                center_y = y + HALF_WINDOW
                # Check if center point is contained within the prepared annotation shapes
                if prepared_annotations.contains(Point(center_x, center_y)):
                    filtered_windows_coords.append((int(x), int(y)))

        num_candidates = len(filtered_windows_coords)
        print(f"Found {num_candidates} candidate windows overlapping annotations.", file=sys.stderr)

        if num_candidates == 0:
            slide.close()
            return # Nothing to process further

        args_list = [(path_Image, target_level, window_size, stride,
                      tissue_percentage_req, match_percentage_req,
                      path_cancer_folder, path_not_cancer_folder,
                      path_cancer_mask_folder, path_not_cancer_mask_folder,
                      patient, x, y,
                      annotations_cancer_level0, annotations_not_cancer_level0,
                      prepared_artifacts_target_level) # Pass the prepared artifacts
                     for x, y in filtered_windows_coords]

        # --- Run Parallel Processing ---
        print(f"Processing {num_candidates} filtered windows for {path_Image} using {NUM_WORKERS} workers...", file=sys.stderr)
        with Pool(processes=NUM_WORKERS) as pool:
            # Use imap_unordered for potentially better memory usage with large iterables
            # results = list(pool.imap_unordered(process_window, args_list)) # Use list() to wait for all
            results = pool.map(process_window, args_list) # map is simpler if memory isn't an issue

        # --- Check Results for Errors ---
        errors = [msg for success, msg in results if not success]
        if errors:
            # Log or raise aggregated errors
            error_summary = f"Errors occurred during parallel processing for {path_Image}: " + "; ".join(set(errors)) # Show unique errors
            raise Exception(error_summary)

        print(f"Finished processing {path_Image}.", file=sys.stderr)

    except Exception as e:
         # Re-raise the exception to be caught by the main block
         raise Exception(f"Failed processing slide {path_Image}: {e.__class__.__name__}: {e}")
    finally:
        # Ensure the slide is closed
        if slide:
            try:
                slide.close()
            except Exception as close_exc:
                 print(f"Warning: Error closing slide {path_Image}: {close_exc}", file=sys.stderr)


# --- Main Execution Block ---
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning, module='PIL')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning) # Ignore potential numpy/shapely future warnings

    status = 'UNKNOWN'
    comments = ''
    parser = argparse.ArgumentParser(description='Extract patches and masks from WSI based on annotations.') # Define parser early
    # Define arguments (moved outside try block)
    parser.add_argument('--path_Image', type=str, required=True, help='Path to the WSI file (.svs)')
    parser.add_argument('--path_cancer_folder', type=str, required=True, help='Output folder for cancer patches')
    parser.add_argument('--path_not_cancer_folder', type=str, required=True, help='Output folder for non-cancer patches')
    parser.add_argument('--path_cancer_mask_folder', type=str, required=True, help='Output folder for cancer masks')
    parser.add_argument('--path_not_cancer_mask_folder', type=str, required=True, help='Output folder for non-cancer masks')
    parser.add_argument('--cancer_color', type=str, required=True, help='LineColor attribute value for cancer annotations in XML')
    parser.add_argument('--not_cancer_color', type=str, required=True, help='LineColor attribute value for non-cancer annotations in XML')
    parser.add_argument('--patient', type=str, required=True, help='Patient identifier')
    parser.add_argument('--path_artifacts_geojson', type=str, required=False, default=None, help='Path to the artifact GeoJSON file')

    try:
        args = parser.parse_args()

        # --- Ensure Output Directories Exist ---
        os.makedirs(args.path_cancer_folder, exist_ok=True)
        os.makedirs(args.path_not_cancer_folder, exist_ok=True)
        os.makedirs(args.path_cancer_mask_folder, exist_ok=True)
        os.makedirs(args.path_not_cancer_mask_folder, exist_ok=True)

        # --- Execute Patch Extraction ---
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
            patient=args.patient,
            path_artifacts_geojson=args.path_artifacts_geojson
        )

        status = 'COMPLETED'
        comments = f'Successfully processed {os.path.basename(args.path_Image)}.' # Use basename for brevity

    except FileNotFoundError as e:
        status = 'FAILED'
        comments = f"Error: Input file or annotation not found. {e}"
    except openslide.OpenSlideError as e:
        # Attempt to get path_Image from args if available, otherwise use generic message
        img_path = args.path_Image if 'args' in locals() and args.path_Image else "input WSI"
        status = 'FAILED'
        comments = f"Error: OpenSlide could not process {os.path.basename(img_path)}. Invalid/Corrupt WSI? Error: {e}"
    except ET.ParseError as e:
        img_path = args.path_Image if 'args' in locals() and args.path_Image else "input WSI"
        status = 'FAILED'
        comments = f"Error: Could not parse XML annotation for {os.path.basename(img_path)}. Error: {e}"
    except Exception as e:
        status = 'FAILED'
        # Provide more specific error info if possible
        comments = f"An unexpected error occurred: {e.__class__.__name__}: {e}"
        # Optional: Add traceback logging here for debugging
        # import traceback
        # comments += f"\nTraceback: {traceback.format_exc()}"
    finally:
        # --- Output JSON for the calling script ---
        output_data = {
            "status": status,
            "comments": comments
        }
        # Ensure output is always printed, even if args parsing failed early
        print(json.dumps(output_data))