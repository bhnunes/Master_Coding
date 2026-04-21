# mypy: ignore-errors

# MAIN LOOP TO PROCESS WSI
import json
from dataclasses import dataclass

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from tqdm import tqdm

from helpers.cv2_compat import ensure_cv2_compat

cv2 = ensure_cv2_compat(cv2)

TISSUE_TILE_MIN_ZERO_PIXELS = 50
MIN_GEOJSON_CONTOUR_POINTS = 4


@dataclass(frozen=True)
class SlideProcessConfig:
    patch_count_w: int
    patch_count_h: int
    patch_size: int
    model_patch_size: int
    colors: list[list[int]]
    encoder_model: str
    encoder_weights: str
    device: str
    back_class: int
    mpp_model: float
    slide_mpp: float
    width_l0: int
    height_l0: int


# Helper functions
def to_tensor_x(x, **_kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(image, preprocessing_fn, model_size):
    if image.size != model_size:
        image = image.resize(model_size)
        print("resized")
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x


def make_1class_map_thr(mask, class_colors):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for class_index in range(1, len(class_colors) + 1):
        idx = mask == class_index
        r[idx] = class_colors[class_index - 1][0]
        g[idx] = class_colors[class_index - 1][1]
        b[idx] = class_colors[class_index - 1][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def _combine_mask_tiles(mask_tiles, patch_size, overhang_x):
    tiles = list(mask_tiles)
    if not tiles:
        raise RuntimeError("Mask tile rows cannot be empty.")
    if len(tiles) > 1 and overhang_x > 0:
        tiles[-1] = tiles[-1][:, patch_size - overhang_x : patch_size]
    return np.concatenate(tiles, axis=1)


def _combine_mask_rows(mask_rows, patch_size, overhang_y):
    rows = list(mask_rows)
    if not rows:
        raise RuntimeError("Mask row collection cannot be empty.")
    if len(rows) > 1 and overhang_y > 0:
        rows[-1] = rows[-1][patch_size - overhang_y : patch_size, :]
    return np.concatenate(rows, axis=0)


def slide_process_single(
    model,
    tis_det_map_mpp,
    slide,
    config: SlideProcessConfig,
):
    """
    Tissue detection map is generated under MPP = 4, therefore model patch size of
    (512, 512) corresponds to a tis_det_map patch size of (128, 128).
    """

    model_size = (config.model_patch_size, config.model_patch_size)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        config.encoder_model,
        config.encoder_weights,
    )

    mask_rows = []
    for he in tqdm(range(config.patch_count_h), total=config.patch_count_h):
        h = he * config.patch_size + 1
        if he == 0:
            h = 0
        row_tiles = []
        for wi in range(config.patch_count_w):
            w = wi * config.patch_size + 1
            if wi == 0:
                w = 0
            td_patch = tis_det_map_mpp[
                he * config.model_patch_size : (he + 1) * config.model_patch_size,
                wi * config.model_patch_size : (wi + 1) * config.model_patch_size,
            ]
            if td_patch.shape != (config.model_patch_size, config.model_patch_size):
                original_shape = td_patch.shape
                desired_shape = (config.model_patch_size, config.model_patch_size)
                padding = [(0, desired_shape[i] - original_shape[i]) for i in range(2)]
                td_patch_ = np.pad(td_patch, padding, mode="constant")
            else:
                td_patch_ = td_patch

            if np.count_nonzero(td_patch == 0) > TISSUE_TILE_MIN_ZERO_PIXELS:
                # Generate patch
                work_patch = slide.read_region((w, h), 0, (config.patch_size, config.patch_size))
                work_patch = work_patch.convert("RGB")

                # Resize to model patch size
                work_patch = work_patch.resize(model_size, Image.Resampling.LANCZOS)

                image_pre = get_preprocessing(work_patch, preprocessing_fn, model_size)
                x_tensor = torch.from_numpy(image_pre).to(config.device).unsqueeze(0)
                predictions = model.predict(x_tensor)
                predictions = predictions.squeeze().cpu().numpy()

                mask_raw = np.argmax(predictions, axis=0).astype("int8")
                mask = np.where(td_patch_ == 1, config.back_class, mask_raw)

            else:
                mask = np.full(model_size, config.back_class)

            row_tiles.append(mask)

        mask_rows.append(_combine_mask_tiles(row_tiles, config.model_patch_size, 0))

    end_image = _combine_mask_rows(mask_rows, config.model_patch_size, 0)

    # now get size of padded region (buffer) at Model MPP
    buffer_right_l = int(
        (config.width_l0 - (config.patch_count_w * config.patch_size))
        * config.slide_mpp
        / config.mpp_model
    )
    buffer_bottom_l = int(
        (config.height_l0 - (config.patch_count_h * config.patch_size))
        * config.slide_mpp
        / config.mpp_model
    )
    # firstly bottom
    buffer_bottom = np.full((buffer_bottom_l, end_image.shape[1]), 0)
    temp_image = np.concatenate((end_image, buffer_bottom), axis=0)
    # now right side
    temp_image_he, temp_image_wi = temp_image.shape  # width and height
    buffer_right = np.full((temp_image_he, buffer_right_l), 0)
    end_image = np.concatenate((temp_image, buffer_right), axis=1).astype(np.uint8)

    end_image_1class = make_1class_map_thr(end_image, config.colors)
    end_image_1class = Image.fromarray(end_image_1class)
    end_image_1class = end_image_1class.resize(
        (config.patch_count_w * 50, config.patch_count_h * 50), Image.Resampling.LANCZOS
    )

    return end_image_1class, end_image


def mask_to_geojson(mask_path, output_path, scale_factor=1.0):
    """
    Convert a semantic segmentation mask to GeoJSON with coordinate scaling

    Parameters:
    -----------
    mask_path : str
        Path to the input PNG mask file
    output_path : str
        Path to save the output GeoJSON file
    scale_factor : float, optional, should be: model_mpp / slide_mpp
        Factor to scale coordinates by (default: 1.0)

    Returns:
    --------
    None
    """
    # Define class mapping
    CLASS_MAPPING = {
        1: "Normal Tissue",
        2: "Fold",
        3: "Darkspot & Foreign Object",
        4: "PenMarking",
        5: "Edge & Air Bubble",
        6: "OOF",  # Out of Focus
        7: "Background",
    }

    # Read the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Dictionary to store features for each class
    features = []

    # Iterate through unique class values (1 to 7)
    for class_value in range(2, 7):
        # Create a binary mask for the current class
        class_mask = (mask == class_value).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to GeoJSON features
        for contour in contours:
            # Flatten contour and reshape
            contour_points = contour.reshape(-1, 2)

            # Scale coordinates
            scaled_points = contour_points * scale_factor

            # Skip contours with less than 4 points
            if len(scaled_points) < MIN_GEOJSON_CONTOUR_POINTS:
                continue

            # Ensure polygon is closed by adding first point at the end if needed
            polygon_points = scaled_points.tolist()
            if not np.array_equal(polygon_points[0], polygon_points[-1]):
                polygon_points.append(polygon_points[0])

            # Create feature for this polygon
            feature = {
                "type": "Feature",
                "properties": {
                    "class_id": int(class_value),
                    "classification": CLASS_MAPPING.get(class_value, "Unknown"),
                    "area": cv2.contourArea(contour) * (scale_factor**2),
                },
                "geometry": {"type": "Polygon", "coordinates": [polygon_points]},
            }

            features.append(feature)

    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {"class_mapping": CLASS_MAPPING, "scale_factor": scale_factor},
    }

    # Write to file
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)
