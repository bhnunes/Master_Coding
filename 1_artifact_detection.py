from openslide import OpenSlide, open_slide
import cv2
import numpy as np
import torch
import argparse
from PIL import Image
import segmentation_models_pytorch as smp
from helpers.wsi_tis_detect_helper_fx import get_preprocessing, make_class_map
from pathlib import Path
from helpers.wsi_colors import colors_QC7 as colors
from helpers.wsi_slide_info import slide_info
from helpers.wsi_process import slide_process_single, mask_to_geojson
from helpers.wsi_maps import make_overlay
from tqdm.auto import tqdm
import os, timeit
import shutil
import sys
import shutil
import zipfile
from pathlib import Path

Image.MAX_IMAGE_PIXELS = 1000000000

#INPUT PATHS
SLIDE_DIR = r"/content/drive/MyDrive/CATCH/IMAGES"
OUTPUT_DIR = r"/content/output_folder"
GEOJSON_DRIVER_FOLDER = r"/content/drive/MyDrive/IA_MEDICA/GEOJSON_CATCH"
DST_FOLDER = r"/content/slide_folder"

# DEVICE
DEVICE = 'cuda'

# MODEL TISSUE DETECTION:
MODEL_TD_DIR = './models/td/'
MODEL_TD_NAME = 'Tissue_Detection_MPP10.pth'
MPP_MODEL_TD = 10
M_P_S_MODEL_TD = 512
ENCODER_MODEL_TD = 'timm-efficientnet-b0'
ENCODER_MODEL_TD_WEIGHTS = 'imagenet'

# OVERLAY PARAMETERS (TRANSPARENCY)
OVER_IMAGE = 0.7    # % original image
OVER_MASK = 0.3     # % segmentation mask

MPP_MODEL = 1.5
start = 0
end = -1
create_geojson = "Y"
OVERLAY_FACTOR = 10

IMAGES_ZIP = '/content/drive/MyDrive/IA_MEDICA/Imagens_anotadas.zip'

# MODEL(S)
MODEL_QC_DIR = './models/qc/'
if MPP_MODEL == 1.5:
    MODEL_QC_NAME = 'GrandQC_MPP15.pth'
elif MPP_MODEL == 1.0:
    MODEL_QC_NAME = 'GrandQC_MPP1.pth'
elif MPP_MODEL == 2.0:
    MODEL_QC_NAME = 'GrandQC_MPP2.pth'
else:
    raise Exception("mpp of the model can only be 1.0, 1.5, 2.0")
ENCODER_MODEL = 'timm-efficientnet-b0'
ENCODER_MODEL_WEIGHTS = 'imagenet'

M_P_S_MODEL = 512

# CLASSES
BACK_CLASS = 7

# COLORS for MASK
colors = [[50, 50, 250],    # BLUE: TISSUE
          [128, 128, 128]]  # GRAY: BACKGROUND


def get_wsi_files(root_folder, geojson_folder):
    """
    Scans recursively for WSIs with extensions: .svs (Leica), .ndpi (Hamamatsu), .tiff (Philips),
    and returns only those that do not have a corresponding .geojson file.
    """
    # accepted WSI formats
    wsi_extensions = {".svs", ".ndpi", ".tiff", ".tif"}
    wsi_files = []

    # collect all existing .geojson files for quick lookup
    geojson_files = {str(p.name) for p in Path(geojson_folder).rglob("*.geojson")}

    # recursively search for WSIs
    for path in Path(root_folder).rglob("*"):
        if path.suffix.lower() in wsi_extensions:
            geojson_name = path.with_suffix(".geojson").name
            if geojson_name not in geojson_files:
                wsi_files.append(str(path.resolve()))

    print(f"{'#'*10} {len(wsi_files)} WSIs found! {'#'*10}")
    return wsi_files

def create_folders(output_dir):
  # Create output dirs
  tis_det_dir_mask = os.path.join(output_dir, 'tis_det_mask/')
  tis_det_dir_over = os.path.join(output_dir, 'tis_det_overlay/')
  tis_det_dir_thumb = os.path.join(output_dir, 'tis_det_thumbnail/')
  tis_det_dir_mask_col = os.path.join(output_dir, 'tis_det_mask_col/')
  maps_qc_dir = os.path.join(output_dir, 'maps_qc')
  overlay_qc_dir = os.path.join(output_dir, 'overlays_qc')
  mask_qc_dir = os.path.join(output_dir, 'mask_qc')

  try:
    shutil.rmtree(output_dir)
  except:
    pass

  try:
      os.makedirs(output_dir)
      os.makedirs(tis_det_dir_mask)
      os.makedirs(tis_det_dir_over)
      os.makedirs(tis_det_dir_thumb)
      os.makedirs(tis_det_dir_mask_col)
      os.makedirs(maps_qc_dir)
      os.makedirs(overlay_qc_dir)
      os.makedirs(mask_qc_dir)
  except:
      print('The folders are already there ..')
  return tis_det_dir_mask, tis_det_dir_over, tis_det_dir_thumb, tis_det_dir_mask_col, maps_qc_dir, overlay_qc_dir, mask_qc_dir

def wis_tis_detect(slide_name, tis_det_dir_mask, tis_det_dir_mask_col, tis_det_dir_thumb):

  slide = OpenSlide(slide_name)

  p = Path(slide_name)

  # Use the stem (name without suffix) for all artifacts
  mask_png_name      = f"{p.stem}_MASK.png"
  mask_col_png_name  = f"{p.stem}_MASK_COL.png"
  overlay_jpg_name   = f"{p.stem}_OVERLAY.jpg"
  thumb_jpg_name     = f"{p.stem}.jpg"

  tis_tir_mask_path      = str(Path(tis_det_dir_mask)     / mask_png_name)
  tis_tir_mask_col_path  = str(Path(tis_det_dir_mask_col) / mask_col_png_name)
  tis_overlay_path       = str(Path(tis_det_dir_over)     / overlay_jpg_name)  # uses global 'tis_det_dir_over'
  thumbnail_path         = str(Path(tis_det_dir_thumb)    / thumb_jpg_name)

  w_l0, h_l0 = slide.level_dimensions[0]
  mpp = round(float(slide.properties["openslide.mpp-x"]), 4)
  reduction_factor = MPP_MODEL_TD / mpp

  # Ensure integer thumbnail size
  thumb_w = max(1, int(round(w_l0 / reduction_factor)))
  thumb_h = max(1, int(round(h_l0 / reduction_factor)))

  image_or = slide.get_thumbnail((thumb_w, thumb_h))
  image_or.save(thumbnail_path, quality=80)

  # Match JPEG-compressed training input
  image = np.array(image_or)
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
  _, image = cv2.imencode(".jpg", image, encode_param)
  image = cv2.imdecode(image, 1)
  image = Image.fromarray(image)

  width, height = image.size
  p_s = M_P_S_MODEL_TD

  wi_n = width // p_s
  he_n = height // p_s

  overhang_wi = width - wi_n * p_s
  overhang_he = height - he_n * p_s

  tqdm.write(f"[{slide_name}] Overhang (<1 patch) -> width: {overhang_wi}, height: {overhang_he}")

  # Optional: a single progress bar for all patches
  total_patches = (he_n + 1) * (wi_n + 1)
  patch_bar = tqdm(total=total_patches, desc=f"Patches ({slide_name})", unit="patch", leave=False)

  end_image = None
  end_image_class_map = None

  # Inference loop
  with torch.inference_mode():
      for h in range(he_n + 1):
          temp_image = None
          temp_image_class_map = None

          for w in range(wi_n + 1):
              # Crop patch with border handling
              if w != wi_n and h != he_n:
                  image_work = image.crop((w * p_s, h * p_s, (w + 1) * p_s, (h + 1) * p_s))
              elif w == wi_n and h != he_n:
                  image_work = image.crop((width - p_s, h * p_s, width, (h + 1) * p_s))
              elif w != wi_n and h == he_n:
                  image_work = image.crop((w * p_s, height - p_s, (w + 1) * p_s, height))
              else:
                  image_work = image.crop((width - p_s, height - p_s, width, height))

              # Preprocess & predict
              image_pre = get_preprocessing(image_work, preprocessing_fn)
              x_tensor = torch.from_numpy(image_pre).to(DEVICE).unsqueeze(0)

              predictions = model.predict(x_tensor)  # model-specific forward
              predictions = predictions.squeeze().cpu().numpy()

              mask = np.argmax(predictions, axis=0).astype("int8")
              class_mask = make_class_map(mask, colors)

              # Stitch row (h) horizontally
              if w == 0:
                  temp_image = mask
                  temp_image_class_map = class_mask
              elif w == wi_n:
                  mask_clip = mask[:, p_s - overhang_wi : p_s] if overhang_wi > 0 else mask[:, :0]
                  temp_image = np.concatenate((temp_image, mask_clip), axis=1)

                  class_mask_clip = (
                      class_mask[:, p_s - overhang_wi : p_s, :] if overhang_wi > 0 else class_mask[:, :0, :]
                  )
                  temp_image_class_map = np.concatenate((temp_image_class_map, class_mask_clip), axis=1)
              else:
                  temp_image = np.concatenate((temp_image, mask), axis=1)
                  temp_image_class_map = np.concatenate((temp_image_class_map, class_mask), axis=1)

              patch_bar.update(1)

          # Stitch columns (across h)
          if h == 0:
              end_image = temp_image
              end_image_class_map = temp_image_class_map
          elif h == he_n:
              temp_clip = temp_image[p_s - overhang_he : p_s, :] if overhang_he > 0 else temp_image[:0, :]
              end_image = np.concatenate((end_image, temp_clip), axis=0)

              temp_class_clip = (
                  temp_image_class_map[p_s - overhang_he : p_s, :, :] if overhang_he > 0 else temp_image_class_map[:0, :, :]
              )
              end_image_class_map = np.concatenate((end_image_class_map, temp_class_clip), axis=0)
          else:
              end_image = np.concatenate((end_image, temp_image), axis=0)
              end_image_class_map = np.concatenate((end_image_class_map, temp_image_class_map), axis=0)

  patch_bar.close()

  Image.fromarray(end_image).save(tis_tir_mask_path)
  Image.fromarray(end_image_class_map).save(tis_tir_mask_col_path)

  overlay = cv2.addWeighted(np.array(image), OVER_IMAGE, end_image_class_map, OVER_MASK, 0)
  Image.fromarray(overlay).save(tis_overlay_path)

  return tis_tir_mask_path, tis_tir_mask_col_path, tis_overlay_path, thumbnail_path

from tqdm import tqdm

def copy_large_file(src_path, dst_folder):
    """
    Copies a large file from Google Drive to a clean destination folder in Colab.

    Steps:
    1. Deletes the destination folder if it exists.
    2. Recreates it cleanly.
    3. Copies the source file (with progress bar).
    4. Returns the absolute path to the copied file.
    """

    # --- Validate source ---
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"❌ Source file not found: {src_path}")

    # --- Ensure destination folder is clean ---
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)  # remove all contents
    os.makedirs(dst_folder, exist_ok=True)

    # --- Define destination path ---
    dst_path = os.path.join(dst_folder, os.path.basename(src_path))
    total_size = os.path.getsize(src_path)

    # --- Copy with progress bar ---
    with open(src_path, "rb") as src_file, open(dst_path, "wb") as dst_file, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"Copying {os.path.basename(src_path)}",
        ncols=80
    ) as pbar:
        while True:
            buf = src_file.read(1024 * 1024)  # 1 MB chunks
            if not buf:
                break
            dst_file.write(buf)
            pbar.update(len(buf))

    abs_path = os.path.abspath(dst_path)
    print(f"✅ File copied successfully to: {abs_path}")
    return abs_path

def get_geojson_name(slide_name):
    """
    Given a WSI filename (.svs, .ndpi, .tiff),
    returns the corresponding .geojson filename.
    """
    valid_exts = {".svs", ".ndpi", ".tiff", ".tif"}
    ext = Path(slide_name).suffix.lower()

    if ext not in valid_exts:
        raise ValueError(f"Unsupported WSI extension: {ext}")

    # Replace the extension dynamically
    geojson_name = os.path.basename(Path(slide_name).with_suffix(".geojson"))
    return geojson_name

def main(slide_name, tis_tir_mask_path, tis_tir_mask_col_path, tis_overlay_path, thumbnail_path, maps_qc_dir, overlay_qc_dir, mask_qc_dir):
    start = timeit.default_timer()
    slide_base = os.path.basename(slide_name)

    # We track 7 simple steps below
    with tqdm(total=7, desc=f"{slide_base}", unit="step", leave=False) as pbar:
        tqdm.write(f"Processing: {slide_base}")

        # 1) Open slide
        slide = open_slide(slide_name)
        pbar.update(1)

        # 2) Get slide info
        p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0, obj_power = slide_info(slide, M_P_S_MODEL, MPP_MODEL)
        pbar.set_postfix(mpp=mpp, w=w_l0, h=h_l0)
        pbar.update(1)

        # 3) Load tissue detection map
        tis_det_map = Image.open(tis_tir_mask_path)
        pbar.update(1)

        # 4) Resize TD map to working MPP
        target_w = max(1, int(w_l0 * mpp / MPP_MODEL))
        target_h = max(1, int(h_l0 * mpp / MPP_MODEL))
        tis_det_map_mpp = np.array(tis_det_map.resize((target_w, target_h), Image.Resampling.LANCZOS))
        pbar.update(1)

        model_prim = torch.load(MODEL_QC_DIR + MODEL_QC_NAME, map_location=DEVICE, weights_only=False)

        # 5) Run the heavy processing
        map_img, full_mask = slide_process_single(
            model_prim, tis_det_map_mpp, slide, patch_n_w_l0, patch_n_h_l0, p_s,
            M_P_S_MODEL, colors, ENCODER_MODEL, ENCODER_MODEL_WEIGHTS,
            DEVICE, BACK_CLASS, MPP_MODEL, mpp, w_l0, h_l0
        )
        pbar.update(1)

        # 6) Save map & mask
        p = Path(slide_name)
        map_path  = str(p.parent / f"{p.stem}_map_QC.png")
        mask_path = str(p.parent / f"{p.stem}_mask.png")


        geojson_name = get_geojson_name(slide_name)
        geojson_save_path = os.path.join(GEOJSON_DRIVER_FOLDER, geojson_name)

        map_img.save(map_path)
        cv2.imwrite(mask_path, full_mask)
        pbar.update(1)

        # 7) Export polygons as GeoJSON
        factor = MPP_MODEL / mpp
        mask_to_geojson(mask_path, geojson_save_path, factor)
        pbar.update(1)

    # Clean up & timing
    del full_mask
    stop = timeit.default_timer()
    tqdm.write(f"[done] {slide_base} in {stop - start:.1f}s → {map_path}, {mask_path}, {geojson_save_path}")


try:
    with zipfile.ZipFile(IMAGES_ZIP,'r') as z:
        z.extractall(SLIDE_DIR)
    print("Extracted. Verifying...");
except Exception as e:
    print(f"Extract Err: {e}. Skip.")
    raise Exception("Extract Err")


# Get slide names
slide_names = get_wsi_files(SLIDE_DIR, GEOJSON_DRIVER_FOLDER)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_MODEL_TD, ENCODER_MODEL_TD_WEIGHTS)

model = smp.UnetPlusPlus(
    encoder_name=ENCODER_MODEL_TD,
    encoder_weights=ENCODER_MODEL_TD_WEIGHTS,
    classes=2,
    activation=None,
)

model.load_state_dict(torch.load(os.path.join(MODEL_TD_DIR, MODEL_TD_NAME), map_location='cpu', weights_only=False))
model.to(DEVICE)
model.eval()

for slide_name in tqdm(slide_names, desc="Slides", unit="slide", dynamic_ncols=True):
    slide_name = copy_large_file(slide_name, DST_FOLDER)
    sname = Path(slide_name).name
    try:
        with tqdm(total=3, desc=f"{sname}", unit="step", leave=False, position=1, dynamic_ncols=True) as pbar:
            # 1) Create output folders
            tis_det_dir_mask, tis_det_dir_over, tis_det_dir_thumb, tis_det_dir_mask_col, maps_qc_dir, overlay_qc_dir, mask_qc_dir = create_folders(OUTPUT_DIR)
            pbar.set_postfix_str("folders created"); pbar.update(1)

            # 2) Run tissue detection (inputs for main)
            tis_tir_mask_path, tis_tir_mask_col_path, tis_overlay_path, thumbnail_path = wis_tis_detect(
                slide_name, tis_det_dir_mask, tis_det_dir_mask_col, tis_det_dir_thumb
            )
            pbar.set_postfix_str("td detect done"); pbar.update(1)

            # 3) Main processing (has its own bar inside)
            main(
                slide_name,
                tis_tir_mask_path, tis_tir_mask_col_path, tis_overlay_path, thumbnail_path,
                maps_qc_dir, overlay_qc_dir, mask_qc_dir
            )
            pbar.set_postfix_str("main done"); pbar.update(1)
    except Exception as e:
        tqdm.write(f"[ERROR] {sname}: {e}")
        continue
print(f"{10*'*'} Processed Finished {10*'*'}")