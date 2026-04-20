# mypy: ignore-errors

import cv2
import numpy as np
from PIL import Image

from helpers.cv2_compat import ensure_cv2_compat

cv2 = ensure_cv2_compat(cv2)


# MAKE OVERLAY: HEATMAP ON REDUCED AND CROPPED SLIDE CLON
def make_overlay(slide, wsi_heatmap_im, p_s, patch_n_w_l0, patch_n_h_l0, overlay_factor):
    del p_s, patch_n_w_l0, patch_n_h_l0
    w_l0, h_l0 = slide.level_dimensions[0]

    slide_reduced = slide.get_thumbnail((w_l0 / overlay_factor, h_l0 / overlay_factor))

    heatmap_temp = wsi_heatmap_im.resize(slide_reduced.size, Image.Resampling.LANCZOS)
    overlay = cv2.addWeighted(np.array(slide_reduced), 0.7, np.array(heatmap_temp), 0.3, 0)
    return overlay
