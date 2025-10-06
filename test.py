import cv2
import numpy as np
from pathlib import Path

def overlay_mask_edges(
    image_path: str,
    mask_path: str,
    out_path: str = None,
    color=(0, 0, 255),   # BGR -> red
    thickness: int = 2,  # edge line thickness in pixels
    alpha: float = 1.0   # 1.0 = solid red line; <1.0 = semi-transparent
):
    """
    Draws the boundary of a binary mask on the image.

    Parameters
    ----------
    image_path : str
        Path to the base image (RGB/BGR/gray).
    mask_path : str
        Path to the mask (PNG). Can be binary, grayscale, or RGBA; non-zero/alpha are treated as 'inside'.
    out_path : str, optional
        If provided, saves result to this path. If None, shows a preview window.
    color : tuple(int,int,int)
        BGR color for the edge (default = red).
    thickness : int
        Line thickness in pixels.
    alpha : float
        Opacity of the edge when blended. 1.0 is opaque; e.g. 0.6 is semi-transparent.
    """
    # --- 1) Load image and mask ---
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Read mask with unchanged flag to keep alpha if present
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    # --- 2) Convert mask to a single-channel binary (0/255) ---
    # If mask has 4 channels (e.g., RGBA), use alpha channel as the mask
    if m.ndim == 3 and m.shape[2] == 4:
        mask_gray = m[:, :, 3]
    elif m.ndim == 3:
        # Take one channel and threshold
        mask_gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = m

    # Threshold: any non-zero is considered 'inside'
    _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)

    # --- 3) Check sizes (mask and image must match) ---
    if mask_bin.shape[:2] != img.shape[:2]:
        raise ValueError(
            f"Size mismatch: image={img.shape[:2]} vs mask={mask_bin.shape[:2]}.\n"
            "Please ensure the mask is the same size as the image."
        )

    # --- 4) Find mask boundary as contours (sub-pixel safe and crisp) ---
    # You could also use morphological gradient, but contours give very clean edges.
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # --- 5) Draw edges on an overlay, then blend if alpha < 1.0 ---
    overlay = img.copy()
    cv2.drawContours(overlay, contours, contourIdx=-1, color=color, thickness=thickness)

    if alpha >= 1.0:
        result = overlay
    else:
        result = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)

    # --- 6) Save or show ---
    if out_path is None:
        # Quick preview (press any key to close)
        cv2.imshow("Mask edges overlay", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, result)
        print(f"Saved: {out_path}")

    return result

# -------------------------
# Example usage:
# -------------------------
# result = overlay_mask_edges(
#     image_path="image.png",
#     mask_path="mask.png",
#     out_path="overlay_edges.png",
#     color=(0, 0, 255),   # red (BGR)
#     thickness=2,
#     alpha=1.0
# )
if __name__ == "__main__":
    # Example usage
    result = overlay_mask_edges(
        image_path=r"D:\Usuario\Desktop\Base_de_dados\MASTER_BASE\CANCER\CANCER_PATIENT_00083_10080_30800_9701_20250925_200252_974642.png",
        mask_path=r"D:\Usuario\Desktop\Base_de_dados\MASTER_BASE\CANCER_MASK\CANCER_PATIENT_00083_10080_30800_9701_20250925_200252_974642.png",
        out_path="overlay_edges.png",
        color=(0, 0, 255),   # red (BGR)
        thickness=2,
        alpha=1.0
    )
