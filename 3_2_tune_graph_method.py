import os
import sys
import numpy as np
import time
import itertools
import logging
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path

try:
    from tqdm import tqdm
    import cv2
    if not hasattr(cv2, 'ximgproc'): raise ImportError("cv2.ximgproc not found")
except ImportError:
    print("Please ensure tqdm and opencv-contrib-python are installed.")
    sys.exit(1)

# --- Scientific Logging Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler("optimization.log", mode='a')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


# --- 1. CONFIGURATION ---

# --- SOURCE FOLDERS (Ground Truth) ---
# The optimization now correctly operates on the original, pristine data.
SOURCE_IMAGE_FOLDER = r"C:\Images_IA_MEDICA\CANCER"
SOURCE_MASK_FOLDER = r"C:\Images_IA_MEDICA\CANCER_MASK" 

# --- LABEL FOLDERS (Pointers to Source Data) ---
# These folders only provide the FILENAMES and their classification.
BASE_DIR = r"C:\Images_IA_MEDICA\Optimization_Test\master_candidate_pool"
APPROVED_FOLDER = os.path.join(BASE_DIR, 'APPROVED')
REJECTED_FOLDER = os.path.join(BASE_DIR, 'REJECTED')

# --- HYPERPARAMETER GRID ---
# We now include 'erosion_px' as a tunable parameter, as recommended.
PARAMETER_GRID = {
    'bg_intensity_thresh': [235, 240, 245],
    'k': [150, 300, 450],
    'min_size': [50, 100, 150],
    'erosion_px': [0, 5, 10] # 0=No erosion, 5,10=Erosion radii to test
}

# --- OPTIMIZATION & EVALUATION ---
THRESHOLDS = np.arange(0.05, 0.96, 0.01) # Thresholds for the contamination rate 'tau'
N_SPLITS_INNER_CV = 3
TEST_SET_SIZE = 0.2


# --- 2. CORE FUNCTIONS (Refactored for Scientific Validity) ---

def get_roi_contamination(image_path, mask_path, params):
    """
    Calculates the ROI Contamination Rate, a scientifically valid metric for
    partial annotations, based on the specialist's recommendation.
    
    This function replaces the naive `get_background_percentage`.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image: {image_path}")
            return None

        # 1. Load the specialist's annotation mask (M)
        roi_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            logging.error(f"Failed to read mask for image: {image_path}")
            return None
        
        # Ensure mask is binary (0 or 255)
        _, roi_mask = cv2.threshold(roi_mask, 1, 255, cv2.THRESH_BINARY)

        # 2. Run the graph segmenter to get the predicted background mask (B)
        segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5, k=params['k'], min_size=params['min_size']
        )
        segment_map = segmentator.processImage(image)
        
        # Initialize the predicted background mask (B)
        background_mask = np.zeros(image.shape[:2], dtype=bool)
        num_segments = np.max(segment_map) + 1
        
        for seg_id in range(num_segments):
            segment_pixel_mask = (segment_map == seg_id)
            avg_color = cv2.mean(image, mask=segment_pixel_mask.astype(np.uint8))
            avg_intensity = (avg_color[0] + avg_color[1] + avg_color[2]) / 3
            if avg_intensity > params['bg_intensity_thresh']:
                background_mask[segment_pixel_mask] = True
        
        # 3. Calculate the ROI Contamination Rate: C = |B ∩ M| / |M|
        # As per the specialist's implementation sketch.
        M = (roi_mask > 0).astype(np.uint8)
        erosion_px = params.get('erosion_px', 0) # Use erosion_px from params
        
        if erosion_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erosion_px+1, 2*erosion_px+1))
            M = cv2.erode(M, kernel)
            
        roi_area = M.sum()
        if roi_area == 0:
            logging.warning(f"ROI area for {Path(image_path).name} is zero after erosion. Cannot calculate contamination.")
            return np.nan # Use NaN for invalid calculation

        contamination_pixels = (background_mask & (M.astype(bool))).sum()
        
        return contamination_pixels / roi_area

    except Exception as e:
        logging.error(f"An exception occurred while processing {Path(image_path).name}: {e}")
        return None

def find_best_contamination_threshold(image_mask_pairs, true_labels, params, pbar):
    """
    Finds the optimal contamination rate threshold (tau) for a given set of graph parameters.
    Renamed and adapted from `find_best_bg_threshold`.
    """
    contamination_rates, valid_labels = [], []
    for (img_path, mask_path), label in zip(image_mask_pairs, true_labels):
        rate = get_roi_contamination(img_path, mask_path, params)
        if rate is not None and not np.isnan(rate):
            contamination_rates.append(rate)
            valid_labels.append(label)
        if pbar: pbar.update(1)

    best_f1, best_tau = -1, 0
    for tau in THRESHOLDS:
        # Prediction: 'Rejected' if contamination rate > tau
        predictions = ['Rejected' if rate > tau else 'Approved' for rate in contamination_rates]
        f1 = f1_score(valid_labels, predictions, pos_label='Rejected', zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    
    return best_tau

def evaluate_on_test_set(train_pairs, train_labels, test_pairs, test_labels, best_params):
    """Final evaluation on the held-out test set using the best parameters found."""
    print("\n--- Evaluating final model on the held-out test set ---")
    
    print("Step 1: Finding final contamination threshold using all training data...")
    pbar_fit = tqdm(total=len(train_pairs), desc="Final Fit")
    final_tau = find_best_contamination_threshold(train_pairs, train_labels, best_params, pbar_fit)
    pbar_fit.close()
    print(f" > Final optimal contamination threshold (tau) found: {final_tau:.2f}")

    print("Step 2: Evaluating performance on test set...")
    rates, valid_labels = [], []
    pbar_eval = tqdm(zip(test_pairs, test_labels), total=len(test_pairs), desc="Final Evaluation")
    for (img_path, mask_path), label in pbar_eval:
        rate = get_roi_contamination(img_path, mask_path, best_params)
        if rate is not None and not np.isnan(rate):
            rates.append(rate)
            valid_labels.append(label)

    predictions = ['Rejected' if rate > final_tau else 'Approved' for rate in rates]
    
    acc = accuracy_score(valid_labels, predictions)
    f1 = f1_score(valid_labels, predictions, pos_label='Rejected', zero_division=0)

    print("\n--- Final Test Set Performance ---")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1-Score (Rejected): {f1:.3f}")

    return final_tau


# --- 3. MAIN TUNING SCRIPT ---

def main():
    start_time = time.time()
    print("--- Hyperparameter Tuning for ROI-Contamination Based Classifier ---")
    
    # CORRECTED DATA LOADING:
    # Read filenames from label folders, but create paths to the original source data.
    approved_files = [f for f in os.listdir(APPROVED_FOLDER) if f.lower().endswith(('.png', '.jpg', '.tif'))]
    rejected_files = [f for f in os.listdir(REJECTED_FOLDER) if f.lower().endswith(('.png', '.jpg', '.tif'))]
    
    # Create pairs of (image_path, mask_path)
    all_pairs = np.array(
        [(os.path.join(SOURCE_IMAGE_FOLDER, f), os.path.join(SOURCE_MASK_FOLDER, f)) for f in approved_files] +
        [(os.path.join(SOURCE_IMAGE_FOLDER, f), os.path.join(SOURCE_MASK_FOLDER, f)) for f in rejected_files]
    )
    all_labels = np.array(['Approved'] * len(approved_files) + ['Rejected'] * len(rejected_files))
    
    # Check for missing files
    for img_p, msk_p in all_pairs:
        if not os.path.exists(img_p) or not os.path.exists(msk_p):
            logging.error(f"Missing file pair: {img_p} or {msk_p}. Please check paths. Exiting.")
            return

    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        all_pairs, all_labels, test_size=TEST_SET_SIZE, random_state=42, stratify=all_labels
    )
    print(f"Data split: {len(train_pairs)} training images, {len(test_pairs)} test images.")
    logging.info(f"Data split: {len(train_pairs)} training, {len(test_pairs)} test.")

    param_keys = PARAMETER_GRID.keys()
    param_values = PARAMETER_GRID.values()
    param_combinations = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]
    
    best_f1_score = -1
    best_params = None
    
    pbar_grid = tqdm(param_combinations, desc="Grid Search")
    for params in pbar_grid:
        pbar_grid.set_postfix(params)
        skf = StratifiedKFold(n_splits=N_SPLITS_INNER_CV, shuffle=True, random_state=42)
        fold_f1_scores = []
        
        total_cv_images = len(train_pairs) # Each image is processed once per fold
        pbar_cv = tqdm(total=total_cv_images, desc=f"CV k={params['k']}", leave=False)

        for train_idx, val_idx in skf.split(train_pairs, train_labels):
            X_train_fold, X_val_fold = train_pairs[train_idx], train_pairs[val_idx]
            y_train_fold, y_val_fold = train_labels[train_idx], train_labels[val_idx]
            
            # Find optimal contamination threshold (tau) on the training part of the fold
            optimal_tau = find_best_contamination_threshold(X_train_fold, y_train_fold, params, pbar_cv)
            
            # Evaluate on the validation part of the fold
            rates_val, valid_labels_val = [], []
            for (img_path, mask_path), label in zip(X_val_fold, y_val_fold):
                rate = get_roi_contamination(img_path, mask_path, params)
                if rate is not None and not np.isnan(rate):
                    rates_val.append(rate)
                    valid_labels_val.append(label)
                pbar_cv.update(1)

            predictions = ['Rejected' if rate > optimal_tau else 'Approved' for rate in rates_val]
            f1 = f1_score(valid_labels_val, predictions, pos_label='Rejected', zero_division=0)
            fold_f1_scores.append(f1)
        
        pbar_cv.close()
        avg_f1 = np.mean(fold_f1_scores) if fold_f1_scores else 0
        
        if avg_f1 > best_f1_score:
            best_f1_score = avg_f1
            best_params = params

    print("\n\n--- Grid Search Complete ---")
    print(f"Best cross-validated F1-Score (Rejected): {best_f1_score:.3f}")
    print(f"Best graph parameters found: {best_params}")
    logging.info(f"Grid search complete. Best F1: {best_f1_score:.3f}, Best Params: {best_params}")

    final_tau = evaluate_on_test_set(train_pairs, train_labels, test_pairs, test_labels, best_params)
    
    print(f"\nTotal execution time: {(time.time() - start_time) / 60:.2f} minutes.")
    
    print("\n\n---------------------------------------------------------")
    print("--- Recommended Default Parameters for Pipeline ---")
    print("---------------------------------------------------------")
    print(f"Graph Method `k`:                      {best_params['k']}")
    print(f"Graph Method `min_size`:               {best_params['min_size']}")
    print(f"Graph Method `bg_intensity_thresh`:    {best_params['bg_intensity_thresh']}")
    print(f"ROI Erosion (Safety Margin):           {best_params['erosion_px']} pixels")
    print(f"Contamination Rate Threshold (tau):    {final_tau:.2f}")
    print("---------------------------------------------------------")
    logging.info(f"Final recommended tau: {final_tau:.2f}")


if __name__ == "__main__":
    main()