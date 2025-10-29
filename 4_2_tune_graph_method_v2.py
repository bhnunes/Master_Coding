import os
import sys
import numpy as np
import time
import logging
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score

try:
    from tqdm import tqdm
    import cv2
    if not hasattr(cv2, 'ximgproc'): raise ImportError("cv2.ximgproc not found")
    # --- NEW: Import Bayesian Optimization tools ---
    from skopt import gp_minimize
    from skopt.space import Integer
    from skopt.utils import use_named_args
except ImportError:
    print("Please ensure tqdm, opencv-contrib-python, and scikit-optimize are installed.")
    sys.exit(1)

# --- Logging and Configuration (mostly unchanged) ---
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if logger.hasHandlers(): logger.handlers.clear()
file_handler = logging.FileHandler("bayesian_optimization.log", mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Changed to INFO for better feedback
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

SOURCE_IMAGE_FOLDER = os.path.normpath(r"D:\Usuario\Desktop\Base_de_dados\CHILE\PATCHES\CANCER")
SOURCE_MASK_FOLDER = os.path.normpath(r"D:\Usuario\Desktop\Base_de_dados\CHILE\PATCHES\CANCER_MASK")
BASE_DIR = os.path.normpath(r"D:\Usuario\Desktop\Base_de_dados\CHILE\Optimization_Test\master_candidate_pool")
APPROVED_FOLDER = os.path.join(BASE_DIR, 'APPROVED')
REJECTED_FOLDER = os.path.join(BASE_DIR, 'REJECTED')

# --- NEW: Define the Search Space for Bayesian Optimization ---
# We define a range for each integer parameter.
SEARCH_SPACE = [
    Integer(230, 250, name='bg_intensity_thresh'),
    Integer(100, 500, name='k'),
    Integer(50, 200, name='min_size'),
    Integer(0, 10, name='erosion_px')
]

THRESHOLDS = np.arange(0.05, 0.96, 0.01)
N_SPLITS_INNER_CV = 3
TEST_SET_SIZE = 0.2
# --- NEW: Set a budget for the number of evaluations ---
N_BAYESIAN_CALLS = 30 # We will run 30 evaluations instead of 81

def get_roi_contamination(image_path, mask_path, params):
    """
    Core function to calculate contamination rate.
    FIXED: Explicitly casts parameters to standard Python types for OpenCV compatibility.
    """
    base_name = os.path.basename(image_path)
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.debug(f"[{base_name}] FAILED: cv2.imread returned None for image path.")
            return None

        roi_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            logging.debug(f"[{base_name}] FAILED: cv2.imread returned None for mask path.")
            return None
        
        _, roi_mask = cv2.threshold(roi_mask, 1, 255, cv2.THRESH_BINARY)
        
        M = (roi_mask > 0).astype(np.uint8)
        initial_roi_area = M.sum()

        erosion_px = int(params.get('erosion_px', 0)) # Cast erosion just in case
        if erosion_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1))
            M = cv2.erode(M, kernel)
        
        final_roi_area = M.sum()

        if final_roi_area == 0:
            if initial_roi_area > 0:
                logging.debug(f"[{base_name}] FAILED: ROI area became zero after erosion of {erosion_px}px.")
            else:
                logging.debug(f"[{base_name}] FAILED: Initial ROI area was zero.")
            return np.nan

        # --- THE FIX IS HERE ---
        # Explicitly convert the parameters from NumPy types to standard Python types
        # that OpenCV can safely handle.
        k_val = float(params['k'])
        min_size_val = int(params['min_size'])
        # --- END OF FIX ---

        segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5,
            k=k_val,
            min_size=min_size_val
        )
        segment_map = segmentator.processImage(image)
        
        background_mask = np.zeros(image.shape[:2], dtype=bool)
        num_segments = np.max(segment_map) + 1
        for seg_id in range(num_segments):
            segment_pixel_mask = (segment_map == seg_id)
            if np.any(segment_pixel_mask):
                avg_color = cv2.mean(image, mask=segment_pixel_mask.astype(np.uint8))
                avg_intensity = (avg_color[0] + avg_color[1] + avg_color[2]) / 3
                if avg_intensity > params['bg_intensity_thresh']:
                    background_mask[segment_pixel_mask] = True

        contamination_pixels = (background_mask & (M.astype(bool))).sum()
        contamination_rate = contamination_pixels / final_roi_area
        
        logging.debug(f"[{base_name}] SUCCESS: Rate={contamination_rate:.4f} with params k={params['k']}, min_size={params['min_size']}, erosion={erosion_px}.")
        return contamination_rate
        
    except Exception as e:
        logging.error(f"[{base_name}] CRITICAL EXCEPTION: {e}", exc_info=True)
        return None
    
def find_best_contamination_threshold(image_mask_pairs, true_labels, params):
    # This helper function also remains identical.
    contamination_rates, valid_labels = [], []
    for (img_path, mask_path), label in zip(image_mask_pairs, true_labels):
        rate = get_roi_contamination(img_path, mask_path, params)
        if rate is not None and not np.isnan(rate):
            contamination_rates.append(rate)
            valid_labels.append(label)
    best_f1, best_tau = -1, 0
    for tau in THRESHOLDS:
        predictions = ['Rejected' if rate > tau else 'Approved' for rate in contamination_rates]
        f1 = f1_score(valid_labels, predictions, pos_label='Rejected', zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau

# --- evaluate_on_test_set is also UNCHANGED ---
def evaluate_on_test_set(train_pairs, train_labels, test_pairs, test_labels, best_params):
    # This final validation step remains the same and is crucial for scientific validity.
    # ... (code is identical to your original script)
    logging.info("\n--- Evaluating final model on the held-out test set ---")
    logging.info("Step 1: Finding final contamination threshold using all training data...")
    final_tau = find_best_contamination_threshold(train_pairs, train_labels, best_params)
    logging.info(f" > Final optimal contamination threshold (tau) found: {final_tau:.2f}")
    logging.info("Step 2: Evaluating performance on test set...")
    rates, valid_labels = [], []
    for (img_path, mask_path), label in tqdm(zip(test_pairs, test_labels), total=len(test_pairs), desc="Final Evaluation"):
        rate = get_roi_contamination(img_path, mask_path, best_params)
        if rate is not None and not np.isnan(rate):
            rates.append(rate)
            valid_labels.append(label)
    predictions = ['Rejected' if rate > final_tau else 'Approved' for rate in rates]
    acc = accuracy_score(valid_labels, predictions)
    f1 = f1_score(valid_labels, predictions, pos_label='Rejected', zero_division=0)
    logging.info("\n--- Final Test Set Performance ---")
    logging.info(f"Accuracy: {acc:.3f}")
    logging.info(f"F1-Score (Rejected): {f1:.3f}")
    return final_tau


# --- NEW: The Objective Function for Bayesian Optimization ---
# This function wraps your previous worker logic.
# It is designed to be called by the `gp_minimize` function.

# Global variables to hold the data, to avoid passing it repeatedly
# This is a common pattern when using multiprocessing with Bayesian optimization
g_train_pairs = None
g_train_labels = None

@use_named_args(dimensions=SEARCH_SPACE)
def objective(**params):
    """
    Objective function for Bayesian Optimization.
    Takes a set of hyperparameters, evaluates them using cross-validation,
    and returns the score to be MINIMIZED.
    """
    global g_train_pairs, g_train_labels
    
    skf = StratifiedKFold(n_splits=N_SPLITS_INNER_CV, shuffle=True, random_state=42)
    fold_f1_scores = []

    # Use the globally defined training data
    for train_idx, val_idx in skf.split(g_train_pairs, g_train_labels):
        X_train_fold, X_val_fold = g_train_pairs[train_idx], g_train_pairs[val_idx]
        y_train_fold, y_val_fold = g_train_labels[train_idx], g_train_labels[val_idx]

        optimal_tau = find_best_contamination_threshold(X_train_fold, y_train_fold, params)
        
        rates_val, valid_labels_val = [], []
        for (img_path, mask_path), label in zip(X_val_fold, y_val_fold):
            rate = get_roi_contamination(img_path, mask_path, params)
            if rate is not None and not np.isnan(rate):
                rates_val.append(rate)
                valid_labels_val.append(label)

        predictions = ['Rejected' if rate > optimal_tau else 'Approved' for rate in rates_val]
        f1 = f1_score(valid_labels_val, predictions, pos_label='Rejected', zero_division=0)
        fold_f1_scores.append(f1)
    
    avg_f1 = np.mean(fold_f1_scores) if fold_f1_scores else 0
    
    # We want to MAXIMIZE F1-score, but gp_minimize MINIMIZES the objective.
    # Therefore, we return the negative of the score.
    return -avg_f1

# --- MAIN SCRIPT (Refactored with Pre-flight Check) ---
def main():
    global g_train_pairs, g_train_labels
    start_time = time.time()
    logging.info("--- Hyperparameter Tuning with Bayesian Optimization ---")
    
    # --- Step 1: Load ground truth file lists ---
    try:
        approved_files = [f for f in os.listdir(APPROVED_FOLDER) if f.lower().endswith(('.png', '.jpg', '.tif'))]
        rejected_files = [f for f in os.listdir(REJECTED_FOLDER) if f.lower().endswith(('.png', '.jpg', '.tif'))]
        if not approved_files and not rejected_files:
            raise FileNotFoundError("Both APPROVED and REJECTED folders are empty. Nothing to tune.")
    except FileNotFoundError as e:
        logging.error(f"FATAL: Could not read APPROVED/REJECTED folders in '{BASE_DIR}'. Error: {e}")
        return

    # --- Step 2: Construct full paths to SOURCE data ---
    all_pairs_list = (
        [(os.path.join(SOURCE_IMAGE_FOLDER, f), os.path.join(SOURCE_MASK_FOLDER, f)) for f in approved_files] +
        [(os.path.join(SOURCE_IMAGE_FOLDER, f), os.path.join(SOURCE_MASK_FOLDER, f)) for f in rejected_files]
    )
    all_labels_list = ['Approved'] * len(approved_files) + ['Rejected'] * len(rejected_files)
    
    # --- Step 3: <<< NEW: PRE-FLIGHT CHECK >>> ---
    # Verify that all constructed paths actually point to existing files before proceeding.
    logging.info("Performing pre-flight check on all source file paths...")
    
    valid_pairs = []
    valid_labels = []
    missing_files = False
    for (img_path, mask_path), label in tqdm(zip(all_pairs_list, all_labels_list), total=len(all_labels_list), desc="Verifying files"):
        if not os.path.exists(img_path):
            logging.warning(f"Source image not found, skipping pair: {img_path}")
            missing_files = True
        elif not os.path.exists(mask_path):
            logging.warning(f"Source mask not found, skipping pair: {mask_path}")
            missing_files = True
        else:
            valid_pairs.append((img_path, mask_path))
            valid_labels.append(label)
            
    if missing_files:
        logging.error("Some source files were not found. The process can continue with the valid files, but results may be skewed. Please check the warnings above.")
    
    if not valid_pairs:
        logging.error("FATAL: No valid image/mask pairs found after checking paths. Halting execution.")
        return

    all_pairs = np.array(valid_pairs)
    all_labels = np.array(valid_labels)
    # --- END of PRE-FLIGHT CHECK ---

    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        all_pairs, all_labels, test_size=TEST_SET_SIZE, random_state=42, stratify=all_labels
    )
    
    g_train_pairs = train_pairs
    g_train_labels = train_labels
    
    logging.info(f"Data split: {len(train_pairs)} training, {len(test_pairs)} test (after validation).")
    logging.info(f"Starting Bayesian Optimization with a budget of {N_BAYESIAN_CALLS} evaluations.")

    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        n_calls=N_BAYESIAN_CALLS,
        n_initial_points=10,
        acq_func='EI',
        random_state=42,
        verbose=True
    )
    
    best_f1_score = -result.fun if result.fun is not None else 0.0
    best_params_list = result.x
    best_params = {dimension.name: value for dimension, value in zip(SEARCH_SPACE, best_params_list)}

    logging.info("\n\n--- Bayesian Optimization Complete ---")
    logging.info(f"Best cross-validated F1-Score (Rejected): {best_f1_score:.3f}")
    logging.info(f"Best graph parameters found: {best_params}")

    # --- FINAL EVALUATION (Ensure it doesn't crash on empty test data) ---
    if len(test_pairs) > 0:
        final_tau = evaluate_on_test_set(train_pairs, train_labels, test_pairs, test_labels, best_params)
    else:
        logging.warning("Test set is empty. Skipping final evaluation.")
        final_tau = find_best_contamination_threshold(train_pairs, train_labels, best_params) # Re-calculate on full train set
    
    logging.info(f"\nTotal execution time: {(time.time() - start_time) / 60:.2f} minutes.")
    
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