import os
import sys
import numpy as np
import pandas as pd
import time
import itertools
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score

try:
    from tqdm import tqdm
    from skimage import io as skimage_io
    from skimage.color import rgb2gray
    import cv2
    if not hasattr(cv2, 'ximgproc'): raise ImportError
except ImportError:
    print("Please ensure tqdm, scikit-image, and opencv-contrib-python are installed in your environment.")
    sys.exit(1)

# --- 1. CONFIGURATION ---

# --- FOLDERS ---
BASE_DIR = r"C:\Images_IA_MEDICA\CANCER\master_candidate_pool" # Use the same path as before
APPROVED_FOLDER = os.path.join(BASE_DIR, 'Approved')
REJECTED_FOLDER = os.path.join(BASE_DIR, 'Rejected')

# --- HYPERPARAMETER GRID ---
PARAMETER_GRID = {
    'bg_intensity_thresh': [235, 240, 245],
    'k': [150, 300, 450],
    'min_size': [50, 100, 150]
}

# --- OPTIMIZATION & EVALUATION ---
THRESHOLDS = np.arange(0.05, 0.96, 0.01)
N_SPLITS_INNER_CV = 3
TEST_SET_SIZE = 0.2

# --- 2. CORE FUNCTIONS ---

def get_background_percentage(image_path, params):
    """Processes an image using the graph method with a given set of parameters."""
    try:
        image = cv2.imread(image_path)
        if image is None: raise IOError("Image read failed.")
        
        segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(
            sigma=0.5, k=params['k'], min_size=params['min_size']
        )
        segment_map = segmentator.processImage(image)
        
        num_segments = np.max(segment_map) + 1
        background_pixel_count = 0
        for seg_id in range(num_segments):
            segment_mask = (segment_map == seg_id)
            avg_color = cv2.mean(image, mask=segment_mask.astype(np.uint8))
            avg_intensity = (avg_color[0] + avg_color[1] + avg_color[2]) / 3
            if avg_intensity > params['bg_intensity_thresh']:
                background_pixel_count += np.sum(segment_mask)
        
        return background_pixel_count / image.size
    except Exception:
        return None

def find_best_bg_threshold(image_paths, true_labels, params, pbar):
    """Finds the optimal background percentage threshold for a given set of graph parameters."""
    bg_percentages, valid_labels = [], []
    for img_path, label in zip(image_paths, true_labels):
        bg = get_background_percentage(img_path, params)
        if bg is not None:
            bg_percentages.append(bg)
            valid_labels.append(label)
        if pbar: pbar.update(1)

    best_f1, best_bg_thresh = -1, 0
    for bg_thresh in THRESHOLDS:
        predictions = ['Rejected' if bg > bg_thresh else 'Approved' for bg in bg_percentages]
        f1 = f1_score(valid_labels, predictions, pos_label='Rejected', zero_division=0)
        if f1 > best_f1:
            best_f1, best_bg_thresh = f1, bg_thresh
    
    return best_bg_thresh

def evaluate_on_test_set(train_files, train_labels, test_files, test_labels, best_params):
    """Final evaluation on the held-out test set using the best parameters found."""
    print("\n--- Evaluating final model on the held-out test set ---")
    
    # Re-fit on the ENTIRE training set to find the final background threshold
    print("Step 1: Finding final background threshold using all training data...")
    pbar_fit = tqdm(total=len(train_files), desc="Final Fit")
    final_bg_threshold = find_best_bg_threshold(train_files, train_labels, best_params, pbar_fit)
    pbar_fit.close()
    print(f" > Final optimal background threshold found: {final_bg_threshold:.2f}")

    # Now, evaluate with this threshold on the unseen test set
    print("Step 2: Evaluating performance on test set...")
    bg_percentages, valid_labels = [], []
    pbar_eval = tqdm(zip(test_files, test_labels), total=len(test_files), desc="Final Evaluation")
    for img_path, label in pbar_eval:
        bg = get_background_percentage(img_path, best_params)
        if bg is not None:
            bg_percentages.append(bg)
            valid_labels.append(label)

    predictions = ['Rejected' if bg > final_bg_threshold else 'Approved' for bg in bg_percentages]
    
    acc = accuracy_score(valid_labels, predictions)
    f1 = f1_score(valid_labels, predictions, pos_label='Rejected', zero_division=0)

    print("\n--- Final Test Set Performance ---")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1-Score (Rejected): {f1:.3f}")

    # *** MODIFICATION: Return the final threshold ***
    return final_bg_threshold

# --- 3. MAIN TUNING SCRIPT ---

def main():
    start_time = time.time()
    print("--- Hyperparameter Tuning for Graph-Based Segmentation ---")
    
    approved_files = [os.path.join(APPROVED_FOLDER, f) for f in os.listdir(APPROVED_FOLDER) if f.lower().endswith('.png')]
    rejected_files = [os.path.join(REJECTED_FOLDER, f) for f in os.listdir(REJECTED_FOLDER) if f.lower().endswith('.png')]
    all_files = np.array(approved_files + rejected_files)
    all_labels = np.array(['Approved'] * len(approved_files) + ['Rejected'] * len(rejected_files))
    
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=TEST_SET_SIZE, random_state=42, stratify=all_labels
    )
    print(f"Data split: {len(train_files)} training images, {len(test_files)} test images.")

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
        
        # Estimate total images to process for the inner CV progress bar
        total_cv_images = (len(train_files) // N_SPLITS_INNER_CV) * (N_SPLITS_INNER_CV -1) + (len(train_files) // N_SPLITS_INNER_CV)
        pbar_cv = tqdm(total=total_cv_images * N_SPLITS_INNER_CV, desc=f"CV k={params['k']}", leave=False)

        for train_idx, val_idx in skf.split(train_files, train_labels):
            X_train_fold, X_val_fold = train_files[train_idx], train_files[val_idx]
            y_train_fold, y_val_fold = train_labels[train_idx], train_labels[val_idx]
            
            optimal_bg_thresh = find_best_bg_threshold(X_train_fold, y_train_fold, params, pbar_cv)
            
            bg_percentages_val, valid_labels_val = [], []
            for img_path, label in zip(X_val_fold, y_val_fold):
                bg = get_background_percentage(img_path, params)
                if bg is not None:
                    bg_percentages_val.append(bg)
                    valid_labels_val.append(label)
                pbar_cv.update(1)

            predictions = ['Rejected' if bg > optimal_bg_thresh else 'Approved' for bg in bg_percentages_val]
            f1 = f1_score(valid_labels_val, predictions, pos_label='Rejected', zero_division=0)
            fold_f1_scores.append(f1)
        
        pbar_cv.close()
        avg_f1 = np.mean(fold_f1_scores)
        
        if avg_f1 > best_f1_score:
            best_f1_score = avg_f1
            best_params = params

    print("\n\n--- Grid Search Complete ---")
    print(f"Best cross-validated F1-Score (Rejected): {best_f1_score:.3f}")
    print(f"Best graph parameters found: {best_params}")

    # *** MODIFICATION: Capture the returned final threshold ***
    final_bg_threshold = evaluate_on_test_set(train_files, train_labels, test_files, test_labels, best_params)
    
    print(f"\nTotal execution time: {(time.time() - start_time) / 60:.2f} minutes.")
    
    # *** MODIFICATION: Add new final output section ***
    print("\n\n---------------------------------------------------------")
    print("--- Recommended Default Parameters for Pipeline ---")
    print("---------------------------------------------------------")
    print(f"Graph Method `k`:                      {best_params['k']}")
    print(f"Graph Method `min_size`:               {best_params['min_size']}")
    print(f"Graph Method `bg_intensity_thresh`:    {best_params['bg_intensity_thresh']}")
    print(f"Background Percentage Threshold:       {final_bg_threshold:.2f}")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    main()