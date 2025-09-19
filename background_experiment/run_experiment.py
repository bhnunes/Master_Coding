import os
import sys
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar

# Add tqdm for progress monitoring
try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm is not installed. Please run 'pip install tqdm'")
    sys.exit(1)

# scikit-image for Otsu, Local Adaptive, and image reading
try:
    from skimage import io as skimage_io
    from skimage.color import rgb2gray
    from skimage.filters import threshold_otsu, threshold_local
except ImportError:
    print("Error: scikit-image is not installed. Please run 'pip install scikit-image imageio'")
    sys.exit(1)

# OpenCV for Graph-Based Segmentation
try:
    import cv2
    if not hasattr(cv2, 'ximgproc'): raise ImportError
except ImportError:
    print("Error: opencv-contrib-python is not installed.")
    print("Please run 'pip uninstall opencv-python' and then 'pip install opencv-contrib-python'")
    sys.exit(1)

# --- 1. CONFIGURATION ---

# --- FOLDERS ---
BASE_DIR = r"C:\Images_IA_MEDICA\CANCER\master_candidate_pool"
APPROVED_FOLDER = os.path.join(BASE_DIR, 'Approved')
REJECTED_FOLDER = os.path.join(BASE_DIR, 'Rejected')
POSITIVE_CLASS = 'Rejected' 

# --- METHODS ---
METHODS_TO_TEST = {
    "Graph-Based": "graph",
    "Otsu Global": "otsu",
    "Local Adaptive": "local_adaptive"
}

# --- CROSS-VALIDATION ---
N_SPLITS = 5

# --- OPTIMIZATION ---
THRESHOLDS = np.arange(0.05, 0.96, 0.01)

# --- HEURISTICS ---
BACKGROUND_INTENSITY_THRESHOLD = 240 # For graph-based method

# --- 2. IMPLEMENTATION OF SEGMENTATION FUNCTIONS ---

def get_background_percentage(image_path, method):
    # This function remains the same, but will be called from within a tqdm loop
    try:
        if method == 'otsu':
            image = skimage_io.imread(image_path)
            image_gray = rgb2gray(image)
            thresh = threshold_otsu(image_gray)
            binary_mask = image_gray > thresh
            return np.sum(binary_mask == False) / binary_mask.size

        elif method == 'local_adaptive':
            image = skimage_io.imread(image_path)
            image_gray = rgb2gray(image)
            block_size = int(min(image_gray.shape) * 0.15)
            if block_size % 2 == 0: block_size += 1
            thresh = threshold_local(image_gray, block_size, offset=0)
            binary_mask = image_gray > thresh
            return np.sum(binary_mask == False) / binary_mask.size
        
        elif method == 'graph':
            image = cv2.imread(image_path)
            if image is None: raise IOError("Image could not be read by OpenCV.")
            segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=300, min_size=100)
            segment_map = segmentator.processImage(image)
            num_segments = np.max(segment_map) + 1
            background_pixel_count = 0
            for seg_id in range(num_segments):
                segment_mask = (segment_map == seg_id)
                avg_color = cv2.mean(image, mask=segment_mask.astype(np.uint8))
                avg_intensity = (avg_color[0] + avg_color[1] + avg_color[2]) / 3
                if avg_intensity > BACKGROUND_INTENSITY_THRESHOLD:
                    background_pixel_count += np.sum(segment_mask)
            return background_pixel_count / image.size
        
        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        # Suppress warnings during runs to keep tqdm bar clean
        # print(f"Warning: Could not process image {os.path.basename(image_path)}... Error: {e}. Skipping.")
        return None

# --- 3 & 4. OPTIMIZATION AND EVALUATION ---

def find_optimal_threshold(image_paths, true_labels, method):
    best_accuracy, best_threshold = 0.0, 0.0
    bg_percentages, valid_labels = [], []
    
    # ADDED TQDM: Monitor image processing for optimization
    pbar = tqdm(zip(image_paths, true_labels), total=len(image_paths), desc=f"Optimizing '{method}'", leave=False)
    for img_path, label in pbar:
        bg = get_background_percentage(img_path, method)
        if bg is not None: 
            bg_percentages.append(bg)
            valid_labels.append(label)

    for threshold in THRESHOLDS:
        predictions = ['Rejected' if bg > threshold else 'Approved' for bg in bg_percentages]
        accuracy = accuracy_score(valid_labels, predictions)
        if accuracy > best_accuracy: best_accuracy, best_threshold = accuracy, threshold
    return best_threshold, best_accuracy

def evaluate_model(image_paths, true_labels, method, optimal_threshold):
    bg_percentages, valid_labels, valid_indices = [], [], []
    
    # ADDED TQDM: Monitor image processing for evaluation
    pbar = tqdm(enumerate(zip(image_paths, true_labels)), total=len(image_paths), desc=f"Evaluating '{method}'", leave=False)
    for i, (img_path, label) in pbar:
        bg = get_background_percentage(img_path, method)
        if bg is not None: 
            bg_percentages.append(bg)
            valid_labels.append(label)
            valid_indices.append(i)

    predictions = ['Rejected' if bg > optimal_threshold else 'Approved' for bg in bg_percentages]
    full_predictions = np.array([''] * len(true_labels), dtype=object)
    full_predictions[valid_indices] = predictions
    accuracy = accuracy_score(valid_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(valid_labels, predictions, average=None, labels=['Approved', 'Rejected'], zero_division=0)
    metrics = {'accuracy': accuracy, 'precision_approved': precision[0], 'recall_approved': recall[0], 'f1_approved': f1[0], 'precision_rejected': precision[1], 'recall_rejected': recall[1], 'f1_rejected': f1[1]}
    return metrics, full_predictions

def perform_mcnemar_test(model1_preds, model2_preds, model1_name, model2_name, all_labels):
    # This function remains the same
    print(f"\n--- McNemar's Test: {model1_name} vs. {model2_name} ---")
    valid_mask = (model1_preds != '') & (model2_preds != '')
    valid_labels = all_labels[valid_mask]; m1_preds, m2_preds = model1_preds[valid_mask], model2_preds[valid_mask]
    correct1 = (m1_preds == valid_labels); correct2 = (m2_preds == valid_labels)
    table = np.array([[np.sum(correct1 & correct2), np.sum(~correct1 & correct2)], [np.sum(correct1 & ~correct2), np.sum(~correct1 & ~correct2)]])
    print(f"Contingency Table:\n                {model2_name} Correct | {model2_name} Incorrect")
    print(f"{model1_name} Correct   {table[0,0]:>12.0f} | {table[1,0]:>18.0f}")
    print(f"{model1_name} Incorrect {table[0,1]:>12.0f} | {table[1,1]:>18.0f}")
    if table[0, 1] + table[1, 0] < 25: print("Warning: Low discordant pairs (<25). Test may be less reliable.")
    result = mcnemar(table, exact=False, correction=True)
    print(f"P-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05: print("Conclusion: The difference in error rates IS statistically significant.")
    else: print("Conclusion: There is NO statistically significant difference in error rates.")

# --- 5. MAIN EXPERIMENT SCRIPT ---
def main():
    start_time = time.time()
    print("--- Starting Segmentation Experiment (3-Way Comparison) ---")
    
    if not (os.path.isdir(APPROVED_FOLDER) and os.path.isdir(REJECTED_FOLDER)):
        print(f"Error: Ensure 'Approved' and 'Rejected' folders exist in {BASE_DIR}"); return
    approved_files = [os.path.join(APPROVED_FOLDER, f) for f in os.listdir(APPROVED_FOLDER) if f.lower().endswith('.png')]
    rejected_files = [os.path.join(REJECTED_FOLDER, f) for f in os.listdir(REJECTED_FOLDER) if f.lower().endswith('.png')]
    all_files, all_labels = np.array(approved_files + rejected_files), np.array(['Approved'] * len(approved_files) + ['Rejected'] * len(rejected_files))
    print(f"Loaded {len(approved_files)} 'Approved' and {len(rejected_files)} 'Rejected' images.")
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    results = {name: [] for name in METHODS_TO_TEST.keys()}
    all_predictions = {name: np.array([''] * len(all_labels), dtype=object) for name in METHODS_TO_TEST.keys()}
    
    # ADDED TQDM: Outer progress bar for cross-validation folds
    fold_pbar = tqdm(enumerate(skf.split(all_files, all_labels)), total=N_SPLITS, desc="Overall Progress")
    for fold, (train_idx, test_idx) in fold_pbar:
        fold_pbar.set_description(f"Fold {fold + 1}/{N_SPLITS}")
        train_files, test_files = all_files[train_idx], all_files[test_idx]
        train_labels, test_labels = all_labels[train_idx], all_labels[test_idx]
        
        for name, method_id in METHODS_TO_TEST.items():
            opt_thresh, _ = find_optimal_threshold(train_files, train_labels, method_id)
            metrics, preds = evaluate_model(test_files, test_labels, method_id, opt_thresh)
            results[name].append(metrics)
            all_predictions[name][test_idx] = preds
            
    print("\n\n--- Experiment Summary (Averaged over all folds) ---")
    for name, res_list in results.items():
        df = pd.DataFrame(res_list)
        print(f"\n--- Method: {name} ---")
        print(df.mean().round(3))
    
    print("\n\n--- Statistical Analysis (Pairwise McNemar's Tests) ---")
    method_names = list(METHODS_TO_TEST.keys())
    perform_mcnemar_test(all_predictions[method_names[0]], all_predictions[method_names[1]], method_names[0], method_names[1], all_labels)
    perform_mcnemar_test(all_predictions[method_names[0]], all_predictions[method_names[2]], method_names[0], method_names[2], all_labels)
    perform_mcnemar_test(all_predictions[method_names[1]], all_predictions[method_names[2]], method_names[1], method_names[2], all_labels)
    
    print(f"\nTotal execution time: {(time.time() - start_time) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()