import os
import pandas as pd
import numpy as np
import cv2
import warnings
from tqdm import tqdm

# --- Configuration ---
warnings.simplefilter(action='ignore', category=FutureWarning)

SPLITS_TO_CHECK = ["TRAIN", "VALIDATION", "TEST"]
EXPECTED_AUG_CODES = ["HF", "VF", "RR", "GB", "HED", "HSV"]
EXPECTED_MASK_VALUES = [0, 255]

# --- Analysis Functions (Unchanged as they are robust) ---

def verify_patient_leakage(manifest_df):
    """Verifies that no patient appears in more than one data split."""
    patient_splits = manifest_df.groupby('patient_id')['split'].nunique()
    leaking_patients = patient_splits[patient_splits > 1].index.tolist()
    if not leaking_patients:
        return {"status": "PASS", "details": "No patient leakage detected across splits."}
    else:
        return {"status": "FAIL", "details": f"{len(leaking_patients)} leaking patients found: {leaking_patients}"}

def analyze_file_manifest_parity(split_df, base_path, split_name):
    """Ensures manifest file counts match actual file counts on disk."""
    if split_df.empty:
        is_empty_on_disk = True
        for label in ["CANCER", "NOT_CANCER"]:
            dir_path = os.path.join(base_path, split_name, label)
            if os.path.isdir(dir_path) and os.listdir(dir_path):
                is_empty_on_disk = False
                break
        if is_empty_on_disk:
            return {"status": "PASS", "details": "Manifest and disk are both empty as expected."}
        else:
            return {"status": "FAIL", "details": "Manifest is empty but files were found on disk."}

    manifest_counts = split_df['label'].value_counts().to_dict()
    def _count_pngs(p):
        if not os.path.isdir(p): return 0
        return sum(1 for f in os.listdir(p) if f.lower().endswith('.png'))

    disk_counts = {
        0: _count_pngs(os.path.join(base_path, split_name, "NOT_CANCER")),
        1: _count_pngs(os.path.join(base_path, split_name, "CANCER"))
    }
    errors = []
    if manifest_counts.get(0, 0) != disk_counts.get(0, 0): errors.append(f"NOT_CANCER mismatch (Manifest: {manifest_counts.get(0,0)}, Disk: {disk_counts.get(0,0)})")
    if manifest_counts.get(1, 0) != disk_counts.get(1, 0): errors.append(f"CANCER mismatch (Manifest: {manifest_counts.get(1,0)}, Disk: {disk_counts.get(1,0)})")
    if not errors: return {"status": "PASS", "details": f"Manifest and disk counts match ({sum(disk_counts.values())} files)."}
    else: return {"status": "FAIL", "details": "; ".join(errors)}

def analyze_augmentation_distribution(split_df):
    """Analyzes the mix of augmentation techniques used in a split."""
    # This function needs the updated manifest from the data prep script
    if 'source' not in split_df.columns: return {"status": "N/A", "details": "Column 'source' not in manifest. Cannot check augmentations."}
    
    aug_df = split_df[split_df['source'] == 'AUG'].copy()
    if aug_df.empty: return {"status": "PASS", "details": "0 augmented files found as expected."}

    # Extract augmentation type from filename (e.g., from '_aug_HF_12345.png')
    aug_df['aug_type'] = aug_df['filename'].str.extract(r'_aug_([A-Z]{2,3})_')
    
    dist_percent = (aug_df['aug_type'].value_counts(normalize=True) * 100).round(2).to_dict()
    unexpected_types = set(dist_percent.keys()) - set(EXPECTED_AUG_CODES)
    status = "WARNING" if unexpected_types else "PASS"
    details = f"Mix: {dist_percent}" + (f" | Unexpected types: {list(unexpected_types)}" if unexpected_types else "")
    return {"status": status, "details": details}


def verify_image_mask_shapes(split_df, base_path, split_name, sample_size=5):
    """Samples image/mask pairs to verify their dimensions match."""
    if split_df.empty: return {"status": "PASS", "details": "Split is empty, nothing to check."}
    sample_df = split_df.sample(n=min(sample_size, len(split_df)), random_state=42)
    mismatched = []
    for _, row in sample_df.iterrows():
        label_dir = "CANCER" if row['label'] == 1 else "NOT_CANCER"
        img_path = os.path.join(base_path, split_name, label_dir, row['filename'])
        mask_path = os.path.join(base_path, split_name, f"{label_dir}_MASK", row['filename'])
        try:
            img, mask = cv2.imread(img_path), cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None: mismatched.append(f"{row['filename']} (unreadable)")
            elif img.shape[:2] != mask.shape: mismatched.append(f"{row['filename']} (Img:{img.shape[:2]} vs Mask:{mask.shape})")
        except Exception as e: mismatched.append(f"{row['filename']} (error: {e})")
    if not mismatched: return {"status": "PASS", "details": f"Sampled {len(sample_df)} pairs, all shapes match."}
    else: return {"status": "FAIL", "details": f"Found {len(mismatched)} mismatches: {mismatched}"}

def analyze_mask_pixel_values(base_path, split_name):
    """Verifies that all mask pixels contain only the expected values (e.g., 0, 255)."""
    all_mask_paths = []
    for label_mask_dir in ["CANCER_MASK", "NOT_CANCER_MASK"]:
        mask_dir = os.path.join(base_path, split_name, label_mask_dir)
        if os.path.isdir(mask_dir):
            all_mask_paths.extend([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith('.png')])
    if not all_mask_paths: return {"status": "PASS", "details": "No mask files found to analyze."}

    found_values = set()
    for mask_path in tqdm(all_mask_paths, desc=f"Analyzing Masks in {split_name}", leave=False, unit="mask"):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None: found_values.update(np.unique(mask))

    unexpected = found_values - set(EXPECTED_MASK_VALUES)
    if not unexpected: return {"status": "PASS", "details": f"Analyzed {len(all_mask_paths)} masks. All pixels are in {EXPECTED_MASK_VALUES}."}
    else: return {"status": "FAIL", "details": f"Found unexpected pixel values: {list(unexpected)}."}

# --- Main Orchestrator (REFACTORED for Single Split) ---
def main(base_path):
    """
    Main function to run the full QA report on a single data split directory.
    """
    LEGEND = {
        "Patient Leakage": "Verifies that no single patient's data appears in more than one split (e.g., TRAIN, VALIDATION, TEST).",
        "File Parity": "Ensures the number of image files on disk exactly matches the number of records in the manifest.csv for that split.",
        "Augmentation Dist.": "For the TRAIN split, reports the percentage mix of different augmentation techniques used (e.g., HED, HSV).",
        "Shape Check": "Samples a few image/mask pairs and confirms they have identical height and width dimensions.",
        "Mask Pixel Values": "Scans all mask images in a split to ensure they only contain the expected pixel values (e.g., 0 and 255).",
    }

    print("="*80)
    print("      SCIENTIFIC INTEGRITY AND DATA QUALITY ASSURANCE REPORT")
    print("="*80)
    print(f"Scanning base directory: {base_path}\n")

    print("--- Checks Explained ---")
    for check, explanation in LEGEND.items():
        print(f"  - {check+':':<22} {explanation}")
    print("-" * 80)

    manifest_path = os.path.join(base_path, "manifest.csv")
    if not os.path.isfile(manifest_path):
        print(f"❌ ERROR: 'manifest.csv' not found in '{base_path}'. Please check the path.")
        return

    all_results = []
    manifest_df = pd.read_csv(manifest_path)
    
    # --- Perform data-set wide check (Patient Leakage) ---
    leakage_result = verify_patient_leakage(manifest_df)
    
    # --- Loop through splits to perform split-level checks ---
    for split_name in tqdm(SPLITS_TO_CHECK, desc="Processing Splits"):
        split_df = manifest_df[manifest_df['split'] == split_name]
        
        if split_name == "TRAIN":
             aug_res = analyze_augmentation_distribution(split_df)
        else:
             aug_res = {"status": "N/A", "details": "Check is only applicable to TRAIN split."}

        all_results.append({
            "Split": split_name,
            "File Parity": (res_parity := analyze_file_manifest_parity(split_df, base_path, split_name))['status'],
            "Augmentation Dist.": aug_res['status'],
            "Shape Check": (res_shape := verify_image_mask_shapes(split_df, base_path, split_name))['status'],
            "Mask Pixel Values": (res_mask := analyze_mask_pixel_values(base_path, split_name))['status'],
            "Details": {
                "File Parity": res_parity['details'],
                "Augmentation Dist.": aug_res['details'], 
                "Shape Check": res_shape['details'],
                "Mask Pixel Values": res_mask['details'],
            }
        })

    # --- Generate and Print Report ---
    report_df = pd.DataFrame(all_results).set_index("Split")
    details_series = report_df.pop("Details")
    
    print("\n\n" + "="*80)
    print("                         SUMMARY REPORT TABLE")
    print("="*80)
    print(report_df.to_string())
    
    print("\n\n" + "="*80)
    print("                           DETAILED FINDINGS")
    print("="*80)
    
    # Print the single data-set wide check first
    print(f"--- Data Set Integrity ---")
    print(f"  - {'Patient Leakage:':<22} [{leakage_result['status']}] {leakage_result['details']}")

    # Then iterate through each split for its specific details
    for split_name, row in report_df.iterrows():
        print(f"\n--- Split: {split_name} ---")
        details_dict = details_series.loc[split_name]
        for check_name, status in row.items():
            detail_text = details_dict.get(check_name, "Detail not found.")
            print(f"  - {check_name+':':<22} [{status}] {detail_text}")

if __name__ == '__main__':
    # IMPORTANT: Change this path to the output of your simplified data prep script.
    # e.g., r'D:\Usuario\Desktop\Base_de_dados\ABLATION\NORMALIZED_SPLITS\REINHARD_seed_42'
    MAIN_OUTPUT_DIRECTION = r'D:\Usuario\Desktop\Base_de_dados\ABLATION\NORMALIZED_SPLITS\REINHARD_seed_42'
    main_output_directory = os.path.normpath(MAIN_OUTPUT_DIRECTION)
    main(main_output_directory)