import os
import sys
from collections import defaultdict # Useful for counting types

def analyze_filenames_for_detailed_aug(folder_path, expected_aug_types, aug_marker="_aug"):
    """
    Counts files with specific augmentation types indicated in their filenames.

    Assumes filenames like '...<aug_marker>_<TYPE>_...'

    Args:
        folder_path (str): The path to the directory to scan.
        expected_aug_types (list): A list of strings representing the known
                                   augmentation type codes (e.g., ["HP", "VF"]).
        aug_marker (str): The substring indicating an augmented file (defaults to "_aug").

    Returns:
        dict: A dictionary containing detailed counts and percentages, or None if errors occur.
              Keys include:
              'total_files', 'non_aug_count', 'total_aug_count',
              'aug_counts_by_type' (dict), 'unknown_aug_count',
              'percent_non_aug_vs_total', 'percent_total_aug_vs_total',
              'percent_types_vs_total' (dict), 'percent_unknown_vs_total',
              'percent_types_vs_augmented' (dict), 'percent_unknown_vs_augmented'
        Returns None if the folder_path is invalid or inaccessible.
    """
    non_aug_count = 0
    # Use defaultdict to easily count known and unknown types
    aug_counts_by_type = defaultdict(int)
    unknown_aug_count = 0 # Count files marked _aug but with unknown/missing type
    total_files = 0
    processed_files = 0 # Files successfully analyzed
    skipped_items = 0 # Non-files or files causing errors

    # Prepare the marker without leading/trailing underscores for reliable splitting
    split_marker = aug_marker.strip('_')
    expected_types_set = set(expected_aug_types) # Faster lookup

    print(f"Scanning folder: {folder_path}")
    print(f"Looking for augmentation marker: '{aug_marker}'")
    print(f"Expected augmentation types: {', '.join(expected_aug_types)}")

    # 1. Check if the provided path is a valid directory
    if not os.path.isdir(folder_path):
        print(f"❌ Error: Folder not found or is not a directory: {folder_path}")
        return None

    try:
        # 2. List all items (files and subdirectories) in the folder
        items_in_folder = os.listdir(folder_path)
        print(f"Found {len(items_in_folder)} total items in the folder.")

        # 3. Iterate through items and classify them
        for item_name in items_in_folder:
            item_path = os.path.join(folder_path, item_name)

            if os.path.isfile(item_path):
                total_files += 1
                try:
                    # Check if the file is augmented
                    if aug_marker in item_name:
                        parts = item_name.split('_')
                        try:
                            # Find the index of the augmentation marker part
                            aug_index = parts.index(split_marker)
                            # Check if there's a part *after* the marker
                            if aug_index + 1 < len(parts):
                                aug_type = parts[aug_index + 1]
                                # Check if it's one of the expected types
                                if aug_type in expected_types_set:
                                    aug_counts_by_type[aug_type] += 1
                                else:
                                    # It's augmented, but type is not in the list
                                    print(f"  - Warning: Found augmented file with unexpected type '{aug_type}': {item_name}")
                                    unknown_aug_count += 1
                            else:
                                # Marker is present but nothing follows it
                                print(f"  - Warning: Found augmented file marker '{aug_marker}' but no type follows: {item_name}")
                                unknown_aug_count += 1
                        except ValueError:
                            # This should theoretically not happen if aug_marker was found by 'in'
                            # but good to handle just in case of weird filenames
                            print(f"  - Warning: Marker '{aug_marker}' found but split part '{split_marker}' not located? File: {item_name}")
                            unknown_aug_count += 1 # Treat as unknown augmentation

                    else:
                        # File does not contain the augmentation marker
                        non_aug_count += 1

                    processed_files += 1

                except Exception as e_file:
                    print(f"  - Error processing file '{item_name}': {e_file}")
                    skipped_items += 1 # Count as skipped due to error during analysis
            else:
                # It's a directory or other non-file item
                skipped_items += 1
                # Optional: uncomment to see skipped items
                # print(f"  - Skipping (not a file): {item_name}")

    except OSError as e_os:
        print(f"❌ Error accessing folder contents: {e_os}")
        return None
    except Exception as e_main:
        print(f"❌ An unexpected error occurred during scanning: {e_main}")
        return None

    # 4. Calculate derived counts and percentages
    total_aug_count = sum(aug_counts_by_type.values()) + unknown_aug_count

    # Sanity check
    if total_files != (non_aug_count + total_aug_count):
         print(f"⚠️ Warning: File count mismatch! Total={total_files}, NonAug={non_aug_count}, TotalAug={total_aug_count}. Check warnings.")
         # Adjust total_files based on actual classified counts if mismatch occurs
         # This might happen if a file error occurred *after* total_files was incremented
         # but before classification completed. A safer approach might be to only increment
         # total_files *after* successful classification, but the current way counts all discoverable files.
         # Let's recalculate total_files based on processed counts for percentage safety:
         total_files = non_aug_count + total_aug_count # Recalculate based on classified files

    results = {
        'folder_path': folder_path,
        'aug_marker': aug_marker,
        'expected_aug_types': expected_aug_types,
        'total_items_in_dir': len(items_in_folder) if 'items_in_folder' in locals() else 'N/A',
        'total_files_analyzed': total_files, # Use recalculated total
        'processed_files': processed_files,
        'skipped_items': skipped_items,
        'non_aug_count': non_aug_count,
        'total_aug_count': total_aug_count,
        'aug_counts_by_type': dict(aug_counts_by_type), # Convert defaultdict to dict for output
        'unknown_aug_count': unknown_aug_count,
        'percent_non_aug_vs_total': 0.0,
        'percent_total_aug_vs_total': 0.0,
        'percent_types_vs_total': {t: 0.0 for t in expected_aug_types},
        'percent_unknown_vs_total': 0.0,
        'percent_types_vs_augmented': {t: 0.0 for t in expected_aug_types},
        'percent_unknown_vs_augmented': 0.0,
    }

    if total_files > 0:
        results['percent_non_aug_vs_total'] = (non_aug_count / total_files) * 100
        results['percent_total_aug_vs_total'] = (total_aug_count / total_files) * 100
        results['percent_unknown_vs_total'] = (unknown_aug_count / total_files) * 100
        for aug_type, count in aug_counts_by_type.items():
            results['percent_types_vs_total'][aug_type] = (count / total_files) * 100

        if total_aug_count > 0:
            results['percent_unknown_vs_augmented'] = (unknown_aug_count / total_aug_count) * 100
            for aug_type, count in aug_counts_by_type.items():
                results['percent_types_vs_augmented'][aug_type] = (count / total_aug_count) * 100
        else:
             print("ℹ️ No augmented files found, percentages relative to augmented count are 0.")
    else:
        print("⚠️ Warning: No files found or processed in the specified directory.")


    print("\nAnalysis Complete.")
    return results

# --- Configuration ---
# <<< --- IMPORTANT: CHANGE THIS PATH --- >>>
target_folder = r'D:\Usuario\Desktop\Base_de_dados\NOT_NORMALIZED\cross_val_splits_balanced_geometric_aug\fold_2\TRAIN\NOT_CANCER'
# <<< --- END OF PATH TO CHANGE --- >>>

# Define the exact strings that identify augmentation types after "_aug_"
# Based on your example: HP, VF, RF, GB, CJ
known_augmentation_types = ["HP", "VF", "RF", "GB", "CJ"]

# Define the marker that precedes the augmentation type
augmentation_marker = "_aug"

# --- Execution ---
analysis_results = analyze_filenames_for_detailed_aug(
    target_folder,
    known_augmentation_types,
    augmentation_marker
)

# --- Print Results ---
if analysis_results:
    print("\n" + "="*40)
    print("       Filename Augmentation Analysis Results")
    print("="*40)
    print(f"Folder Scanned:         {analysis_results['folder_path']}")
    print(f"Total Items in Dir:     {analysis_results['total_items_in_dir']}")
    print(f"Total Files Analyzed:   {analysis_results['total_files_analyzed']:,}")
    print(f"Skipped/Non-File Items: {analysis_results['skipped_items']:,}")
    print("-" * 40)
    print(f"Non-Augmented Files:    {analysis_results['non_aug_count']:,} ({analysis_results['percent_non_aug_vs_total']:.2f}% of total)")
    print(f"Total Augmented Files:  {analysis_results['total_aug_count']:,} ({analysis_results['percent_total_aug_vs_total']:.2f}% of total)")
    print("-" * 40)

    print("Augmentation Breakdown (Counts):")
    for aug_type in analysis_results['expected_aug_types']:
        count = analysis_results['aug_counts_by_type'].get(aug_type, 0)
        print(f"  - Type '{aug_type}': {count:,}")
    if analysis_results['unknown_aug_count'] > 0:
        print(f"  - Unknown/Malformed: {analysis_results['unknown_aug_count']:,}")

    print("\nAugmentation Breakdown (% of TOTAL Files):")
    total_percent_vs_total = 0
    for aug_type in analysis_results['expected_aug_types']:
        percent = analysis_results['percent_types_vs_total'].get(aug_type, 0.0)
        print(f"  - Type '{aug_type}': {percent:.2f}%")
        total_percent_vs_total += percent
    if analysis_results['unknown_aug_count'] > 0:
         percent_unknown = analysis_results['percent_unknown_vs_total']
         print(f"  - Unknown/Malformed: {percent_unknown:.2f}%")
         total_percent_vs_total += percent_unknown
    print(f"  ---------------------------")
    print(f"  Sum of Aug Types (% Total): {total_percent_vs_total:.2f}% (should match Total Augmented % above)")


    if analysis_results['total_aug_count'] > 0:
        print("\nAugmentation Breakdown (% of AUGMENTED Files):")
        total_percent_vs_aug = 0
        for aug_type in analysis_results['expected_aug_types']:
            percent = analysis_results['percent_types_vs_augmented'].get(aug_type, 0.0)
            print(f"  - Type '{aug_type}': {percent:.2f}%")
            total_percent_vs_aug += percent
        if analysis_results['unknown_aug_count'] > 0:
            percent_unknown = analysis_results['percent_unknown_vs_augmented']
            print(f"  - Unknown/Malformed: {percent_unknown:.2f}%")
            total_percent_vs_aug += percent_unknown
        print(f"  ---------------------------")
        print(f"  Sum of Aug Types (% Augmented): {total_percent_vs_aug:.2f}% (should be 100%)")
    else:
        print("\nNo augmented files found, cannot calculate percentages relative to augmented total.")

    print("="*40)

else:
    print("\nScript finished with errors or no folder processed.")