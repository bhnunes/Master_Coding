import os
import cv2  # OpenCV for image reading
import numpy as np
from collections import defaultdict
import time # To measure execution time

def analyze_mask_pixels(folder_paths, expected_values=[0, 1, 2]):
    """
    Analyzes pixel value distribution in image masks across multiple folders.

    It specifically counts occurrences of values in `expected_values` and groups
    all other pixel values found into an 'other' category.

    Args:
        folder_paths (list): A list of paths to the folders containing mask images.
        expected_values (list): A list of integer pixel values to specifically count (e.g., [0, 1, 2]).

    Returns:
        tuple: A tuple containing:
            - dict: Counts for each expected value and 'other'.
            - int: Total number of pixels analyzed across all valid images.
            - dict: Percentages for each expected value and 'other'.
                   Returns None for percentages if total_pixels is 0.
            - int: Number of image files processed.
            - int: Number of files skipped due to errors or format.
    """
    # Use defaultdict for convenient counting
    pixel_counts = defaultdict(int)
    total_pixels = 0
    files_processed = 0
    files_skipped = 0

    # Define common image extensions (add more if needed)
    supported_extensions = ('.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg', '.gif')

    print("Starting analysis...")
    start_time = time.time()

    # Create a set from expected_values for faster lookup
    expected_values_set = set(expected_values)

    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found or is not a directory: {folder_path}. Skipping.")
            continue

        print(f"\nProcessing folder: {folder_path}")
        try:
            filenames = os.listdir(folder_path)
            print(f"Found {len(filenames)} items in the folder.")
            folder_files_processed = 0
            folder_files_skipped = 0

            for filename in filenames:
                file_path = os.path.join(folder_path, filename)

                # Check if it's a file and has a supported extension
                if os.path.isfile(file_path) and filename.lower().endswith(supported_extensions):
                    try:
                        # Read image in grayscale mode - essential for masks
                        # Grayscale ensures we get a 2D array with single intensity values
                        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                        if mask is None:
                            print(f"  - Warning: Could not read image file: {filename}. Skipping.")
                            files_skipped += 1
                            folder_files_skipped += 1
                            continue

                        # --- Pixel Counting ---
                        # Method 1: Using np.unique (generally efficient)
                        unique_vals, counts = np.unique(mask, return_counts=True)
                        current_file_pixels = mask.size # Total pixels in this mask
                        total_pixels += current_file_pixels

                        # Add counts to the global counter dictionary
                        for value, count in zip(unique_vals, counts):
                            if value in expected_values_set:
                                pixel_counts[value] += count
                            else:
                                pixel_counts['other'] += count # Count unexpected values separately

                        # # Method 2: Direct comparison (might be faster if only few values expected)
                        # current_file_pixels = mask.size
                        # total_pixels += current_file_pixels
                        # for val in expected_values:
                        #     pixel_counts[val] += np.sum(mask == val)
                        # # This method requires an extra step to find 'other' if needed

                        files_processed += 1
                        folder_files_processed +=1

                    except Exception as e:
                        print(f"  - Error processing file {filename}: {e}")
                        files_skipped += 1
                        folder_files_skipped += 1
                elif os.path.isfile(file_path):
                    print(f"  - Skipping non-image file: {filename}") # Optional: uncomment for more verbose output
                    files_skipped += 1
                    folder_files_skipped += 1
                # else: it's a sub-directory, ignore it

            print(f"Finished processing folder '{os.path.basename(folder_path)}': {folder_files_processed} images processed, {folder_files_skipped} items skipped.")

        except Exception as e:
             print(f"Error listing files in folder {folder_path}: {e}")


    end_time = time.time()
    print(f"\nAnalysis Complete. Time taken: {end_time - start_time:.2f} seconds.")
    print(f"Total images processed successfully: {files_processed}")
    print(f"Total files/items skipped: {files_skipped}")
    print(f"Total pixels analyzed: {total_pixels}")

    percentages = None
    if total_pixels > 0:
        percentages = {}
        # Ensure all expected values and 'other' are in the percentage dict
        all_keys_to_calculate = list(expected_values) + ['other']
        for value in all_keys_to_calculate:
             count = pixel_counts.get(value, 0) # Use .get() in case a value never appeared
             percentages[value] = (count / total_pixels) * 100
    else:
        print("No pixels were analyzed (maybe no valid images found?).")

    # Ensure all expected values are present in the final counts dictionary, even if count is 0
    final_counts = {val: pixel_counts.get(val, 0) for val in expected_values}
    final_counts['other'] = pixel_counts.get('other', 0) # Add 'other' count

    return final_counts, total_pixels, percentages, files_processed, files_skipped

# --- Configuration ---
# <<< --- IMPORTANT: CHANGE THESE PATHS --- >>>
folder1_mask_path = r'D:\Usuario\Desktop\Base_de_dados\FINAL_SPLIT_DATA\TRAIN\CANCER_MASK'
folder2_mask_path = r'D:\Usuario\Desktop\Base_de_dados\FINAL_SPLIT_DATA\TRAIN\NOT_CANCER_MASK'
# <<< --- END OF PATHS TO CHANGE --- >>>

# Define the pixel values you expect and want to count specifically
# Based on your request for 0, 1, and 2 percentages
pixel_values_to_count = [0, 1, 2]

# --- Execution ---
all_folders = [folder1_mask_path, folder2_mask_path]
counts, total_px, percentages, processed_count, skipped_count = analyze_mask_pixels(
    all_folders,
    pixel_values_to_count
)

# --- Print Results ---
print("\n--- Final Results Summary ---")
if total_px > 0 and percentages is not None:
    print("\nPixel Counts:")
    for value in pixel_values_to_count:
        print(f"  Value {value}: {counts.get(value, 0):,}") # Formatted count
    print(f"  Other Values: {counts.get('other', 0):,}")

    print("\nPixel Percentages:")
    for value in pixel_values_to_count:
         print(f"  Value {value}: {percentages.get(value, 0):.4f}%")
    print(f"  Other Values: {percentages.get('other', 0):.4f}%")

    # Explicitly print the percentages you asked for:
    print("\nRequested Percentages:")
    print(f"  Percentage of 0s: {percentages.get(0, 0):.4f}%")
    print(f"  Percentage of 1s: {percentages.get(1, 0):.4f}%")
    print(f"  Percentage of 2s: {percentages.get(2, 0):.4f}%") # Added based on your text

    # Sanity check: percentages should sum close to 100%
    total_percentage = sum(percentages.values())
    print(f"\nTotal Percentage Sum: {total_percentage:.4f}% (should be close to 100%)")

    if counts.get('other', 0) > 0:
        print("\nNote: 'Other Values' were detected in your masks.")
        print("This means some pixels had values other than", pixel_values_to_count)

elif processed_count == 0:
     print("\nNo valid image files were found or processed in the specified folders.")
else:
     print("\nAn issue occurred, or no pixels were found in the processed images.")