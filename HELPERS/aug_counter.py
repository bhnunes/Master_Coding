import os
import sys # To check if path is provided and exit gracefully

def analyze_filenames_for_aug(folder_path, suffix="_aug"):
    """
    Counts files with and without a specific suffix in their name within a folder.

    Args:
        folder_path (str): The path to the directory to scan.
        suffix (str): The substring to look for in filenames (defaults to "_aug").

    Returns:
        tuple: A tuple containing:
            - int: Count of files containing the suffix.
            - int: Count of files NOT containing the suffix.
            - int: Total number of files scanned.
            - float: Percentage of files containing the suffix (or 0.0 if no files).
            - float: Percentage of files NOT containing the suffix (or 0.0 if no files).
        Returns None if the folder_path is invalid or inaccessible.
    """
    aug_count = 0
    non_aug_count = 0
    total_files = 0

    print(f"Scanning folder: {folder_path}")
    print(f"Looking for suffix: '{suffix}'")

    # 1. Check if the provided path is a valid directory
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found or is not a directory: {folder_path}")
        return None # Indicate failure

    try:
        # 2. List all items (files and subdirectories) in the folder
        items_in_folder = os.listdir(folder_path)
        print(f"Found {len(items_in_folder)} total items in the folder.")

        # 3. Iterate through items and count files based on the suffix
        for item_name in items_in_folder:
            item_path = os.path.join(folder_path, item_name)

            # IMPORTANT: Only process items that are actual files
            if os.path.isfile(item_path):
                total_files += 1 # Increment total file count
                if suffix in item_name:
                    aug_count += 1
                else:
                    non_aug_count += 1
            # Optional: uncomment to see skipped items
            # else:
            #     print(f"  - Skipping (not a file): {item_name}")


    except OSError as e:
        print(f"Error accessing folder contents: {e}")
        return None # Indicate failure
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None # Indicate failure


    # 4. Calculate percentages (handle division by zero if no files found)
    if total_files > 0:
        percentage_aug = (aug_count / total_files) * 100
        percentage_non_aug = (non_aug_count / total_files) * 100
    else:
        percentage_aug = 0.0
        percentage_non_aug = 0.0
        print("Warning: No files found in the specified directory.")

    print("\nAnalysis Complete.")
    return aug_count, non_aug_count, total_files, percentage_aug, percentage_non_aug

# --- Configuration ---
# <<< --- IMPORTANT: CHANGE THIS PATH --- >>>
target_folder = r'D:\Usuario\Desktop\Base_de_dados\cross_val_splits_balanced\fold_5\TRAIN\CANCER'
# <<< --- END OF PATH TO CHANGE --- >>>

# Suffix to search for (you requested "_aug")
search_suffix = "_aug"

# --- Execution ---
results = analyze_filenames_for_aug(target_folder, search_suffix)

# --- Print Results ---
if results:
    aug_files, non_aug_files, total, percent_aug, percent_non_aug = results

    print("\n--- Results ---")
    print(f"Folder Scanned:       {target_folder}")
    print(f"Suffix Searched For:  '{search_suffix}'")
    print("-" * 30)
    print(f"Total Files Found:    {total:,}") # Add comma for readability
    print(f"Files WITH suffix:    {aug_files:,}")
    print(f"Files WITHOUT suffix: {non_aug_files:,}")
    print("-" * 30)
    print(f"Percentage WITH suffix:    {percent_aug:.2f}%")
    print(f"Percentage WITHOUT suffix: {percent_non_aug:.2f}%")
    print("-" * 30)
    # Sanity check
    if total > 0:
      print(f"Check Sum (should be 100%): {percent_aug + percent_non_aug:.2f}%")
else:
    print("\nScript finished with errors or no folder processed.")