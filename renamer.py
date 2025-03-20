import os
import shutil
import concurrent.futures
import time

def rename_file(filepath):
    """Renames a single file, removing 'adj_' if present."""
    try:
        dirname, filename = os.path.split(filepath)
        if filename.startswith("adj_"):
            new_filename = filename[4:]  # Remove 'adj_'
            new_filepath = os.path.join(dirname, new_filename)
            shutil.move(filepath, new_filepath)
            return True, None  # Success
        return False, None  # File not renamed (not an error)
    except OSError as e:
        return False, str(e)  # Return error message
    except Exception as e:
        return False, str(e)

def rename_files_in_parallel(directory):
    """Renames files in parallel, removing 'adj_' from PNG filenames."""

    start_time = time.time()
    files_to_rename = []
    renamed_count = 0
    skipped_count = 0
    error_count = 0
    errors = []

    print(f"Scanning files in {directory}...")

    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.startswith("adj_") and entry.name.endswith(".png"):
            files_to_rename.append(entry.path)

    print(f"Found {len(files_to_rename)} files to rename.")

    # Automatically determine the number of worker threads (number of CPU cores)
    max_workers = os.cpu_count()
    print(f"Using {max_workers} worker threads.")


    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(rename_file, filepath) for filepath in files_to_rename]

        for future in concurrent.futures.as_completed(futures):
            success, error_message = future.result()
            if success:
                renamed_count += 1
            elif error_message:
                error_count += 1
                errors.append(error_message)
            else:
                skipped_count += 1

    end_time = time.time()
    print(f"\nRenamed {renamed_count} files.")
    print(f"Skipped {skipped_count} files.")
    print(f"Encountered {error_count} errors.")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(error)
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    directory = r'D:\Usuario\Desktop\Base_de_dados\MASTER_ADJUSTED\NOT_CANCER'  # Change this to the directory containing the files
    # Basic validation: check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
    else:
        rename_files_in_parallel(directory)