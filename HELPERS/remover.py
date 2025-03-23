import os
import shutil  # Import shutil
from tqdm import tqdm

def delete_original_files(root_dir):
    """
    Deletes files containing 'original' in their names within a directory tree.

    Args:
        root_dir: The root directory to scan.
    """
    files_to_delete = []

    # Efficiently walk the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if 'original' in filename.lower():  # Case-insensitive check
                files_to_delete.append(os.path.join(dirpath, filename))

    # Use tqdm for progress display during deletion
    with tqdm(total=len(files_to_delete), desc="Deleting 'original' files") as pbar:
        for file_path in files_to_delete:
            try:
                os.remove(file_path)  # Delete the file
                # OR, for safer operation, move to a "deleted" folder:
                # deleted_folder = os.path.join(root_dir, "deleted_files")
                # os.makedirs(deleted_folder, exist_ok=True)
                # shutil.move(file_path, os.path.join(deleted_folder, os.path.basename(file_path)))
                pbar.update(1)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")


if __name__ == '__main__':
    root_directory = r'D:\Usuario\Desktop\Base_de_dados\cross_val_splits'  # Replace with your directory
    delete_original_files(root_directory)
    print("Deletion process completed.")