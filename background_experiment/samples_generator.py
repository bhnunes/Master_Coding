import os
import random
import shutil
import math

def calculate_cochran_sample_size(confidence_level=0.95, margin_of_error=0.05, proportion=0.5):
    """
    Calculates the sample size using Cochran's formula for a large population.
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    if confidence_level not in z_scores:
        raise ValueError("Confidence level must be one of 0.90, 0.95, or 0.99.")
    z = z_scores[confidence_level]
    p = proportion
    e = margin_of_error
    n = (z**2 * p * (1 - p)) / (e**2)
    return math.ceil(n)

def create_image_samples(source_folder):
    """
    Analyzes a folder of images, calculates the required sample size, and creates
    a pilot sample folder and a master candidate pool folder.
    """
    if not os.path.isdir(source_folder):
        print(f"Error: The provided path '{source_folder}' is not a valid directory.")
        return

    print(f"Scanning source folder: {source_folder}")
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'}
    try:
        all_files = [f for f in os.listdir(source_folder) if os.path.splitext(f)[1].lower() in image_extensions]
    except OSError as e:
        print(f"Error reading directory: {e}")
        return

    total_images = len(all_files)
    if total_images == 0:
        print("Error: No image files found in the source directory.")
        return

    print(f"\nFound {total_images} total images.")

    required_sample_size = calculate_cochran_sample_size()
    pilot_sample_size = 100
    master_pool_size = math.ceil(total_images * 0.10)

    print("\n--- Experiment Parameters ---")
    print(f"Statistically Required Sample Size (95% confidence, 5% error): {required_sample_size}")
    print(f"Pilot Sample Size (for estimating proportions): {pilot_sample_size}")
    print(f"Master Candidate Pool Size (10% of total): {master_pool_size}")
    
    if total_images < (pilot_sample_size + master_pool_size):
        print("\nError: Not enough images in the source folder to create the required non-overlapping samples.")
        return

    pilot_folder_path = os.path.join(source_folder, 'pilot_sample')
    master_pool_path = os.path.join(source_folder, 'master_candidate_pool')

    print(f"\nCreating directory: {pilot_folder_path}")
    os.makedirs(pilot_folder_path, exist_ok=True)
    
    print(f"Creating directory: {master_pool_path}")
    os.makedirs(master_pool_path, exist_ok=True)

    random.shuffle(all_files)
    master_pool_files = all_files[:master_pool_size]
    pilot_sample_files = all_files[master_pool_size : master_pool_size + pilot_sample_size]
    
    print(f"\nCopying {len(pilot_sample_files)} images to '{pilot_folder_path}'...")
    for filename in pilot_sample_files:
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(pilot_folder_path, filename)
        shutil.copy2(src_path, dst_path)
    print("Pilot sample creation complete.")

    print(f"\nCopying {len(master_pool_files)} images to '{master_pool_path}'...")
    for filename in master_pool_files:
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(master_pool_path, filename)
        shutil.copy2(src_path, dst_path)
    print("Master candidate pool creation complete.")

    print("\n--- Setup Complete! ---")
    print("\nYour next steps are:")
    print("1. Go to the 'pilot_sample' folder and manually label the 100 images as 'Approved' or 'Rejected'.")
    print("2. Calculate the proportion of each class (e.g., 70% Approved, 30% Rejected).")
    print(f"3. Use this proportion to determine your final target numbers from the {required_sample_size} total samples needed.")
    print(f"   - Example: 0.70 * {required_sample_size} = {math.ceil(0.7 * required_sample_size)} 'Approved' images.")
    print(f"   - Example: 0.30 * {required_sample_size} = {math.ceil(0.3 * required_sample_size)} 'Rejected' images.")
    print("4. Go to the 'master_candidate_pool' folder and randomly select/label images until you reach your target numbers for both classes.")
    print("5. This final collection of labeled images will be your statistically representative dataset for the experiment.")

if __name__ == "__main__":        
    folder_path =r"C:\Images_IA_MEDICA\CANCER"
    create_image_samples(folder_path)