import os
import shutil
from tqdm import tqdm

origin = "D:\\Usuario\\Desktop\\Base_de_dados\\CAMELYON16\\PATCHES\\CANCER"  # <-- change this
origin_mask = "D:\\Usuario\\Desktop\\Base_de_dados\\CAMELYON16\\PATCHES\\CANCER_MASK"  # <-- change this

destination = "D:\\Usuario\\Desktop\\Base_de_dados\\CAMELYON16\\PATCHES_TRAIN\\CANCER"  # <-- change this
destination_mask = "D:\\Usuario\\Desktop\\Base_de_dados\\CAMELYON16\\PATCHES_TRAIN\\CANCER_MASK"  # <-- change this

target_patient = ['100076',
'100117',
'100079',
'100073',
'100078',
'100119',
'100069',
'100115',
'100065',
'100116',
'100074',
'100112']

for filename in tqdm(os.listdir(origin), desc="Processing files"):
    if any(f"PATIENT_{patient}_" in filename for patient in target_patient):
        file_path = os.path.join(origin, filename)
        mask_path = os.path.join(origin_mask, filename)

        try:
            # Move files to destination folder
            destination_path = os.path.join(destination, filename)
            destination_mask_path = os.path.join(destination_mask, filename)

            shutil.move(file_path, destination_path)
            shutil.move(mask_path, destination_mask_path)

        except Exception as e:
            tqdm.write(
                f"Error moving {file_path} and {mask_path} to "
                f"{destination_path} and {destination_mask_path}: {e}"
            )
