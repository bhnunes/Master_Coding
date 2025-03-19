import os
from PIL import Image
import numpy as np

def create_and_save_masks(source_folder, target_folder,label):
    os.makedirs(target_folder, exist_ok=True)
    files_total=len(os.listdir(source_folder))
    i=1

    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):
            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path).convert("L")
            if "not_cancer" in filename.lower():
                #mask = np.zeros_like(img)
                mask = np.zeros_like(np.array(img), dtype=np.uint8)
            else:
                #mask = np.ones_like(img) * 255  # Assign 255 to all pixels for cancer
                mask = np.ones_like(np.array(img), dtype=np.uint8)  # Use 1 for cancer instead of 255
            mask_path = os.path.join(target_folder,filename)
            Image.fromarray(mask).save(mask_path)  # Multiply by 255 to save as visible image
            print(f'Type: {label} - Processing file {str(i)} of {str(files_total)}')
            i=i+1


# source_cancer_folder = r"D:\Usuario\Desktop\Base_de_dados\MASTER_SET\CANCER"
# source_not_cancer_folder = r"D:\Usuario\Desktop\Base_de_dados\MASTER_SET\NOT_CANCER"
# target_cancer_folder = r"D:\Usuario\Desktop\Base_de_dados\MASTER_SET\CANCER_MASK"
# target_not_cancer_folder = r"D:\Usuario\Desktop\Base_de_dados\MASTER_SET\CANCER\NOT_CANCER_MASK"

# create_and_save_masks(source_cancer_folder, target_cancer_folder, "CANCER")
# create_and_save_masks(source_not_cancer_folder, target_not_cancer_folder, "NOT_CANCER")


for k in ['VALIDATION', 'TEST', 'TRAIN']:
    #for i in range(4,11,1):
    source_cancer_folder = f'D:\\Usuario\\Desktop\\Base_de_dados\\MASTER_SET\\{k}\\CANCER'
    source_not_cancer_folder = f'D:\\Usuario\\Desktop\\Base_de_dados\\MASTER_SET\\{k}\\NOT_CANCER'
    target_cancer_folder = f'D:\\Usuario\\Desktop\\Base_de_dados\\MASTER_SET\\{k}\\CANCER_MASK'
    target_not_cancer_folder = f'D:\\Usuario\\Desktop\\Base_de_dados\\MASTER_SET\\{k}\\NOT_CANCER_MASK'
    create_and_save_masks(source_cancer_folder, target_cancer_folder, "CANCER")
    create_and_save_masks(source_not_cancer_folder, target_not_cancer_folder, "NOT_CANCER")
    print(f'Finished all {k}')
