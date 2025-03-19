import numpy as np
import cv2
import os
from PIL import Image
from joblib import Parallel, delayed

TEMPLATE = r"D:\Usuario\Desktop\ProjetoMestrado\Extractor_Pipeline\Slice_Template\CANCER_PATIENT_00083_105_06_08_2024_17_14_27.png"
INPUT = r"D:\Usuario\Desktop\Base_de_dados\MASTER\NOT_CANCER"
OUTPUT = r"D:\Usuario\Desktop\Base_de_dados\CORRECTED_IMAGES\NOT_CANCER"

# Load the template image only once
template_img = cv2.imread(TEMPLATE)
template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB)
template_mean, template_std = cv2.meanStdDev(template_img)
template_mean = np.hstack(np.around(template_mean, 2))
template_std = np.hstack(np.around(template_std, 2))

def process_image(img, input_dir, output_dir):
	input_img = Image.open(os.path.join(input_dir, img))
	input_img = np.array(input_img)
	input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)

	img_mean, img_std = cv2.meanStdDev(input_img)
	img_mean = np.hstack(np.around(img_mean, 2))
	img_std = np.hstack(np.around(img_std, 2))

	# Vectorized Reinhard color normalization
	input_img = ((input_img - img_mean) * (template_std / img_std)) + template_mean
	input_img = np.clip(input_img, 0, 255).astype(np.uint8)

	input_img = cv2.cvtColor(input_img, cv2.COLOR_LAB2BGR)
	cv2.imwrite(os.path.join(output_dir, f"adj_{img}"), input_img)

if __name__ == "__main__":
    input_image_list = os.listdir(INPUT)
    j=1
    batch_size = 500 # Adjust batch size based on available memory
    total=len(input_image_list)//batch_size
    remainer=len(input_image_list)%batch_size
    if remainer>0:
          total=total+1

    # Process images in batches
    for i in range(0, len(input_image_list), batch_size):
        image_batch = input_image_list[i:i + batch_size]
        
        # You can still parallelize within a batch if needed:
        Parallel(n_jobs=-1)(
            delayed(process_image)(img, INPUT, OUTPUT) for img in image_batch 
        ) 
        print("Processed batch "+str(j)+" Of "+str(total))
        j=j+1
