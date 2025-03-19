import numpy as np
import cv2
import os
import random  # Import the random module
from PIL import Image
from joblib import Parallel, delayed

TEMPLATE_DIR = r"D:\Usuario\Desktop\ProjetoMestrado\Extractor_Pipeline\template_images"  # Directory for templates
INPUT = r"D:\Usuario\Desktop\Base_de_dados\MASTER\NOT_CANCER"
OUTPUT = r"D:\Usuario\Desktop\Base_de_dados\CORRECTED_IMAGES\NOT_CANCER"

def load_template():
    """Randomly selects and loads a template image."""
    template_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]  # Handle different image types
    if not template_files:
        raise ValueError("No template images found in the specified directory.")
    template_path = os.path.join(TEMPLATE_DIR, random.choice(template_files))
    template_img = cv2.imread(template_path)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB)
    template_mean, template_std = cv2.meanStdDev(template_img)
    return np.hstack(np.around(template_mean, 2)), np.hstack(np.around(template_std, 2))

def process_image(img, input_dir, output_dir, template_mean, template_std):  # Add template_mean and template_std as arguments
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
    j = 1
    batch_size = 500  # Adjust batch size

    total = len(input_image_list) // batch_size
    remainder = len(input_image_list) % batch_size
    if remainder > 0:
        total += 1

    # Process images in batches
    for i in range(0, len(input_image_list), batch_size):
        image_batch = input_image_list[i:i + batch_size]

        # Load a random template image *for each batch*
        template_mean, template_std = load_template()

        # Pass the template mean and std to process_image
        Parallel(n_jobs=-1)(
            delayed(process_image)(img, INPUT, OUTPUT, template_mean, template_std) for img in image_batch
        )
        print("Processed batch " + str(j) + " Of " + str(total))
        j = j + 1
