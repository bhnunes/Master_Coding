import numpy as np
import cv2
import os
import random
from PIL import Image, UnidentifiedImageError  # Import UnidentifiedImageError
from joblib import Parallel, delayed
import logging  # Import the logging module


TEMPLATE_DIR = r"D:\Usuario\Desktop\Base_de_dados\MASTER_ADJUSTED\Slice_Template"  # Directory for templates
INPUT = r"D:\Usuario\Desktop\Base_de_dados\MASTER\CANCER"
OUTPUT = r"D:\Usuario\Desktop\Base_de_dados\MASTER_ADJUSTED\CANCER"
LOG_FILE = os.path.join(OUTPUT, "D:\\Usuario\\Desktop\\Base_de_dados\\MASTER_ADJUSTED\\error_log.txt")  # Define the log file path


# --- Configure logging ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,  # Log only errors and above
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",  # Overwrite log file each run.  Use 'a' to append.
)


def load_template():
    """Randomly selects and loads a template image."""
    template_files = [
        f
        for f in os.listdir(TEMPLATE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]  # Handle different image types, case-insensitively
    if not template_files:
        raise ValueError("No template images found in the specified directory.")
    template_path = os.path.join(TEMPLATE_DIR, random.choice(template_files))
    template_img = cv2.imread(template_path)
    if template_img is None:  # check if image is read correctly
        raise ValueError(f"Could not read template image at {template_path}")
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2LAB)
    template_mean, template_std = cv2.meanStdDev(template_img)
    return np.hstack(np.around(template_mean, 2)), np.hstack(
        np.around(template_std, 2)
    )


def process_image(img, input_dir, output_dir, template_mean, template_std):  # Add template_mean and template_std as arguments
    input_path = os.path.join(input_dir, img)
    output_path = os.path.join(output_dir, f"adj_{img}")
    try:
        # check if the current file is really an image
        input_img = Image.open(input_path)
        input_img = np.array(input_img)

        # Check if conversion to numpy array was successful (handles some edge cases)
        if input_img.size == 0:
            print(f"Warning: Skipping empty image: {img}")
            logging.error(f"Skipping empty image: {input_path}")  # Log the error
            return

        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR *before* LAB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)

        img_mean, img_std = cv2.meanStdDev(input_img)
        img_mean = np.hstack(np.around(img_mean, 2))
        img_std = np.hstack(np.around(img_std, 2))

        # Vectorized Reinhard color normalization
        input_img = ((input_img - img_mean) * (template_std / img_std)) + template_mean
        input_img = np.clip(input_img, 0, 255).astype(np.uint8)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_LAB2BGR)
        cv2.imwrite(output_path, input_img)

    except UnidentifiedImageError:
        print(f"Warning: Skipping non-image file or corrupted image: {img}")
        logging.error(f"Skipping non-image file or corrupted image: {input_path}") #log the error
    except Exception as e:
        print(f"Error processing image {img}: {e}")  # Catch other exceptions
        logging.exception(f"Error processing image: {input_path} - {e}")  # Log the full exception



if __name__ == "__main__":
    # --- Input Validation ---
    if not os.path.isdir(INPUT):
        raise ValueError(f"Input directory '{INPUT}' does not exist.")
    if not os.path.isdir(OUTPUT):
      os.makedirs(OUTPUT) #creates the output directory if it doesn't exist

    input_image_list = [f for f in os.listdir(INPUT) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] # Consider only images

    if not input_image_list:
        print("No images found in the input directory.")
        exit()
    
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
        try:
            template_mean, template_std = load_template()
        except ValueError as ve:
            print(ve)
            exit()

        # Pass the template mean and std to process_image
        Parallel(n_jobs=-1)(
            delayed(process_image)(img, INPUT, OUTPUT, template_mean, template_std) for img in image_batch
        )
        print("Processed batch " + str(j) + " Of " + str(total))
        j = j + 1
    print(f"Processing complete.  Check '{LOG_FILE}' for any errors.")
