import cv2
import albumentations as A
import numpy as np
import os

# ----------------------------
# Simple augmentation factory
# ----------------------------
def get_transform(code):
    if code == "HF":  # Horizontal Flip
        return A.HorizontalFlip(p=1.0)
    if code == "VF":  # Vertical Flip
        return A.VerticalFlip(p=1.0)
    if code == "RR":  # Random 90-degree rotation
        return A.RandomRotate90(p=1.0)
    if code == "GB":  # Gaussian blur
        return A.GaussianBlur(blur_limit=(3, 7), p=1.0)
    if code == "HED":  # H&E stain perturbation
        return A.HEStain(
            method="random_preset",
            intensity_shift_range=(-0.2, 0.2),
            intensity_scale_range=(0.7, 1.3),
            p=1.0
        )
    if code == "HSV":  # Hue/Saturation/Value change
        return A.HueSaturationValue(
            hue_shift_limit=25,
            sat_shift_limit=50,
            val_shift_limit=50,
            p=1.0
        )
    raise ValueError("Unknown transform code")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    input_img_path = "D:\\Usuario\\Desktop\\AUG\\INPUT\\CANCER_PATIENT_100052_139431_66168_2437_20251115_015504_259868.png"   # <-- Change to your example image
    output_dir = "D:\\Usuario\\Desktop\\AUG\\OUTPUT"  # <-- Change to your desired output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read input image (color)
    image = cv2.imread(input_img_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {input_img_path}")

    # Augmentations to preview
    AUGS = ["HF", "VF", "RR", "GB", "HED", "HSV"]

    for code in AUGS:
        transform = get_transform(code)
        augmented = transform(image=image)

        output_path = os.path.join(output_dir, f"example_{code}.png")
        cv2.imwrite(output_path, augmented["image"])
        print(f"Saved: {output_path}")

    print("\nDone! All augmentation examples saved in:", output_dir)
