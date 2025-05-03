# LETTER SEGMENTATION USING EXISTING MASKS
# ----------------------------------------

import os
import cv2
import numpy as np

# ========= SETTINGS ==========
mask_folder = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\generated_masks_v2"  # Use pre-made masks
input_folder = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\resized_binarized"  # Original binarized images
output_letters_folder = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\segmented_letters_v2"

min_letter_width = 10
min_letter_height = 10
letter_size = (28, 28)
# =============================

# ========= PREPARE OUTPUT FOLDER ==========
os.makedirs(output_letters_folder, exist_ok=True)

# ========= PROCESS EACH IMAGE ==========
for filename in os.listdir(mask_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        mask_path = os.path.join(mask_folder, filename)
        image_path = os.path.join(input_folder, filename.replace("mask_", "") if filename.startswith("mask_") else filename)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if mask is None or image is None:
            print(f"⚠️ Skipped {filename}: mask or image not found.")
            continue

        # Threshold to binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"✅ {filename}: Found {len(contours)} letter regions.")

        # Sort contours left-to-right
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        contours = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][0])]

        # Crop and save letters
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            if w > min_letter_width and h > min_letter_height:
                letter_crop = image[y:y+h, x:x+w]
                letter_crop = cv2.resize(letter_crop, letter_size, interpolation=cv2.INTER_AREA)

                save_path = os.path.join(output_letters_folder, f"{filename.split('.')[0]}_letter_{idx}.png")
                cv2.imwrite(save_path, letter_crop)

print("\n🎯 Letter segmentation using existing masks completed!")
print(f"Letters saved to: {output_letters_folder}")
