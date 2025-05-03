import os
import cv2

# === Configuration ===
original_folder = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\binarized"
resized_folder = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\resized_binarized"
resize_scale = 0.7  # 70% size

# Create output folder
os.makedirs(resized_folder, exist_ok=True)

# Process each image
for filename in os.listdir(original_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(original_folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"⚠️ Failed to read {filename}, skipping.")
            continue

        # Resize image
        new_width = int(image.shape[1] * resize_scale)
        new_height = int(image.shape[0] * resize_scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Save resized image
        save_path = os.path.join(resized_folder, filename)
        cv2.imwrite(save_path, resized)

        print(f"✅ Resized and saved: {filename}")

print("\n🎉 All images resized successfully.")
print(f"📁 Output folder: {resized_folder}")
