import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===== SETTINGS =====
resized_folder = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\resized_binarized"
mask_folder = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\generated_masks_v2"
model_save_path = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\simple_unet_letter_segmentation_v2.pth"

batch_size = 2
num_epochs = 600
learning_rate = 0.001
target_size = (256, 256)

os.makedirs(mask_folder, exist_ok=True)

# ===== STEP 1: Generate Masks from Resized Images =====
for filename in os.listdir(resized_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(resized_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        if np.mean(image) > 127:
            image = 255 - image

        mask = image
        cv2.imwrite(os.path.join(mask_folder, filename), mask)

print("✅ Masks saved to:", mask_folder)

# ===== STEP 2: U-Net Model Definition =====
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ===== Dataset Definition =====
class LetterSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images = sorted(os.listdir(images_dir))
        self.images_dir = images_dir
        self.masks_dir = masks_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size)

        image = image / 255.0
        mask = mask / 255.0

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.FloatTensor(image), torch.FloatTensor(mask)

# ===== Train U-Net =====
dataset = LetterSegmentationDataset(resized_folder, mask_folder)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), model_save_path)
print("✅ Model training complete and saved to:", model_save_path)
