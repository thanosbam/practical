import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ===== SETTINGS =====
data_dir = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\monkbrill-jpg"
model_save_path = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\cnn_letter_classifier.pth"
batch_size = 32
num_epochs = 100
learning_rate = 0.001
image_size = 28
val_split = 0.2
weight_decay = 1e-4  # L2 regularization

# Early Stopping Config
early_stopping_patience = 10  # Number of epochs with no improvement before stopping
best_val_loss = float('inf')
patience_counter = 0

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===== LOAD DATASET & SPLIT =====
dataset = datasets.ImageFolder(data_dir, transform=transform)
total_size = len(dataset)
val_size = int(val_split * total_size)
train_size = total_size - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class_names = dataset.classes
num_classes = len(class_names)

# ===== CNN MODEL =====
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# ===== TRAINING SETUP =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ===== TRAINING LOOP =====
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # ===== VALIDATION LOOP =====
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            v_loss = criterion(val_outputs, val_labels)
            val_loss += v_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += (val_preds == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_acc = 100 * val_correct / val_total
    val_loss_avg = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%")

    # ===== EARLY STOPPING =====
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), model_save_path)
        patience_counter = 0  # Reset if improvement
    else:
        patience_counter += 1
        print(f"📉 No improvement in val loss. Patience: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            print("⏹ Early stopping triggered!")
            break

print(f"\n✅ Best model saved to: {model_save_path}")
