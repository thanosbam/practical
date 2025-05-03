import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
import random

# ===== SETTINGS =====
model_path = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\cnn_letter_classifier.pth"
letter_folder = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\segmented_letters_v2"
monkbrill_dataset = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\monkbrill-jpg"
save_csv_path = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\results\predictions.csv"
results_folder = r"C:\Users\Thanos\Desktop\Uni\Deep Learning Practical\results"
image_size = 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== CREATE RESULTS FOLDER =====
os.makedirs(results_folder, exist_ok=True)

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===== GET CLASS LABELS =====
class_names = sorted([
    d for d in os.listdir(monkbrill_dataset)
    if os.path.isdir(os.path.join(monkbrill_dataset, d))
])
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

# ===== LOAD MODEL =====
model = SimpleCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===== CLASSIFY ALL LETTERS =====
predictions = []
filenames = []

for filename in sorted(os.listdir(letter_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(letter_folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"⚠️ Could not read: {filename}")
            continue

        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            pred_index = torch.argmax(output, dim=1).item()
            predicted_label = class_names[pred_index]

        predictions.append(predicted_label)
        filenames.append(filename)

        print(f"{filename}: {predicted_label}")

# ===== SAVE PREDICTIONS TO CSV =====
df = pd.DataFrame({
    'Filename': filenames,
    'Predicted Label': predictions
})
df.to_csv(save_csv_path, index=False)
print(f"\n✅ Predictions saved to {save_csv_path}")

# ===== PLOT AND SAVE BAR PLOT =====
counter = Counter(predictions)
labels = class_names
values = [counter.get(label, 0) for label in labels]

plt.figure(figsize=(12, 6))
plt.bar(labels, values)
plt.xticks(rotation=90)
plt.xlabel('Predicted Letter')
plt.ylabel('Count')
plt.title('Number of Times Each Letter was Predicted')
plt.tight_layout()

# Save bar plot
barplot_path = os.path.join(results_folder, 'predictions_barplot.png')
plt.savefig(barplot_path)
print(f"✅ Bar plot saved to {barplot_path}")
plt.show()

# ===== PLOT AND SAVE CONFUSION MATRIX =====
# Simulated true labels for now
true_labels = random.choices(class_names, k=len(predictions))

cm = confusion_matrix(true_labels, predictions, labels=class_names)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Simulated)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

conf_matrix_path = os.path.join(results_folder, 'confusion_matrix.png')
plt.savefig(conf_matrix_path)
print(f"✅ Confusion matrix saved to {conf_matrix_path}")
plt.show()
