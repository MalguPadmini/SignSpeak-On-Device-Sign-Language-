import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import os

# Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Split into train/val
img_dir = r"data\frames"
all_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
classes = sorted(set(f.rsplit('_', 1)[0] for f in all_files))
class_to_idx = {c: i for i, c in enumerate(classes)}
print(f"Classes: {len(classes)}, Images: {len(all_files)}")

# Create dataset
class SignDataset(torch.utils.data.Dataset):
    def __init__(self, files, root):
        self.files = files
        self.root = root
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        f = self.files[idx]
        img = Image.open(os.path.join(self.root, f)).convert('RGB')
        img = transform(img)
        label = class_to_idx[f.rsplit('_', 1)[0]]
        return img, label

# 80/20 split
import random
random.shuffle(all_files)
split = int(0.8 * len(all_files))
train_files, val_files = all_files[:split], all_files[split:]

train_ds = SignDataset(train_files, img_dir)
val_ds = SignDataset(val_files, img_dir)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Model: ResNet18 (pretrained, fine-tune)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(30):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    val_acc = correct / total

    if (epoch + 1) % 5 == 0 or val_acc > best_acc:
        print(f"Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.3f}")
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/sign_cnn.pth")
        print(f"  → Saved (acc={val_acc:.3f})")

# Save class names for later use
import json
with open("models/classes.json", "w") as f:
    json.dump(classes, f)

print(f"\nBest val accuracy: {best_acc:.3f}")   