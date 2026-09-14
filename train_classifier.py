import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

DATA_DIR = r"data\landmarks"
samples = []
labels = []
label_map = {}

for f in sorted(os.listdir(DATA_DIR)):
    if not f.endswith('.npy'):
        continue
    sign = f.rsplit('_', 1)[0]
    if sign not in label_map:
        label_map[sign] = len(label_map)
    data = np.load(os.path.join(DATA_DIR, f))
    samples.append(data.reshape(30, 126))
    labels.append(label_map[sign])

X = np.array(samples).astype(np.float32)
y = np.array(labels)
num_classes = len(label_map)
print(f"Samples: {X.shape[0]}, Classes: {num_classes}")

idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, val_idx = idx[:split], idx[split:]

train_ds = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx]))
val_ds = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

class ISLClassifier(nn.Module):
    def __init__(self, num_classes, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(126, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = ISLClassifier(num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
for epoch in range(50):
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

    if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
        print(f"Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save({"model": model.state_dict(), "label_map": label_map}, "models/isl_classifier.pth")
        print(f"  → Saved best model (acc={val_acc:.3f})")

print(f"\nBest val accuracy: {best_val_acc:.3f}")   