import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image

# -------------------
# CONFIG (edit these)
# -------------------
DATA_DIR = Path("data")          # put your Kaggle folder here (or ".")
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR  = DATA_DIR / "test" / "test"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4
SEED = 314
NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)

# -------------------
# Transforms
# -------------------
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------
# Dataset (train)
# -------------------
full_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
class_names = full_ds.classes
num_classes = len(class_names)
print("Classes:", class_names)

# split train/val
val_ratio = 0.15
val_size = int(len(full_ds) * val_ratio)
train_size = len(full_ds) - val_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

# IMPORTANT: val should not use augmentation
# random_split keeps same transform object, so we override:
val_ds.dataset = datasets.ImageFolder(TRAIN_DIR, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# -------------------
# Model (ResNet18)
# -------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# -------------------
# Train loop
# -------------------
def evaluate():
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total

for epoch in range(1, EPOCHS + 1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=float(loss.item()))

    val_loss, val_acc = evaluate()
    print(f"Epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

# -------------------
# Predict test
# -------------------
# Read sample submission to match required columns
sample = pd.read_csv(SAMPLE_SUB)
print("Sample submission columns:", list(sample.columns))

# Usually: ["id", "label"] or ["filename", "category"] etc.
id_col = sample.columns[0]
pred_col = sample.columns[1]

# build list of test image files
test_files = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

# If sample expects filenames, use file.name. If expects numeric ids, you might need to strip extension.
# We'll default to filename (most common).
ids = [p.name for p in test_files]

model.eval()
pred_labels = []
with torch.no_grad():
    for p in tqdm(test_files, desc="Predicting"):
        img = Image.open(p).convert("RGB")
        x = val_tfms(img).unsqueeze(0).to(DEVICE)
        logits = model(x)
        pred_idx = int(logits.argmax(dim=1).item())
        pred_labels.append(class_names[pred_idx])

submission = pd.DataFrame({id_col: ids, pred_col: pred_labels})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")