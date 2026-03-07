import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

# CONFIG (edit these)
DATA_DIR   = Path(r"C:\Users\user\Desktop\2025-2026\STA314\J-model\STA314_99percentile-main\train\train1\src\data")
TRAIN_DIR  = DATA_DIR / "train"
TEST_DIR   = DATA_DIR / "test"
SAMPLE_SUB = Path(r"C:\Users\user\Desktop\2025-2026\STA314\J-model\STA314_99percentile-main\train\train1\sample_submission.csv")


BATCH_SIZE   = 32
EPOCHS       = 10          # total epochs (Stage 1 + Stage 2)
FREEZE_EPOCHS = 4          # Stage 1: head-only (set to EPOCHS to skip fine-tuning)
LR           = 1e-3        # Stage 1 LR
FINETUNE_LR  = 1e-4        # Stage 2 LR (backbone unfrozen)
FINE_TUNE    = True        # set False to skip Stage 2
LR_SCHEDULER = True        # cosine annealing within each stage
WEIGHT_DECAY = 1e-4
SEED         = 314
NUM_WORKERS  = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)


# Transforms
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
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

# Dataset
full_ds    = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
class_names = full_ds.classes
num_classes = len(class_names)
print("Classes:", class_names)

val_ratio  = 0.15
val_size   = int(len(full_ds) * val_ratio)
train_size = len(full_ds) - val_size
g = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=g)

# val split should NOT use augmentation
val_ds.dataset = datasets.ImageFolder(TRAIN_DIR, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# Model helpers

def build_model(feature_extract: bool = True) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    if feature_extract:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def unfreeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
    print("  🔓 All layers unfrozen for fine-tuning.")


# Eval helper
criterion = nn.CrossEntropyLoss()

def evaluate(model: nn.Module):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * x.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += y.size(0)
    return loss_sum / total, correct / total


# STAGE 1 — Head only
model = build_model(feature_extract=True)

stage1_epochs = FREEZE_EPOCHS if FINE_TUNE else EPOCHS
print(f"\n{'='*55}")
print(f"Stage 1: Head-only  ({stage1_epochs} epochs, lr={LR})")
print(f"{'='*55}")

opt1 = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY
)
sch1 = (torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=stage1_epochs)
        if LR_SCHEDULER else None)

for epoch in range(1, stage1_epochs + 1):
    model.train()
    for x, y in tqdm(train_loader, desc=f"[S1] Epoch {epoch}/{stage1_epochs}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt1.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt1.step()
    if sch1: sch1.step()

    val_loss, val_acc = evaluate(model)
    print(f"[S1] Epoch {epoch:02d} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
          f"| lr={opt1.param_groups[0]['lr']:.2e}")

# STAGE 2 — Fine-tune
if FINE_TUNE:
    stage2_epochs = EPOCHS - FREEZE_EPOCHS
    print(f"\n{'='*55}")
    print(f"Stage 2: Full fine-tune  ({stage2_epochs} epochs, lr={FINETUNE_LR})")
    unfreeze_model(model)
    print(f"{'='*55}")

    opt2 = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    sch2 = (torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=stage2_epochs)
            if LR_SCHEDULER else None)

    for epoch in range(1, stage2_epochs + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"[S2] Epoch {epoch}/{stage2_epochs}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt2.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt2.step()
        if sch2: sch2.step()

        val_loss, val_acc = evaluate(model)
        global_ep = FREEZE_EPOCHS + epoch
        print(f"[S2] Epoch {global_ep:02d} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"| lr={opt2.param_groups[0]['lr']:.2e}")


# Predict test
sample  = pd.read_csv(SAMPLE_SUB)
id_col  = sample.columns[0]
pred_col = sample.columns[1]

test_files  = sorted([p for p in TEST_DIR.iterdir()
                      if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
ids         = [p.name for p in test_files]

model.eval()
pred_labels = []
with torch.no_grad():
    for p in tqdm(test_files, desc="Predicting"):
        img  = Image.open(p).convert("RGB")
        x    = val_tfms(img).unsqueeze(0).to(DEVICE)
        pred = int(model(x).argmax(dim=1).item())
        pred_labels.append(class_names[pred])

submission = pd.DataFrame({id_col: ids, pred_col: pred_labels})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
