import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# config
train_dir = "train/train"
test_dir = "test/test"
sample_sub_file = "sample_submission.csv"

batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)
print("train_dir:", train_dir)
print("test_dir:", test_dir)
print("sample_sub_file:", sample_sub_file)

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")
if not os.path.exists(sample_sub_file):
    raise FileNotFoundError(f"Sample submission file not found: {sample_sub_file}")

# transform
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# dataset
full_ds = datasets.ImageFolder(train_dir, transform=tfms)

print("classes:", full_ds.classes)
print("number of images:", len(full_ds))

# split train / val
val_size = int(0.15 * len(full_ds))
train_size = len(full_ds) - val_size

generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)

print("train size:", len(train_ds))
print("val size:", len(val_ds))

# dataloader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

print("train batches:", len(train_loader))
print("val batches:", len(val_loader))

# ResNet18 feature extractor
resnet = models.resnet18(weights=None)

# remove final classification layer
resnet.fc = nn.Identity()

resnet = resnet.to(device)
resnet.eval()

print("feature extractor ready")

# feature extraction
def extract_features(loader):
    feature_list = []
    label_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            features = resnet(images)
            features = features.cpu().numpy()

            feature_list.append(features)
            label_list.append(labels.numpy())

    X = np.vstack(feature_list)
    y = np.hstack(label_list)

    return X, y

# extract train / val features
X_train, y_train = extract_features(train_loader)
X_val, y_val = extract_features(val_loader)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)

# logistic regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# validation prediction
y_pred = logreg.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print("Validation accuracy (logistic regression):", acc)

# extract test features
test_files = sorted(os.listdir(test_dir))

test_ids = []
test_features = []

with torch.no_grad():
    for file_name in test_files:
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(test_dir, file_name)
            img = Image.open(img_path).convert("RGB")
            img = tfms(img).unsqueeze(0).to(device)

            features = resnet(img)
            features = features.cpu().numpy()

            test_ids.append(file_name)
            test_features.append(features)

X_test = np.vstack(test_features)

print("X_test shape:", X_test.shape)

# predict test
test_pred = logreg.predict(X_test)

# map numeric labels back to class names
pred_labels = [full_ds.classes[i] for i in test_pred]

# read sample submission
sample = pd.read_csv(sample_sub_file)
id_col = sample.columns[0]
pred_col = sample.columns[1]

# make submission
submission_logreg = pd.DataFrame({
    id_col: test_ids,
    pred_col: pred_labels
})

submission_logreg.to_csv("submission_logreg.csv", index=False)

print(submission_logreg.head())
print("saved submission_logreg.csv")
