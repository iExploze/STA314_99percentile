import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# =====================
# config
# =====================
train_dir = "data/train/train"
test_dir = "data/test/test"
sample_sub_file = "data/sample_submission.csv"

batch_size = 32
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)
print("train_dir:", train_dir)
print("test_dir:", test_dir)

# =====================
# transform
# =====================
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================
# dataset
# =====================
full_ds = datasets.ImageFolder(train_dir, transform=train_tfms)

print("classes:", full_ds.classes)
print("number of images:", len(full_ds))

# split train / val
val_size = int(0.15 * len(full_ds))
train_size = len(full_ds) - val_size

train_ds, val_ds = random_split(full_ds, [train_size, val_size])

# validation does not use augmentation
val_ds.dataset = datasets.ImageFolder(train_dir, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

print("train size:", len(train_ds))
print("val size:", len(val_ds))
print("train batches:", len(train_loader))
print("val batches:", len(val_loader))

# =====================
# model
# =====================
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)
    return model

# =====================
# train
# =====================
def train_model(model, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch:", epoch + 1, "Loss:", total_loss)

# =====================
# validation
# =====================
def check_accuracy(model):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

# =====================
# submission
# =====================
def make_submission(model, save_name):
    model.eval()

    sample = pd.read_csv(sample_sub_file)
    id_col = sample.columns[0]
    pred_col = sample.columns[1]

    test_files = sorted(os.listdir(test_dir))

    ids = []
    pred_labels = []

    with torch.no_grad():
        for file_name in test_files:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = test_dir + "/" + file_name
                img = Image.open(img_path).convert("RGB")
                img = val_tfms(img).unsqueeze(0).to(device)

                outputs = model(img)
                _, predicted = torch.max(outputs, 1)

                ids.append(file_name)
                pred_labels.append(full_ds.classes[predicted.item()])

    submission = pd.DataFrame({
        id_col: ids,
        pred_col: pred_labels
    })

    submission.to_csv(save_name, index=False)
    print("saved", save_name)

# =====================
# experiment 1: 5 epochs
# =====================
print("\n===== ResNet18: 5 epochs =====")
model_5 = build_model()
train_model(model_5, 5)
acc_5 = check_accuracy(model_5)
print("Validation accuracy (5 epochs):", acc_5)
make_submission(model_5, "submission_resnet18_5epochs.csv")

# =====================
# experiment 2: 10 epochs
# =====================
print("\n===== ResNet18: 10 epochs =====")
model_10 = build_model()
train_model(model_10, 10)
acc_10 = check_accuracy(model_10)
print("Validation accuracy (10 epochs):", acc_10)
make_submission(model_10, "submission_resnet18_10epochs.csv")

# =====================
# final result
# =====================
print("\n===== Final Result =====")
print("5 epochs accuracy:", acc_5)
print("10 epochs accuracy:", acc_10)
