import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Define data load function
X = []
y = []

# 图片大小
img_size = (64,64)

base_dir = "C:/Users/user/Desktop/2025-2026/STA314/J-model/STA314_99percentile-main/train/train/"

happy_dir = base_dir + "Happy/"
sad_dir = base_dir + "Sad/"
angry_dir = base_dir + "Angry/"

# Happy
for i in range(253,500):

    filename = f"img_{i:06d}.jpg"
    path = happy_dir + filename

    if os.path.exists(path):

        img = Image.open(path).convert("RGB")
        img = img.resize(img_size)

        img = np.array(img) / 255.0

        X.append(img)
        y.append(0)

# Sad

for i in range(501,785):

    filename = f"img_{i:06d}.jpg"
    path = sad_dir + filename

    if os.path.exists(path):

        img = Image.open(path).convert("RGB")
        img = img.resize(img_size)

        img = np.array(img) / 255.0

        X.append(img)
        y.append(1)

# Angry

for i in range(1,250):

    filename = f"img_{i:06d}.jpg"
    path = angry_dir + filename

    if os.path.exists(path):

        img = Image.open(path).convert("RGB")
        img = img.resize(img_size)

        img = np.array(img) / 255.0

        X.append(img)
        y.append(2)

# Data transforming

X = np.array(X)
y = np.array(y)

X = np.transpose(X, (0,3,1,2))

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for images, labels in loader:
    print(images.shape)
    print(labels.shape)
    break
print(X.shape)

#CNN
class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
print(model)

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# train
epochs = 10
loss_history = []
for epoch in range(epochs):

    total_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    loss_history.append(total_loss)

    print("Epoch:", epoch, "Loss:", total_loss)

# accuracy
correct = 0
total = 0

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print("Accuracy:", correct / total)

# loss graph

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# save model
torch.save(model.state_dict(), "cnn_model.pth")