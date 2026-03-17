from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    EfficientNet_B2_Weights,
)


# -----------------------------
# Device
# -----------------------------
def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device_arg == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Transforms
# -----------------------------
def build_eval_transform(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# -----------------------------
# Model builders
# -----------------------------
def build_resnet18(num_classes: int, dropout: float = 0.0) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50(num_classes: int, dropout: float = 0.0) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b2(num_classes: int, dropout: float = 0.0) -> nn.Module:
    model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_model_from_checkpoint_args(
    checkpoint_args: Dict,
    num_classes: int,
) -> nn.Module:
    dropout = float(checkpoint_args.get("dropout", 0.0))

    # train_resnet50_finetune.py usually has no backbone arg
    if "backbone" not in checkpoint_args:
        return build_resnet50(num_classes=num_classes, dropout=dropout)

    backbone = checkpoint_args.get("backbone", "resnet18")

    if backbone == "resnet18":
        return build_resnet18(num_classes=num_classes, dropout=dropout)
    if backbone == "efficientnet":
        return build_efficientnet_b2(num_classes=num_classes, dropout=dropout)

    raise ValueError(f"Unsupported backbone in checkpoint args: {backbone}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission.csv from a saved checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument("--sample_submission", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    test_dir = Path(args.test_dir)
    sample_submission_path = Path(args.sample_submission)
    output_csv = Path(args.output_csv)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not sample_submission_path.exists():
        raise FileNotFoundError(f"Sample submission not found: {sample_submission_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "class_to_idx" not in ckpt:
        raise KeyError("Checkpoint is missing 'class_to_idx'")

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    checkpoint_args = ckpt.get("args", {})

    print(f"Classes: {idx_to_class}")

    model = build_model_from_checkpoint_args(checkpoint_args, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tfm = build_eval_transform(args.img_size)

    sample = pd.read_csv(sample_submission_path)
    if sample.shape[1] < 2:
        raise ValueError("sample_submission must have at least 2 columns")

    id_col = sample.columns[0]
    pred_col = sample.columns[1]

    test_files = sorted([
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    ids = [p.name for p in test_files]
    pred_labels = []

    with torch.no_grad():
        for p in test_files:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            pred_idx = int(model(x).argmax(dim=1).item())
            pred_labels.append(idx_to_class[pred_idx])

    submission = pd.DataFrame({
        id_col: ids,
        pred_col: pred_labels,
    })
    submission.to_csv(output_csv, index=False)

    print(f"Saved submission to: {output_csv}")


if __name__ == "__main__":
    main()