from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    EfficientNet_B2_Weights,
    ConvNeXt_Tiny_Weights,
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
# Eval transforms
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
def build_resnet18_simple(num_classes: int, dropout: float = 0.0) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet18_deep(num_classes: int, dropout: float = 0.2) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=max(0.05, dropout * 0.5)),
        nn.Linear(128, num_classes),
    )
    return model


def build_resnet50_simple(num_classes: int, dropout: float = 0.0) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50_deep(num_classes: int, dropout: float = 0.2) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=max(0.05, dropout * 0.5)),
        nn.Linear(128, num_classes),
    )
    return model


def build_efficientnet_b2_simple(num_classes: int, dropout: float = 0.0) -> nn.Module:
    model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_efficientnet_b2_deep(num_classes: int, dropout: float = 0.2) -> nn.Module:
    model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=max(0.05, dropout * 0.5)),
        nn.Linear(128, num_classes),
    )
    return model


def build_convnext_tiny_deep(num_classes: int, dropout: float = 0.2) -> nn.Module:
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=max(0.05, dropout * 0.5)),
        nn.Linear(128, num_classes),
    )
    return model


# -----------------------------
# Robust checkpoint loader
# -----------------------------
def load_model_from_checkpoint(
    ckpt: Dict[str, Any],
    num_classes: int,
) -> tuple[nn.Module, str]:
    checkpoint_args = ckpt.get("args", {})
    state_dict = ckpt["model_state_dict"]
    dropout = float(checkpoint_args.get("dropout", 0.0))
    backbone = checkpoint_args.get("backbone", None)

    candidates: List[tuple[str, nn.Module]] = []

    # Newer ResNet50 training script had no backbone arg
    if backbone is None:
        candidates.extend([
            ("resnet50_deep", build_resnet50_deep(num_classes, dropout=max(dropout, 0.2))),
            ("resnet50_simple", build_resnet50_simple(num_classes, dropout=dropout)),
        ])
    elif backbone == "resnet18":
        candidates.extend([
            ("resnet18_deep", build_resnet18_deep(num_classes, dropout=max(dropout, 0.2))),
            ("resnet18_simple", build_resnet18_simple(num_classes, dropout=dropout)),
        ])
    elif backbone == "efficientnet":
        candidates.extend([
            ("efficientnet_b2_deep", build_efficientnet_b2_deep(num_classes, dropout=max(dropout, 0.2))),
            ("efficientnet_b2_simple", build_efficientnet_b2_simple(num_classes, dropout=dropout)),
        ])
    elif backbone == "convnext":
        candidates.extend([
            ("convnext_tiny_deep", build_convnext_tiny_deep(num_classes, dropout=max(dropout, 0.2))),
        ])
    else:
        raise ValueError(f"Unsupported backbone in checkpoint args: {backbone}")

    last_error = None
    for name, model in candidates:
        try:
            model.load_state_dict(state_dict)
            return model, name
        except RuntimeError as e:
            last_error = e

    raise RuntimeError(
        f"Could not load checkpoint into any supported architecture. Last error:\n{last_error}"
    )


# -----------------------------
# Test file resolver
# -----------------------------
def build_test_lookup(test_dir: Path) -> Dict[str, Path]:
    valid_exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

    lookup: Dict[str, Path] = {}
    for p in files:
        lookup[p.name.lower()] = p
        lookup[p.stem.lower()] = p
    return lookup


def resolve_test_path(raw_id: str, lookup: Dict[str, Path]) -> Path:
    raw = str(raw_id).strip()
    raw_lower = raw.lower()

    candidates = [
        raw_lower,
        Path(raw_lower).name,
        Path(raw_lower).stem,
    ]

    if "." not in Path(raw).name:
        candidates.extend([
            f"{raw_lower}.jpg",
            f"{raw_lower}.jpeg",
            f"{raw_lower}.png",
        ])

    for key in candidates:
        if key in lookup:
            return lookup[key]

    raise FileNotFoundError(f"Could not resolve test file for sample_submission id: {raw_id}")


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

    model, architecture_name = load_model_from_checkpoint(ckpt, num_classes=num_classes)
    model.to(device)
    model.eval()

    print(f"Loaded model as: {architecture_name}")
    print(f"Classes: {idx_to_class}")

    tfm = build_eval_transform(args.img_size)

    sample = pd.read_csv(sample_submission_path)
    if sample.shape[1] < 2:
        raise ValueError("sample_submission must have at least 2 columns")

    id_col = sample.columns[0]
    pred_col = sample.columns[1]

    lookup = build_test_lookup(test_dir)

    predictions = []
    with torch.no_grad():
        for raw_id in sample[id_col].tolist():
            img_path = resolve_test_path(raw_id, lookup)
            img = Image.open(img_path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            logits = model(x)
            pred_idx = int(logits.argmax(dim=1).item())
            predictions.append(idx_to_class[pred_idx])

    submission = sample.copy()
    submission[pred_col] = predictions
    submission.to_csv(output_csv, index=False)

    print(f"Saved submission to: {output_csv}")


if __name__ == "__main__":
    main()

#.\.venv\Scripts\python.exe src\train_resnet50_finetune.py --data_dir data --fine_tune --lr_scheduler --device cuda --num_workers 4 --lr 1e-3 --finetune_lr 2e-5 --dropout 0.2 --label_smooth 0.0 --amp --save_path checkpoints/best_resnet50.pt
# command to run ^