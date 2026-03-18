from __future__ import annotations

import os
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models.convnext import LayerNorm2d
from tqdm.auto import tqdm


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
# Model builders
# -----------------------------
def build_resnet50(num_classes: int, dropout: float = 0.2) -> nn.Module:
    model = models.resnet50(weights=None)
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



def build_convnext_tiny(num_classes: int, dropout: float = 0.3) -> nn.Module:
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        LayerNorm2d(in_features, eps=1e-6),
        nn.Flatten(1),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return model


# -----------------------------
# Dataset
# -----------------------------
class OrderedTestDataset(Dataset):
    def __init__(self, test_dir: str, filenames: Sequence[str], transform) -> None:
        self.test_dir = Path(test_dir)
        self.filenames = list(filenames)
        self.transform = transform

        missing = [name for name in self.filenames if not (self.test_dir / name).exists()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Could not find {len(missing)} test files under {self.test_dir}. First missing: {preview}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]
        img = Image.open(self.test_dir / name).convert("RGB")
        x = self.transform(img)
        return x, name


# -----------------------------
# Transforms / TTA
# -----------------------------
def build_base_transform(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])



def predict_with_tta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    tta: int,
) -> torch.Tensor:
    model.eval()
    all_probs: List[torch.Tensor] = []
    amp_ctx = autocast if (use_amp and device.type == "cuda") else nullcontext

    with torch.no_grad():
        pbar = tqdm(loader, desc="predict", dynamic_ncols=True, leave=False, unit="batch")
        for x, _ in pbar:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            with amp_ctx():
                probs = torch.softmax(model(x), dim=1)
                if tta >= 2:
                    probs = probs + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
                probs = probs / float(tta)
            all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)


# -----------------------------
# Loaders
# -----------------------------
def assert_same_mapping(reference: Dict[str, int] | None, current: Dict[str, int], source_name: str) -> Dict[str, int]:
    if reference is None:
        return current
    if reference != current:
        raise ValueError(f"class_to_idx mismatch for source: {source_name}")
    return reference



def predict_resnet_group(
    checkpoint_paths: Sequence[Path],
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    tta: int,
    class_to_idx_ref: Dict[str, int] | None,
):
    group_probs = None
    class_to_idx = class_to_idx_ref

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        class_to_idx = assert_same_mapping(class_to_idx, ckpt["class_to_idx"], str(ckpt_path))
        dropout = ckpt.get("args", {}).get("dropout", 0.2)

        model = build_resnet50(num_classes=len(class_to_idx), dropout=dropout)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)

        print(f"Predicting ResNet50 source: {ckpt_path}")
        probs = predict_with_tta(model, loader, device, use_amp=use_amp, tta=tta)
        group_probs = probs if group_probs is None else group_probs + probs

    if group_probs is None:
        return None, class_to_idx
    return group_probs / float(len(checkpoint_paths)), class_to_idx



def collect_fold_checkpoints(dirs: Sequence[Path]) -> List[Path]:
    ckpts: List[Path] = []
    for d in dirs:
        found = sorted(d.glob("fold*_best.pt"))
        if not found:
            raise FileNotFoundError(f"No fold*_best.pt files found in {d}")
        ckpts.extend(found)
    return ckpts



def predict_convnext_group(
    checkpoint_dirs: Sequence[Path],
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    tta: int,
    class_to_idx_ref: Dict[str, int] | None,
    group_name: str,
):
    checkpoint_paths = collect_fold_checkpoints(checkpoint_dirs)
    group_probs = None
    class_to_idx = class_to_idx_ref

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        class_to_idx = assert_same_mapping(class_to_idx, ckpt["class_to_idx"], str(ckpt_path))
        dropout = ckpt.get("args", {}).get("dropout", 0.3)

        model = build_convnext_tiny(num_classes=len(class_to_idx), dropout=dropout)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)

        print(f"Predicting {group_name} source: {ckpt_path}")
        probs = predict_with_tta(model, loader, device, use_amp=use_amp, tta=tta)
        group_probs = probs if group_probs is None else group_probs + probs

    return group_probs / float(len(checkpoint_paths)), class_to_idx


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weighted hybrid ensemble for ResNet50 + ConvNeXt (+ optional crop model)."
    )
    parser.add_argument("--sample_submission", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--crop_test_dir", type=str, default="", help="Optional cropped test folder for crop models.")
    parser.add_argument("--output_csv", type=str, default="submission_hybrid_ensemble.csv")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", type=int, default=2, choices=[1, 2])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--resnet_checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--convnext_dirs", type=str, nargs="*", default=[])
    parser.add_argument("--crop_convnext_dirs", type=str, nargs="*", default=[])

    parser.add_argument("--resnet_weight", type=float, default=1.0)
    parser.add_argument("--convnext_weight", type=float, default=1.0)
    parser.add_argument("--crop_weight", type=float, default=1.0)

    args = parser.parse_args()

    if not args.resnet_checkpoints and not args.convnext_dirs and not args.crop_convnext_dirs:
        raise ValueError("Provide at least one source: resnet checkpoints, convnext dirs, or crop convnext dirs.")

    device = get_device(args.device)
    print(f"Device: {device}")

    sample_df = pd.read_csv(args.sample_submission)
    if sample_df.shape[1] < 2:
        raise ValueError("sample_submission.csv should have at least 2 columns.")

    id_col = sample_df.columns[0]
    target_col = sample_df.columns[1]
    filenames = sample_df[id_col].astype(str).tolist()

    base_dataset = OrderedTestDataset(
        test_dir=args.test_dir,
        filenames=filenames,
        transform=build_base_transform(args.img_size),
    )
    loader = DataLoader(
        base_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    crop_loader = loader
    if args.crop_convnext_dirs:
        crop_dir = args.crop_test_dir or args.test_dir
        crop_dataset = OrderedTestDataset(
            test_dir=crop_dir,
            filenames=filenames,
            transform=build_base_transform(args.img_size),
        )
        crop_loader = DataLoader(
            crop_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

    class_to_idx: Dict[str, int] | None = None
    weighted_sum = None
    weight_total = 0.0

    if args.resnet_checkpoints:
        probs, class_to_idx = predict_resnet_group(
            checkpoint_paths=[Path(p) for p in args.resnet_checkpoints],
            loader=loader,
            device=device,
            use_amp=args.amp,
            tta=args.tta,
            class_to_idx_ref=class_to_idx,
        )
        if probs is not None and args.resnet_weight > 0:
            weighted_sum = probs * args.resnet_weight if weighted_sum is None else weighted_sum + probs * args.resnet_weight
            weight_total += args.resnet_weight

    if args.convnext_dirs:
        probs, class_to_idx = predict_convnext_group(
            checkpoint_dirs=[Path(p) for p in args.convnext_dirs],
            loader=loader,
            device=device,
            use_amp=args.amp,
            tta=args.tta,
            class_to_idx_ref=class_to_idx,
            group_name="ConvNeXt",
        )
        if args.convnext_weight > 0:
            weighted_sum = probs * args.convnext_weight if weighted_sum is None else weighted_sum + probs * args.convnext_weight
            weight_total += args.convnext_weight

    if args.crop_convnext_dirs:
        probs, class_to_idx = predict_convnext_group(
            checkpoint_dirs=[Path(p) for p in args.crop_convnext_dirs],
            loader=crop_loader,
            device=device,
            use_amp=args.amp,
            tta=args.tta,
            class_to_idx_ref=class_to_idx,
            group_name="Crop ConvNeXt",
        )
        if args.crop_weight > 0:
            weighted_sum = probs * args.crop_weight if weighted_sum is None else weighted_sum + probs * args.crop_weight
            weight_total += args.crop_weight

    if weighted_sum is None or class_to_idx is None or weight_total <= 0:
        raise RuntimeError("No valid predictions were produced.")

    ensemble_probs = weighted_sum / weight_total
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pred_idx = torch.argmax(ensemble_probs, dim=1).tolist()
    pred_labels = [idx_to_class[i] for i in pred_idx]

    out_df = sample_df.copy()
    out_df[target_col] = pred_labels
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("\nDone.")
    print(f"Wrote weighted ensemble submission to: {out_path}")
    print(f"Weights used -> resnet: {args.resnet_weight}, convnext: {args.convnext_weight}, crop: {args.crop_weight}")


if __name__ == "__main__":
    main()
