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
from typing import Dict, List

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
# Model
# -----------------------------
def build_model(num_classes: int, dropout: float = 0.3) -> nn.Module:
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
    def __init__(self, test_dir: str, filenames: List[str], transform) -> None:
        self.test_dir = Path(test_dir)
        self.filenames = filenames
        self.transform = transform

        missing = [name for name in filenames if not (self.test_dir / name).exists()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Could not find {len(missing)} test files under {self.test_dir}. "
                f"First missing: {preview}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]
        path = self.test_dir / name
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        return x, name


# -----------------------------
# TTA
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
    img_size: int,
) -> torch.Tensor:
    model.eval()
    probs_chunks: List[torch.Tensor] = []
    amp_ctx = autocast if (use_amp and device.type == "cuda") else nullcontext

    with torch.no_grad():
        pbar = tqdm(loader, desc="predict", dynamic_ncols=True, leave=False, unit="batch")
        for x, _names in pbar:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            prob_sum = None

            with amp_ctx():
                logits = model(x)
                prob_sum = torch.softmax(logits, dim=1)

                if tta >= 2:
                    logits_flip = model(torch.flip(x, dims=[3]))
                    prob_sum = prob_sum + torch.softmax(logits_flip, dim=1)

                if tta >= 4:
                    pad = max(8, img_size // 18)
                    x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
                    crop1 = x_pad[:, :, 0:img_size, 0:img_size]
                    crop2 = x_pad[:, :, -img_size:, -img_size:]
                    logits_crop1 = model(crop1)
                    logits_crop2 = model(crop2)
                    prob_sum = prob_sum + torch.softmax(logits_crop1, dim=1)
                    prob_sum = prob_sum + torch.softmax(logits_crop2, dim=1)

            probs_chunks.append((prob_sum / float(tta)).cpu())

    return torch.cat(probs_chunks, dim=0)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict with a 5-fold ConvNeXt-Tiny ensemble + TTA"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--sample_submission", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="submission_convnext_tiny_cv.csv")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("fold*_best.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold checkpoints found in {checkpoint_dir}")

    print(f"Found {len(checkpoint_paths)} checkpoints:")
    for p in checkpoint_paths:
        print(f"  - {p}")

    sample_df = pd.read_csv(args.sample_submission)
    if sample_df.shape[1] < 2:
        raise ValueError("sample_submission.csv should have at least 2 columns.")

    id_col = sample_df.columns[0]
    target_col = sample_df.columns[1]
    filenames = sample_df[id_col].astype(str).tolist()

    base_transform = build_base_transform(args.img_size)
    dataset = OrderedTestDataset(args.test_dir, filenames, base_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    ensemble_probs = None
    idx_to_class: Dict[int, str] | None = None

    for ckpt_path in checkpoint_paths:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        class_to_idx = checkpoint["class_to_idx"]
        idx_to_class_local = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        dropout = checkpoint.get("args", {}).get("dropout", 0.3)

        if idx_to_class is None:
            idx_to_class = idx_to_class_local
        elif idx_to_class != idx_to_class_local:
            raise ValueError("Checkpoint class mappings do not match across folds.")

        model = build_model(num_classes=num_classes, dropout=dropout)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(device)

        print(f"\nPredicting with {ckpt_path.name} | saved val acc={checkpoint.get('best_val_acc', 'n/a')}")
        probs = predict_with_tta(
            model=model,
            loader=loader,
            device=device,
            use_amp=args.amp,
            tta=args.tta,
            img_size=args.img_size,
        )

        if ensemble_probs is None:
            ensemble_probs = probs
        else:
            ensemble_probs += probs

    if ensemble_probs is None or idx_to_class is None:
        raise RuntimeError("No predictions were generated.")

    ensemble_probs = ensemble_probs / float(len(checkpoint_paths))
    pred_idx = torch.argmax(ensemble_probs, dim=1).tolist()
    pred_labels = [idx_to_class[i] for i in pred_idx]

    out_df = sample_df.copy()
    out_df[target_col] = pred_labels
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nDone. Wrote ensemble submission to: {out_path}")
    print(f"Target column filled: {target_col}")
    print(f"TTA mode: {args.tta}")
    print(f"Models ensembled: {len(checkpoint_paths)}")


if __name__ == "__main__":
    main()
