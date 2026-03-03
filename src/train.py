from __future__ import annotations

"""
STA314 baseline training script (ResNet18 feature extraction / fine-tuning).
- Reads data via ImageFolder using your src/data_loader.py
- Splits train/val (default 85/15)
- Trains pretrained ResNet18 quickly (feature extraction by default)
- Prints train/val loss + accuracy each epoch
- Saves best checkpoint by val accuracy to checkpoints/best.pt

Folder expected:
  data/train/<class_name>/*.jpg
"""

# ---- SSL / certificates fix (must be AFTER __future__ import) ----
import os
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights

from data_loader import make_dataloaders  # train.py and data_loader.py both in src/


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device_arg == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    """
    If optimizer is provided -> train mode
    Else -> eval mode
    Returns: (avg_loss, avg_acc)
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_seen += y.size(0)

    avg_loss = total_loss / max(total_seen, 1)
    avg_acc = total_correct / max(total_seen, 1)
    return avg_loss, avg_acc


def build_resnet18(num_classes: int, feature_extract: bool = True, pretrained: bool = True) -> nn.Module:
    """
    feature_extract=True: freeze backbone, train only final layer
    pretrained=True: load ImageNet weights (downloads once, caches)
    """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    if feature_extract:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    epoch: int,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "args": vars(args),
    }
    torch.save(payload, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="STA314 baseline training (ResNet18).")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_subdir", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--fine_tune", action="store_true", help="Train all layers (not just final layer).")
    parser.add_argument("--no_pretrained", action="store_true", help="Do NOT download pretrained weights.")
    parser.add_argument("--save_path", type=str, default="checkpoints/best.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, class_names, class_to_idx, n_train, n_val = make_dataloaders(
        data_dir=args.data_dir,
        train_subdir=args.train_subdir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train samples: {n_train} | Val samples: {n_val}")

    # Default = feature extraction unless you pass --fine_tune
    feature_extract = not args.fine_tune
    pretrained = not args.no_pretrained

    try:
        model = build_resnet18(num_classes=num_classes, feature_extract=feature_extract, pretrained=pretrained).to(device)
    except Exception as e:
        # If pretrained download fails, fall back to non-pretrained so you can still run end-to-end.
        print(f"⚠️ Could not load pretrained weights ({e}). Falling back to weights=None.")
        model = build_resnet18(num_classes=num_classes, feature_extract=feature_extract, pretrained=False).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    save_path = Path(args.save_path)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(save_path, model, class_to_idx, epoch, best_val_acc, args)
            print(f"  ✅ Saved best checkpoint to {save_path} (val acc {best_val_acc:.4f})")

    print(f"Done. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {save_path}")


if __name__ == "__main__":
    main()