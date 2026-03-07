from __future__ import annotations

"""
STA314 training script optimised for small datasets (~700 images).
Improvements over the baseline:
  1. EfficientNet-B2 backbone (stronger than ResNet18, fewer params)
  2. Strong data augmentation (most effective boost for small datasets)
  3. Label smoothing (reduces overfitting)
  4. Mixup augmentation (further regularisation)
  5. Two-stage fine-tuning + cosine LR scheduler
  6. Dropout regularisation

Recommended usage:
  python train.py \
    --data_dir "C:\\your\\full\\path\\src\\data" \
    --train_subdir train \
    --fine_tune --lr_scheduler --mixup \
    --num_workers 0
"""

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
from torch.utils.data import DataLoader, random_split
from torchvision import models, datasets, transforms
from torchvision.models import EfficientNet_B2_Weights, ResNet18_Weights


# Device

def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cpu":  return torch.device("cpu")
    if device_arg == "cuda": return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device_arg == "mps":  return torch.device("mps")  if torch.backends.mps.is_available() else torch.device("cpu")
    if torch.cuda.is_available():          return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# Strong data augmentation

def build_transforms(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tfm, val_tfm

# DataLoader

def make_loaders(data_dir: str, train_subdir: str, batch_size: int,
                 img_size: int, val_ratio: float, num_workers: int, seed: int):
    train_dir = Path(data_dir) / train_subdir
    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    train_tfm, val_tfm = build_transforms(img_size)

    full_ds      = datasets.ImageFolder(str(train_dir), transform=train_tfm)
    class_names  = full_ds.classes
    class_to_idx = full_ds.class_to_idx

    n_val   = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    # val set uses no augmentation
    val_ds.dataset = datasets.ImageFolder(str(train_dir), transform=val_tfm)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, class_names, class_to_idx, n_train, n_val


# Mixup

def mixup_data(x, y, alpha=0.3):
    lam = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha)).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# Model

def build_model(num_classes: int, backbone: str = "efficientnet",
                feature_extract: bool = True, dropout: float = 0.4) -> nn.Module:
    if backbone == "efficientnet":
        model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        if feature_extract:
            for p in model.parameters():
                p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        if feature_extract:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
    return model


def unfreeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
    print("  🔓 All layers unfrozen for fine-tuning.")

def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# One epoch

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.3,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_seen = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if is_train and use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                ref_y = y_a
            else:
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                ref_y = y

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * y.size(0)
            total_correct += (torch.argmax(logits, 1) == ref_y).sum().item()
            total_seen    += y.size(0)

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


# Checkpoint

def save_checkpoint(path: Path, model: nn.Module, class_to_idx: Dict[str, int],
                    epoch: int, best_val_acc: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx":     class_to_idx,
        "epoch":            epoch,
        "best_val_acc":     best_val_acc,
        "args":             vars(args),
    }, str(path))


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="STA314 training script (EfficientNet-B2 + Mixup + Label Smoothing)")

    # Data
    parser.add_argument("--data_dir",      type=str,   default="data")
    parser.add_argument("--train_subdir",  type=str,   default="train")
    parser.add_argument("--val_ratio",     type=float, default=0.15)
    parser.add_argument("--img_size",      type=int,   default=224)
    parser.add_argument("--batch_size",    type=int,   default=16,
                        help="Small batch size gives more diverse augmentation for small datasets")
    parser.add_argument("--num_workers",   type=int,   default=0)
    parser.add_argument("--seed",          type=int,   default=314)

    # Training
    parser.add_argument("--epochs",        type=int,   default=30,
                        help="Total epochs; small datasets need more epochs to converge")
    parser.add_argument("--lr",            type=float, default=1e-3,
                        help="Stage 1 learning rate (head only)")
    parser.add_argument("--finetune_lr",   type=float, default=2e-5,
                        help="Stage 2 learning rate (all layers); should be 10-50x smaller than --lr")
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--freeze_epochs", type=int,   default=8,
                        help="Number of Stage 1 epochs to train head only before unfreezing")
    parser.add_argument("--device",        type=str,   default="auto",
                        choices=["auto","cpu","cuda","mps"])

    # Techniques
    parser.add_argument("--fine_tune",     action="store_true")
    parser.add_argument("--lr_scheduler",  action="store_true")
    parser.add_argument("--mixup",         action="store_true",
                        help="Enable Mixup augmentation in Stage 2")
    parser.add_argument("--mixup_alpha",   type=float, default=0.3)
    parser.add_argument("--label_smooth",  type=float, default=0.1)
    parser.add_argument("--dropout",       type=float, default=0.4)
    parser.add_argument("--backbone",      type=str,   default="efficientnet",
                        choices=["efficientnet","resnet18"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--save_path",     type=str,   default="checkpoints/best.pt")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone} | Mixup: {args.mixup} | "
          f"Label Smooth: {args.label_smooth} | Dropout: {args.dropout}")

    # ---- Data ----
    train_loader, val_loader, class_names, class_to_idx, n_train, n_val = make_loaders(
        data_dir=args.data_dir, train_subdir=args.train_subdir,
        batch_size=args.batch_size, img_size=args.img_size,
        val_ratio=args.val_ratio, num_workers=args.num_workers, seed=args.seed,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {n_train} | Val: {n_val}")

    # ---- Model ----
    try:
        model = build_model(num_classes, backbone=args.backbone,
                            feature_extract=True, dropout=args.dropout).to(device)
    except Exception as e:
        print(f"⚠️ EfficientNet failed to load ({e}), falling back to ResNet18.")
        model = build_model(num_classes, backbone="resnet18",
                            feature_extract=True, dropout=args.dropout).to(device)

    criterion    = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    save_path    = Path(args.save_path)
    best_val_acc = -1.0


    # STAGE 1 — Head only (backbone frozen)

    stage1_epochs = args.freeze_epochs if args.fine_tune else args.epochs

    print(f"\n{'='*60}")
    print(f"Stage 1: Head-only  ({stage1_epochs} epochs | lr={args.lr})")
    print(f"  Trainable params: {count_trainable(model):,}")
    print(f"{'='*60}")

    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    sch1 = (torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=stage1_epochs)
            if args.lr_scheduler else None)

    for ep in range(1, stage1_epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, device, opt1,
                                        use_mixup=False)
        vl_loss, vl_acc = run_one_epoch(model, val_loader, criterion, device)
        if sch1: sch1.step()

        print(f"[S1] Ep {ep:02d}/{stage1_epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | "
              f"lr={opt1.param_groups[0]['lr']:.2e}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_checkpoint(save_path, model, class_to_idx, ep, best_val_acc, args)
            print(f"  ✅ Best val acc {best_val_acc:.4f} → {save_path}")


    # STAGE 2 — Full fine-tuning
    if args.fine_tune:
        stage2_epochs = args.epochs - stage1_epochs
        if stage2_epochs <= 0:
            print("\n⚠️  --freeze_epochs >= --epochs, no Stage 2. Increase --epochs.")
        else:
            print(f"\n{'='*60}")
            print(f"Stage 2: Full fine-tune  ({stage2_epochs} epochs | lr={args.finetune_lr})")
            unfreeze_model(model)
            print(f"  Trainable params: {count_trainable(model):,}")
            print(f"{'='*60}")

            opt2 = torch.optim.AdamW(
                model.parameters(),
                lr=args.finetune_lr, weight_decay=args.weight_decay
            )
            sch2 = (torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=stage2_epochs)
                    if args.lr_scheduler else None)

            for ep in range(1, stage2_epochs + 1):
                tr_loss, tr_acc = run_one_epoch(
                    model, train_loader, criterion, device, opt2,
                    use_mixup=args.mixup, mixup_alpha=args.mixup_alpha
                )
                vl_loss, vl_acc = run_one_epoch(model, val_loader, criterion, device)
                if sch2: sch2.step()

                global_ep = stage1_epochs + ep
                print(f"[S2] Ep {global_ep:02d}/{args.epochs} | "
                      f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                      f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | "
                      f"lr={opt2.param_groups[0]['lr']:.2e}")

                if vl_acc > best_val_acc:
                    best_val_acc = vl_acc
                    save_checkpoint(save_path, model, class_to_idx, global_ep, best_val_acc, args)
                    print(f"  ✅ Best val acc {best_val_acc:.4f} → {save_path}")

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()
