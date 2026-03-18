from __future__ import annotations

import os
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import argparse
import logging
import math
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from torchvision import models, datasets, transforms
from torchvision.models import ResNet50_Weights


# -----------------------------
# Losses
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


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
# Face-friendlier augmentation
# -----------------------------
def build_transforms(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.08,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tfm, val_tfm


# -----------------------------
# Stratified split
# -----------------------------
def stratified_split_indices(
    targets: List[int],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = torch.Generator().manual_seed(seed)
    by_class: Dict[int, List[int]] = {}

    for idx, y in enumerate(targets):
        by_class.setdefault(y, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for _, idxs in by_class.items():
        perm = torch.randperm(len(idxs), generator=rng).tolist()
        shuffled = [idxs[i] for i in perm]

        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


# -----------------------------
# DataLoader
# -----------------------------
def make_loaders(
    data_dir: str,
    train_subdir: str,
    batch_size: int,
    img_size: int,
    val_ratio: float,
    num_workers: int,
    seed: int,
    device: torch.device,
):
    train_dir = Path(data_dir) / train_subdir
    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    train_tfm, val_tfm = build_transforms(img_size)

    train_full = datasets.ImageFolder(str(train_dir), transform=train_tfm)
    val_full = datasets.ImageFolder(str(train_dir), transform=val_tfm)

    class_names = train_full.classes
    class_to_idx = train_full.class_to_idx
    targets = list(train_full.targets)

    train_idx, val_idx = stratified_split_indices(targets, val_ratio=val_ratio, seed=seed)

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    n_train = len(train_ds)
    n_val = len(val_ds)

    # train class counts after split
    train_targets = [targets[i] for i in train_idx]
    train_class_counts = torch.bincount(
        torch.tensor(train_targets, dtype=torch.long),
        minlength=len(class_names),
    )

    pin = device.type == "cuda"
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    return (
        train_loader,
        val_loader,
        class_names,
        class_to_idx,
        n_train,
        n_val,
        train_class_counts,
    )


# -----------------------------
# Mixup
# -----------------------------
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    lam = torch.distributions.Beta(
        torch.tensor(alpha, device=x.device),
        torch.tensor(alpha, device=x.device),
    ).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# -----------------------------
# Model
# -----------------------------
def build_model(
    num_classes: int,
    feature_extract: bool = True,
    dropout: float = 0.2,
    pretrained: bool = True,
) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if feature_extract:
        for p in model.parameters():
            p.requires_grad = False

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


def unfreeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
    print("  🔓 All layers unfrozen for fine-tuning.")


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# Logging
# -----------------------------
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_resnet50_finetune")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


# -----------------------------
# Loss factory
# -----------------------------
def build_criterion(
    loss_name: str,
    label_smooth: float,
    focal_gamma: float,
    class_weights: torch.Tensor | None,
) -> nn.Module:
    if loss_name == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            label_smoothing=label_smooth,
        )
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smooth,
    )


# -----------------------------
# One epoch
# -----------------------------
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    progress_label: str = "",
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_seen = 0.0, 0, 0

    amp_ctx = autocast if (use_amp and device.type == "cuda") else nullcontext

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(
            loader,
            desc=progress_label or ("train" if is_train else "val"),
            dynamic_ncols=True,
            leave=False,
            unit="batch",
        )

        for x, y in pbar:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with amp_ctx():
                if is_train and use_mixup:
                    x_mixed, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)
                    logits = model(x_mixed)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

                    # only approximate under mixup
                    preds = torch.argmax(logits, dim=1)
                    batch_correct = (preds == y).sum().item()
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
                    preds = torch.argmax(logits, dim=1)
                    batch_correct = (preds == y).sum().item()

            if is_train:
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_correct += batch_correct
            total_seen += y.size(0)

            pbar.set_postfix(
                loss=f"{total_loss / max(total_seen, 1):.4f}",
                acc=f"{total_correct / max(total_seen, 1):.4f}",
            )

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


# -----------------------------
# Checkpoint
# -----------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    epoch: int,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "args": vars(args),
        },
        str(path),
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="STA314 training script (ResNet50 + better head + stratified split)"
    )

    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_subdir", type=str, default="train")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_epochs", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    # Techniques
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--label_smooth", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no_pretrained", action="store_true")

    # Better stuff
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Cross-entropy is the safer default; try focal if class imbalance seems real.",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--class_weights",
        action="store_true",
        help="Use inverse-frequency weights from the training split.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision on CUDA.",
    )

    # Outputs
    parser.add_argument("--save_path", type=str, default="checkpoints/best_resnet50.pt")
    parser.add_argument("--log_path", type=str, default="logs/train_resnet50_finetune.log")

    args = parser.parse_args()
    logger = setup_logger(Path(args.log_path))

    if args.full_finetune and args.fine_tune:
        print("⚠️ Both --full_finetune and --fine_tune were set. Using --full_finetune mode.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info(f"Training started at {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"Device: {device}")

    print(f"Device: {device}")
    print(
        f"Backbone: ResNet50 | Mixup: {args.mixup} | "
        f"Label Smooth: {args.label_smooth} | Dropout: {args.dropout} | "
        f"Loss: {args.loss} | Full FT: {args.full_finetune} | AMP: {args.amp}"
    )

    (
        train_loader,
        val_loader,
        class_names,
        class_to_idx,
        n_train,
        n_val,
        train_class_counts,
    ) = make_loaders(
        data_dir=args.data_dir,
        train_subdir=args.train_subdir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )

    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {n_train} | Val: {n_val}")
    print(f"Train class counts: {dict(zip(class_names, train_class_counts.tolist()))}")

    logger.info(f"Classes ({num_classes}): {class_names}")
    logger.info(
        f"Train samples: {n_train} | Val samples: {n_val} | "
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}"
    )
    logger.info(f"Train class counts: {dict(zip(class_names, train_class_counts.tolist()))}")

    model = build_model(
        num_classes=num_classes,
        feature_extract=not args.full_finetune,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)

    weight_tensor = None
    if args.class_weights:
        counts = train_class_counts.float()
        weights = counts.sum() / (len(counts) * counts.clamp_min(1.0))
        weight_tensor = weights.to(device)
        print(f"Using class weights: {weights.tolist()}")
        logger.info(f"Using class weights: {weights.tolist()}")

    criterion = build_criterion(
        loss_name=args.loss,
        label_smooth=args.label_smooth,
        focal_gamma=args.focal_gamma,
        class_weights=weight_tensor,
    )

    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))

    save_path = Path(args.save_path)
    best_val_acc = -1.0

    if args.full_finetune:
        total_epochs = args.epochs

        print(f"\n{'=' * 60}")
        print("Full Fine-tune Mode: all ResNet50 layers trainable from epoch 1")
        print(f"Training for {total_epochs} epochs | lr={args.finetune_lr}")
        print(f"  Trainable params: {count_trainable(model):,}")
        print(f"{'=' * 60}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.finetune_lr,
            weight_decay=args.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
            if args.lr_scheduler else None
        )

        for ep in range(1, total_epochs + 1):
            epoch_start = time.perf_counter()

            tr_loss, tr_acc = run_one_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                use_mixup=args.mixup,
                mixup_alpha=args.mixup_alpha,
                progress_label=f"FULL train {ep}/{total_epochs}",
                scaler=scaler,
                use_amp=args.amp,
            )
            vl_loss, vl_acc = run_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                progress_label=f"FULL val {ep}/{total_epochs}",
                scaler=None,
                use_amp=args.amp,
            )

            if scheduler:
                scheduler.step()

            epoch_time = time.perf_counter() - epoch_start
            msg = (
                f"[FULL] Ep {ep:02d}/{total_epochs} | "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"time={epoch_time:.1f}s"
            )
            print(msg)
            logger.info(msg)

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                save_checkpoint(save_path, model, class_to_idx, ep, best_val_acc, args)
                print(f"  ✅ Best val acc {best_val_acc:.4f} → {save_path}")
                logger.info(
                    f"Best val acc improved to {best_val_acc:.4f}; checkpoint saved to {save_path}"
                )

    else:
        stage1_epochs = args.freeze_epochs if args.fine_tune else args.epochs

        print(f"\n{'=' * 60}")
        print(f"Stage 1: Head-only ({stage1_epochs} epochs | lr={args.lr})")
        print(f"  Trainable params: {count_trainable(model):,}")
        print(f"{'=' * 60}")

        opt1 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        sch1 = (
            torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=stage1_epochs)
            if args.lr_scheduler else None
        )

        for ep in range(1, stage1_epochs + 1):
            epoch_start = time.perf_counter()

            tr_loss, tr_acc = run_one_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=opt1,
                use_mixup=False,
                progress_label=f"S1 train {ep}/{stage1_epochs}",
                scaler=scaler,
                use_amp=args.amp,
            )
            vl_loss, vl_acc = run_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                progress_label=f"S1 val {ep}/{stage1_epochs}",
                scaler=None,
                use_amp=args.amp,
            )

            if sch1:
                sch1.step()

            epoch_time = time.perf_counter() - epoch_start
            msg = (
                f"[S1] Ep {ep:02d}/{stage1_epochs} | "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | "
                f"lr={opt1.param_groups[0]['lr']:.2e} | "
                f"time={epoch_time:.1f}s"
            )
            print(msg)
            logger.info(msg)

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                save_checkpoint(save_path, model, class_to_idx, ep, best_val_acc, args)
                print(f"  ✅ Best val acc {best_val_acc:.4f} → {save_path}")
                logger.info(
                    f"Best val acc improved to {best_val_acc:.4f}; checkpoint saved to {save_path}"
                )

        if args.fine_tune:
            stage2_epochs = args.epochs - stage1_epochs
            if stage2_epochs <= 0:
                print("\n⚠️ --freeze_epochs >= --epochs, no Stage 2. Increase --epochs.")
            else:
                print(f"\n{'=' * 60}")
                print(f"Stage 2: Full fine-tune ({stage2_epochs} epochs | lr={args.finetune_lr})")
                unfreeze_model(model)
                print(f"  Trainable params: {count_trainable(model):,}")
                print(f"{'=' * 60}")

                opt2 = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.finetune_lr,
                    weight_decay=args.weight_decay,
                )
                sch2 = (
                    torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=stage2_epochs)
                    if args.lr_scheduler else None
                )

                for ep in range(1, stage2_epochs + 1):
                    epoch_start = time.perf_counter()

                    tr_loss, tr_acc = run_one_epoch(
                        model,
                        train_loader,
                        criterion,
                        device,
                        optimizer=opt2,
                        use_mixup=args.mixup,
                        mixup_alpha=args.mixup_alpha,
                        progress_label=f"S2 train {ep}/{stage2_epochs}",
                        scaler=scaler,
                        use_amp=args.amp,
                    )
                    vl_loss, vl_acc = run_one_epoch(
                        model,
                        val_loader,
                        criterion,
                        device,
                        progress_label=f"S2 val {ep}/{stage2_epochs}",
                        scaler=None,
                        use_amp=args.amp,
                    )

                    if sch2:
                        sch2.step()

                    global_ep = stage1_epochs + ep
                    epoch_time = time.perf_counter() - epoch_start
                    msg = (
                        f"[S2] Ep {global_ep:02d}/{args.epochs} | "
                        f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                        f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | "
                        f"lr={opt2.param_groups[0]['lr']:.2e} | "
                        f"time={epoch_time:.1f}s"
                    )
                    print(msg)
                    logger.info(msg)

                    if vl_acc > best_val_acc:
                        best_val_acc = vl_acc
                        save_checkpoint(save_path, model, class_to_idx, global_ep, best_val_acc, args)
                        print(f"  ✅ Best val acc {best_val_acc:.4f} → {save_path}")
                        logger.info(
                            f"Best val acc improved to {best_val_acc:.4f}; checkpoint saved to {save_path}"
                        )

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {save_path}")
    logger.info(f"Done. Best val accuracy: {best_val_acc:.4f}")
    logger.info(f"Best checkpoint saved to: {save_path}")
    logger.info(f"Training log saved to: {args.log_path}")


if __name__ == "__main__":
    main()