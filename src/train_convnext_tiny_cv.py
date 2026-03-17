from __future__ import annotations

import os
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import argparse
import csv
import json
import logging
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.models.convnext import LayerNorm2d
from tqdm.auto import tqdm


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



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



def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_convnext_tiny_cv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


# -----------------------------
# Data
# -----------------------------
def build_transforms(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.Resize((img_size + 24, img_size + 24)),
        transforms.RandomResizedCrop(img_size, scale=(0.80, 1.00), ratio=(0.90, 1.10)),
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



def stratified_kfold_indices(
    targets: Sequence[int],
    n_splits: int,
    seed: int,
) -> List[Tuple[List[int], List[int]]]:
    by_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(int(y), []).append(idx)

    rng = random.Random(seed)
    folds: List[List[int]] = [[] for _ in range(n_splits)]

    for _, idxs in sorted(by_class.items()):
        idxs = idxs[:]
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            folds[i % n_splits].append(idx)

    all_indices = set(range(len(targets)))
    split_indices: List[Tuple[List[int], List[int]]] = []
    for fold_id in range(n_splits):
        val_idx = sorted(folds[fold_id])
        train_idx = sorted(all_indices.difference(val_idx))
        split_indices.append((train_idx, val_idx))
    return split_indices



def make_loaders_for_fold(
    data_dir: str,
    train_subdir: str,
    batch_size: int,
    img_size: int,
    num_workers: int,
    device: torch.device,
    train_idx: List[int],
    val_idx: List[int],
):
    train_dir = Path(data_dir) / train_subdir
    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    train_tfm, val_tfm = build_transforms(img_size)
    train_full = datasets.ImageFolder(str(train_dir), transform=train_tfm)
    val_full = datasets.ImageFolder(str(train_dir), transform=val_tfm)

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    class_names = train_full.classes
    class_to_idx = train_full.class_to_idx
    targets = list(train_full.targets)
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
        len(train_ds),
        len(val_ds),
        train_class_counts,
    )


# -----------------------------
# Mixup
# -----------------------------
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(
        torch.tensor(alpha, device=x.device),
        torch.tensor(alpha, device=x.device),
    ).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    return mixed_x, y, y[idx], lam



def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# -----------------------------
# Model
# -----------------------------
def build_model(
    num_classes: int,
    feature_extract: bool = True,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)

    if feature_extract:
        for p in model.parameters():
            p.requires_grad = False

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



def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
    print("  🔓 All ConvNeXt layers unfrozen for fine-tuning.")



def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# Train / Eval
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

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

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
# Checkpointing
# -----------------------------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    fold_id: int,
    epoch: int,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "fold_id": fold_id,
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
        description="ConvNeXt-Tiny 5-fold CV training for pet facial expression classification"
    )

    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_subdir", type=str, default="train")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n_folds", type=int, default=5)

    # Training
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--freeze_epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--amp", action="store_true")

    # Techniques
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--label_smooth", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--class_weights", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")

    # Outputs
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/convnext_tiny_cv_seed2025",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/train_convnext_tiny_cv.log",
    )

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(Path(args.log_path))

    set_seed(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info(f"Training started at {datetime.now().isoformat(timespec='seconds')}")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    print(f"Device: {device}")
    print(
        f"Backbone: ConvNeXt-Tiny | folds={args.n_folds} | epochs={args.epochs} | "
        f"Mixup={args.mixup} | Label Smooth={args.label_smooth} | Dropout={args.dropout} | AMP={args.amp}"
    )

    train_dir = Path(args.data_dir) / args.train_subdir
    probe_ds = datasets.ImageFolder(str(train_dir))
    class_names = probe_ds.classes
    class_to_idx = probe_ds.class_to_idx
    targets = list(probe_ds.targets)

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Total training images: {len(targets)}")
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(f"Total training images: {len(targets)}")

    folds = stratified_kfold_indices(targets, n_splits=args.n_folds, seed=args.seed)

    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))
    fold_rows: List[Dict[str, float | int | str]] = []
    best_paths: List[str] = []

    for fold_num, (train_idx, val_idx) in enumerate(folds, start=1):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold_num}/{args.n_folds}")
        print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)}")
        print(f"{'=' * 72}")
        logger.info(f"Fold {fold_num}/{args.n_folds} | train={len(train_idx)} val={len(val_idx)}")

        (
            train_loader,
            val_loader,
            _class_names,
            _class_to_idx,
            n_train,
            n_val,
            train_class_counts,
        ) = make_loaders_for_fold(
            data_dir=args.data_dir,
            train_subdir=args.train_subdir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            device=device,
            train_idx=train_idx,
            val_idx=val_idx,
        )

        print(f"Train class counts: {dict(zip(class_names, train_class_counts.tolist()))}")
        logger.info(f"Fold {fold_num} train class counts: {dict(zip(class_names, train_class_counts.tolist()))}")

        model = build_model(
            num_classes=len(class_names),
            feature_extract=True,
            dropout=args.dropout,
            pretrained=not args.no_pretrained,
        ).to(device)

        weight_tensor = None
        if args.class_weights:
            counts = train_class_counts.float()
            weights = counts.sum() / (len(counts) * counts.clamp_min(1.0))
            weight_tensor = weights.to(device)
            print(f"Using class weights: {weights.tolist()}")
            logger.info(f"Fold {fold_num} class weights: {weights.tolist()}")

        criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=args.label_smooth,
        )

        fold_best_val_acc = -1.0
        fold_best_epoch = -1
        fold_ckpt_path = save_dir / f"fold{fold_num}_best.pt"

        # Stage 1: head only
        stage1_epochs = min(args.freeze_epochs, args.epochs)
        if stage1_epochs > 0:
            print(f"\nStage 1 (fold {fold_num}): head-only for {stage1_epochs} epochs | lr={args.lr}")
            print(f"  Trainable params: {count_trainable(model):,}")

            opt1 = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=stage1_epochs)

            for ep in range(1, stage1_epochs + 1):
                epoch_start = time.perf_counter()
                tr_loss, tr_acc = run_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer=opt1,
                    use_mixup=False,
                    progress_label=f"F{fold_num} S1 train {ep}/{stage1_epochs}",
                    scaler=scaler,
                    use_amp=args.amp,
                )
                vl_loss, vl_acc = run_one_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    progress_label=f"F{fold_num} S1 val {ep}/{stage1_epochs}",
                    scaler=None,
                    use_amp=args.amp,
                )
                sch1.step()

                epoch_time = time.perf_counter() - epoch_start
                msg = (
                    f"[Fold {fold_num} S1] Ep {ep:02d}/{stage1_epochs} | "
                    f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                    f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | "
                    f"lr={opt1.param_groups[0]['lr']:.2e} | time={epoch_time:.1f}s"
                )
                print(msg)
                logger.info(msg)

                if vl_acc > fold_best_val_acc:
                    fold_best_val_acc = vl_acc
                    fold_best_epoch = ep
                    save_checkpoint(
                        fold_ckpt_path,
                        model,
                        _class_to_idx,
                        fold_id=fold_num,
                        epoch=ep,
                        best_val_acc=fold_best_val_acc,
                        args=args,
                    )
                    print(f"  ✅ Fold {fold_num} best val acc {fold_best_val_acc:.4f} -> {fold_ckpt_path}")

        # Stage 2: full fine-tune
        stage2_epochs = max(0, args.epochs - stage1_epochs)
        if stage2_epochs > 0:
            print(f"\nStage 2 (fold {fold_num}): full fine-tune for {stage2_epochs} epochs | lr={args.finetune_lr}")
            unfreeze_all(model)
            print(f"  Trainable params: {count_trainable(model):,}")

            opt2 = torch.optim.AdamW(
                model.parameters(),
                lr=args.finetune_lr,
                weight_decay=args.weight_decay,
            )
            sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=stage2_epochs)

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
                    progress_label=f"F{fold_num} S2 train {ep}/{stage2_epochs}",
                    scaler=scaler,
                    use_amp=args.amp,
                )
                vl_loss, vl_acc = run_one_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    progress_label=f"F{fold_num} S2 val {ep}/{stage2_epochs}",
                    scaler=None,
                    use_amp=args.amp,
                )
                sch2.step()

                global_ep = stage1_epochs + ep
                epoch_time = time.perf_counter() - epoch_start
                msg = (
                    f"[Fold {fold_num} S2] Ep {global_ep:02d}/{args.epochs} | "
                    f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                    f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | "
                    f"lr={opt2.param_groups[0]['lr']:.2e} | time={epoch_time:.1f}s"
                )
                print(msg)
                logger.info(msg)

                if vl_acc > fold_best_val_acc:
                    fold_best_val_acc = vl_acc
                    fold_best_epoch = global_ep
                    save_checkpoint(
                        fold_ckpt_path,
                        model,
                        _class_to_idx,
                        fold_id=fold_num,
                        epoch=global_ep,
                        best_val_acc=fold_best_val_acc,
                        args=args,
                    )
                    print(f"  ✅ Fold {fold_num} best val acc {fold_best_val_acc:.4f} -> {fold_ckpt_path}")

        fold_rows.append(
            {
                "fold": fold_num,
                "train_size": n_train,
                "val_size": n_val,
                "best_epoch": fold_best_epoch,
                "best_val_acc": fold_best_val_acc,
                "checkpoint": str(fold_ckpt_path),
            }
        )
        best_paths.append(str(fold_ckpt_path))
        logger.info(
            f"Fold {fold_num} done | best_epoch={fold_best_epoch} | best_val_acc={fold_best_val_acc:.4f} | checkpoint={fold_ckpt_path}"
        )

    mean_val_acc = sum(float(row["best_val_acc"]) for row in fold_rows) / max(len(fold_rows), 1)
    print(f"\n{'=' * 72}")
    print(f"Finished {args.n_folds}-fold CV. Mean best val acc: {mean_val_acc:.4f}")
    print(f"Checkpoints saved under: {save_dir}")
    print(f"{'=' * 72}")
    logger.info(f"Mean best val acc across folds: {mean_val_acc:.4f}")

    results_csv = save_dir / "cv_results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "train_size", "val_size", "best_epoch", "best_val_acc", "checkpoint"],
        )
        writer.writeheader()
        for row in fold_rows:
            writer.writerow(row)

    manifest = {
        "model": "convnext_tiny",
        "n_folds": args.n_folds,
        "seed": args.seed,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "mean_best_val_acc": mean_val_acc,
        "checkpoints": best_paths,
        "args": vars(args),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (save_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved fold summary to: {results_csv}")
    print(f"Saved manifest to: {save_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
