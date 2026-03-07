from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

print(torch.__version__)
print("CUDA in torch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())


def build_transforms(img_size: int = 224) -> transforms.Compose:
    """Standard transforms (works well with pretrained ImageNet backbones)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def make_dataloaders(
    data_dir: str | Path = "data",
    train_subdir: str = "train",
    batch_size: int = 32,
    img_size: int = 224,
    val_ratio: float = 0.15,
    num_workers: int = 2,
    seed: int = 314,
) -> Tuple[DataLoader, DataLoader, List[str], Dict[str, int], int, int]:
    """
    Expects folder structure:
      data/train/<class_name>/*.jpg

    Returns:
      train_loader, val_loader, class_names, class_to_idx, n_train, n_val
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / train_subdir

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Expected training folder not found: {train_dir.resolve()}\n"
            f"Your structure should look like: data/train/<class_name>/*.jpg"
        )

    tfm = build_transforms(img_size=img_size)

    full_ds = datasets.ImageFolder(root=str(train_dir), transform=tfm)
    class_names = full_ds.classes
    class_to_idx = full_ds.class_to_idx

    n_total = len(full_ds)
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val

    # Reproducible split
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    # pin_memory is useful mainly for CUDA; on MPS it does nothing and warns.
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, class_names, class_to_idx, n_train, n_val


def main() -> None:
    train_loader, val_loader, class_names, class_to_idx, n_train, n_val = make_dataloaders(
        data_dir="data",
        train_subdir="train",
        batch_size=32,
        img_size=224,
        val_ratio=0.15,
        num_workers=2,
        seed=314,
    )

    print(f"Number of samples (train): {n_train}")
    print(f"Number of samples (val):   {n_val}")
    print(f"Class names: {class_names}")
    print(f"Class -> index mapping: {class_to_idx}")

    x, y = next(iter(train_loader))
    print(f"One batch X shape: {tuple(x.shape)}")  # (B, 3, 224, 224)
    print(f"One batch y shape: {tuple(y.shape)}")  # (B,)
    print(f"y labels example: {y[:10].tolist()}")


if __name__ == "__main__":
    main()