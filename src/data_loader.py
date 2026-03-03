from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_transforms(img_size: int = 224) -> transforms.Compose:
    """Basic transforms for CNN training."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # ImageNet normalization (standard for pretrained backbones)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def make_dataloaders(
    data_dir: str | Path = "data",
    train_subdir: str = "train",
    batch_size: int = 32,
    img_size: int = 224,
    val_ratio: float = 0.15,
    num_workers: int = 2,
    seed: int = 314,
) -> Tuple[DataLoader, DataLoader, List[str], int, int]:
    """
    Returns:
      train_loader, val_loader, class_names, n_train, n_val
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / train_subdir  # expects data/train/<class folders>

    if not train_dir.exists():
        raise FileNotFoundError(f"Expected folder not found: {train_dir.resolve()}")

    tfm = build_transforms(img_size=img_size)

    full_ds = datasets.ImageFolder(root=str(train_dir), transform=tfm)
    class_names = full_ds.classes  # ['Angry', 'Happy', 'Sad'] (alphabetical)
    n_total = len(full_ds)

    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val

    # Reproducible split
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names, n_train, n_val


def main() -> None:
    train_loader, val_loader, class_names, n_train, n_val = make_dataloaders(
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

    # Peek one batch
    x, y = next(iter(train_loader))
    print(f"One batch X shape: {tuple(x.shape)}")  # (B, 3, 224, 224)
    print(f"One batch y shape: {tuple(y.shape)}")  # (B,)
    print(f"y labels example: {y[:10].tolist()}")


if __name__ == "__main__":
    main()