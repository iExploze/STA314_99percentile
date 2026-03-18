from __future__ import annotations

import os
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from tqdm.auto import tqdm

COCO_CAT = 17
COCO_DOG = 18


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


def list_image_files(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


def expand_box_to_square(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    margin: float,
    face_bias: float,
) -> Tuple[int, int, int, int]:
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    # Bias crop upward a bit so the square favors the head/face region.
    cx = (x1 + x2) / 2.0
    cy = y1 + face_bias * h

    size = max(w, h) * (1.0 + margin)
    half = size / 2.0

    left = cx - half
    top = cy - half
    right = cx + half
    bottom = cy + half

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img_w:
        left -= (right - img_w)
        right = img_w
    if bottom > img_h:
        top -= (bottom - img_h)
        bottom = img_h

    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(img_w, int(round(right)))
    bottom = min(img_h, int(round(bottom)))

    if right <= left:
        right = min(img_w, left + 1)
    if bottom <= top:
        bottom = min(img_h, top + 1)

    return left, top, right, bottom


def fallback_center_crop(img_w: int, img_h: int, keep_ratio: float) -> Tuple[int, int, int, int]:
    size = int(round(min(img_w, img_h) * keep_ratio))
    size = max(1, min(size, img_w, img_h))
    left = (img_w - size) // 2
    top = (img_h - size) // 2
    return left, top, left + size, top + size


@torch.no_grad()
def detect_crop_box(
    model,
    image: Image.Image,
    device: torch.device,
    score_thresh: float,
    margin: float,
    face_bias: float,
    fallback_keep_ratio: float,
) -> Tuple[int, int, int, int, str]:
    img_w, img_h = image.size
    x = F.to_tensor(image).to(device)
    outputs = model([x])[0]

    boxes = outputs["boxes"].detach().cpu()
    labels = outputs["labels"].detach().cpu()
    scores = outputs["scores"].detach().cpu()

    best_idx = None
    best_score = -1.0
    for i, (label, score) in enumerate(zip(labels.tolist(), scores.tolist())):
        if label in (COCO_CAT, COCO_DOG) and score >= score_thresh and score > best_score:
            best_idx = i
            best_score = score

    if best_idx is not None:
        x1, y1, x2, y2 = boxes[best_idx].tolist()
        crop = expand_box_to_square(x1, y1, x2, y2, img_w, img_h, margin=margin, face_bias=face_bias)
        return crop[0], crop[1], crop[2], crop[3], f"detected(score={best_score:.3f})"

    left, top, right, bottom = fallback_center_crop(img_w, img_h, keep_ratio=fallback_keep_ratio)
    return left, top, right, bottom, "fallback_center"


def process_images(
    input_paths: Sequence[Path],
    input_root: Path,
    output_root: Path,
    model,
    device: torch.device,
    score_thresh: float,
    margin: float,
    face_bias: float,
    fallback_keep_ratio: float,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    detected = 0
    fallback = 0

    for path in tqdm(input_paths, desc=f"Cropping {input_root.name}", dynamic_ncols=True):
        rel = path.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(path).convert("RGB")
        left, top, right, bottom, mode = detect_crop_box(
            model=model,
            image=image,
            device=device,
            score_thresh=score_thresh,
            margin=margin,
            face_bias=face_bias,
            fallback_keep_ratio=fallback_keep_ratio,
        )
        cropped = image.crop((left, top, right, bottom))
        cropped.save(out_path, quality=95)

        if mode.startswith("detected"):
            detected += 1
        else:
            fallback += 1

    total = max(1, len(input_paths))
    print(f"\n{input_root.name}: {len(input_paths)} images")
    print(f"  detected pet crop: {detected} ({detected / total:.1%})")
    print(f"  fallback center crop: {fallback} ({fallback / total:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an automatically cropped pet-region dataset using COCO cat/dog detection."
    )
    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data_petcrop")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--score_thresh", type=float, default=0.50)
    parser.add_argument("--margin", type=float, default=0.35, help="Expand detected pet box by this fraction.")
    parser.add_argument(
        "--face_bias",
        type=float,
        default=0.30,
        help="Vertical bias inside detected pet box. Lower values bias upward toward head/face.",
    )
    parser.add_argument(
        "--fallback_keep_ratio",
        type=float,
        default=0.80,
        help="When no pet is detected, crop this much of the shorter image side from center.",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    train_in = input_dir / "train"
    test_in = input_dir / "test"

    if not train_in.exists():
        raise FileNotFoundError(f"Missing training folder: {train_in}")
    if not test_in.exists():
        print(f"Warning: test folder not found at {test_in}; only training data will be processed.")

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=args.score_thresh)
    model.to(device)
    model.eval()

    train_paths = list_image_files(train_in)
    if not train_paths:
        raise RuntimeError(f"No images found under {train_in}")
    process_images(
        input_paths=train_paths,
        input_root=train_in,
        output_root=output_dir / "train",
        model=model,
        device=device,
        score_thresh=args.score_thresh,
        margin=args.margin,
        face_bias=args.face_bias,
        fallback_keep_ratio=args.fallback_keep_ratio,
    )

    if test_in.exists():
        test_paths = list_image_files(test_in)
        if test_paths:
            process_images(
                input_paths=test_paths,
                input_root=test_in,
                output_root=output_dir / "test",
                model=model,
                device=device,
                score_thresh=args.score_thresh,
                margin=args.margin,
                face_bias=args.face_bias,
                fallback_keep_ratio=args.fallback_keep_ratio,
            )

    print(f"\nDone. Cropped dataset written to: {output_dir}")
    print("This is an automatic pet-region crop, not a guaranteed face detector.")


if __name__ == "__main__":
    main()
