from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models.convnext import LayerNorm2d


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


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


@st.cache_resource
def load_ensemble(checkpoint_dir: str, device_str: str):
    device = torch.device(device_str)
    ckpt_dir = Path(checkpoint_dir)

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpt_paths = sorted(ckpt_dir.glob("fold*_best.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No fold*_best.pt checkpoints found in: {ckpt_dir}")

    models_list: list[nn.Module] = []
    class_to_idx = None

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device)

        this_class_to_idx = ckpt["class_to_idx"]
        if class_to_idx is None:
            class_to_idx = this_class_to_idx
        elif this_class_to_idx != class_to_idx:
            raise ValueError(f"class_to_idx mismatch in {ckpt_path}")

        args = ckpt.get("args", {})
        dropout = args.get("dropout", 0.3)

        model = build_convnext_tiny(
            num_classes=len(class_to_idx),
            dropout=dropout,
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
        models_list.append(model)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return models_list, class_to_idx, idx_to_class, [str(p) for p in ckpt_paths]


@torch.no_grad()
def predict_ensemble(
    models_list: list[nn.Module],
    img: Image.Image,
    tfm,
    device: torch.device,
    use_flip_tta: bool = True,
) -> torch.Tensor:
    x = tfm(img).unsqueeze(0).to(device)

    x_list = [x]
    if use_flip_tta:
        x_list.append(torch.flip(x, dims=[3]))  # horizontal flip

    probs_total = None
    num_preds = 0

    for model in models_list:
        for x_aug in x_list:
            logits = model(x_aug)
            probs = torch.softmax(logits, dim=1)
            probs_total = probs if probs_total is None else probs_total + probs
            num_preds += 1

    probs_mean = (probs_total / num_preds).squeeze(0).cpu()
    return probs_mean


def find_happy_index(class_to_idx: dict[str, int]) -> int | None:
    for name, idx in class_to_idx.items():
        if name.lower() == "happy":
            return idx
    return None


def main():
    st.set_page_config(page_title="Pet Expression Ensemble Tester", page_icon="🐶", layout="centered")
    st.title("🐶 Pet Expression Ensemble Tester")
    st.caption("Loads multiple ConvNeXt fold checkpoints and averages predictions.")

    ckpt_dir_default = "checkpoints/convnext_tiny_cv_seed2025"
    checkpoint_dir = st.text_input("Checkpoint directory", value=ckpt_dir_default)

    device = get_device()
    st.write(f"Using device: `{device.type}`")

    try:
        models_list, class_to_idx, idx_to_class, loaded_paths = load_ensemble(checkpoint_dir, str(device))
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.write(f"Loaded **{len(models_list)}** checkpoints.")
    with st.expander("Loaded checkpoint files"):
        for p in loaded_paths:
            st.write(p)

    st.write("Classes:", list(class_to_idx.keys()))

    threshold = st.slider("Happy threshold", 0.0, 1.0, 0.50, 0.01)
    use_flip_tta = st.checkbox("Use horizontal flip TTA", value=True)

    uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        st.info("Upload a dog or cat pic to test it 🙂")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your upload", use_container_width=True)

    tfm = build_transform(224)
    probs = predict_ensemble(models_list, img, tfm, device, use_flip_tta=use_flip_tta)

    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx_to_class[pred_idx]
    pred_conf = float(probs[pred_idx].item())

    st.subheader("Prediction")
    st.write(f"**Top guess:** `{pred_label}` (confidence: **{pred_conf:.3f}**)")

    happy_idx = find_happy_index(class_to_idx)
    if happy_idx is not None:
        happy_prob = float(probs[happy_idx].item())
        verdict = "HAPPY ✅" if happy_prob >= threshold else "NOT HAPPY ❌"
        st.write(f"**Happy probability:** **{happy_prob:.3f}** → **{verdict}**")
        st.progress(min(max(happy_prob, 0.0), 1.0))

    st.subheader("All class probabilities")
    rows = [(idx_to_class[i], float(probs[i].item())) for i in range(len(probs))]
    rows.sort(key=lambda x: x[1], reverse=True)
    for label, p in rows:
        st.write(f"- `{label}`: **{p:.3f}**")


if __name__ == "__main__":
    main()