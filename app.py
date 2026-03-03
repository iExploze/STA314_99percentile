from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


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


def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@st.cache_resource
def load_model(ckpt_path: str, device_str: str):
    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device)

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_resnet18(num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_to_idx, idx_to_class


@torch.no_grad()
def predict(model: nn.Module, img: Image.Image, tfm, device: torch.device) -> torch.Tensor:
    x = tfm(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()  # (C,)
    return probs


def main():
    st.set_page_config(page_title="Is my dog happy?", page_icon="🐶", layout="centered")
    st.title("🐶 Is my dog happy?")
    st.caption("Drag & drop a photo. This is for fun — model may be wrong on dogs depending on your training data.")

    ckpt_default = "checkpoints/best.pt"
    ckpt_path = st.text_input("Checkpoint path", value=ckpt_default)

    if not Path(ckpt_path).exists():
        st.error(f"Checkpoint not found: {Path(ckpt_path).resolve()}")
        st.stop()

    device = get_device()
    st.write(f"Using device: `{device.type}`")

    model, class_to_idx, idx_to_class = load_model(ckpt_path, str(device))
    classes = list(class_to_idx.keys())
    st.write("Classes:", classes)

    # Threshold slider for “Happy / Not Happy”
    threshold = st.slider("Happy threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        st.info("Upload a dog pic to get a prediction 🙂")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your upload", use_container_width=True)

    tfm = build_transform(img_size=224)
    probs = predict(model, img, tfm, device)

    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx_to_class[pred_idx]
    pred_conf = float(probs[pred_idx].item())

    happy_idx = class_to_idx.get("Happy", None)
    happy_prob = float(probs[happy_idx].item()) if happy_idx is not None else None

    st.subheader("Prediction")
    st.write(f"**Top guess:** `{pred_label}` (confidence: **{pred_conf:.3f}**)")

    if happy_prob is not None:
        verdict = "HAPPY ✅" if happy_prob >= threshold else "NOT HAPPY ❌"
        st.write(f"**Happy probability:** **{happy_prob:.3f}** → **{verdict}**")
        st.progress(min(max(happy_prob, 0.0), 1.0))
    else:
        st.warning("This model doesn't have a 'Happy' class, so I can't do happy/not-happy mode.")

    st.subheader("All class probabilities")
    # Show sorted probs
    rows = []
    for i in range(len(probs)):
        rows.append((idx_to_class[i], float(probs[i].item())))
    rows.sort(key=lambda x: x[1], reverse=True)
    for label, p in rows:
        st.write(f"- `{label}`: **{p:.3f}**")


if __name__ == "__main__":
    main()