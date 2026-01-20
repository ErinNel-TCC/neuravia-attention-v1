import streamlit as st
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Model placeholder (V1)
# -----------------------------
class SaliencyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

model = SaliencyNet()
model.eval()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def preprocess(img):
    return transform(img).unsqueeze(0)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NeuraVia.ai — Visual Attention", layout="centered")
st.title("NeuraVia.ai — Visual Attention Predictor")

uploaded = st.file_uploader("Upload a social image", type=["jpg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    tensor = preprocess(img)

    with torch.no_grad():
        saliency = model(tensor).squeeze().numpy()
        saliency = normalize(saliency)

    heatmap = cv2.resize(saliency, img.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Predicted Visual Attention Heatmap")

    st.subheader("Attention Summary")
    st.write(
        "Predicted attention is driven primarily by high-contrast regions "
        "and visually dominant elements. Early fixation is likely to occur "
        "within the central visual field, with attention dispersing rapidly "
        "in high-density compositions."
    )
