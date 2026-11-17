"""Streamlit interface for OCT retinal disease classification."""
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt

from config import MODEL_DIR, IMAGE_SIZE
from data_utils import make_transforms, load_oct_dataset, get_class_names
from models import build_model
from gradcam import make_gradcam

st.set_page_config(page_title="Retinal OCT AI", page_icon="👁️", layout="wide")
st.title("👁️ Retinal Disease Detection Using Deep Learning on OCT Images")
st.warning("Educational prototype only. It is not a clinical diagnostic tool and must not guide treatment.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def get_ds():
    return load_oct_dataset()

@st.cache_resource
def load_model(checkpoint_path, modified_time):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt

tab1, tab2, tab3 = st.tabs(["Dataset explorer", "Image prediction + Grad-CAM", "Evaluation results"])

with tab1:
    st.subheader("OCT2017 dataset")
    if st.button("Load dataset information"):
        try:
            ds = get_ds()
            names = get_class_names(ds["train"])
            st.write("Dataset: `zacharielegault/Kermany2017-OCT`")
            st.write(f"Training images: **{len(ds['train']):,}**")
            st.write(f"Testing images: **{len(ds['test']):,}**")
            st.write("Classes:", ", ".join(names))
            counts = np.bincount(np.asarray(ds["train"]["label"]), minlength=len(names))
            st.bar_chart({names[i]: int(counts[i]) for i in range(len(names))})
            cols = st.columns(len(names))
            for class_id, col in enumerate(cols):
                idx = next(i for i, label in enumerate(ds["train"]["label"]) if label == class_id)
                col.image(ds["train"][idx]["image"], caption=names[class_id], use_container_width=True)
        except Exception as exc:
            st.error(f"Dataset load failed: {exc}")

with tab2:
    model_files = sorted(Path(MODEL_DIR).glob("*_best.pth")) if Path(MODEL_DIR).exists() else []
    if not model_files:
        st.info("Train a model first using the commands in README.md. The saved checkpoint will appear in models/.")
    else:
        selected = st.selectbox("Select trained model", model_files, format_func=lambda p: p.stem)
        upload = st.file_uploader("Upload an OCT image", type=["png", "jpg", "jpeg", "bmp"])
        if upload:
            pil_image = Image.open(upload).convert("RGB")
            st.image(pil_image, caption="Uploaded OCT scan", width=480)
            if st.button("Classify image"):
                try:
                    model, ckpt = load_model(str(selected), selected.stat().st_mtime)
                    transform = make_transforms(ckpt["image_size"], training=False)
                    tensor = transform(pil_image).unsqueeze(0).to(device)
                    cam, class_id, probs = make_gradcam(model, tensor, ckpt["model_name"])
                    names = ckpt["class_names"]
                    st.success(f"Predicted class: **{names[class_id]}**")
                    st.metric("Prediction confidence", f"{probs[class_id]*100:.2f}%")
                    st.bar_chart({names[i]: float(probs[i]) for i in range(len(names))})
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.imshow(pil_image.convert("L"), cmap="gray")
                    ax.imshow(cam, cmap="jet", alpha=0.4)
                    ax.set_title("Grad-CAM — regions influencing the prediction")
                    ax.axis("off")
                    st.pyplot(fig)
                except Exception as exc:
                    st.error(f"Prediction/Grad-CAM failed: {exc}")

with tab3:
    result_files = sorted(Path("outputs").glob("*_metrics.json")) if Path("outputs").exists() else []
    if not result_files:
        st.info("No evaluation files yet. Run train.py to generate metrics.")
    else:
        import json
        for path in result_files:
            st.markdown(f"**{path.stem}**")
            st.json(json.loads(path.read_text()))
            cm_path = path.with_name(path.stem.replace("_metrics", "_confusion_matrix") + ".png")
            if cm_path.exists():
                st.image(str(cm_path), caption=f"Confusion matrix: {path.stem}")
