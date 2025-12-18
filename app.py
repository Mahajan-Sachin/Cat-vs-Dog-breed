import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
import pandas as pd

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Cat & Dog Breed Classifier",
    page_icon="🐶",
    layout="centered"
)

# ---------------- LOAD MODEL & METADATA ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("breed_classifier_mobilenet.h5")

@st.cache_resource
def load_metadata():
    return joblib.load("metadata.pkl")

model = load_model()
metadata = load_metadata()

label_to_breed = metadata["label_to_breed"]
IMG_SIZE = metadata["img_size"]

# ---------------- UTILS ----------------
def format_breed(name):
    return name.replace("_", " ").title()

def predict_top_k(image, k=3):
    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    top_indices = preds.argsort()[-k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "Breed": format_breed(label_to_breed[idx]),
            "Confidence (%)": round(preds[idx] * 100, 2)
        })

    return results

# ---------------- SIDEBAR ----------------
st.sidebar.title("ℹ️ About")
st.sidebar.write(
    """
    **Cat & Dog Breed Classifier**
    
    - MobileNetV2 (Transfer Learning)
    - 90%+ Validation Accuracy
    - Built with TensorFlow & Streamlit
    
    Upload an image to see predictions.
    """
)

# ---------------- MAIN UI ----------------
st.title("🐶🐱 Cat & Dog Breed Classifier")
st.write("Upload an image and get the **top predictions with confidence**.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting breed..."):
        results = predict_top_k(image, k=3)

    st.subheader("🔍 Top Predictions")

    # Show best prediction clearly
    best = results[0]
    st.success(f"**{best['Breed']}** — {best['Confidence (%)']}%")

    # Bar chart for all predictions
    df = pd.DataFrame(results)
    st.bar_chart(df.set_index("Breed"))

    # Detailed table
    with st.expander("See detailed probabilities"):
        st.table(df)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built by Sachin • TensorFlow + Streamlit")
