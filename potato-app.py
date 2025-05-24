# -- Imports --
import streamlit as st
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -- Page Configuration --
st.set_page_config(page_title="🥔 Potato Disease Classifier 🥔", layout="centered")

# -- Constants --
MODEL_URL = "https://huggingface.co/datasets/jralipio18/potato/resolve/main/potato_model.h5"
MODEL_PATH = "potato_model.h5"
IMG_SIZE = (150, 150)  # Update if your model uses a different size
CLASS_NAMES = ['Common Scab', 'Blackleg', 'Dry Rot', 'Pink Rot', 'Black Scurf', 'Healthy', 'Miscellaneous']

# -- Load Model with Caching --
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = load_model(MODEL_PATH)
    return model

model = download_and_load_model()

# -- UI Styling --
st.markdown("""
    <style>
        .stButton>button {
            background-color: #28a745;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #1e7e34;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #888;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

# -- App Title and Upload --
st.title("🥔 Potato Disease Classifier 🥔")
st.markdown("Upload a potato image to identify whether it's **Healthy** or has a disease like **Common Scab**, **Dry Rot**, etc.")

uploaded_file = st.file_uploader("📤 Upload a potato image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=False, width=300)

    # Preprocess
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Classifying..."):
        prediction = model.predict(img_array)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = prediction[predicted_class_index] * 100

    # Display Results
    st.success(f"🎯 **Prediction:** `{predicted_class}`")
    st.metric("🔒 Confidence", f"{confidence:.2f} %")
