import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Image-Based Cancer Detection", layout="centered")

@st.cache_resource
def load_cnn_model():
    return load_model("models/skin_cancer_model5.keras")

model = load_cnn_model()
cancer_classes = [
    'actinic keratosis','basal cell carcinoma','dermatofibroma','melanoma',
    'nevus','pigmented benign keratosis','seborrheic keratosis',
    'squamous cell carcinoma','vascular lesion'
]

st.markdown("##  Cancer Type Detection from Image")

uploaded = st.file_uploader("Upload a medical image", type=["jpg","jpeg","png"])

def predict_cancer(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)
    idx = np.argmax(pred, axis=1)[0]
    conf = float(np.max(pred)) * 100 
    return cancer_classes[idx], conf

if uploaded:
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.read())
    st.image(path, caption="Uploaded Image", use_container_width=True)

    if st.button("🔮 Predict Cancer Type"):
        with st.spinner("Analyzing image..."):
            disease, conf = predict_cancer(path)
        st.success(f"Predicted: {disease}")
        