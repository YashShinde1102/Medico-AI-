import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Skin Lesion Segmentation", layout="centered")

@st.cache_resource
def load_unet():
    return load_model(r"E:/AI_medico/models/unet_model.keras")

model = load_unet()

st.title("🧠 Skin Lesion Segmentation")
uploaded = st.file_uploader("Upload a skin image", type=["jpg","jpeg","png"])


def predict_mask(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0]   # (128,128,1)
    confidence = float(np.mean(pred))
    mask = (pred > 0.5).astype(np.uint8)
    mask = mask.squeeze()

    return arr[0], mask,confidence


if uploaded:
    img = image.load_img(uploaded, target_size=(128,128))
    st.image(img, caption="Uploaded Image", use_container_width=True)
    

    if st.button("🔍 Segment Lesion"):
        with st.spinner("Analyzing..."):
            img_arr, mask,confidence = predict_mask(uploaded)

        st.subheader("Predicted Lesion Mask")
        st.image(mask * 255, use_container_width=True)
        #st.write(f"Mask Confidence:{confidence:.4f}")