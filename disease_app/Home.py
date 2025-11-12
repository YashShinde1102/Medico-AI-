import streamlit as st

st.set_page_config(page_title="Medical AI System", layout="centered")

st.title(" AI-Medico: Disease Diagnosis system")
st.markdown("""
Welcome to the **AI Medical Assistant** — choose a feature from the sidebar:
-  **Disease Predictor** using symptoms and ML  
-  **Cancer Image Detection** using CNN model  
-  **Symptom-Based Chatbot** using NLP similarity model  
-  **Report Generation of the patient**             
""")

import os
st.image(os.path.join("disease_app", "cover_image.jpg"), caption="Your Health Companion",use_container_width=True)

