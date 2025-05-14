# app.py
import streamlit as st
from model import predict_outfit_score
from PIL import Image
import os

st.title("👗 FitCheck AI - How Good Is Your Outfit?")

uploaded_file = st.file_uploader("Upload an outfit image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Outfit", use_column_width=True)

    with st.spinner("Judging your outfit..."):
        score, category = predict_outfit_score(image)

    st.markdown(f"### 👚 Style Category: `{category}`")
    st.markdown(f"### 🔥 Fashion Score: `{score}/10`")
