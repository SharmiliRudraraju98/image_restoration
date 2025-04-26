# app.py

import streamlit as st
import torch
import os
from PIL import Image
from utils import load_real_esrgan, load_bsrgan, load_swinir, enhance_image_real_esrgan, enhance_image_bsrgan, enhance_image_swinir

st.set_page_config(page_title="AI Image Restoration App", layout="centered")

st.title("\ud83d\ude80 Restore Image Quality with AI Models!")

# Sidebar - Choose Model
model_choice = st.sidebar.selectbox(
    "Choose a model for enhancement:",
    ("Real-ESRGAN", "BSRGAN", "SwinIR")
)

# Sidebar - Upload Image
uploaded_file = st.sidebar.file_uploader("Upload a low-quality image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display input image
    input_image = Image.open(uploaded_file).convert('RGB')
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    enhance_button = st.button("Enhance Image")

    if enhance_button:
        st.spinner("Enhancing image, please wait...")
        # Model loading and enhancement
        if model_choice == "Real-ESRGAN":
            model = load_real_esrgan()
            output_image = enhance_image_real_esrgan(input_image, model)

        elif model_choice == "BSRGAN":
            model = load_bsrgan()
            output_image = enhance_image_bsrgan(input_image, model)

        elif model_choice == "SwinIR":
            model = load_swinir()
            output_image = enhance_image_swinir(input_image, model)

        st.success("Image enhancement complete!")
        st.image(output_image, caption=f"Enhanced with {model_choice}", use_column_width=True)

        # Download button
        output_path = "enhanced_output.png"
        output_image.save(output_path)
        with open(output_path, "rb") as file:
            st.download_button(
                label="\ud83d\udcbe Download Enhanced Image",
                data=file,
                file_name="enhanced_image.png",
                mime="image/png"
            )
else:
    st.info("\ud83d\udcf7 Please upload an image to get started.")
