import streamlit as st
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import generate_caption
from io import BytesIO

# Load model and tokenizer
model = tf.keras.models.load_model(r'artifacts\model.keras')
fe = tf.keras.models.load_model(r'artifacts\fe.keras')
max_length = 34

with open(r'artifacts\tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Custom CSS for animations and styling
st.markdown(
    """
    <style>
    /* Background gradient */
    body {
        background: linear-gradient(135deg, #74ebd5, #ACB6E5);
        color: #fff;
        font-family: 'Roboto', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #111;
        color: #fff;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.4);
    }

    /* Sidebar title */
    [data-testid="stSidebar"] .css-1d391kg p {
        color: #74ebd5;
        font-weight: bold;
        font-size: 1.2rem;
    }

    /* Drag-and-drop file upload styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.2);
        border: 2px dashed #fff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        animation: glow 1.5s infinite;
    }

    /* Glow effect animation */
    @keyframes glow {
        0% { box-shadow: 0 0 10px #fff; }
        50% { box-shadow: 0 0 20px #74ebd5; }
        100% { box-shadow: 0 0 10px #fff; }
    }

    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
        margin-top: 50px;
    }

    /* Add hover animation to buttons */
    button {
        background-color: #74ebd5;
        color: #111;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        transition: transform 0.2s ease-in-out;
    }
    button:hover {
        transform: scale(1.1);
    }

    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
        background: rgba(0, 0, 0, 0.5);
        color: #fff;
        font-size: 0.9rem;
    }

    /* Animated moving symbol */
    @keyframes moveSymbol {
        0% { transform: translateX(0); }
        50% { transform: translateX(200px); }
        100% { transform: translateX(0); }
    }

    .moving-symbol {
        font-size: 3rem;
        color: #fff;
        animation: moveSymbol 2s infinite;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<div class='main-title'>Image Captioning Application</div>", unsafe_allow_html=True)

# Sidebar menu
st.sidebar.title("Group - Luminaries")
st.sidebar.write("Illuminating Data Excellence")

# Image uploader in the sidebar
st.sidebar.title("Image Upload")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# st.title("Image Upload Example")

# Main functionality
if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    img = load_img(uploaded_file, target_size=(224, 224))
    
    # Generate the caption (function call replaced with a placeholder for demonstration)
    predicted_caption = generate_caption(model, tokenizer, fe, img, max_length)
    
    # Resize the image to a maximum width and height
    max_size = (300, 500)  # Set the maximum dimensions (width, height)
    image.thumbnail(max_size)

    st.image(image, caption="Uploaded Image (Resized)")
    st.write("**Generated Caption:**", predicted_caption)
else:
    st.markdown("<div class='moving-symbol'>👈</div>", unsafe_allow_html=True)

# # Footer
st.markdown(
    "<div class='footer'>Created with ❤️ by the Luminaries</div>",
    unsafe_allow_html=True
)
