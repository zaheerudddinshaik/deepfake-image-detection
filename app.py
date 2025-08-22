# app.py (Professional Version)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- App Configuration ---
# Use st.set_page_config() as the first Streamlit command.
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Parameters ---
MODEL_PATH = 'deepfake_detector_finetuned.h5'
IMAGE_SIZE = (224, 224)

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    """Loads and caches the trained Keras model for performance."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# --- Label Mapping ---
labels_map = {0: 'fake', 1: 'real'}

# --- Helper Function for Prediction ---
def predict(image_to_predict):
    """Preprocesses the image and returns the prediction label and confidence."""
    img = image_to_predict.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = (img_array / 127.5) - 1.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    pred_prob = model.predict(img_batch)[0][0]
    pred_label_index = 1 if pred_prob > 0.5 else 0
    pred_label_name = labels_map[pred_label_index]
    
    # Calculate confidence based on the predicted class
    if pred_label_name == 'real':
        confidence = pred_prob * 100
    else:
        confidence = (1 - pred_prob) * 100
        
    return pred_label_name, confidence

# --- Sidebar ---
with st.sidebar:
    st.title("딥페이크 탐지기")
    st.markdown("Upload an image and the AI will determine if the face is real or a deepfake.")
    
    uploaded_file = st.file_uploader(
        "Choose an image file...", 
        type=["png", "jpg", "jpeg"],
        help="Upload an image of a person's face."
    )
    
    st.markdown("---")
    with st.expander("ℹ️ About this App"):
        st.write("""
            This application uses a deep learning model (based on ResNet50) to classify images as either 'REAL' or 'FAKE'.
            
            It was built as part of a hackathon to demonstrate the ability to detect manipulated media. The model is not perfect but serves as a strong proof-of-concept.
        """)

# --- Main Page ---
st.header("Deepfake Detection Analysis")

if uploaded_file is None:
    st.info("👈 Please upload an image using the sidebar to begin analysis.")
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM2F3bm1pcXl2NW95Z3g5bzQza2t4MGQ1c3BiaWVucTZyOGo0NmY4cyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L2zV2bXvG2fC742x28/giphy.gif", caption="Waiting for your image...")


if uploaded_file is not None and model is not None:
    # --- Prediction runs automatically after upload ---
    pil_image = Image.open(uploaded_file)
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Uploaded Image")
        st.image(pil_image, use_column_width=True)
        
    with col2:
        st.subheader("Analysis Result")
        with st.spinner('The AI is thinking...'):
            label, confidence = predict(pil_image)
            
            if label == 'real':
                st.metric(label="Prediction", value="✅ REAL", delta=f"{confidence:.2f}% Confidence")
            else:
                st.metric(label="Prediction", value="❌ FAKE", delta=f"{confidence:.2f}% Confidence", delta_color="inverse")
            
            st.write("Confidence Score:")
            st.progress(int(confidence))
            
            st.markdown("---")
            st.info("""
                **Disclaimer:** This AI model is for educational and demonstration purposes only. Its predictions are not 100% accurate and should not be used to make definitive judgments.
            """)
