import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import logging
import os
from typing import Optional, Tuple, Union
import requests
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow logging
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Config:
    """Application configuration"""
    MODEL_PATH = "models/tb_model.h5"
    IMAGE_SIZE: Tuple[int, int] = (300, 300)
    GITHUB_REPO_OWNER = 'Krishna-Acroplans'
    GITHUB_REPO_NAME = 'medical-image-enhancement'
    GITHUB_BRANCH = 'main'

def get_github_file_url(file_path):
    return f"https://raw.githubusercontent.com/{Config.GITHUB_REPO_OWNER}/{Config.GITHUB_REPO_NAME}/{Config.GITHUB_BRANCH}/{file_path}"

@st.cache_data
def load_metadata():
    metadata_url = get_github_file_url('MetaData.csv')
    try:
        return pd.read_csv(metadata_url)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None

@st.cache_data
def load_image_and_mask(image_url, mask_url):
    try:
        # Load image
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image = cv2.imdecode(np.asarray(bytearray(image_response.content)), cv2.IMREAD_GRAYSCALE)

        # Load mask
        mask_response = requests.get(mask_url)
        mask_response.raise_for_status()
        mask = cv2.imdecode(np.asarray(bytearray(mask_response.content)), cv2.IMREAD_GRAYSCALE)

        return image, mask
    except Exception as e:
        st.error(f"Error loading image or mask: {e}")
        return None, None

def preprocess_image(image: Union[np.ndarray, Image.Image], target_size: Tuple[int, int] = Config.IMAGE_SIZE) -> Optional[np.ndarray]:
    """
    Preprocess image for model prediction
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize image to 300x300
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        st.error(f"Failed to preprocess image: {str(e)}")
        return None

@st.cache_resource
def load_model() -> Optional[tf.keras.Model]:
    """Load and cache the model"""
    try:
        model = tf.keras.models.load_model(Config.MODEL_PATH)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Failed to load model: {str(e)}")
        return None

def predict_tuberculosis(image: Union[np.ndarray, Image.Image], model: tf.keras.Model) -> Optional[float]:
    """Make prediction on the input image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None
            
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][0])
        return confidence
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        st.error(f"Failed to make prediction: {str(e)}")
        return None

def apply_color_mask(mask, color):
    """Apply color to the segmentation mask"""
    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if color == 'Blue':
        color_mask[:, :, 1] = 0
        color_mask[:, :, 0] = 0
    elif color == 'Green':
        color_mask[:, :, 2] = 0
        color_mask[:, :, 0] = 0
    elif color == 'Red':
        color_mask[:, :, 2] = 0
        color_mask[:, :, 1] = 0
    return color_mask

def overlay_mask(image, mask, color, alpha=0.5):
    """Overlay segmentation mask on the image"""
    color_mask = apply_color_mask(mask, color)
    
    # Ensure image is in BGR format for OpenCV
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    overlay = cv2.addWeighted(color_mask, alpha, image, 1 - alpha, 0)
    return overlay

@st.cache_data
def enhance_image(image, technique, **kwargs):
    """Apply various enhancement techniques"""
    if technique == 'Histogram Equalization':
        return cv2.equalizeHist(image)
    elif technique == 'Gaussian Blur':
        ksize = kwargs.get('ksize', 5)
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif technique == 'Canny Edge Detection':
        threshold1 = kwargs.get('threshold1', 100)
        threshold2 = kwargs.get('threshold2', 200)
        return cv2.Canny(image, threshold1, threshold2)
    elif technique == 'CLAHE':
        clip_limit = kwargs.get('clip_limit', 2.0)
        tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    else:
        return image

def main():
    st.title("Medical Image Enhancement and Tuberculosis Detection")
    
    # Load TB detection model
    model = load_model()
    if model is None:
        st.error("Failed to load TB detection model. Please check the model file.")
        st.stop()
    
    # Sidebar for navigation
    app_mode = st.sidebar.selectbox("Choose App Mode", 
        ["Sample Images", "Upload Your Own Image"])
    
    if app_mode == "Upload Your Own Image":
        # File uploader for user images
        uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file is not None:
            # Read the image
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
            
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption='Original Image', use_column_width=True)
                
                # Optional mask upload
                uploaded_mask = st.file_uploader("Upload Segmentation Mask (Optional)", 
                    type=["png", "jpg", "jpeg", "webp"], key="user_mask")
                
                if uploaded_mask is not None:
                    mask = np.array(bytearray(uploaded_mask.read()), dtype=np.uint8)
                    mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
                    
                    # Mask visualization options
                    show_mask = st.checkbox("Show Segmentation Mask", value=True)
                    if show_mask:
                        st.image(mask, caption="Uploaded Segmentation Mask")
                    
                    # Mask overlay
                    overlay = st.checkbox("Overlay Segmentation Mask", value=False)
                    if overlay:
                        overlay_alpha = st.slider('Overlay Opacity', 0.0, 1.0, 0.5)
                        mask_color = st.selectbox('Mask Color', ['Red', 'Green', 'Blue'])
                        overlay_image = overlay_mask(image, mask, mask_color, alpha=overlay_alpha)
                        st.image(overlay_image, caption="Image with Mask Overlay")
            
            with col2:
                # Image Enhancement Techniques
                technique = st.selectbox("Select Enhancement Technique", 
                    ['Original', 'Histogram Equalization', 'Gaussian Blur', 
                     'Canny Edge Detection', 'CLAHE'])
                
                # Dynamic parameter adjustments
                enhanced_image = image
                if technique == 'Gaussian Blur':
                    ksize = st.slider('Kernel Size', 1, 31, 5, step=2)
                    enhanced_image = enhance_image(image, technique, ksize=ksize)
                elif technique == 'Canny Edge Detection':
                    threshold1 = st.slider('Threshold 1', 0, 300, 100)
                    threshold2 = st.slider('Threshold 2', 0, 300, 200)
                    enhanced_image = enhance_image(image, technique, 
                        threshold1=threshold1, threshold2=threshold2)
                elif technique == 'CLAHE':
                    clip_limit = st.slider('Clip Limit', 1.0, 40.0, 2.0)
                    tile_grid_size = st.slider('Tile Grid Size', 1, 16, 8)
                    enhanced_image = enhance_image(image, technique, 
                        clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size))
                elif technique == 'Histogram Equalization':
                    enhanced_image = enhance_image(image, technique)
                
                # Display enhanced image
                st.image(enhanced_image, caption='Enhanced Image')
            
            # TB Detection
            if st.button("Analyze for Tuberculosis"):
                with st.spinner("Detecting Tuberculosis..."):
                    # Convert grayscale to RGB for prediction
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    confidence = predict_tuberculosis(rgb_image, model)
                    
                    if confidence is not None:
                        # Display results
                        if confidence > 0.5:
                            st.error(f"Tuberculosis Detected (Confidence: {confidence:.2%})")
                        else:
                            st.success(f"No Tuberculosis Detected (Confidence: {(1-confidence):.2%})")
                        
                        st.info("Note: This is a screening tool and should not replace professional medical diagnosis.")
    
    else:  # Sample Images Mode
        # Load metadata
        metadata = load_metadata()
        if metadata is not None:
            # Randomly select an image if not already selected
            if 'image_id' not in st.session_state:
                selected_image = metadata.sample()
                st.session_state.image_id = selected_image['id'].values[0]
                st.session_state.county = selected_image['county'].values[0]
                st.session_state.ptb = selected_image['ptb'].values[0]
            
            # Construct image and mask URLs
            image_url = get_github_file_url(f'image/{st.session_state.image_id}.webp')
            mask_url = get_github_file_url(f'mask/{st.session_state.image_id}.webp')
            
            # Load image and mask
            image, mask = load_image_and_mask(image_url, mask_url)
            
            if image is not None and mask is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Original Image Display
                    st.image(image, caption=f"Original Image - {st.session_state.county}")
                    
                    # Mask Visualization
                    show_mask = st.checkbox("Show Segmentation Mask", value=True)
                    if show_mask:
                        st.image(mask, caption="Segmentation Mask")
                    
                    # Mask Overlay
                    overlay = st.checkbox("Overlay Segmentation Mask", value=False)
                    if overlay:
                        overlay_alpha = st.slider('Overlay Opacity', 0.0, 1.0, 0.5)
                        mask_color = st.selectbox('Mask Color', ['Red', 'Green', 'Blue'])
                        overlay_image = overlay_mask(image, mask, mask_color, alpha=overlay_alpha)
                        st.image(overlay_image, caption="Image with Mask Overlay")
                
                with col2:
                    # Enhancement Techniques
                    technique = st.selectbox("Select Enhancement Technique", 
                        ['Original', 'Histogram Equalization', 'Gaussian Blur', 
                         'Canny Edge Detection', 'CLAHE'])
                    
                    # Dynamic parameter adjustments
                    enhanced_image = image
                    if technique == 'Gaussian Blur':
                        ksize = st.slider('Kernel Size', 1, 31, 5, step=2)
                        enhanced_image = enhance_image(image, technique, ksize=ksize)
                    elif technique == 'Canny Edge Detection':
                        threshold1 = st.slider('Threshold 1', 0, 300, 100)
                        threshold2 = st.slider('Threshold 2', 0, 300, 200)
                        enhanced_image = enhance_image(image, technique, 
                            threshold1=threshold1, threshold2=threshold2)
                    elif technique == 'CLAHE':
                        clip_limit = st.slider('Clip Limit', 1.0, 40.0, 2.0)
                        tile_grid_size = st.slider('Tile Grid Size', 1, 16, 8)
                        enhanced_image = enhance_image(image, technique, 
                            clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size))
                    elif technique == 'Histogram Equalization':
                        enhanced_image = enhance_image(image, technique)
                    
                    # Display enhanced image
                    st.image(enhanced_image, caption='Enhanced Image')
                
                # TB Detection for Sample Image
                if st.button("Analyze for Tuberculosis"):
                    with st.spinner("Detecting Tuberculosis..."):
                        # Convert grayscale to RGB for prediction
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        confidence = predict_tuberculosis(rgb_image, model)
                        
                        if confidence is not None:
                            # Display results
                            
                            if confidence > 0.5:
                                st.error(f"Tuberculosis Detected (Confidence: {confidence:.2%})")
                            else:
                                st.success(f"No Tuberculosis Detected (Confidence: {(1-confidence):.2%})")
                            
                            st.info("Note: This is a screening tool and should not replace professional medical diagnosis.")
            else:
                st.error("Error loading sample image. Please try again.")
        else:
            st.error("No metadata available. Unable to load sample images.")

    # Additional context and information
    st.sidebar.header("About the App")
    st.sidebar.info("""
    Medical Image Enhancement and TB Detection App
    
    Features:
    - Image Enhancement Techniques
    - Segmentation Mask Visualization
    - Tuberculosis Screening
    
    Disclaimer: This tool is for educational purposes 
    and should not replace professional medical advice.
    """)

if __name__ == "__main__":
    main()
