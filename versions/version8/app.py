# with enhancements ....

# Enhancement 1: Add Contrast Limited Adaptive Histogram Equalization (CLAHE)
# CLAHE is a technique for enhancing the contrast of images, particularly useful for improving the visibility of features in medical images.

# Enhancement 2: More Interactivity
# Improved interactivity in application by adding the following features:
# 1.	Real-Time Parameter Adjustments: Allow users to adjust parameters in real-time and see the results immediately.
# 2.	Comparative Visualization: Show side-by-side comparison of the original and enhanced images.
# 3. with checkbox option to Show/No-Show option for segmentation mask

# Enhancement 3: Improving Visualization
# We can further enhance visualization by adding the following features:
# 1.	Overlaying Segmentation Mask: Allow users to overlay the segmentation mask on the original or enhanced image for better visual analysis.
#       Function to apply color to the mask
# 2.	Image Zoom and Pan: Implement zoom and pan functionality for detailed inspection of the images.

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import logging
import os
from typing import Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow logging
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Config:
    """Application configuration"""
    MODEL_PATH = "models/tb_model.h5"
    IMAGE_SIZE: Tuple[int, int] = (300, 300)  # Changed from 224x224 to 300x300
    
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

def main():
    st.title("Tuberculosis Detection System")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        
        # Make prediction
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                confidence = predict_tuberculosis(image, model)
                
                if confidence is not None:
                    # Display results
                    if confidence > 0.5:
                        st.error(f"Tuberculosis detected (Confidence: {confidence:.2%})")
                    else:
                        st.success(f"No tuberculosis detected (Confidence: {(1-confidence):.2%})")
                    
                    st.info("Note: This is a screening tool and should not be used as a definitive diagnosis.")

if __name__ == "__main__":
    main()
