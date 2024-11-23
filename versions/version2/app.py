import streamlit as st
import os
import cv2
import pandas as pd

# Define paths directly
image_dir = 'image'
mask_dir = 'mask'
metadata_file = 'MetaData.csv'

# Function to load metadata
def load_metadata():
    if os.path.exists(metadata_file):
        return pd.read_csv(metadata_file)
    else:
        st.error("Metadata file not found.")
        return None

# Function to check if the image and mask files exist
def file_exists(path):
    if os.path.exists(path):
        return True
    else:
        st.error(f"File does not exist: {path}")
        return False

# Function to load and display an image and its mask
def load_image_and_mask(image_path, mask_path):
    # st.write(f"Trying to load image from: {image_path}")
    # st.write(f"Trying to load mask from: {mask_path}")

    # Check if files exist
    if file_exists(image_path) and file_exists(mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            st.error(f"Failed to load image from: {image_path}")
        if mask is None:
            st.error(f"Failed to load mask from: {mask_path}")
            
        return image, mask
    else:
        return None, None

# Enhancement techniques
def enhance_image(image, technique, **kwargs):
    if technique == 'Histogram Equalization':
        return cv2.equalizeHist(image)
    elif technique == 'Gaussian Blur':
        ksize = kwargs.get('ksize', 5)
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif technique == 'Canny Edge Detection':
        threshold1 = kwargs.get('threshold1', 100)
        threshold2 = kwargs.get('threshold2', 200)
        return cv2.Canny(image, threshold1, threshold2)
    else:
        return image

# Streamlit app
st.title("Medical Image Enhancement and Segmentation Demo")

# Load metadata
metadata = load_metadata()
if metadata is not None:
    # Select an image
    selected_image = metadata.sample()  # Randomly select an image
    # st.write(f"Selected image ID: {selected_image['id'].values[0]}")

    image_path = os.path.join(image_dir, f"{selected_image['id'].values[0]}.png")
    mask_path = os.path.join(mask_dir, f"{selected_image['id'].values[0]}.png")

    # Load and display the image and its mask
    image, mask = load_image_and_mask(image_path, mask_path)
    if image is not None and mask is not None:
        st.image(image, caption=f"Original Image - {selected_image['county'].values[0]} - {selected_image['ptb'].values[0]}")
        st.image(mask, caption="Segmentation Mask")

        # Select enhancement technique
        technique = st.selectbox("Select Enhancement Technique", ['Original', 'Histogram Equalization', 'Gaussian Blur', 'Canny Edge Detection'])

        # Set parameters for techniques
        if technique == 'Gaussian Blur':
            ksize = st.slider('Kernel Size', 1, 31, 5, step=2)
            enhanced_image = enhance_image(image, technique, ksize=ksize)
        elif technique == 'Canny Edge Detection':
            threshold1 = st.slider('Threshold 1', 0, 300, 100)
            threshold2 = st.slider('Threshold 2', 0, 300, 200)
            enhanced_image = enhance_image(image, technique, threshold1=threshold1, threshold2=threshold2)
        else:
            enhanced_image = enhance_image(image, technique)

        # Display enhanced image
        if enhanced_image is not None:
            st.image(enhanced_image, caption='Enhanced Image')
        else:
            st.error("Error enhancing image. Please select a valid technique.")
    else:
        st.error("Error loading image or mask. Please check the file paths.")
else:
    st.error("No metadata available. Unable to proceed.")

st.write("This demo showcases different image enhancement techniques and highlights lung segmentation masks. Select a technique and adjust the parameters to see the effect on the image.")
