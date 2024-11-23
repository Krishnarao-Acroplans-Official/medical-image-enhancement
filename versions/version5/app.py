import streamlit as st
import cv2
import numpy as np
import pandas as pd
import requests

# Define GitHub parameters
repo_owner = 'Krishna-Acroplans'
repo_name = 'medical-image-enhancement'
branch = 'main'

# Function to get GitHub raw URL
def get_github_file_url(file_path):
    return f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file_path}"

# Function to load metadata
def load_metadata():
    metadata_url = get_github_file_url('MetaData.csv')
    try:
        return pd.read_csv(metadata_url)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return None

# Function to load image and mask from GitHub
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

# Streamlit app
st.title("Medical Image Enhancement and Segmentation Demo")

# Load metadata
metadata = load_metadata()
if metadata is not None:
    # Select an image
    selected_image = metadata.sample()  # Randomly select an image
    image_id = selected_image['id'].values[0]

    image_url = get_github_file_url(f'image/{image_id}.webp')
    mask_url = get_github_file_url(f'mask/{image_id}.webp')

    # Load and display the image and its mask
    image, mask = load_image_and_mask(image_url, mask_url)
    if image is not None and mask is not None:
        st.image(image, caption=f"Original Image - {selected_image['county'].values[0]} - {selected_image['ptb'].values[0]}")
        st.image(mask, caption="Segmentation Mask")

        # Select enhancement technique and display enhanced image
        technique = st.selectbox("Select Enhancement Technique", ['Original', 'Histogram Equalization', 'Gaussian Blur', 'Canny Edge Detection'])
        if technique == 'Gaussian Blur':
            ksize = st.slider('Kernel Size', 1, 31, 5, step=2)
            enhanced_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif technique == 'Canny Edge Detection':
            threshold1 = st.slider('Threshold 1', 0, 300, 100)
            threshold2 = st.slider('Threshold 2', 0, 300, 200)
            enhanced_image = cv2.Canny(image, threshold1, threshold2)
        elif technique == 'Histogram Equalization':
            enhanced_image = cv2.equalizeHist(image)
        else:
            enhanced_image = image

        st.image(enhanced_image, caption='Enhanced Image')
    else:
        st.error("Error loading image or mask. Please check the file paths.")
else:
    st.error("No metadata available. Unable to proceed.")

st.write("This demo showcases different image enhancement techniques and highlights lung segmentation masks. Select a technique and adjust the parameters to see the effect on the image.")
