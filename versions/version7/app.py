# with enhancements ....

# Enhancement 1: Add Contrast Limited Adaptive Histogram Equalization (CLAHE)
# CLAHE is a technique for enhancing the contrast of images, particularly useful for improving the visibility of features in medical images.

# Enhancement 2: More Interactivity
# Improved interactivity in application by adding the following features:
# 1.	Real-Time Parameter Adjustments: Allow users to adjust parameters in real-time and see the results immediately.
# 2.	Comparative Visualization: Show side-by-side comparison of the original and enhanced images.
# 3. with checkbox option to Show/No-Show option for segmentation mask


# Add File Uploader: 
# o Allow users to upload their own images. 
# Update the App to Process and Display the Uploaded Images: 
# o Process the uploaded images and display them in the app. 
# o Apply enhancement techniques to the uploaded images.


# Enhancement 3: Improving Visualization
# We can further enhance visualization by adding the following features:
# 1.	Overlaying Segmentation Mask: Allow users to overlay the segmentation mask on the original or enhanced image for better visual analysis.
#       Function to apply color to the mask
# 2.	Image Zoom and Pan: Implement zoom and pan functionality for detailed inspection of the images.


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

# Function to apply color to the mask
def apply_color_mask(mask, color):
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

# Function to overlay mask
def overlay_mask(image, mask, color, alpha=0.5):
    color_mask = apply_color_mask(mask, color)
    overlay = cv2.addWeighted(color_mask, alpha, image, 1 - alpha, 0)
    return overlay

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
    elif technique == 'CLAHE':
        clip_limit = kwargs.get('clip_limit', 2.0)
        tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    else:
        return image

# Streamlit app
st.title("Medical Image Enhancement and Segmentation Demo")

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    # Process the uploaded file
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image')

        # Add a segmentation mask upload option for the uploaded image
        uploaded_mask_file = st.file_uploader("Upload Segmentation Mask (Optional)", type=["png", "jpg", "jpeg", "webp"], key="mask")
        if uploaded_mask_file is not None:
            mask = np.array(bytearray(uploaded_mask_file.read()), dtype=np.uint8)
            mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
            show_mask = st.checkbox("Show Segmentation Mask", value=True)
            if show_mask:
                st.image(mask, caption="Uploaded Segmentation Mask")

            overlay = st.checkbox("Overlay Segmentation Mask on Image", value=False)
            if overlay:
                overlay_alpha = st.slider('Overlay Alpha', 0.0, 1.0, 0.5)
                mask_color = st.selectbox('Select Mask Color', ['Red', 'Green', 'Blue'], index=0)
                image_with_overlay = overlay_mask(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), mask, mask_color, alpha=overlay_alpha)
                st.image(image_with_overlay, caption="Image with Segmentation Mask Overlay")

    with col2:
        # Select enhancement technique and display enhanced image
        technique = st.selectbox("Select Enhancement Technique", ['Original', 'Histogram Equalization', 'Gaussian Blur', 'Canny Edge Detection', 'CLAHE'])
        enhanced_image = image  # Default to original image

        if technique == 'Gaussian Blur':
            ksize = st.slider('Kernel Size', 1, 31, 5, step=2)
            enhanced_image = enhance_image(image, technique, ksize=ksize)
        elif technique == 'Canny Edge Detection':
            threshold1 = st.slider('Threshold 1', 0, 300, 100)
            threshold2 = st.slider('Threshold 2', 0, 300, 200)
            enhanced_image = enhance_image(image, technique, threshold1=threshold1, threshold2=threshold2)
        elif technique == 'CLAHE':
            clip_limit = st.slider('Clip Limit', 1.0, 40.0, 2.0)
            tile_grid_size = st.slider('Tile Grid Size', 1, 16, 8)
            enhanced_image = enhance_image(image, technique, clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size))
        elif technique == 'Histogram Equalization':
            enhanced_image = enhance_image(image, technique)

        st.image(enhanced_image, caption='Enhanced Image')

else:
    # Load metadata
    metadata = load_metadata()
    if metadata is not None:
        # Maintain the same image selection using session state
        if 'image_id' not in st.session_state:
            selected_image = metadata.sample()  # Randomly select an image
            st.session_state.image_id = selected_image['id'].values[0]
            st.session_state.county = selected_image['county'].values[0]
            st.session_state.ptb = selected_image['ptb'].values[0]

        image_url = get_github_file_url(f'image/{st.session_state.image_id}.webp')
        mask_url = get_github_file_url(f'mask/{st.session_state.image_id}.webp')

        # Load and display the image and its mask
        image, mask = load_image_and_mask(image_url, mask_url)
        if image is not None and mask is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption=f"Original Image - {st.session_state.county} - {st.session_state.ptb}")

                # Show/No-Show option for segmentation mask
                show_mask = st.checkbox("Show Segmentation Mask", value=True)
                if show_mask:
                    st.image(mask, caption="Segmentation Mask")

                # Overlay mask option
                overlay = st.checkbox("Overlay Segmentation Mask on Image", value=False)
                if overlay:
                    overlay_alpha = st.slider('Overlay Alpha', 0.0, 1.0, 0.5)
                    mask_color = st.selectbox('Select Mask Color', ['Red', 'Green', 'Blue'], index=0)
                    image_with_overlay = overlay_mask(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), mask, mask_color, alpha=overlay_alpha)
                    st.image(image_with_overlay, caption="Image with Segmentation Mask Overlay")

            with col2:
                # Select enhancement technique and display enhanced image
                technique = st.selectbox("Select Enhancement Technique", ['Original', 'Histogram Equalization', 'Gaussian Blur', 'Canny Edge Detection', 'CLAHE'])
                enhanced_image = enhance_image(image, technique)

                if technique == 'Gaussian Blur':
                    ksize = st.slider('Kernel Size', 1, 31, 5, step=2)
                    enhanced_image = enhance_image(image, technique, ksize=ksize)
                elif technique == 'Canny Edge Detection':
                    threshold1 = st.slider('Threshold 1', 0, 300, 100)
                    threshold2 = st.slider('Threshold 2', 0, 300, 200)
                    enhanced_image = enhance_image(image, technique, threshold1=threshold1, threshold2=threshold2)
                elif technique == 'CLAHE':
                    clip_limit = st.slider('Clip Limit', 1.0, 40.0, 2.0)
                    tile_grid_size = st.slider('Tile Grid Size', 1, 16, 8)
                    enhanced_image = enhance_image(image, technique, clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size))
                elif technique == 'Histogram Equalization':
                    enhanced_image = enhance_image(image, technique)

                st.image(enhanced_image, caption='Enhanced Image')
        else:
            st.error("Error loading image or mask. Please check the file paths.")
    else:
        st.error("No metadata available. Unable to proceed.")

st.write("This demo showcases different image enhancement techniques and highlights lung segmentation masks. Select a technique and adjust the parameters to see the effect on the image.")
