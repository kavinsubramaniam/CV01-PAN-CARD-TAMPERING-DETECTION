import streamlit as st
import cv2
from detectTampering import DetectPancardTampering
import numpy as np
from PIL import Image


# Function to load and process the uploaded image
def load_image(image_file):
    """
    Load an image file using PIL and return it.

    Args:
        image_file: The uploaded image file.

    Returns:
        img (PIL.Image): The loaded image.
    """
    img = Image.open(image_file)
    return img


# Function to predict tampering on the uploaded image
def predict_image(image):
    """
    Detect potential tampering on a PAN card image using the DetectPancardTampering model.

    Args:
        image (numpy.ndarray): The image array of the PAN card.

    Returns:
        predictions (list): A list of tampering predictions (1 for tampered, 0 for not tampered).
        processed_image (numpy.ndarray): The image with detection bounding boxes drawn.
    """
    obj = DetectPancardTampering()
    return obj.detect(image)


# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar navigation
st.sidebar.title("Navigation")

# Radio button widget for page selection
st.session_state.page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Model Testing"],
    index=["Home", "Model Testing"].index(st.session_state.page),
)

# Content display based on the selected page
if st.session_state.page == "Home":
    # Home page with project description
    st.title("Project Description")
    st.write(
        """
    ## PAN Card Tampering Detection
    The PAN Card Tampering Detection project is designed to identify fraudulent or altered PAN cards 
    using advanced deep learning and computer vision techniques. The project integrates YOLO (You Only Look Once) 
    object detection, OpenCV image processing, and Optical Character Recognition (OCR) to analyze PAN cards 
    and detect any tampering in the provided images.
    """
    )

    st.write("""### Key Components:""")
    st.write(
        """#### 1. Preprocessing:
    - The image undergoes skew correction to properly align the PAN card, enhancing detection and OCR accuracy."""
    )

    st.write(
        """#### 2. YOLO-Based PAN Card Detection:
    - A YOLO model detects the PAN card in the image and outputs bounding boxes around it. 
    This enables cropping for further processing."""
    )

    st.write(
        """#### 3. OCR (Optical Character Recognition):
    - OCR is applied to extract text from the detected and cropped PAN card, including the PAN number and date of birth."""
    )

    st.write(
        """#### 4. Tampering Detection:
    - Regular expressions validate the PAN number and date of birth formats. 
    Mismatches in the extracted text may indicate tampering."""
    )

    st.write(
        """### Workflow:
    - **Input:** Image with a PAN card.
    - **Preprocessing:** Skew correction to align the PAN card.
    - **Detection:** YOLO model detects and crops the PAN card from the image.
    - **OCR Processing:** Text extraction using OCR.
    - **Validation:** Text analysis for tampering detection based on format validation.
    - **Output:** Prediction of whether the PAN card is tampered with or not."""
    )

    st.write(
        """### Usage:
    - The project can be used in automated systems for PAN card verification in financial institutions 
    and government services."""
    )

    st.write(
        """### Future Enhancements:
    - **Enhanced Model:** Further training the YOLO model on a more diverse dataset.
    - **Integration:** Incorporating this system into broader document verification frameworks."""
    )

    st.write(
        """### Conclusion:
    The PAN Card Tampering Detection project offers an efficient and automated solution for ensuring 
    the integrity of PAN cards by leveraging deep learning and image processing techniques."""
    )

    st.markdown("### References and Resources:")
    st.markdown(
        """
    1. **YOLO (You Only Look Once) - Ultralytics**
       - **Website**: [Ultralytics YOLO](https://ultralytics.com/yolov5)
       - **GitHub Repository**: [Ultralytics YOLO GitHub](https://github.com/ultralytics/yolov5)

    2. **Tesseract OCR - Google**
       - **Website**: [Tesseract OCR](https://opensource.google/projects/tesseract)
       - **GitHub Repository**: [Tesseract OCR GitHub](https://github.com/tesseract-ocr/tesseract)

    3. **Streamlit**
       - **Website**: [Streamlit](https://streamlit.io/)
       - **GitHub Repository**: [Streamlit GitHub](https://github.com/streamlit/streamlit)
    """
    )

elif st.session_state.page == "Model Testing":
    # Model Testing page with image upload functionality
    st.title("Model Testing")

    image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        col1, col2 = st.columns(2)
        img = load_image(image_file)

        # Get the result from the model
        result, image = predict_image(np.array(img))

        # Display the uploaded image
        col1.image(image, caption="Uploaded Image", use_column_width=True)

        # Display the prediction results
        for i, val in enumerate(result):
            col2.markdown(
                f"Result for PAN card {i + 1}: {'Tampered' if val == 1 else 'Not Tampered'}"
            )
