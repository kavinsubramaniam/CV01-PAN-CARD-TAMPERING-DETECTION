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

if st.session_state.page == "Home":
    st.title("Project Description")
    st.markdown(
        """
    <h2>PAN Card Tampering Detection</h2>
    <p>The PAN Card Tampering Detection project is designed to identify fraudulent or altered PAN cards 
    using advanced deep learning and computer vision techniques. The project integrates YOLO (You Only Look Once) 
    object detection, OpenCV image processing, and Optical Character Recognition (OCR) to analyze PAN cards 
    and detect any tampering in the provided images.</p>
    """, unsafe_allow_html=True
    )

    st.markdown(
        """
    <h3>Key Components:</h3>
    <ul>
        <li><strong>Preprocessing:</strong> The image undergoes skew correction to properly align the PAN card, enhancing detection and OCR accuracy.</li>
        <li><strong>YOLO-Based PAN Card Detection:</strong> A YOLO model detects the PAN card in the image and outputs bounding boxes around it. This enables cropping for further processing.</li>
        <li><strong>OCR (Optical Character Recognition):</strong> OCR is applied to extract text from the detected and cropped PAN card, including the PAN number and date of birth.</li>
        <li><strong>Tampering Detection:</strong> Regular expressions validate the PAN number and date of birth formats. Mismatches in the extracted text may indicate tampering.</li>
    </ul>
    """, unsafe_allow_html=True
    )

    st.markdown(
        """
    <h3>Workflow:</h3>
    <ul>
        <li><strong>Input:</strong> Image with a PAN card.</li>
        <li><strong>Preprocessing:</strong> Skew correction to align the PAN card.</li>
        <li><strong>Detection:</strong> YOLO model detects and crops the PAN card from the image.</li>
        <li><strong>OCR Processing:</strong> Text extraction using OCR.</li>
        <li><strong>Validation:</strong> Text analysis for tampering detection based on format validation.</li>
        <li><strong>Output:</strong> Prediction of whether the PAN card is tampered with or not.</li>
    </ul>
    """, unsafe_allow_html=True
    )

    st.markdown(
        """
    <h3>Usage:</h3>
    <p>The project can be used in automated systems for PAN card verification in financial institutions and government services.</p>
    """, unsafe_allow_html=True
    )

    st.markdown(
        """
    <h3>Future Enhancements:</h3>
    <ul>
        <li><strong>Enhanced Model:</strong> Further training the YOLO model on a more diverse dataset.</li>
        <li><strong>Integration:</strong> Incorporating this system into broader document verification frameworks.</li>
    </ul>
    """, unsafe_allow_html=True
    )

    st.markdown(
        """
    <h3>Conclusion:</h3>
    <p>The PAN Card Tampering Detection project offers an efficient and automated solution for ensuring 
    the integrity of PAN cards by leveraging deep learning and image processing techniques.</p>
    """, unsafe_allow_html=True
    )

    st.markdown(
        """
    <h3>References and Resources:</h3>
    <ul>
        <li><strong>YOLO (You Only Look Once) - Ultralytics</strong><br>
            <a href="https://ultralytics.com/yolov5">Ultralytics YOLO</a><br>
            <a href="https://github.com/ultralytics/yolov5">Ultralytics YOLO GitHub</a></li>
        <li><strong>Tesseract OCR - Google</strong><br>
            <a href="https://opensource.google/projects/tesseract">Tesseract OCR</a><br>
            <a href="https://github.com/tesseract-ocr/tesseract">Tesseract OCR GitHub</a></li>
        <li><strong>Streamlit</strong><br>
            <a href="https://streamlit.io/">Streamlit</a><br>
            <a href="https://github.com/streamlit/streamlit">Streamlit GitHub</a></li>
    </ul>
    """, unsafe_allow_html=True
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
                f"Result for PAN card {i + 1}: {'Tampered' if val == 0 else 'Not Tampered'}"
            )

    st.markdown(">Sometimes The result will be affected by the quality of photo which is provided.")
