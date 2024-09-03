import streamlit as st
import cv2
from detectTampering import DetectPancardTampering
import numpy as np
from PIL import Image

# model = DetectPancardTampering()
# Function to load and process the uploaded image
def load_image(image_file):
    img = Image.open(image_file)
    return img


# Function to simulate model prediction (replace with your actual model prediction code)
def predict_image(image):
    obj = DetectPancardTampering()
    return obj.detect(image)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sidebar navigation with radio buttons
st.sidebar.title("Navigation")

# Create a radio button widget inside the sidebar
st.session_state.page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Model Testing"],
    index=["Home", "Model Testing"].index(st.session_state.page)
)

# Display content based on the current page
if st.session_state.page == "Home":

    # Home page with project description
    st.title("Project Description ")
    st.write("""
    ## PAN Card Tampering Detection
    The PAN Card Tampering Detection project aims to identify fraudulent or altered PAN cards by leveraging 
    state-of-the-art deep learning and computer vision techniques. This project integrates YOLO (You Only Look Once) 
    object detection, OpenCV image processing, and Optical Character Recognition (OCR) to detect and analyze PAN 
    cards from images. The goal is to determine whether the PAN card in an image has been tampered with by analyzing 
    the extracted text and its alignment with expected patterns.""")
    st.write("")
    st.write("""### Key Components:""")

    st.write("""#### 1. Preprocessing:
- The image undergoes skew correction to align the PAN card properly. This ensures that the text 
extraction and detection processes are more accurate.""")

    st.write("""#### 2. YOLO-Based PAN Card Detection:
- A YOLO model, trained specifically to detect PAN cards, is used to identify the location of the card in the 
image. The detection is performed after the image has been preprocessed for skew.
- The YOLO model outputs bounding boxes around the detected PAN card, which are then used to crop 
the image for further processing.""")

    st.write("""#### 3. OCR (Optical Character Recognition):
- OCR is applied to the detected and cropped PAN card to extract the text. This text includes critical
 elements such as the PAN number and date of birth.
- The text is extracted from the binary image and the grayscale image to enhance 
the quality of OCR outputs.""")

    st.write("""#### 4. Tampering Detection:
- Regular expressions are used to validate the format of the PAN number and date of birth. 
If the extracted text matches the expected patterns, the card is considered genuine; otherwise, tampering is suspected.""")
    st.write("")
    st.write("""### Workflow:
- **Input:** &nbsp; The system takes an image containing a PAN card.
- **Preprocessing:** &nbsp; The image undergoes skew correction to align the PAN card properly.
- **Detection:** &nbsp; YOLO model detects the PAN card in the image, and the relevant portion is cropped.
- **OCR Processing:** &nbsp; Text is extracted from the cropped image using OCR.
- **Validation:** &nbsp; Extracted text is analyzed to detect tampering by checking the format and presence of essential elements like the PAN number and date of birth.
- **Output:** &nbsp; The system outputs a tampering prediction—whether the PAN card is likely tampered with or not.""")
    st.write("")
    st.write("""### Usage:
- This project can be used in automated systems for verifying the authenticity of PAN cards in banks, financial 
institutions, and government services.
- By identifying discrepancies in the text on PAN cards, this system can aid in detecting fraudulent activities.""")
    st.write("")
    st.write("""### Future Enhancements:
- **Enhanced Model:** Further training the YOLO model on a more diverse dataset to improve detection accuracy.
- **Integration:** Integrating this system into larger document verification frameworks.""")
    st.write("")
    st.write("""### Conclusion:
The PAN Card Tampering Detection project offers an efficient and automated way to ensure the integrity of PAN cards, 
leveraging advanced deep learning and image processing techniques. By combining YOLO-based detection with OCR, 
this system provides a robust solution for identifying tampered documents.""")
    st.write("")
    st.markdown("### References and Resources:")
    st.markdown("""
    1. **YOLO (You Only Look Once) - Ultralytics**  
       YOLO is a real-time object detection system developed by Joseph Redmon and Ali Farhadi. The specific implementation used in this project is by Ultralytics, which provides an accessible and high-performance version of YOLO.  
       - **Website**: [Ultralytics YOLO](https://ultralytics.com/yolov5)  
       - **GitHub Repository**: [Ultralytics YOLO GitHub](https://github.com/ultralytics/yolov5)  
       - **Paper**: Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.

    2. **Tesseract OCR - Google**  
       Tesseract is an open-source OCR (Optical Character Recognition) engine originally developed by HP and later maintained by Google. It is widely used for extracting text from images and supports multiple languages.  
       - **Website**: [Tesseract OCR](https://opensource.google/projects/tesseract)  
       - **GitHub Repository**: [Tesseract OCR GitHub](https://github.com/tesseract-ocr/tesseract)  
       - **Documentation**: [Tesseract Documentation](https://tesseract-ocr.github.io/)

    3. **Streamlit**  
       Streamlit is an open-source framework that allows for the rapid development and deployment of machine learning and data science applications with an easy-to-use Python interface.  
       - **Website**: [Streamlit](https://streamlit.io/)  
       - **GitHub Repository**: [Streamlit GitHub](https://github.com/streamlit/streamlit)  
       - **Documentation**: [Streamlit Documentation](https://docs.streamlit.io/)
    """)
elif st.session_state.page == "Model Testing":
    # Model Testing page with image upload
    st.title("Model Testing")

    image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        col1, col2 = st.columns(2)
        img = load_image(image_file)

        # Get the result from the model
        result, image = predict_image(np.array(img))

        # Display the uploaded image
        for i, val in enumerate(result):
            col1.image(image, caption='Uploaded Image', use_column_width=True)
            col2.markdown(f"Result for pancard {i+1} : {"Positive" if val == 1 else "Negative"}")

