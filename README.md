# PAN Card Tampering Detection

## Overview

The PAN Card Tampering Detection project aims to identify fraudulent or altered PAN cards using advanced deep learning and computer vision techniques. The project integrates YOLO (You Only Look Once) object detection, OpenCV for image processing, and Optical Character Recognition (OCR) to analyze PAN cards from images and detect any tampering.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Features

- **Preprocessing:** Skew correction using Hough Lines and Canny Edge detection to align the PAN card properly.
- **YOLO-Based Detection:** Utilizes YOLOv8 for detecting PAN cards in images.
- **OCR Extraction:** Extracts key information like PAN number and date of birth using OCR.
- **Tampering Detection:** Validates extracted text to detect tampering by comparing with expected formats.

## Project Structure

├── data
│   ├── raw                        # Raw input images
│   ├── preprocessed               # Skew corrected images
│   ├── detected_pancards          # Cropped images with detected PAN cards
│   └── annotated                  # Annotated data for training
├── model
│   ├── YOLO model weights
│   ├── PanCardDetection
│   │   ├── runs
│   │   │   └── detects            # All the weights of fine-tuned YOLOv8
│   │   ├── yolov8Training.py      # Fine-tuning YOLOv8n
│   │   └── yolov8n.pt             # YOLOv8n model file
│   └── TokenClassification
│       ├── layoutLMv2Training.py  # Fine-tuning LayoutLMv2 for PAN cards
│       └── layoutLMv2Testing.py   # Testing LayoutLMv2
├── src
│   ├── preprocessor
│   │   └── preprocessor.py        # Preprocessing module where skew correction is done
│   ├── dataSetPreparation
│   │   ├── dataPreprocessing.py   # The preprocessing for dataset creation
│   │   └── panCardDetection.py    # Detecting the PAN cards using YOLOv8
│   ├── app.py                     # Script to run the Streamlit app
│   └── detectTampering.py         # Script to identify tampering
└── README.md                      # Project documentation


## Setup guide

1. **Clone the repository:**

    ```bash
    git clone https://github.com/kavinsubramaniam/CV01-PAN-CARD-TAMPERING-DETECTION.git
    cd CV01-PAN-CARD-TAMPERING-DETECTION
    ```
2. **Create a virtual environment:** 

	```bash 
	python -m venv env source env/bin/activate # On Windows, use 
	`env\Scripts\activate`
	```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```
4. **Run the Project**
	```bash
	cd src
	streamlit run app.py
	```

## Model Training

If you wish to train the YOLO model on your own dataset, follow these steps:

1. Prepare your dataset and update the `data.yaml` file under the `data/annotated/` directory.
2. Run the following command to start training:

    ```bash
    cd model/PanCardDetection
    python yolov8Training.py
    ```

3. The trained model will be saved in the `model/PanCardDetection/runs/detects` directory.

## Future Enhancements

- **Enhanced Model:** Further training the YOLO model on a more diverse dataset to improve detection accuracy.
- **Integration:** Integrating layoutLM to understand the layout features of the pancard  which improves the tampering detection.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the   Apache License - see the [LICENSE](LICENSE) file for details.

## References

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
---