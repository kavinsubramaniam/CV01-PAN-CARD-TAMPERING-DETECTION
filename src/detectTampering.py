import cv2
from ultralytics import YOLO
import pytesseract
import re
from preprocessor import Preprocessing


class DetectPancardTampering:
    """
    Class to detect tampering in PAN cards using YOLO for detection and Tesseract for OCR.
    """

    def __init__(self):
        """
        Initializes the preprocessor and the YOLO model.
        """
        self.preprocessor = Preprocessing()
        self.detection_model = YOLO(
            "../model/PanCardDetection/runs/detect/train12/weights/best.pt"
        )

    def detect(self, image, show=False, show_ocr=False):
        """
        Detects PAN card tampering in the given image.

        Args:
            image (np.ndarray): The input image containing a PAN card.
            show (bool): Flag to display the detection results.
            show_ocr (bool): Flag to display the OCR results.

        Returns:
            tuple: A tuple containing:
                - predictions (list): A list of tampering predictions for each detected PAN card.
                - pancard_returned_image (np.ndarray): The image with bounding boxes drawn around detected PAN cards.
        """
        image_copy = image.copy()
        preprocessed_img = self.preprocessor.skewCorrection(image_copy, angle=10)
        detections, pancard_returned_image = self.pancard_detection(
            preprocessed_img, show=show
        )
        predictions = [
            self.predict_tampering(detections[idx], show_ocr)
            for idx in range(len(detections))
        ]
        return predictions, pancard_returned_image

    def pancard_detection(self, skew_corrected_image, show=False):
        """
        Detects PAN cards in the given image using the YOLO model.

        Args:
            skew_corrected_image (np.ndarray): The input image after skew correction.
            show (bool): Flag to display the detection results.

        Returns:
            tuple: A tuple containing:
                - detections (list): A list of detected PAN cards with bounding boxes and confidence scores.
                - image_copy (np.ndarray): The image with bounding boxes drawn around detected PAN cards.
        """
        image_copy = skew_corrected_image.copy()
        detection_results = self.detection_model(image_copy)
        detections = []

        for result in detection_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pancard = image_copy[y1:y2, x1:x2]
                confidence = box.conf[0]
                detections.append(
                    {"pancard": pancard, "bbox": (x1, y1, x2, y2), "conf": confidence}
                )

                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                self._draw_bbox(image_copy, class_name, confidence, x1, y1, x2, y2)

                if show:
                    cv2.imshow("YOLO Detection", image_copy)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        return detections, image_copy

    def _draw_bbox(self, image, label, confidence, x1, y1, x2, y2):
        """
        Draws bounding box and label on the image.

        Args:
            image (np.ndarray): The image on which to draw.
            label (str): The label to display.
            confidence (float): The confidence score.
            x1 (int): Top-left x-coordinate of the bounding box.
            y1 (int): Top-left y-coordinate of the bounding box.
            x2 (int): Bottom-right x-coordinate of the bounding box.
            y2 (int): Bottom-right y-coordinate of the bounding box.
        """
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label}: {confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - label_height - baseline),
            (x1 + label_width, y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    def ocr(self, pancard, show_ocr=False):
        """
        Extracts text from the PAN card image using OCR.

        Args:
            pancard (np.ndarray): The PAN card image.
            show_ocr (bool): Flag to display the OCR results.

        Returns:
            list: A list of OCR detections containing bounding boxes, confidence scores, and extracted text.
        """
        data = pytesseract.image_to_data(pancard)
        detections = []

        for i in data.split("\n")[1:]:
            temp_data = i.split("\t")[-6:]
            if len(temp_data[-1].strip()) > 1:
                try:
                    bbox = tuple(map(int, temp_data[:4]))
                    confidence = temp_data[4]
                    text = temp_data[5]
                    detections.append({"bbox": bbox, "conf": confidence, "text": text})
                    self._draw_ocr_text(pancard, bbox, text)
                except:
                    continue

        if show_ocr:
            cv2.imshow("OCR Results", pancard)
            cv2.waitKey(0)

        return detections

    def _draw_ocr_text(self, image, bbox, text):
        """
        Draws OCR text on the image.

        Args:
            image (np.ndarray): The image on which to draw.
            bbox (tuple): The bounding box coordinates for the text.
            text (str): The extracted text.
        """
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            text,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def predict_tampering(self, detection, show_ocr=False):
        """
        Predicts whether the PAN card is tampered with based on OCR results.

        Args:
            detection (dict): A dictionary containing detected PAN card information.
            show_ocr (bool): Flag to display the OCR results.

        Returns:
            int: 1 if tampering is detected, 0 otherwise.
        """
        pancard = detection["pancard"]
        gray = cv2.cvtColor(pancard, cv2.COLOR_BGR2GRAY)

        # Apply different preprocessing techniques for OCR
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        ocr_extractions = (
            self.ocr(thresh, show_ocr)
            + self.ocr(pancard, show_ocr)
            + self.ocr(gray, show_ocr)
        )

        return self._analyze_ocr_data(ocr_extractions)

    def _analyze_ocr_data(self, ocr_data):
        """
        Analyzes OCR data to detect tampering based on PAN card patterns.

        Args:
            ocr_data (list): A list of OCR detections.

        Returns:
            int: 1 if tampering is detected, 0 otherwise.
        """
        patterns = [r"[A-Z]{5}[0-9]{3}[A-Z]", r"\b\d{2}/\d{2}/\d{4}\b"]
        flags = [0, 0]

        for data in ocr_data:
            if flags[0] == 0 and re.search(patterns[0], data["text"]):
                flags[0] = 1
            if flags[1] == 0 and re.search(patterns[1], data["text"]):
                flags[1] = 1

        return 1 if all(flags) else 0


if __name__ == "__main__":
    img = cv2.imread("../data/raw/4.jpg")
    detector = DetectPancardTampering()
    predictions, processed_image = detector.detect(img)
    print(predictions)
