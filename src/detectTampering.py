from preprocessor import Preprocessing
import cv2
from ultralytics import YOLO
import pytesseract
import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'best.pt')

class DetectPancardTampering:
    def __init__(self):
        self.preprocessor = Preprocessing()
        self.detectionModel = YOLO(model_path)

    def detect(self, image, show=False, show_ocr=False):
        image_copy = image.copy()
        preprocessed_img = self.preprocessor.skewCorrection(image_copy, 10)
        detection_of_pancard, pancard_returned_image = self.pancardDetection(preprocessed_img, show=show)
        predictions = [0] * len(detection_of_pancard)
        for idx, detections in enumerate(detection_of_pancard):
            pancard = detections['pancard']
            gray = cv2.cvtColor(pancard, cv2.COLOR_BGR2GRAY)

            thresh = cv2.adaptiveThreshold(gray, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
            ocr_extraction_thresh = self.ocr(thresh, pancard.copy(), show_ocr)
            ocr_extraction_org = self.ocr(pancard, pancard.copy(), show_ocr)
            ocr_extraction_gray = self.ocr(gray, pancard.copy(), show_ocr)
            ocr_extraction = ocr_extraction_thresh+ocr_extraction_org+ocr_extraction_gray
            predictions[idx] = self.prediction(ocr_extraction)
        return predictions, pancard_returned_image

    def pancardDetection(self, skew_corrected_image, show=False):
        image_copy_1 = skew_corrected_image.copy()
        detection = self.detectionModel(image_copy_1)
        detections = []
        for result in detection:
            for count, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pancard = image_copy_1[y1:y2, x1:x2]
                confidence = box.conf[0]
                detections.append({"pancard": pancard,
                                   "bbox": (x1, y1, x2, y2),
                                   "conf": confidence})
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                cv2.rectangle(image_copy_1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f} , PANCARD NO: {count+1}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image_copy_1, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0),
                              -1)
                cv2.putText(image_copy_1, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return detections, image_copy_1

    def ocr(self, pancard, original, show_ocr=False):
        image_copy_2 = pancard.copy()
        data = pytesseract.image_to_data(image_copy_2)
        detections = []
        for i in data.split('\n')[1:]:
            temp_d = i.split('\t')[-6:]
            if len(temp_d[-1].strip()) > 1:
                try:
                    detections.append({"bbox": tuple(temp_d[:4]), "conf": temp_d[4], "text": temp_d[5]})
                except:
                    continue

                x, y, w, h = temp_d[:4]
                cv2.rectangle(original, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (int(x) + 10, int(y) - 10)
                fontScale = 0.8
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(original, f'{temp_d[-1]}', org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        return detections

    def prediction(self, ocr_data):
        flag = [0, 0]
        for data in ocr_data:
            if flag[0] == 0 and re.search(r"[A-Z]{5}[0-9]{3}.[A-Z]", data["text"]) is not None:
                flag[0] = 1
            if flag[1] == 0 and re.search(r"\b\d{2}/\d{2}/\d{4}\b", data["text"]) is not None:
                flag[1] = 1
        if all(flag):
            return 1
        return 0
