import cv2
import os
import numpy as np


class Preprocessing:
    def skewCorrection(self, img_org, threshold):
        """
        description: This function will correct the skew by using canny edge detection and
        hough Lines predictions
        :param img_org: Image in BGR format
        :param threshold: The threshold for median angle
        :return: Rotated Image in BGR format
        """

        # Making a copy of the original image for safe processing
        img = img_org.copy()

        # Making sure the image is valid
        if img is not None:

            # Converting the image to Gray scale for further processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Hough Lines Prediction
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50
            )

            # for storing the angles
            angles = []
            try:
                for line in lines:
                    # The coordinates from the lines detected from Hough Lines Prediction
                    x1, y1, x2, y2 = line[0]

                    # Finding the angles
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    angles.append(angle)

                # Finding the average angle
                median_angle = np.median(angles)

                # Making sure the average angle is within the threshold
                median_angle = (
                    0
                    if median_angle > threshold or median_angle < -threshold
                    else median_angle
                )

            except:
                # if any error is raised the image will not be corrected
                print("An error occurred, The image is not Corrected!")
                median_angle = 0

            # Taking the height and width of the image
            rows, cols = img.shape[:2]

            # Rotating the image
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), median_angle, 1)

            # Wrapping the image to its original height and width
            rotated = cv2.warpAffine(img, M, (cols, rows))

            return rotated

        else:
            return None
