import cv2
import numpy as np


class Preprocessing:
    def skewCorrection(self, img_org, threshold):
        """
        This function corrects the skew of an image using Canny edge detection and
        Hough Line Transformation.

        Args:
            img_org (numpy.ndarray): Image in BGR format.
            threshold (float): Threshold to ignore small skew angles.

        Returns:
            numpy.ndarray: Rotated image in BGR format.
        """

        # Make a copy of the original image
        img = img_org.copy()

        # Ensure the image is valid
        if img is not None:
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Hough Line Transformation to detect lines
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50
            )

            angles = []
            try:
                for line in lines:
                    # Extract the coordinates of the line
                    x1, y1, x2, y2 = line[0]

                    # Calculate the angle of the line
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    angles.append(angle)

                # Calculate the median angle
                median_angle = np.median(angles)

                # Ignore small angles (within the threshold)
                if abs(median_angle) > threshold:
                    print(
                        f"Applying skew correction with angle: {median_angle} degrees"
                    )
                else:
                    print(
                        f"Skew angle {median_angle} is within the threshold. No correction needed."
                    )
                    median_angle = 0

            except Exception as e:
                print(f"An error occurred during skew correction: {e}")
                median_angle = 0

            # Get the image dimensions
            rows, cols = img.shape[:2]

            # Calculate the rotation matrix
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), median_angle, 1)

            # Rotate the image
            rotated = cv2.warpAffine(img, M, (cols, rows))

            return rotated

        else:
            print("Invalid image input!")
            return None
