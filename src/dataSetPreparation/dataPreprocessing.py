from src.preprocessor import Preprocessing
import os
import cv2

# Define source and destination paths
path = os.path.join(os.getcwd().split("src")[0], "data", "raw")
des = os.path.join(os.getcwd().split("src")[0], "data", "preprocessed")

# Create the destination directory if it doesn't exist
os.makedirs(des, exist_ok=True)

# List files in the source directory
files = os.listdir(path)

for i in files:
    # Process only image files
    if i.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            # Read the image
            img = cv2.imread(os.path.join(path, i))

            if img is None:
                print(f"Failed to load image {i}. Skipping...")
                continue

            # Create an instance of Preprocessing
            obj = Preprocessing()

            # Apply skew correction
            corrected_img = obj.skewCorrection(img, 20)

            # Save the corrected image
            save_path = os.path.join(des, f"{i.split('.')[0]}_skew_corrected.jpg")
            cv2.imwrite(save_path, corrected_img)

        except Exception as e:
            print(f"An error occurred while processing {i}: {e}")

print("The images have been preprocessed successfully.")
