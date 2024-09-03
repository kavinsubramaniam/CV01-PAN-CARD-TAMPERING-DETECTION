from ultralytics import YOLO
import os
import cv2

# Load the YOLO model
model = YOLO("../../model/PanCardDetection/runs/detect/train9/weights/best.pt")

# Source and destination paths
src = os.path.join(os.getcwd().split("src")[0], "data", "preprocessed")
des = os.path.join(os.getcwd().split("src")[0], "data", "detected_pancards")

# Create the destination directory if it doesn't exist
os.makedirs(des, exist_ok=True)

# Extract the names of the files in the src folder
files = os.listdir(src)

for i in files:
    # Check if the file is an image
    if i.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            # Reading the image
            img = cv2.imread(os.path.join(src, i))

            if img is None:
                print(f"Failed to load image {i}. Skipping...")
                continue

            # Run the model on the image
            results = model(img)

            # Iterate through each result (usually just one result per image)
            for result in results:
                # Extract bounding boxes from the result
                for j, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Crop the image using the bounding box coordinates
                    cropped_img = img[y1:y2, x1:x2]

                    # Save the cropped image with a unique name
                    cropped_img_path = os.path.join(
                        des, f"{i.split('.')[0]}_detected_pancard_{j}.jpg"
                    )
                    cv2.imwrite(cropped_img_path, cropped_img)

        except Exception as e:
            print(f"An error occurred while processing {i}: {e}")

print("Successfully detected and cropped the PAN cards. Saved in:", des)
