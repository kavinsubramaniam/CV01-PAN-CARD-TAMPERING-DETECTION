from ultralytics import YOLO
import os
import cv2

model = YOLO("../../model/PanCardDetection/runs/detect/train9/weights/best.pt")

# Source and destination paths
src = os.getcwd().split("src")[0] + r"data\preprocessed"
des = os.getcwd().split("src")[0] + r"data\detected_pancards"

# Creating the destination location if it doesn't exist
if os.path.exists(des):
    pass
else:
    os.makedirs(des)
    print("Destination Directory created successfully!")

# Extracting the names of the files in the src folder
files = os.listdir(src)

for i in files:
    # Reading the image
    img = cv2.imread(os.path.join(src, i))

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
            cropped_img_path = os.path.join(des, f"{i.split('.')[0]}_detected_pancard_{j}.jpg")
            cv2.imwrite(cropped_img_path, cropped_img)

print("Successfully detected and cropped the pan cards. Saved in : ", des)
