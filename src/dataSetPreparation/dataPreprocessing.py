from src.preprocessor import Preprocessing
import os
import cv2

path = os.getcwd().split("src")[0] + r"data\raw"
des = os.getcwd().split("src")[0] + r"data\preprocessed"

if os.path.exists(des):
    pass
else:
    os.makedirs(des)
    print("Destination Directory created successfully!")

files = os.listdir(path)
for i in files:
    img = cv2.imread(os.path.join(path, i))
    obj = Preprocessing()
    cv2.imwrite(os.path.join(des, f"{i.split('.')[0]}_skew_corrected.jpg"), obj.skewCorrection(img, 20))

print("The images the preprocessed successfully")
