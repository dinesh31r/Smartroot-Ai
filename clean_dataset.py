import os
import cv2

DATASET_DIR = "dataset"
IMG_SIZE = 128

def clean_and_resize(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            img = cv2.imread(path)
            if img is None:
                os.remove(path)
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(path, img)
        except:
            if os.path.exists(path):
                os.remove(path)

for cls in ["healthy", "low_moisture", "low_nutrient"]:
    folder_path = os.path.join(DATASET_DIR, cls)
    if os.path.exists(folder_path):
        clean_and_resize(folder_path)

print("Dataset cleaned and resized successfully")
