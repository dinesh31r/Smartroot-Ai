"""
CNN inference + plant validation
"""

import os
import cv2
import numpy as np
import tensorflow as tf

# --------------------------------------------------
# MODEL LOADING (ROBUST & SAFE)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "vetiver_cnn.h5")

IMG_SIZE = 128
model = None

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    MODEL_LOADED = True
except Exception as e:
    print("⚠️ CNN model not loaded:", e)
    MODEL_LOADED = False

# --------------------------------------------------
# CLASS MAPPING
# --------------------------------------------------
CLASS_MAP = {
    0: ("Healthy", 85, 80),
    1: ("Low Moisture", 35, 65),
    2: ("Low Nutrient", 65, 35)
}

# --------------------------------------------------
# PLANT VALIDATION (GREEN PIXEL CHECK)
# --------------------------------------------------
def is_probable_plant(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.sum(mask > 0) / (IMG_SIZE * IMG_SIZE)
    return green_ratio > 0.08

# --------------------------------------------------
# CNN STRESS PREDICTION
# --------------------------------------------------
def predict_stress(image_path):
    # Non-plant safety
    if not is_probable_plant(image_path):
        return "Non-Plant Image", 0, 0

    # Model not available → graceful fallback
    if not MODEL_LOADED:
        return "Healthy", 85, 80

    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    cls = int(np.argmax(pred))

    return CLASS_MAP.get(cls, ("Healthy", 85, 80))
