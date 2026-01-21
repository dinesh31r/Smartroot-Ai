"""
CNN Inference + Plant Validation
SmartRoot-AI
Python 3.10 | TensorFlow 2.12
Exact architecture + confidence-aware inference
"""

# --------------------------------------------------
# ENVIRONMENT SETUP
# --------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf
from functools import lru_cache

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "vetiver_cnn.h5")

IMG_SIZE = 128
MODEL_LOADED = False

# --------------------------------------------------
# EXACT CNN ARCHITECTURE (FROM INSPECTION)
# --------------------------------------------------
def build_vetiver_cnn():
    model = tf.keras.Sequential(name="vetiver_cnn")

    model.add(tf.keras.layers.Input(shape=(128, 128, 3), name="input_layer"))
    model.add(tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling"))

    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu", name="conv2d"))
    model.add(tf.keras.layers.MaxPooling2D(2, 2, name="max_pooling2d"))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2d_1"))
    model.add(tf.keras.layers.MaxPooling2D(2, 2, name="max_pooling2d_1"))

    model.add(tf.keras.layers.Flatten(name="flatten"))
    model.add(tf.keras.layers.Dense(64, activation="relu", name="dense"))
    model.add(tf.keras.layers.Dense(3, activation="softmax", name="dense_1"))

    return model

# --------------------------------------------------
# LOAD CNN (EXACT MATCH)
# --------------------------------------------------
@lru_cache(maxsize=1)
def get_cnn_model():
    global MODEL_LOADED
    try:
        model = build_vetiver_cnn()
        model.build((None, 128, 128, 3))
        model.load_weights(MODEL_PATH)
        MODEL_LOADED = True
        print("✅ CNN loaded successfully")
        return model
    except Exception as e:
        MODEL_LOADED = False
        print("⚠️ CNN model not loaded:", e)
        return None

# --------------------------------------------------
# CLASS MAP
# --------------------------------------------------
CLASS_MAP = {
    0: ("Healthy", 85, 80),
    1: ("Low Moisture", 35, 65),
    2: ("Low Nutrient", 65, 35)
}

# --------------------------------------------------
# PLANT VALIDATION
# --------------------------------------------------
def is_probable_plant(image, pre_resized=False):
    if image is None:
        return False

    img = image if pre_resized else cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([25, 40, 40]),
        np.array([95, 255, 255])
    )

    green_ratio = np.sum(mask > 0) / (IMG_SIZE * IMG_SIZE)
    return green_ratio > 0.08

# --------------------------------------------------
# CNN PREDICTION (CONFIDENCE + WEAKNESS FLAG)
# --------------------------------------------------
def predict_stress(image_path):
    """
    Returns:
    label, moisture, nutrient, confidence, cnn_weak
    """

    img = cv2.imread(image_path)
    if img is None:
        return "Invalid Image", 0, 0, 0.0, True

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    if not is_probable_plant(resized, pre_resized=True):
        return "Non-Plant Image", 0, 0, 0.0, True

    model = get_cnn_model()
    if not model:
        return "Healthy", 85, 80, 0.0, True

    img = np.expand_dims(resized.astype("float32"), axis=0)

    pred = model.predict(img, verbose=0)
    confidence = float(np.max(pred))
    entropy = -np.sum(pred * np.log(pred + 1e-9))

    cnn_weak = (confidence < 0.60) or (entropy > 0.90)

    cls = int(np.argmax(pred))
    label, moisture, nutrient = CLASS_MAP.get(cls, ("Healthy", 85, 80))

    return label, moisture, nutrient, confidence, cnn_weak
