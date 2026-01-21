import tensorflow as tf
from tensorflow.keras import layers, models

def build_vetiver_cnn(input_shape=(128, 128, 3), num_classes=3):
    model = models.Sequential([
        layers.Input(shape=input_shape),   # 🔥 NO batch_shape
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model
