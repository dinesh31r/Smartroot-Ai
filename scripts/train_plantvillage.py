"""
Train MobileNetV2 for PlantVillage
SmartRoot-AI Support Script
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DATASET_PATH = "data/plantvillage"  # Directory where dataset will be extracted
IMG_SIZE = (128, 128)               # Matches SmartRoot-AI requirement
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 38                    # Typical PlantVillage count, adjusts automatically

# --------------------------------------------------
# DATA PREPARATION (AUGMENTATION)
# --------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

def get_data_loaders(path):
    train_gen = train_datagen.flow_from_directory(
        path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = train_datagen.flow_from_directory(
        path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_gen, val_gen

# --------------------------------------------------
# MODEL DEFINITION (TRANSFER LEARNING)
# --------------------------------------------------
def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(128, 128, 3), 
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze the base model to preserve pretrained weights
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}.")
        print("Please download PlantVillage and extract it to this folder.")
        exit(1)
        
    train_gen, val_gen = get_data_loaders(DATASET_PATH)
    num_classes = train_gen.num_classes
    
    print(f"✅ Training for {num_classes} classes...")
    model = build_model(num_classes)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )
    
    # Save for SmartRoot-AI
    os.makedirs("model", exist_ok=True)
    model_save_path = "model/mobilenetv2_plantvillage.h5"
    model.save(model_save_path)
    print(f"🚀 Training Complete! Model saved to: {model_save_path}")
