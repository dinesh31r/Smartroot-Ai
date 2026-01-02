import tensorflow as tf
import os

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 8

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Build CNN
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(train_ds, epochs=EPOCHS)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/vetiver_cnn.h5")

print("✅ CNN model trained and saved at model/vetiver_cnn.h5")
