import tensorflow as tf

model = tf.keras.models.load_model("vetiver_cnn.h5")

# Save in modern format
model.save("vetiver_cnn.keras")
