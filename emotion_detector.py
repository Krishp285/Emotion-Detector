# Real-Time Emotion Detector by [Your Name]
import tensorflow as tf
import cv2
import numpy as np

# Paths
train_dir = "C:/emotion_detector_project/train"
test_dir = "C:/emotion_detector_project/test"

# Load data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    label_mode='categorical'
).map(lambda x, y: (x / 255.0, y))

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    label_mode='categorical'
).map(lambda x, y: (x / 255.0, y))

# Build CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')  # 7 emotions
])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_dataset, epochs=15, validation_data=test_dataset)

# Save
model.save("emotion_model.h5")