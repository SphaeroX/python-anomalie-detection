import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from config import use_contour, resize_x, resize_y, video_path, contour_threshold

def apply_contour_detection(frame, threshold=contour_threshold):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blurred_frame, threshold, threshold * 2)
    return edges

def preprocess_video(video_path, use_contour=True):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if use_contour:
            frame = apply_contour_detection(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        resized_frame = cv2.resize(frame, (resize_x, resize_y))
        frames.append(resized_frame)

    cap.release()
    return np.array(frames)

def create_autoencoder_model():
    input_shape = (resize_x, resize_y, 3)

    inputs = Input(input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    encoded = x

    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(encoded)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2)(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

frames = preprocess_video(video_path, use_contour)
x_train = frames / 255.0

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.1,
    horizontal_flip=True
)

# Splitting data into training and validation
validation_split = 0.2
split_index = int((1 - validation_split) * len(x_train))
x_train_data = x_train[:split_index]
x_validation_data = x_train[split_index:]

train_generator = datagen.flow(x_train_data, x_train_data, batch_size=4)
validation_generator = datagen.flow(x_validation_data, x_validation_data, batch_size=4)

autoencoder = create_autoencoder_model()

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training the model
history = autoencoder.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping]
)

autoencoder.save('autoencoder_model.h5')

# Verlustwert plotten
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

print(f"Anzahl der trainierten Frames: {len(x_train)}")
