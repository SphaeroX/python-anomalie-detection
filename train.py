import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
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

autoencoder = create_autoencoder_model()
history = autoencoder.fit(x_train, x_train, epochs=10, batch_size=4)

autoencoder.save('autoencoder_model.h5')

# Verlustwert plotten
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()

print(f"Anzahl der trainierten Frames: {len(x_train)}")