import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def preprocess_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (224, 224))
        frames.append(resized_frame)

    cap.release()
    return np.array(frames)

def create_autoencoder_model():
    input_shape = (224, 224, 3)

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

video_path = 'soll_zustand.avi'
frames = preprocess_video(video_path)
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
