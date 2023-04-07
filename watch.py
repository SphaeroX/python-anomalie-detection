import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime


def reconstruct_image(autoencoder, image):
    input_tensor = np.expand_dims(image, axis=0)
    reconstructed_image = autoencoder.predict(input_tensor,verbose = 0)[0]
    return reconstructed_image

def calculate_anomaly(image, reconstructed_image, threshold):
    mse = np.mean((image - reconstructed_image) ** 2)
    anomaly_mask = mse > threshold
    return anomaly_mask

def highlight_anomaly(frame, anomaly_mask, coloring_anomaly, transparency=0.3):
    if not coloring_anomaly:
        return frame

    highlighted_frame = frame.copy()
    anomaly_overlay = np.zeros_like(highlighted_frame, dtype=np.uint8)
    anomaly_overlay[anomaly_mask] = [0, 0, 255]

    cv2.addWeighted(anomaly_overlay, transparency, highlighted_frame, 1 - transparency, 0, highlighted_frame)
    return highlighted_frame


def calibrate_threshold(autoencoder, calibration_time=10, calibration_percent=1.06):
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    mse_values = []

    while time.time() - start_time < calibration_time:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (224, 224)) / 255.0
        reconstructed_frame = reconstruct_image(autoencoder, resized_frame)
        mse = np.mean((resized_frame - reconstructed_frame) ** 2)
        mse_values.append(mse)

    cap.release()
    average_mse = np.mean(mse_values)
    threshold = average_mse * calibration_percent
    return threshold

def monitor_and_detect_anomalies(autoencoder, coloring_anomaly=True):
    anomaly_threshold = calibrate_threshold(autoencoder)
    print(f"Anomaly threshold set to: {anomaly_threshold}")

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    recording = False
    recorded_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (224, 224)) / 255.0
        reconstructed_frame = reconstruct_image(autoencoder, resized_frame)
        anomaly_mask = calculate_anomaly(resized_frame, reconstructed_frame, anomaly_threshold)

        highlighted_frame = highlight_anomaly(frame, anomaly_mask, coloring_anomaly)
        cv2.imshow('Ist-Zustand', highlighted_frame)

        if np.any(anomaly_mask):
            if not recording:
                recording = True
            recorded_frames.append(frame)
        else:
            if recording:
                if len(recorded_frames) > 20:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    output_filename = f'anomaly_{timestamp}.avi'
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (width, height))

                    for recorded_frame in recorded_frames:
                        out.write(recorded_frame)

                    out.release()

                recorded_frames = []
                recording = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

autoencoder = tf.keras.models.load_model('autoencoder_model.h5')
monitor_and_detect_anomalies(autoencoder)