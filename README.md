# Anomaly Detection in Video Streams

This project uses an autoencoder neural network to detect anomalies in video streams. The autoencoder is trained on a reference video and can be used to monitor live video streams for deviations from the reference. Detected anomalies are saved as separate video files.

## Files

### 1. record.py

This script records a reference video using a webcam. The video is saved to the specified path and has a predefined duration in seconds.

### 2. train.py

This script preprocesses the reference video, optionally applying contour detection. It then trains an autoencoder neural network on the video frames and saves the trained model as an .h5 file. The model loss is plotted over time to visualize the training progress.

### 3. watch.py

This script monitors a live video stream using a webcam and compares it to the trained autoencoder model. Detected anomalies are saved as separate video files with a timestamp in their names. The script also calibrates the anomaly detection threshold based on the autoencoder's performance.

### 4. config.py

This file contains adjustable variables such as the use of contour detection, resizing dimensions, calibration percentage, video path, minimum recorded frames, contour threshold, and recording duration.

## Usage

1. Record a reference video with `record.py`.
2. Train the autoencoder model using the reference video with `train.py`.
3. Monitor the live video stream for anomalies with `watch.py`.
