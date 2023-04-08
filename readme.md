# Real-Time Anomaly Detection with Python and OpenCV

This project demonstrates how to perform real-time anomaly detection using Python, OpenCV, and a pre-trained autoencoder model. The project consists of three main parts: capturing the reference state, training the autoencoder, and monitoring the current state for anomalies.

## Prerequisites

- Python 3.7+
- OpenCV
- TensorFlow

You can install the required packages using the following command:

```pip install opencv-python opencv-python-headless tensorflow```


## Usage

1. **Capturing the reference state:** Run the `capture_reference_state.py` script to record a video of the reference state using your webcam. The recorded video will be saved as `soll_zustand.avi`.

```
python capture_reference_state.py
```

2. **Training the autoencoder:** Run the `train_autoencoder.py` script to train the autoencoder model using the captured reference video. The trained model will be saved as `autoencoder_model.h5`. The script will also display a plot of the model's loss during training and print the number of trained frames.

```
python train_autoencoder.py
```

3. **Monitoring the current state for anomalies:** Run the `monitor_current_state.py` script to monitor the current state using your webcam. The script will compare the current state to the reference state and detect anomalies in real time. Detected anomalies will be highlighted in the video with a partially transparent red overlay. If an anomaly lasts longer than 20 frames, the video sequence will be saved as `anomaly_TIMESTAMP.avi`, where `TIMESTAMP` is the time when the anomaly was detected.

```
python monitor_current_state.py
```

## Customization

You can customize the behavior of the anomaly detection by modifying the following variables in the `monitor_current_state.py` script:

- `coloring_anomaly`: Set this to `True` to enable coloring the detected anomalies in the video. Set it to `False` to disable coloring.
- `transparency`: Adjust the transparency of the anomaly overlay. A value of 0 makes the overlay fully transparent, while a value of 1 makes it fully opaque.

## License

This project is licensed under the MIT License.
