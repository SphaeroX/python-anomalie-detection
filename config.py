# Allgemein
resize_x = 224
resize_y = 224

#Konturen
use_contour = False
contour_threshold = 5

# Training
epochs = 30
batch_size = 4

# Recording
video_path = 'soll_zustand.avi'
recording_duration = 20

# Auslöser für Detection
min_recorded_frames = 20
calibration_percent = 1.005
calibration_method = 'median'  # 'average' or 'median', or 'percentile'
adaptive_threshold = False
adaptive_threshold_alpha = 0.1  # for exponential smoothing (adjust between 0 and 1)