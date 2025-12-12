import numpy as np
import os

# --- Paths ---
SOURCE_VIDEO_PATH = os.path.join("data", "vehicles.mp4")
TARGET_VIDEO_PATH = os.path.join("data", "vehicles-result-5.mp4")
MODEL_PATH = "yolo11m.pt"
CAPTURES_DIR = "capturas"

# --- Model & Tracking ---
CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.8
# classes: 2-car, 3-motorcycle, 5-bus, 7-truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# --- Speed Estimation ---
SPEED_LIMIT_KMH = 80
TARGET_WIDTH = 25
TARGET_HEIGHT = 200

# Polygon zone for detection
SOURCE_POLYGON = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

# Perspective transform target
TARGET_RECT = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)
