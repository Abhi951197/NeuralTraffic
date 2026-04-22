"""
Configuration for the Smart Traffic Management System.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# ── API Keys ──────────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = "rNZWSRJ2xVRaq8AWduv2"

# ── Video Sources ─────────────────────────────────────────────────────────────
_VIDEOS = os.path.join(BASE_DIR, "static", "videos")
VIDEO_SOURCES = [
    os.path.join(_VIDEOS, "27260-362770008_small.mp4"),
    os.path.join(_VIDEOS, "agra-india-november-17-2012-traffic-on-indian-street-in-agra-india-17-nov-2012-SBV-347430175-preview.mp4"),
    os.path.join(_VIDEOS, "AMB.mp4"),
    os.path.join(_VIDEOS, "udaipur-india-november-24-2012-traffic-on-indian-street-in-udaipur-SBV-347557199-preview.mp4"),
]

NUM_LANES = 4

# ── Traffic Controller ────────────────────────────────────────────────────────
TRAFFIC_SETTINGS = {
    "min_green_time":    10,
    "max_green_time":    60,
    "yellow_time":        3,
    "all_red_time":       1,
    "max_wait_threshold": 120,
}

# ── Vehicle Priority Weights ──────────────────────────────────────────────────
# Matches best.pt classes + Roboflow emergency
VEHICLE_WEIGHTS = {
    "car":              1.0,
    "truck":            1.5,
    "bus":              2.0,
    "motorcycle":       0.8,
    "autorickshaw":     1.2,
    "bicycle":          0.5,
    "vehicle_fallback": 1.0,
    "emergency":       10.0,   # ambulance detected by Roboflow
}

# ── Detection Settings ────────────────────────────────────────────────────────
DETECTION_SETTINGS = {
    "confidence_threshold":    0.5,
    "yolo_detection_interval": 1.0,   # seconds between YOLO runs
    "ambulance_check_interval": 2.0,  # seconds between Roboflow runs
}
