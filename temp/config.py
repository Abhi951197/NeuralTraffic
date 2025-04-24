"""
Configuration file for the Smart Traffic Management System
"""

# API Keys
ROBOFLOW_API_KEY = "rNZWSRJ2xVRaq8AWduv2"

# Camera Sources
# Use integers for webcams (0, 1, 2, etc.)
# Use file paths for video files (e.g., "videos/traffic1.mp4")
# Use URLs for streaming sources
VIDEO_SOURCES = [
    "static/videos/27260-362770008_small.mp4",  # Lane 1 - Main camera or default webcam
    "static/videos/WhatsApp Video 2025-04-24 at 8.27.31 PM.mp4",  # Lane 2 - Example video file
    "static/videos/traffic-congestion-and-street-life-in-the-city-of-jaipur-pink-gate-city-walls--SBV-300214180-preview.mp4",  # Lane 3 - Example video file
    "static/videos/udaipur-india-november-24-2012-traffic-on-indian-street-in-udaipur-SBV-347557199-preview.mp4"   # Lane 4 - Example video file
]

# Number of lanes to monitor
NUM_LANES = 4

# Traffic Controller Settings
TRAFFIC_SETTINGS = {
    "min_green_time": 10,     # Minimum green light duration in seconds
    "max_green_time": 60,     # Maximum green light duration in seconds
    "yellow_time": 3,         # Yellow light duration in seconds
    "all_red_time": 1,        # All-red clearance interval in seconds
    "max_wait_threshold": 120 # Maximum wait time threshold in seconds
}

# Vehicle Priority Weights
VEHICLE_WEIGHTS = {
    "car": 1.0,
    "truck": 5,
    "bus": 5.0,
    "motorcycle": 3,
    "emergency": 50.0
}

# Detection Settings
DETECTION_SETTINGS = {
    "confidence_threshold": 0.5,  # Minimum confidence for detections
    "roi": [                      # Region of interest per lane (as percentages)
        {"x": 0.1, "y": 0.3, "width": 0.8, "height": 0.4},  # Lane 1
        {"x": 0.1, "y": 0.3, "width": 0.8, "height": 0.4},  # Lane 2
        {"x": 0.1, "y": 0.3, "width": 0.8, "height": 0.4},  # Lane 3
        {"x": 0.1, "y": 0.3, "width": 0.8, "height": 0.4}   # Lane 4
    ],
    "update_interval": 5,         # Time in seconds between detection updates
    "frame_skip": 5               # Process every Nth frame to optimize performance
}