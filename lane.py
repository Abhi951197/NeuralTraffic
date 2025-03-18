import cv2
import numpy as np
from ultralytics import YOLO

class LaneProcessor:
    def __init__(self, lane_count):
        """Initialize lane processor for vehicle detection and priority calculation"""
        self.lane_count = lane_count
        
        # Frame processing parameters
        self.chunk_size = 5  # seconds
        self.frame_rate = 1  # 1 frame per second
        
        # Priority scores for different vehicle types
        self.priority_scores = {
            'car': 3,
            'motorcycle': 1,
            'truck': 5,
            'bus': 4,
            'emergency': 10  # Special priority for emergency vehicles
        }

        # Class mapping for YOLOv8 (COCO dataset) to our categories
        self.class_mapping = {
            2: 'car',         # car
            3: 'motorcycle',  # motorcycle 
            5: 'bus',         # bus
            7: 'truck'        # truck
            # Could add emergency vehicle detection with custom model
        }
        
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Using nano model for performance
    
    def process_lane_frame(self, frame):
        """Process a single frame from a lane, detect vehicles, and calculate priority"""
        # Clone the frame to avoid modifying the original
        processed_frame = frame.copy()
        
        # Process with YOLOv8
        results = self.model(processed_frame)
        detections = results[0]
        
        # Track detected vehicles
        vehicle_counts = {}
        
        # Process detected objects
        boxes = detections.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Only process if confidence is high enough and class is in our mapping
            if conf > 0.5 and cls in self.class_mapping:
                vehicle_type = self.class_mapping[cls]
                
                # Draw bounding box
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add text label
                text = f"{vehicle_type}: {conf:.2f}"
                cv2.putText(processed_frame, text, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Update vehicle count
                if vehicle_type in vehicle_counts:
                    vehicle_counts[vehicle_type] += 1
                else:
                    vehicle_counts[vehicle_type] = 1
        
        # Calculate lane priority based on vehicle types and counts
        priority = 0
        for vehicle_type, count in vehicle_counts.items():
            priority += count * self.priority_scores.get(vehicle_type, 0)
        
        return processed_frame, vehicle_counts, priority