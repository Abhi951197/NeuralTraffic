import cv2
import numpy as np
import io
from PIL import Image
from inference_sdk import InferenceHTTPClient
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
            'emergency': 50  # Special priority for emergency vehicles
            # Any emergency vehicles detected by Roboflow will be mapped to this
        }

        # Class mapping for YOLOv8 (COCO dataset) to our categories
        self.class_mapping = {
            2: 'car',         # car
            3: 'motorcycle',  # motorcycle 
            5: 'bus',         # bus
            7: 'truck'        # truck
        }
        
        # Emergency vehicle types from Roboflow model
        self.emergency_types = ['ambulance', 'police', 'firetruck']
        
        # Load YOLOv8 model
        self.yolo_model = YOLO('yolov8n.pt')  # Using nano model for performance
        
        # Initialize Roboflow client for emergency vehicle detection
        self.roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="rNZWSRJ2xVRaq8AWduv2"
        )
        
        # Roboflow model ID
        self.roboflow_model_id = "vehicle-detection-eckrb/4"
    
    def process_lane_frame(self, frame):
        """Process a single frame from a lane, detect vehicles, and calculate priority"""
        # Clone the frame to avoid modifying the original
        processed_frame = frame.copy()
        
        # Track detected vehicles
        vehicle_counts = {}
        
        # STEP 1: Process with YOLOv8 for regular vehicles
        yolo_results = self.yolo_model(processed_frame)
        yolo_detections = yolo_results[0]
        print(f"YOLOv8 Detections: {yolo_detections.names}")
        
        # Process detected objects from YOLO
        boxes = yolo_detections.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Only process if confidence is high enough and class is in our mapping
            if conf > 0.5 and cls in self.class_mapping:
                vehicle_type = self.class_mapping[cls]
                
                # Update vehicle count
                if vehicle_type in vehicle_counts:
                    vehicle_counts[vehicle_type] += 1
                else:
                    vehicle_counts[vehicle_type] = 1
                
                # Draw bounding box (green for regular vehicles)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add text label
                text = f"{vehicle_type}: {conf:.2f}"
                cv2.putText(processed_frame, text, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # STEP 2: Process with Roboflow specifically for emergency vehicles
        # Convert OpenCV BGR format to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Process with Roboflow API
        emergency_detected = False
        try:
            results = self.roboflow_client.infer(pil_image, model_id=self.roboflow_model_id)
            
            # Process detected objects from Roboflow
            for prediction in results.get("predictions", []):
                class_name = prediction["class"].lower()
                print(f"Roboflow Prediction: {class_name}")
                confidence = prediction["confidence"]
                
                # Only process emergency vehicles with sufficient confidence
                if class_name in self.emergency_types and confidence > 0.5:
                    emergency_detected = True
                    
                    # Get bounding box coordinates
                    x = prediction["x"]
                    y = prediction["y"]
                    width = prediction["width"]
                    height = prediction["height"]
                    
                    # Calculate box corners
                    x1 = int(x - width/2)
                    y1 = int(y - height/2)
                    x2 = int(x + width/2)
                    y2 = int(y + height/2)
                    
                    # Ensure box stays within image boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(processed_frame.shape[1], x2)
                    y2 = min(processed_frame.shape[0], y2)
                    
                    # Draw bounding box with red for emergency vehicles
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Add text label with emergency vehicle type
                    text = f"{class_name}: {confidence:.2f} - EMERGENCY"
                    cv2.putText(processed_frame, text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Add to emergency vehicle count
                    if 'emergency' in vehicle_counts:
                        vehicle_counts['emergency'] += 1
                    else:
                        vehicle_counts['emergency'] = 1
                    
                    # Also track specific type if needed
                    if class_name in vehicle_counts:
                        vehicle_counts[class_name] += 1
                    else:
                        vehicle_counts[class_name] = 1
            
            # Add visual indicator for emergency vehicles
            if emergency_detected:
                cv2.putText(processed_frame, "EMERGENCY VEHICLE DETECTED", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        except Exception as e:
            # Handle API errors gracefully
            print(f"Error during Roboflow inference: {e}")
            cv2.putText(processed_frame, "Roboflow API Error", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate lane priority based on vehicle types and counts
        priority = 0
        for vehicle_type, count in vehicle_counts.items():
            priority += count * self.priority_scores.get(vehicle_type, 0)
        
        # If emergency vehicle detected, give high priority regardless of other vehicles
        if emergency_detected:
            priority = max(priority, 50)  # Ensure high priority for emergency vehicles
        
        return processed_frame, vehicle_counts, priority