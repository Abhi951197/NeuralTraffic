import cv2
import numpy as np
import io
from PIL import Image
from inference_sdk import InferenceHTTPClient

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
            'ambulance': 50,  # Emergency vehicle
            'police': 50,     # Emergency vehicle
            'firetruck': 50   # Emergency vehicle
        }
        
        # Initialize Roboflow client
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="rNZWSRJ2xVRaq8AWduv2"
        )
        
        # Roboflow model ID
        self.model_id = "vehicle-detection-eckrb/4"
    
    def process_lane_frame(self, frame):
        """Process a single frame from a lane, detect vehicles, and calculate priority"""
        # Clone the frame to avoid modifying the original
        processed_frame = frame.copy()
        
        # Convert OpenCV BGR format to RGB for PIL
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Process with Roboflow API
        try:
            results = self.client.infer(pil_image, model_id=self.model_id)
            
            # Track detected vehicles
            vehicle_counts = {}
            
            # Process detected objects
            for prediction in results.get("predictions", []):
                class_name = prediction["class"].lower()
                confidence = prediction["confidence"]
                
                # Only process if confidence is high enough
                if confidence > 0.5:
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
                    
                    # Map to emergency vehicle category if needed
                    vehicle_type = class_name
                    if class_name in ['ambulance', 'police', 'firetruck']:
                        vehicle_type = class_name
                        # Add special emergency indicator
                        if 'emergency' not in vehicle_counts:
                            vehicle_counts['emergency'] = 0
                        vehicle_counts['emergency'] += 1
                    
                    # Draw bounding box
                    # Use red for emergency vehicles, green for others
                    color = (0, 0, 255) if class_name in ['ambulance', 'police', 'firetruck'] else (0, 255, 0)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add text label
                    text = f"{class_name}: {confidence:.2f}"
                    cv2.putText(processed_frame, text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Update vehicle count
                    if vehicle_type in vehicle_counts:
                        vehicle_counts[vehicle_type] += 1
                    else:
                        vehicle_counts[vehicle_type] = 1
        
        except Exception as e:
            # Handle API errors gracefully
            print(f"Error during Roboflow inference: {e}")
            cv2.putText(processed_frame, "API Error", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            vehicle_counts = {}
        
        # Calculate lane priority based on vehicle types and counts
        priority = 0
        for vehicle_type, count in vehicle_counts.items():
            if vehicle_type != 'emergency':  # Skip the special emergency counter
                priority += count * self.priority_scores.get(vehicle_type, 0)
        
        return processed_frame, vehicle_counts, priority