import cv2
import time
import threading
import numpy as np
from queue import Queue
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io

class VideoProcessor:
    def __init__(self, source_id, lane_id, api_key="rNZWSRJ2xVRaq8AWduv2"):
        """
        Initialize the video processor for a specific lane
        
        Args:
            source_id: Camera source (0 for webcam, or file path/URL)
            lane_id: Identifier for the lane this processor is monitoring
            api_key: Roboflow API key
        """
        self.source_id = source_id
        print("ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss",self.source_id)
        self.lane_id = lane_id
        self.cap = None
        self.running = False
        self.frame_queue = Queue(maxsize=10)
        self.results_queue = Queue(maxsize=10)
        self.current_frame = None
        self.current_results = []
        self.last_detection_time = time.time()
        self.detection_interval = 1.0  # Perform detection every second
        self.processor_thread = None
        self.detector_thread = None
        
        # Vehicle counts
        self.vehicle_counts = {
            "car": 0,
            "truck": 0,
            "bus": 0,
            "motorcycle": 0,
            "emergency": 0
        }
        
        # Cumulative counts (total vehicles seen)
        self.total_vehicles = 0
        
        # Detection client
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        
        # Detection confidence threshold
        self.confidence_threshold = 0.5
        
        # For tracking wait time
        self.last_empty = time.time()
        self.current_wait_time = 0
    
    def start(self):
        """Start video processing"""
        if self.running:
            return
            
        self.running = True
        self.cap = cv2.VideoCapture(self.source_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.source_id}")
            self.running = False
            return False
        
        # Start processing threads
        self.processor_thread = threading.Thread(target=self._process_frames)
        self.detector_thread = threading.Thread(target=self._detect_vehicles)
        
        self.processor_thread.daemon = True
        self.detector_thread.daemon = True
        
        self.processor_thread.start()
        self.detector_thread.start()
        
        return True
    
    def stop(self):
        """Stop video processing"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)
        if self.detector_thread:
            self.detector_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
    
    def _process_frames(self):
        """Process video frames in a separate thread"""
        while self.running:
            ret, frame = self.cap.read()

            # If reading failed (likely end of video), restart the video
            if not ret:
    # Rewind the video to the beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"Restarting video source {self.source_id}")
                continue

            # Update current frame for GUI display
            self.current_frame = frame.copy()

            # Add to queue for detection if not full
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            time.sleep(0.01)  # Small delay to prevent high CPU usage

    
    def _detect_vehicles(self):
        """Detect vehicles in frames using Roboflow"""
        while self.running:
            # Only process if we have frames and enough time has passed
            if not self.frame_queue.empty() and (time.time() - self.last_detection_time) > self.detection_interval:
                frame = self.frame_queue.get()
                self.last_detection_time = time.time()
                
                try:
                    # Convert frame to bytes for Roboflow API
                    is_success, buffer = cv2.imencode(".jpg", frame)
                    if not is_success:
                        continue
                        
                    # Send to Roboflow for inference
                    # Convert bytes to a NumPy array, then to PIL
                    image_array = np.frombuffer(buffer, dtype=np.uint8)
                    pil_image = Image.open(io.BytesIO(image_array))

# Now pass the PIL image to Roboflow
                    results = self.client.infer(pil_image, model_id="vehicle-detection-eckrb/4")


                    
                    # Filter results by confidence
                    filtered_results = [r for r in results["predictions"] if r["confidence"] >= self.confidence_threshold]
                    
                    # Update vehicle counts
                    self._update_vehicle_counts(filtered_results)
                    
                    # Store results
                    self.current_results = filtered_results
                    if not self.results_queue.full():
                        self.results_queue.put((frame, filtered_results))
                        
                except Exception as e:
                    print(f"Detection error in lane {self.lane_id}: {str(e)}")
            
            time.sleep(0.1)
    
    def _update_vehicle_counts(self, results):
        """Update vehicle counts based on detection results"""
        # Reset counts for this detection cycle
        temp_counts = {
            "car": 0,
            "truck": 0,
            "bus": 0,
            "motorcycle": 0,
            "emergency": 0
        }
        
        # Count vehicles in current frame
        for detection in results:
            class_name = detection["class"].lower()
            
            # Map detected classes to our categories
            if class_name in ["car", "sedan", "suv"]:
                temp_counts["car"] += 1
            elif class_name in ["truck", "pickup", "lorry"]:
                temp_counts["truck"] += 1
            elif class_name == "bus":
                temp_counts["bus"] += 1
            elif class_name in ["motorcycle", "motorbike", "bicycle"]:
                temp_counts["motorcycle"] += 1
            elif class_name in ["ambulance", "police", "fire truck"]:
                temp_counts["emergency"] += 1
        
        # Update current counts
        self.vehicle_counts = temp_counts
        
        # Update total count
        self.total_vehicles = sum(temp_counts.values())
        
        # Update wait time
        if self.total_vehicles == 0:
            self.last_empty = time.time()
        else:
            self.current_wait_time = time.time() - self.last_empty
    
    def get_annotated_frame(self):
        """Return the current frame with vehicle detection annotations"""
        if self.current_frame is None:
            return None
            
        annotated = self.current_frame.copy()
        
        # Draw bounding boxes
        for detection in self.current_results:
            x = int(detection["x"])
            y = int(detection["y"])
            width = int(detection["width"])
            height = int(detection["height"])
            confidence = detection["confidence"]
            class_name = detection["class"]
            
            # Calculate bounding box coordinates
            x1 = int(x - width/2)
            y1 = int(y - height/2)
            x2 = int(x + width/2)
            y2 = int(y + height/2)
            
            # Choose color based on class
            if class_name.lower() in ["ambulance", "police", "fire truck"]:
                color = (0, 0, 255)  # Red for emergency
            elif class_name.lower() in ["truck", "pickup", "lorry"]:
                color = (255, 165, 0)  # Orange for trucks
            elif class_name.lower() == "bus":
                color = (255, 0, 255)  # Purple for buses
            elif class_name.lower() in ["motorcycle", "motorbike", "bicycle"]:
                color = (0, 255, 255)  # Yellow for motorcycles
            else:
                color = (0, 255, 0)  # Green for cars
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add lane ID and vehicle counts
        text = f"Lane {self.lane_id} | Cars: {self.vehicle_counts['car']} | " \
               f"Trucks: {self.vehicle_counts['truck']} | Buses: {self.vehicle_counts['bus']} | " \
               f"Motorcycles: {self.vehicle_counts['motorcycle']} | Emergency: {self.vehicle_counts['emergency']}"
        
        cv2.putText(annotated, text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add wait time
        wait_text = f"Wait: {int(self.current_wait_time)}s"
        cv2.putText(annotated, wait_text, (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def get_vehicle_counts(self):
        """Return the current vehicle counts"""
        return self.vehicle_counts
    
    def get_wait_time(self):
        """Return the current wait time for this lane"""
        return self.current_wait_time