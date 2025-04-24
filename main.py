import cv2
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import random
import os
import datetime
from collections import deque

# Import our modules
from lane import LaneProcessor
from traffic_control import TrafficController
from gui_components import SystemGUI


class TrafficManagementSystem:
    def __init__(self, video_sources):
        # Initialize video sources (paths or camera indices)
        self.video_sources = video_sources
        self.lane_count = len(video_sources)
        
        # Initialize video captures
        self.captures = [cv2.VideoCapture(src) for src in video_sources]
        
        # System control flags
        self.is_running = False
        
        # Create directories for frame storage
        self.frame_dirs = []
        for i in range(self.lane_count):
            dir_path = f"lane_{i}_frames"
            os.makedirs(dir_path, exist_ok=True)
            self.frame_dirs.append(dir_path)
        
        # Initialize components
        self.lane_processor = LaneProcessor(self.lane_count)
        self.traffic_controller = TrafficController(self.lane_count)
        
        # Pre-processed frames for each lane
        self.processed_frames = [deque() for _ in range(self.lane_count)]
        self.latest_frames = [None] * self.lane_count
        
        # Create GUI
        self.gui = SystemGUI(self)
        self.root = self.gui.root
        
        # Thread management
        self.preprocess_threads = []
        self.detection_thread = None
        self.traffic_control_thread = None
        self.display_thread = None
    
    def initialize_threads(self):
        """Initialize all processing threads"""
        # Clear existing threads
        self.preprocess_threads = []
        
        # Create preprocessing threads
        for i in range(self.lane_count):
            thread = threading.Thread(target=self.preprocess_video, args=(i,))
            thread.daemon = True
            self.preprocess_threads.append(thread)
        
        # Create other system threads
        self.detection_thread = threading.Thread(target=self.process_frames)
        self.detection_thread.daemon = True
        
        self.traffic_control_thread = threading.Thread(target=self.traffic_controller.control_traffic_lights,
                                                      args=(self,))  # Pass self reference
        self.traffic_control_thread.daemon = True
        
        self.display_thread = threading.Thread(target=self.update_display)
        self.display_thread.daemon = True
    
    def toggle_system(self):
        """Start or stop the system"""
        if self.is_running:
            self.is_running = False
            self.gui.update_system_status("System Stopped", "Start System")
        else:
            self.is_running = True
            self.gui.update_system_status("System Running", "Stop System")
            
            # Initialize threads if needed
            self.initialize_threads()
            
            # Start all threads
            for i, thread in enumerate(self.preprocess_threads):
                if not thread.is_alive():
                    self.preprocess_threads[i].start()
            
            if not self.detection_thread.is_alive():
                self.detection_thread.start()
            
            if not self.traffic_control_thread.is_alive():
                self.traffic_control_thread.start()
                
            if not self.display_thread.is_alive():
                self.display_thread.start()

    def preprocess_video(self, lane_idx):
        """Preprocess video for a specific lane by extracting frames at regular intervals"""
        cap = self.captures[lane_idx]
        
        while self.is_running:
            if not cap.isOpened():
                time.sleep(1)
                continue
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / self.lane_processor.frame_rate)
            chunk_frames = int(fps * self.lane_processor.chunk_size)
            
            # Process chunk by chunk
            start_frame = 0
            
            while self.is_running:
                # Set video position to start of chunk
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Extract frames from chunk
                chunk_images = []
                for i in range(chunk_frames):
                    # Only capture frames at specified interval
                    if i % frame_interval == 0:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        chunk_images.append(frame)
                
                # If we reached the end of video, loop back
                if len(chunk_images) == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start_frame = 0
                    continue
                
                # Select a random frame from the chunk
                if chunk_images:
                    selected_frame = random.choice(chunk_images)
                    selected_frame = cv2.resize(selected_frame, (640, 480))
                    
                    # Add to processing queue
                    if len(self.processed_frames[lane_idx]) >= 5:
                        self.processed_frames[lane_idx].popleft()
                    self.processed_frames[lane_idx].append(selected_frame)
                    
                    # Update latest frame
                    self.latest_frames[lane_idx] = selected_frame.copy()
                
                # Move to next chunk
                start_frame += chunk_frames
                
                # Wait before processing next chunk
                time.sleep(0.1)
    
    def process_frames(self):
        """Process frames from all lanes simultaneously"""
        while self.is_running:
            for lane_idx in range(self.lane_count):
                # Skip if no frames available
                if not self.processed_frames[lane_idx]:
                    continue
                
                # Get a frame from the queue
                frame = self.processed_frames[lane_idx].popleft()
                
                # Process frame with vehicle detection
                processed_frame, vehicle_counts, priority = self.lane_processor.process_lane_frame(frame)
                
                # Update controller with new vehicle data
                self.traffic_controller.lane_vehicle_counts[lane_idx] = vehicle_counts
                self.traffic_controller.lane_priorities[lane_idx] = priority
                
                # Update latest frame with detection results
                self.latest_frames[lane_idx] = processed_frame
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)
    
    def update_display(self):
        """Update the GUI display with the latest frames and information"""
        while self.is_running:
            # Update all lane displays
            for lane_idx in range(self.lane_count):
                # Skip if no frame available
                if self.latest_frames[lane_idx] is None:
                    continue
                
                # Get current frame
                frame = self.latest_frames[lane_idx].copy()
                
                # Add traffic light status overlay to frame
                status = self.traffic_controller.traffic_states[lane_idx].upper()
                color = (0, 0, 255)  # Red
                if status == "GREEN":
                    color = (0, 255, 0)  # Green
                elif status == "YELLOW":
                    color = (0, 255, 255)  # Yellow
                
                # Draw signal status on the frame
                cv2.rectangle(frame, (10, 10), (150, 50), (0, 0, 0), -1)
                cv2.putText(frame, f"Signal: {status}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, color, 2)
                
                # Draw priority score on the frame
                priority = self.traffic_controller.lane_priorities[lane_idx]
                cv2.rectangle(frame, (10, 60), (150, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"Priority: {priority}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 2)
                
                # Update GUI with new frame and information
                self.gui.update_lane_display(lane_idx, frame, 
                                         self.traffic_controller.traffic_states[lane_idx],
                                         self.traffic_controller.lane_priorities[lane_idx],
                                         self.traffic_controller.lane_vehicle_counts[lane_idx])
            
            # Update GUI periodically
            time.sleep(0.1)
            
    def run(self):
        """Run the traffic management system"""
        # Start GUI main loop
        self.root.mainloop()
        
        # Clean up when GUI closes
        self.is_running = False
        for cap in self.captures:
            cap.release()

def main():
    # Replace with your video sources
    # Could be camera indices (0, 1, 2, 3) or video file paths
    video_sources = [
        'static/videos/27260-362770008_small.mp4',
        'C:/Users/abhis/OneDrive/Desktop/NeuraLTraffic/static/videos/agra-india-november-17-2012-traffic-on-indian-street-in-agra-india-17-nov-2012-SBV-347430175-preview.mp4',
        'static/videos/WhatsApp Video 2025-04-24 at 20.27.45_8550a03e.mp4', 
        'static/videos/udaipur-india-november-24-2012-traffic-on-indian-street-in-udaipur-SBV-347557199-preview.mp4'
    ]
    
    # Create and run the system
    system = TrafficManagementSystem(video_sources)
    system.run()

if __name__ == "__main__":
    main()