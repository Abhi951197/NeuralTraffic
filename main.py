import cv2
import numpy as np
import torch
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
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
        self.video_sources = video_sources
        self.lane_count = len(video_sources)
        self.captures = [cv2.VideoCapture(src) for src in video_sources]
        self.is_running = False
        
        self.frame_dirs = []
        for i in range(self.lane_count):
            dir_path = f"lane_{i}_frames"
            os.makedirs(dir_path, exist_ok=True)
            self.frame_dirs.append(dir_path)
        
        self.lane_processor = LaneProcessor(self.lane_count)
        self.traffic_controller = TrafficController(self.lane_count)
        self.processed_frames = [deque() for _ in range(self.lane_count)]
        self.latest_frames = [None] * self.lane_count
        
        self.gui = SystemGUI(self)
        self.root = self.gui.root
        
        self.preprocess_threads = []
        self.detection_thread = None
        self.traffic_control_thread = None
        self.display_thread = None
    
    def initialize_threads(self):
        self.preprocess_threads = []
        
        for i in range(self.lane_count):
            thread = threading.Thread(target=self.preprocess_video, args=(i,))
            thread.daemon = True
            self.preprocess_threads.append(thread)
        
        self.detection_thread = threading.Thread(target=self.process_frames)
        self.detection_thread.daemon = True
        
        self.traffic_control_thread = threading.Thread(target=self.traffic_controller.control_traffic_lights, args=(self,))
        self.traffic_control_thread.daemon = True
        
        self.display_thread = threading.Thread(target=self.update_display)
        self.display_thread.daemon = True
    
    def toggle_system(self):
        if self.is_running:
            self.is_running = False
            self.gui.update_system_status("System Stopped", "Start System")
        else:
            self.is_running = True
            self.gui.update_system_status("System Running", "Stop System")
            self.initialize_threads()
            
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
        cap = self.captures[lane_idx]
        frame_interval = 5  # process every nth frame
        chunk_frames = 30  # number of frames per chunk

        while self.is_running:
            chunk_images = []
            count = 0
            
            while count < chunk_frames and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                if count % frame_interval == 0:
                    chunk_images.append(frame)
                
                count += 1
            
            if chunk_images:
                selected_frame = random.choice(chunk_images)
                selected_frame = cv2.resize(selected_frame, (640, 480))
                
                if len(self.processed_frames[lane_idx]) >= 5:
                    self.processed_frames[lane_idx].popleft()
                self.processed_frames[lane_idx].append(selected_frame)
                
                self.latest_frames[lane_idx] = selected_frame.copy()
            
            time.sleep(0.1)

    def process_frames(self):
        while self.is_running:
            for lane_idx in range(self.lane_count):
                if not self.processed_frames[lane_idx]:
                    continue
                
                frame = self.processed_frames[lane_idx].popleft()
                processed_frame, vehicle_counts, priority = self.lane_processor.process_lane_frame(frame)
                self.traffic_controller.lane_vehicle_counts[lane_idx] = vehicle_counts
                self.traffic_controller.lane_priorities[lane_idx] = priority
                self.latest_frames[lane_idx] = processed_frame
            
            time.sleep(0.1)
    
    def update_display(self):
        while self.is_running:
            for lane_idx in range(self.lane_count):
                if self.latest_frames[lane_idx] is None:
                    continue
                
                frame = self.latest_frames[lane_idx].copy()
                status = self.traffic_controller.traffic_states[lane_idx].upper()
                color = (0, 0, 255)
                if status == "GREEN":
                    color = (0, 255, 0)
                elif status == "YELLOW":
                    color = (0, 255, 255)
                
                cv2.rectangle(frame, (10, 10), (150, 50), (0, 0, 0), -1)
                cv2.putText(frame, f"Signal: {status}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, color, 2)
                
                priority = self.traffic_controller.lane_priorities[lane_idx]
                cv2.rectangle(frame, (10, 60), (150, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"Priority: {priority}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 2)
                
                self.gui.update_lane_display(lane_idx, frame, 
                                         self.traffic_controller.traffic_states[lane_idx],
                                         self.traffic_controller.lane_priorities[lane_idx],
                                         self.traffic_controller.lane_vehicle_counts[lane_idx])
            time.sleep(0.1)
    
    def run(self):
        self.root.mainloop()
        self.is_running = False
        for cap in self.captures:
            cap.release()

def main():
    # Use camera indices instead of video file paths
    # Ensure the system has at least 4 cameras or virtual cameras for testing
    video_sources = [0, 1, 2,"static/videos/istockphoto-866517852-640_adpp_is.mp4" ]
    
    system = TrafficManagementSystem(video_sources)
    system.run()

if __name__ == "__main__":
    main()