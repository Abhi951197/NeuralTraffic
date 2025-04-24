import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

class SystemGUI:
    def __init__(self, system):
        self.system = system
        self.root = tk.Tk()
        self.root.title("Smart Traffic Management System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2f3e46")
        
        # For image display conversion
        self.photo_images = [None] * system.lane_count  # Store PhotoImage references
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        self.setup_gui()
    
    def setup_gui(self):
        header = tk.Frame(self.root, bg="#2c3e50")
        header.pack(fill=tk.X)

        title = tk.Label(header, text="NEURALTRAFFIC", font=("Arial", 20, "bold"), bg="#2c3e50", fg="white", pady=10)
        title.pack(side=tk.LEFT, padx=20)

        self.status_label = tk.Label(header, text="Status: Ready", font=("Arial", 12), bg="#2c3e50", fg="lightgreen")
        self.status_label.pack(side=tk.RIGHT, padx=20)

        main_frame = tk.Frame(self.root, bg="#2f3e46")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_labels = []
        self.priority_labels = []
        self.vehicle_count_labels = []
        self.wait_time_labels = []
        self.signal_indicators = []

        grid_frame = tk.Frame(main_frame, bg="#2f3e46")
        grid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        for i in range(self.system.lane_count):
            frame = tk.Frame(grid_frame, bg="#1e2a33", bd=2, relief=tk.RIDGE)
            frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")

            # Header with lane label and signal indicator
            header_frame = tk.Frame(frame, bg="#34495e")
            header_frame.pack(fill=tk.X)
            
            lane_label = tk.Label(header_frame, text=f"Lane {i+1}", font=("Arial", 12, "bold"), bg="#34495e", fg="white", padx=5)
            lane_label.pack(side=tk.LEFT)
            
            # Traffic signal indicator
            signal_indicator = tk.Canvas(header_frame, width=20, height=20, bg="#34495e", highlightthickness=0)
            signal_indicator.create_oval(2, 2, 18, 18, fill="red", outline="white", width=1, tags="signal")
            signal_indicator.pack(side=tk.RIGHT, padx=10)
            self.signal_indicators.append(signal_indicator)

            video_canvas = tk.Label(frame, bg="black", height=10)
            video_canvas.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
            self.video_labels.append(video_canvas)

            info_text = tk.StringVar()
            info_label = tk.Label(frame, textvariable=info_text, bg="#ecf0f1", font=("Arial", 10), anchor="w", justify="left")
            info_label.pack(fill=tk.X, padx=5)
            self.vehicle_count_labels.append(info_text)

            bottom_frame = tk.Frame(frame, bg="#ecf0f1")
            bottom_frame.pack(fill=tk.X, padx=5, pady=2)

            tk.Label(bottom_frame, text="Priority:", bg="#ecf0f1").pack(side=tk.LEFT)
            priority = tk.Label(bottom_frame, text="0", width=5, bg="white")
            priority.pack(side=tk.LEFT, padx=5)
            self.priority_labels.append(priority)

            wait_time = tk.Label(bottom_frame, text="Wait: 0s", bg="#ecf0f1")
            wait_time.pack(side=tk.RIGHT)
            self.wait_time_labels.append(wait_time)

        grid_frame.grid_rowconfigure(0, weight=1)
        grid_frame.grid_rowconfigure(1, weight=1)
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)

        # Control panel
        control_panel = tk.Frame(self.root, bg="#34495e", padx=10, pady=5)
        control_panel.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Time control settings
        control_frame = tk.Frame(control_panel, bg="#34495e")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(control_frame, text="Green Time (s):", bg="#34495e", fg="white").grid(row=0, column=0, padx=5, pady=2)
        green_time_var = tk.IntVar(value=self.system.traffic_controller.green_time)
        green_spinner = tk.Spinbox(control_frame, from_=5, to=60, width=5, textvariable=green_time_var)
        green_spinner.grid(row=0, column=1, padx=5, pady=2)
        
        tk.Button(control_frame, text="Set", bg="#95a5a6", 
                 command=lambda: self.system.traffic_controller.set_green_time(green_time_var.get())).grid(row=0, column=2, padx=5)
        
        tk.Label(control_frame, text="Yellow Time (s):", bg="#34495e", fg="white").grid(row=1, column=0, padx=5, pady=2)
        yellow_time_var = tk.IntVar(value=self.system.traffic_controller.yellow_time)
        yellow_spinner = tk.Spinbox(control_frame, from_=3, to=10, width=5, textvariable=yellow_time_var)
        yellow_spinner.grid(row=1, column=1, padx=5, pady=2)
        
        tk.Button(control_frame, text="Set", bg="#95a5a6", 
                 command=lambda: self.system.traffic_controller.set_yellow_time(yellow_time_var.get())).grid(row=1, column=2, padx=5)

        self.start_stop_button = tk.Button(control_panel, text="Start System", bg="green", fg="white", 
                                          command=self.system.toggle_system, width=15, height=2)
        self.start_stop_button.pack(side=tk.RIGHT, padx=20)

        self.fps_label = tk.Label(control_panel, text="FPS: 0", bg="#34495e", fg="white")
        self.fps_label.pack(side=tk.RIGHT, padx=10)
    
    def update_lane_display(self, lane_idx, frame, traffic_state, priority, vehicle_counts):
        """Update the display for a specific lane with current data"""
        # Count frames for FPS calculation
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_update
        
        # Update FPS every second
        if elapsed > 1.0:
            fps = self.frame_count / elapsed
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_update = current_time
        
        # Convert OpenCV BGR to RGB for tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit the display area if needed
        height, width, _ = rgb_frame.shape
        target_width = 400  # Adjust as needed for your GUI
        target_height = int(height * (target_width / width))
        resized_frame = cv2.resize(rgb_frame, (target_width, target_height))
        
        # Convert to PhotoImage
        img = Image.fromarray(resized_frame)
        photo = ImageTk.PhotoImage(image=img)
        
        # Store reference to prevent garbage collection
        self.photo_images[lane_idx] = photo
        
        # Update the image in the label
        self.video_labels[lane_idx].config(image=photo)
        
        # Update priority label
        self.priority_labels[lane_idx].config(
            text=str(int(priority)),
            bg=self.get_priority_color(priority)
        )
        
        # Update vehicle count info
        count_text = "Vehicles: "
        if vehicle_counts:
            count_details = []
            for vehicle_type, count in vehicle_counts.items():
                count_details.append(f"{vehicle_type}: {count}")
            count_text += ", ".join(count_details)
        else:
            count_text += "None detected"
        self.vehicle_count_labels[lane_idx].set(count_text)
        
        # Wait time (from traffic controller)
        wait_time = self.system.traffic_controller.lane_wait_times[lane_idx]
        self.wait_time_labels[lane_idx].config(text=f"Wait: {int(wait_time)}s")
        
        # Update traffic light state
        self.update_traffic_indicator(lane_idx)
    
    def get_priority_color(self, priority):
        """Return a color based on the priority value"""
        if priority >= 50:  # Emergency vehicle
            return "#e74c3c"  # Red
        elif priority >= 20:
            return "#f39c12"  # Orange
        elif priority >= 10:
            return "#f1c40f"  # Yellow
        else:
            return "#2ecc71"  # Green
    
    def update_traffic_indicator(self, lane_idx):
        """Update the traffic signal indicator for the specified lane"""
        light_state = self.system.traffic_controller.traffic_states[lane_idx].lower()
        
        canvas = self.signal_indicators[lane_idx]
        canvas.delete("signal")
        
        if light_state == "green":
            color = "green"
        elif light_state == "yellow":
            color = "yellow" 
        else:  # red
            color = "red"
            
        canvas.create_oval(2, 2, 18, 18, fill=color, outline="white", width=1, tags="signal")
    
    def update_system_status(self, status_text, button_text):
        """Update the system status display and button text"""
        self.status_label.config(text=f"Status: {status_text}")
        self.start_stop_button.config(text=button_text)
        
        # Change button color based on system state
        if "Start" in button_text:
            self.start_stop_button.config(bg="green")
        else:
            self.start_stop_button.config(bg="red")