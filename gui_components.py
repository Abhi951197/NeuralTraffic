import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

class SystemGUI:
    def __init__(self, system):
        """Initialize the GUI for the traffic management system"""
        self.system = system
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("AI-Powered Traffic Management System")
        self.root.geometry("1200x800")
        
        # Create GUI components
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the GUI layout and components"""
        # Create two main sections: video grid and traffic control panel
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Create video grid
        video_grid = ttk.Frame(left_frame)
        video_grid.pack(fill=tk.BOTH, expand=True)
        
        # Create traffic control panel
        traffic_panel = ttk.LabelFrame(right_frame, text="Traffic Control")
        traffic_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create traffic light indicators and priority displays
        self.traffic_indicators = []
        self.priority_labels = []
        self.vehicle_count_labels = []
        
        for i in range(self.system.lane_count):
            # Create lane frame in traffic panel
            lane_frame = ttk.LabelFrame(traffic_panel, text=f"Lane {i+1}")
            lane_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Create traffic light indicator
            indicator_frame = ttk.Frame(lane_frame)
            indicator_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(indicator_frame, text="Signal:").pack(side=tk.LEFT, padx=5)
            traffic_indicator = ttk.Label(indicator_frame, text="RED", background="red", foreground="white", width=10)
            traffic_indicator.pack(side=tk.LEFT, padx=5)
            self.traffic_indicators.append(traffic_indicator)
            
            # Create priority display
            priority_frame = ttk.Frame(lane_frame)
            priority_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(priority_frame, text="Priority:").pack(side=tk.LEFT, padx=5)
            priority_label = ttk.Label(priority_frame, text="0")
            priority_label.pack(side=tk.LEFT, padx=5)
            self.priority_labels.append(priority_label)
            
            # Create vehicle count display
            vehicle_count_frame = ttk.Frame(lane_frame)
            vehicle_count_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(vehicle_count_frame, text="Vehicles:").pack(side=tk.LEFT, padx=5)
            vehicle_count_label = ttk.Label(vehicle_count_frame, text="No vehicles")
            vehicle_count_label.pack(side=tk.LEFT, padx=5)
            self.vehicle_count_labels.append(vehicle_count_label)
        
        # Create video displays in grid
        self.video_frames = []
        self.video_labels = []
        
        for i in range(self.system.lane_count):
            # Create frame for each lane
            row, col = i // 2, i % 2
            lane_frame = ttk.LabelFrame(video_grid, text=f"Lane {i+1}")
            lane_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Create video display
            video_frame = ttk.Frame(lane_frame)
            video_frame.pack(pady=5, fill=tk.BOTH, expand=True)
            video_label = ttk.Label(video_frame)
            video_label.pack(fill=tk.BOTH, expand=True)
            
            self.video_frames.append(video_frame)
            self.video_labels.append(video_label)
        
        # Configure grid weights
        for i in range(2):
            video_grid.grid_columnconfigure(i, weight=1)
            video_grid.grid_rowconfigure(i, weight=1)
        
        # Create control panel
        control_frame = ttk.LabelFrame(right_frame, text="System Control")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Green time slider
        ttk.Label(control_frame, text="Green Light Duration (sec):").pack(anchor=tk.W, padx=5, pady=5)
        self.green_slider = ttk.Scale(control_frame, from_=5, to=60, orient=tk.HORIZONTAL)
        self.green_slider.set(self.system.traffic_controller.green_time)
        self.green_slider.pack(fill=tk.X, padx=5, pady=5)
        self.green_slider.bind("<ButtonRelease-1>", lambda e: self.set_green_time())
        
        # Yellow time slider
        ttk.Label(control_frame, text="Yellow Light Duration (sec):").pack(anchor=tk.W, padx=5, pady=5)
        self.yellow_slider = ttk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        self.yellow_slider.set(self.system.traffic_controller.yellow_time)
        self.yellow_slider.pack(fill=tk.X, padx=5, pady=5)
        self.yellow_slider.bind("<ButtonRelease-1>", lambda e: self.set_yellow_time())
        
        # Start/Stop button
        self.start_stop_button = ttk.Button(control_frame, text="Start System", command=self.system.toggle_system)
        self.start_stop_button.pack(fill=tk.X, padx=5, pady=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="System Ready")
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Add analytics section
        analytics_frame = ttk.LabelFrame(right_frame, text="Traffic Analytics")
        analytics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.congestion_label = ttk.Label(analytics_frame, text="System Congestion: Normal")
        self.congestion_label.pack(fill=tk.X, padx=5, pady=5)
        
        self.pattern_label = ttk.Label(analytics_frame, text="Traffic Pattern: Normal")
        self.pattern_label.pack(fill=tk.X, padx=5, pady=5)
        
        self.wait_time_labels = []
        for i in range(self.system.lane_count):
            wait_time_label = ttk.Label(analytics_frame, text=f"Lane {i+1} Wait Time: 0 sec")
            wait_time_label.pack(fill=tk.X, padx=5, pady=2)
            self.wait_time_labels.append(wait_time_label)
    
    def set_green_time(self):
        """Update green time based on slider value"""
        new_time = int(self.green_slider.get())
        self.system.traffic_controller.set_green_time(new_time)
    
    def set_yellow_time(self):
        """Update yellow time based on slider value"""
        new_time = int(self.yellow_slider.get())
        self.system.traffic_controller.set_yellow_time(new_time)
    
    def update_system_status(self, status_text, button_text):
        """Update system status and button text"""
        self.status_label.config(text=status_text)
        self.start_stop_button.config(text=button_text)
    
    def update_traffic_indicator(self, lane_idx):
        """Update traffic light indicator for a specific lane"""
        state = self.system.traffic_controller.traffic_states[lane_idx]
        indicator = self.traffic_indicators[lane_idx]
        
        if state == 'red':
            indicator.config(text="RED", background="red", foreground="white")
        elif state == 'yellow':
            indicator.config(text="YELLOW", background="yellow", foreground="black")
        else:  # green
            indicator.config(text="GREEN", background="green", foreground="white")
    
    def update_lane_display(self, lane_idx, frame, traffic_state, priority, vehicle_counts):
        """Update display for a specific lane"""
        # Convert frame to TkInter format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img = ImageTk.PhotoImage(image=img)
        
        # Update image
        self.video_labels[lane_idx].config(image=img)
        self.video_labels[lane_idx].image = img
        
        # Update priority label
        self.priority_labels[lane_idx].config(text=str(priority))
        
        # Update vehicle count label
        count_text = ", ".join([f"{v_type}: {count}" for v_type, count in vehicle_counts.items()])
        if not count_text:
            count_text = "No vehicles"
        self.vehicle_count_labels[lane_idx].config(text=count_text)
        
        # Update traffic light indicator
        self.update_traffic_indicator(lane_idx)
        
        # Update wait time label if available
        if hasattr(self, 'wait_time_labels') and lane_idx < len(self.wait_time_labels):
            wait_time = self.system.traffic_controller.lane_wait_times[lane_idx]
            self.wait_time_labels[lane_idx].config(text=f"Lane {lane_idx+1} Wait Time: {wait_time:.1f} sec")
    
    def update_analytics_display(self):
        """Update analytics display with current system state"""
        # Update congestion status
        system_congested, severe_imbalance = self.system.traffic_controller.detect_system_congestion()
        
        if severe_imbalance:
            congestion_text = "System Congestion: Severe Imbalance"
        elif system_congested:
            congestion_text = "System Congestion: High"
        else:
            congestion_text = "System Congestion: Normal"
        
        self.congestion_label.config(text=congestion_text)
        
        # Update time pattern
        pattern = self.system.traffic_controller.get_time_of_day_pattern()
        pattern_names = {
            'morning_rush': 'Morning Rush Hour',
            'evening_rush': 'Evening Rush Hour',
            'night': 'Night Mode',
            'normal': 'Normal Daytime'
        }
        
        self.pattern_label.config(text=f"Traffic Pattern: {pattern_names.get(pattern, 'Normal')}")