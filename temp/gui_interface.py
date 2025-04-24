import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time

class TrafficSystemGUI:
    def __init__(self, video_processors, traffic_controller):
        """
        Initialize the GUI for the traffic management system
        
        Args:
            video_processors: List of VideoProcessor objects
            traffic_controller: TrafficController object
        """
        self.video_processors = video_processors
        self.traffic_controller = traffic_controller
        self.num_lanes = len(video_processors)
        
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Smart Traffic Management System")
        self.root.geometry("1280x800")
        self.root.configure(bg="#f0f0f0")
        
        # Set up the GUI components
        self._setup_gui()
        
        # Update flag
        self.running = False
        
        # For fps calculation
        self.frame_times = []
        self.last_frame_time = time.time()
    
    def _setup_gui(self):
        """Set up the GUI components"""
        # Create main frames
        self.header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        self.header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.footer_frame = tk.Frame(self.root, bg="#2c3e50", height=40)
        self.footer_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        # Header elements
        self.title_label = tk.Label(
            self.header_frame, 
            text="SMART TRAFFIC MANAGEMENT SYSTEM",
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        self.title_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.status_label = tk.Label(
            self.header_frame,
            text="Status: Ready",
            font=("Arial", 12),
            bg="#2c3e50",
            fg="#2ecc71"
        )
        self.status_label.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Main content
        # Create frames for each lane
        self.lane_frames = []
        self.video_labels = []
        self.status_indicators = []
        self.count_labels = []
        self.priority_bars = []
        self.wait_time_labels = []
        
        # Organize lanes in a grid: 2x2 for 4 lanes
        rows, cols = 2, 2
        for i in range(self.num_lanes):
            row = i // cols
            col = i % cols
            
            lane_frame = tk.Frame(self.main_frame, bg="white", bd=2, relief=tk.RAISED)
            lane_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            # Lane header
            lane_header = tk.Frame(lane_frame, bg="#34495e", height=30)
            lane_header.pack(fill=tk.X)
            
            lane_title = tk.Label(
                lane_header,
                text=f"Lane {i+1}",
                font=("Arial", 12, "bold"),
                bg="#34495e",
                fg="white"
            )
            lane_title.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Traffic light indicator
            light_indicator = tk.Canvas(lane_header, width=20, height=20, bg="#34495e", highlightthickness=0)
            light_indicator.pack(side=tk.RIGHT, padx=5, pady=5)
            light_indicator.create_oval(2, 2, 18, 18, fill="red", tags="light")
            
            # Video display
            video_label = tk.Label(lane_frame, bg="black")
            video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Lane footer with stats
            lane_footer = tk.Frame(lane_frame, bg="#ecf0f1")
            lane_footer.pack(fill=tk.X)
            
            # Vehicle counts
            count_label = tk.Label(
                lane_footer,
                text="Cars: 0 | Trucks: 0 | Buses: 0 | MC: 0 | Emg: 0",
                font=("Arial", 10),
                bg="#ecf0f1"
            )
            count_label.pack(side=tk.TOP, padx=5, pady=(5,0), anchor="w")
            
            # Priority and wait time in same row
            stats_frame = tk.Frame(lane_footer, bg="#ecf0f1")
            stats_frame.pack(fill=tk.X, padx=5, pady=2)
            
            priority_label = tk.Label(
                stats_frame,
                text="Priority:",
                font=("Arial", 10),
                bg="#ecf0f1"
            )
            priority_label.pack(side=tk.LEFT, padx=(0,5))
            
            priority_bar = ttk.Progressbar(
                stats_frame,
                orient=tk.HORIZONTAL,
                length=100,
                mode='determinate'
            )
            priority_bar.pack(side=tk.LEFT, padx=5)
            
            wait_label = tk.Label(
                stats_frame,
                text="Wait: 0s",
                font=("Arial", 10),
                bg="#ecf0f1"
            )
            wait_label.pack(side=tk.RIGHT, padx=5)
            
            # Store references
            self.lane_frames.append(lane_frame)
            self.video_labels.append(video_label)
            self.status_indicators.append(light_indicator)
            self.count_labels.append(count_label)
            self.priority_bars.append(priority_bar)
            self.wait_time_labels.append(wait_label)
        
        # Configure grid weights so cells expand proportionally
        for i in range(rows):
            self.main_frame.grid_rowconfigure(i, weight=1)
        for i in range(cols):
            self.main_frame.grid_columnconfigure(i, weight=1)
        
        # Footer with controls
        self.mode_var = tk.StringVar(value="Automatic")
        
        mode_label = tk.Label(
            self.footer_frame,
            text="Mode:",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="white"
        )
        mode_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        mode_auto = tk.Radiobutton(
            self.footer_frame,
            text="Automatic",
            variable=self.mode_var,
            value="Automatic",
            command=self._mode_changed,
            bg="#2c3e50",
            fg="white",
            selectcolor="#2c3e50",
            activebackground="#2c3e50",
            activeforeground="white"
        )
        mode_auto.pack(side=tk.LEFT, padx=5, pady=5)
        
        mode_manual = tk.Radiobutton(
            self.footer_frame,
            text="Manual",
            variable=self.mode_var,
            value="Manual",
            command=self._mode_changed,
            bg="#2c3e50",
            fg="white",
            selectcolor="#2c3e50",
            activebackground="#2c3e50",
            activeforeground="white"
        )
        mode_manual.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Manual control buttons (initially disabled)
        self.manual_buttons = []
        for i in range(self.num_lanes):
            btn = tk.Button(
                self.footer_frame,
                text=f"Set Lane {i+1} Green",
                command=lambda lane=i: self._manual_set_green(lane),
                state=tk.DISABLED
            )
            btn.pack(side=tk.LEFT, padx=5, pady=5)
            self.manual_buttons.append(btn)
        
        # FPS counter
        self.fps_label = tk.Label(
            self.footer_frame,
            text="FPS: 0",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="white"
        )
        self.fps_label.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Start/Stop button
        self.start_button = tk.Button(
            self.footer_frame,
            text="Start System",
            command=self._toggle_system,
            bg="#27ae60",
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.start_button.pack(side=tk.RIGHT, padx=15, pady=5)
    
    def _toggle_system(self):
        """Toggle system between running and stopped states"""
        if not self.running:
            # Start system
            success = True
            
            # Start video processors
            for processor in self.video_processors:
                if not processor.start():
                    success = False
            
            # Start traffic controller
            if not self.traffic_controller.start():
                success = False
            
            if success:
                self.running = True
                self.start_button.config(text="Stop System", bg="#c0392b")
                self.status_label.config(text="Status: Running", fg="#2ecc71")
                
                # Start update loop
                self.update_thread = threading.Thread(target=self._update_loop)
                self.update_thread.daemon = True
                self.update_thread.start()
        else:
            # Stop system
            self.running = False
            
            # Stop video processors
            for processor in self.video_processors:
                processor.stop()
            
            # Stop traffic controller
            self.traffic_controller.stop()
            
            self.start_button.config(text="Start System", bg="#27ae60")
            self.status_label.config(text="Status: Stopped", fg="#e74c3c")
    
    def _update_loop(self):
        """Main update loop for the GUI"""
        while self.running:
            # Update GUI with latest information
            try:
                self._update_gui()
            except Exception as e:
                print(f"GUI update error: {str(e)}")
            
            # Sleep briefly to avoid consuming too much CPU
            time.sleep(0.033)  # ~30 FPS
    
    def _update_gui(self):
        """Update the GUI with latest data"""
        # Calculate FPS
        current_time = time.time()
        self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        # Keep only last 30 frames for FPS calculation
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Calculate average FPS
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        
        # Get light states
        light_states = self.traffic_controller.get_light_states()
        priorities = self.traffic_controller.get_priorities()
        wait_times = self.traffic_controller.get_wait_times()
        
        # Update for each lane
        for i in range(self.num_lanes):
            # Update video frame
            processor = self.video_processors[i]
            frame = processor.get_annotated_frame()
            
            if frame is not None:
                # Scale down for display if needed
                h, w = frame.shape[:2]
                display_w = self.video_labels[i].winfo_width()
                display_h = self.video_labels[i].winfo_height()
                
                if display_w > 0 and display_h > 0:
                    # Calculate scale ratio to fit
                    scale = min(display_w/w, display_h/h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # Resize the frame
                    frame = cv2.resize(frame, (new_w, new_h))
                
                # Convert to PhotoImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update label
                self.video_labels[i].config(image=img_tk)
                self.video_labels[i].image = img_tk  # Keep reference to prevent garbage collection
            
            # Update traffic light indicator
            light_state = light_states[i]
            if light_state.name == "RED":
                color = "red"
            elif light_state.name == "YELLOW":
                color = "yellow"
            else:  # GREEN
                color = "green"
            
            self.status_indicators[i].itemconfig("light", fill=color)
            
            # Update vehicle counts
            counts = processor.get_vehicle_counts()
            count_text = f"Cars: {counts['car']} | Trucks: {counts['truck']} | " \
                         f"Buses: {counts['bus']} | MC: {counts['motorcycle']} | Emg: {counts['emergency']}"
            self.count_labels[i].config(text=count_text)
            
            # Update priority bar - scale to 0-100
            max_priority = max(priorities) if max(priorities) > 0 else 1
            priority_percent = (priorities[i] / max_priority) * 100
            self.priority_bars[i].config(value=priority_percent)
            
            # Update wait time
            self.wait_time_labels[i].config(text=f"Wait: {int(wait_times[i])}s")
            
            # Update traffic controller with latest data
            self.traffic_controller.update_vehicle_counts(i, counts)
            self.traffic_controller.update_wait_time(i, processor.get_wait_time())
        
        # Update root window to refresh display
        self.root.update()
    
    def _mode_changed(self):
        """Handle mode change between Automatic and Manual"""
        mode = self.mode_var.get()
        self.traffic_controller.set_operation_mode(mode)
        
        # Enable/disable manual control buttons
        button_state = tk.NORMAL if mode == "Manual" else tk.DISABLED
        for button in self.manual_buttons:
            button.config(state=button_state)
    
    def _manual_set_green(self, lane):
        """Manually set a lane to green (Manual mode only)"""
        self.traffic_controller.manual_set_green(lane)
    
    def start(self):
        """Start the GUI main loop"""
        self.root.mainloop()

    def _update_video_frames(self):
        """Update video frames and UI components"""
        for i, processor in enumerate(self.video_processors):
            frame = processor.get_frame()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            display_w, display_h = self.video_labels[i].winfo_width(), self.video_labels[i].winfo_height()

            if display_w > 0 and display_h > 0:
            # Calculate scale ratio to fit
                scale = min(display_w / w, display_h / h)
                new_size = (int(w * scale), int(h * scale))
                resized_frame = cv2.resize(frame, new_size)
            else:
                resized_frame = frame

        # Convert BGR to RGB and show in Tkinter
            image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image=image)

        # Avoid garbage collection
            self.video_labels[i].image = image_tk
            self.video_labels[i].config(image=image_tk)

        # Update traffic light indicator
            light_color = "green" if self.traffic_controller.get_light_state(i) else "red"
            canvas = self.status_indicators[i]
            canvas.itemconfig("light", fill=light_color)

        # Update vehicle counts
            counts = processor.get_vehicle_counts()
            count_text = f"Cars: {counts.get('car', 0)} | Trucks: {counts.get('truck', 0)} | Buses: {counts.get('bus', 0)} | MC: {counts.get('motorcycle', 0)} | Emg: {counts.get('emergency', 0)}"
            self.count_labels[i].config(text=count_text)

        # Update priority bar
            self.priority_bars[i]['value'] = self.traffic_controller.get_priority(i)

        # Update wait time
            self.wait_time_labels[i].config(text=f"Wait: {self.traffic_controller.get_wait_time(i)}s")

    def _mode_changed(self):
        """Handle mode switch between Automatic and Manual"""
        mode = self.mode_var.get()
        is_manual = mode == "Manual"

        for btn in self.manual_buttons:
            btn.config(state=tk.NORMAL if is_manual else tk.DISABLED)

        self.traffic_controller.set_mode(mode)

    def _manual_set_green(self, lane):
        """Manually set the specified lane's light to green"""
        self.traffic_controller.set_manual_green(lane)

    def run(self):
        """Run the GUI application"""
        self.root.mainloop()
