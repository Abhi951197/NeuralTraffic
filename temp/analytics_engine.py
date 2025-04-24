import time
import threading
import numpy as np
from collections import deque
import csv
import os
from datetime import datetime

class AnalyticsEngine:
    def __init__(self, traffic_controller):
        """
        Initialize the analytics engine
        
        Args:
            traffic_controller: TrafficController object
        """
        self.traffic_controller = traffic_controller
        self.num_lanes = traffic_controller.num_lanes
        
        # Historical data storage
        self.time_window = 3600  # Store data for the last hour
        self.sample_interval = 5  # Sample every 5 seconds
        
        # Data structures
        self.vehicle_history = [[] for _ in range(self.num_lanes)]
        self.wait_time_history = [[] for _ in range(self.num_lanes)]
        self.light_state_history = [[] for _ in range(self.num_lanes)]
        self.timestamps = []
        
        # For congestion prediction
        self.prediction_model = SimpleCongestionPredictor(self.num_lanes)
        
        # Reporting
        self.reports_dir = "traffic_reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Running state
        self.running = False
        self.collection_thread = None
    
    def start(self):
        """Start the analytics engine"""
        if self.running:
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_data)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        return True
    
    def stop(self):
        """Stop the analytics engine"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)
    
    def _collect_data(self):
        """Collect and store traffic data"""
        last_collection = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Collect data at the specified interval
            if current_time - last_collection >= self.sample_interval:
                last_collection = current_time
                
                # Get current system data
                priorities = self.traffic_controller.get_priorities()
                wait_times = self.traffic_controller.get_wait_times()
                light_states = self.traffic_controller.get_light_states()
                
                # Store timestamp
                self.timestamps.append(current_time)
                
                # Store data for each lane
                for lane in range(self.num_lanes):
                    self.vehicle_history[lane].append(priorities[lane])
                    self.wait_time_history[lane].append(wait_times[lane])
                    self.light_state_history[lane].append(light_states[lane].value)
                
                # Trim old data outside the time window
                self._trim_old_data()
                
                # Update prediction model
                self.prediction_model.update(priorities, wait_times)
            
            time.sleep(0.1)  # Small delay to prevent high CPU usage
    
    def _trim_old_data(self):
        """Remove data outside the specified time window"""
        if not self.timestamps:
            return
            
        current_time = time.time()
        cutoff_time = current_time - self.time_window
        
        # Find the index of the oldest data to keep
        cutoff_idx = 0
        for i, ts in enumerate(self.timestamps):
            if ts >= cutoff_time:
                cutoff_idx = i
                break
        
        # If we need to trim data
        if cutoff_idx > 0:
            self.timestamps = self.timestamps[cutoff_idx:]
            
            for lane in range(self.num_lanes):
                self.vehicle_history[lane] = self.vehicle_history[lane][cutoff_idx:]
                self.wait_time_history[lane] = self.wait_time_history[lane][cutoff_idx:]
                self.light_state_history[lane] = self.light_state_history[lane][cutoff_idx:]
    
    def get_lane_statistics(self, lane):
        """Get statistics for a specific lane"""
        if lane < 0 or lane >= self.num_lanes or not self.timestamps:
            return None
            
        # Calculate avg, min, max for vehicle counts and wait times
        vehicle_data = np.array(self.vehicle_history[lane])
        wait_data = np.array(self.wait_time_history[lane])
        
        stats = {
            "avg_priority": np.mean(vehicle_data) if len(vehicle_data) > 0 else 0,
            "max_priority": np.max(vehicle_data) if len(vehicle_data) > 0 else 0,
            "avg_wait": np.mean(wait_data) if len(wait_data) > 0 else 0,
            "max_wait": np.max(wait_data) if len(wait_data) > 0 else 0,
            "green_time_percent": self._calculate_green_time_percent(lane)
        }
        
        return stats
    
    def _calculate_green_time_percent(self, lane):
        """Calculate percentage of time the lane had green light"""
        if not self.light_state_history[lane]:
            return 0
            
        # Count green states (value 2 is GREEN)
        green_count = sum(1 for state in self.light_state_history[lane] if state == 2)
        return (green_count / len(self.light_state_history[lane])) * 100
    
    def get_congestion_prediction(self):
        """Get congestion predictions for all lanes"""
        return self.prediction_model.get_predictions()
    
    def generate_report(self):
        """Generate a CSV report with traffic statistics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.reports_dir, f"traffic_report_{timestamp}.csv")
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['lane', 'avg_priority', 'max_priority', 
                         'avg_wait', 'max_wait', 'green_time_percent']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for lane in range(self.num_lanes):
                stats = self.get_lane_statistics(lane)
                if stats:
                    writer.writerow({
                        'lane': lane,
                        'avg_priority': stats['avg_priority'],
                        'max_priority': stats['max_priority'],
                        'avg_wait': stats['avg_wait'],
                        'max_wait': stats['max_wait'],
                        'green_time_percent': stats['green_time_percent']
                    })
        
        return filename


class SimpleCongestionPredictor:
    """A simple predictive model for traffic congestion"""
    
    def __init__(self, num_lanes):
        self.num_lanes = num_lanes
        self.window_size = 12  # Use last 12 data points (1 minute at 5s intervals)
        self.priority_history = [deque(maxlen=self.window_size) for _ in range(num_lanes)]
        self.wait_history = [deque(maxlen=self.window_size) for _ in range(num_lanes)]
        
        # Thresholds for congestion prediction
        self.priority_threshold = 10.0
        self.wait_threshold = 30.0
        self.trend_factor = 1.2  # Factor for trend detection
    
    def update(self, priorities, wait_times):
        """Update the model with new data"""
        for lane in range(self.num_lanes):
            self.priority_history[lane].append(priorities[lane])
            self.wait_history[lane].append(wait_times[lane])
    
    def get_predictions(self):
        """Get congestion predictions for all lanes"""
        predictions = []
        
        for lane in range(self.num_lanes):
            if len(self.priority_history[lane]) < 3:  # Need at least 3 data points
                predictions.append({
                    "lane": lane,
                    "congestion_risk": "Unknown",
                    "trend": "Stable",
                    "estimated_time_to_congestion": None
                })
                continue
            
            # Calculate trends
            priority_trend = self._calculate_trend(self.priority_history[lane])
            wait_trend = self._calculate_trend(self.wait_history[lane])
            
            # Current values
            current_priority = self.priority_history[lane][-1]
            current_wait = self.wait_history[lane][-1]
            
            # Determine congestion risk
            if current_priority >= self.priority_threshold or current_wait >= self.wait_threshold:
                risk = "High"
            elif priority_trend > 0 and current_priority >= self.priority_threshold * 0.7:
                risk = "Medium"
            elif wait_trend > 0 and current_wait >= self.wait_threshold * 0.7:
                risk = "Medium"
            else:
                risk = "Low"
            
            # Determine trend direction
            if priority_trend > 0 and wait_trend > 0:
                trend = "Worsening"
                # Estimate time to congestion
                if current_priority < self.priority_threshold:
                    priority_time = (self.priority_threshold - current_priority) / priority_trend
                else:
                    priority_time = 0
                    
                if current_wait < self.wait_threshold:
                    wait_time = (self.wait_threshold - current_wait) / wait_trend
                else:
                    wait_time = 0
                
                if priority_time > 0 or wait_time > 0:
                    time_to_congestion = min(filter(lambda x: x > 0, [priority_time, wait_time]))
                else:
                    time_to_congestion = 0
            elif priority_trend < 0 and wait_trend < 0:
                trend = "Improving"
                time_to_congestion = None
            else:
                trend = "Stable"
                time_to_congestion = None
            
            predictions.append({
                "lane": lane,
                "congestion_risk": risk,
                "trend": trend,
                "estimated_time_to_congestion": time_to_congestion
            })
        
        return predictions
    
    def _calculate_trend(self, data_queue):
        """Calculate the trend of data points"""
        data = list(data_queue)
        if len(data) < 3:
            return 0
            
        # Simple linear trend calculation
        x = np.arange(len(data))
        y = np.array(data)
        
        # Use linear regression to get slope
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return m  # Return slope as trend indicator