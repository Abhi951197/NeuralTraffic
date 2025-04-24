import time
import numpy as np
import datetime

class TrafficController:
    def __init__(self, lane_count):
        """Initialize the traffic controller"""
        self.lane_count = lane_count
        
        # Traffic control parameters
        self.green_time = 15  # Default green time in seconds
        self.yellow_time = 3  # Yellow time in seconds
        self.green_time_factor = 1.0  # Multiplier for green time based on congestion
        
        # Traffic light states and control
        self.traffic_states = ['red'] * lane_count
        self.current_green_lane = None
        
        # Lane metrics
        self.lane_priorities = [0] * lane_count
        self.lane_vehicle_counts = [{} for _ in range(lane_count)]
        self.lane_wait_times = [0] * lane_count
        self.last_green_time = [0] * lane_count
        self.priority_history = [[] for _ in range(lane_count)]
        self.trend_window = 10  # Number of cycles to calculate trend
    
    def calculate_green_time(self, lane_idx):
        """Calculate dynamic green time based on priority and vehicle count"""
        base_time = 10  # Minimum green time in seconds
    
        # Calculate additional time based on vehicle count and types
        vehicle_count = sum(self.lane_vehicle_counts[lane_idx].values())
        priority = self.lane_priorities[lane_idx]
        print(f"Lane {lane_idx + 1} - Vehicle Count: {vehicle_count}, Priority: {priority}")
    
        # Logarithmic scaling to prevent excessively long green times
        additional_time = min(30, 5 * np.log2(priority + 1))
        print(f"Lane {lane_idx + 1} - Additional Time: {additional_time:.2f} seconds")
    
        return base_time + additional_time
    
    def update_wait_times(self):
        """Update wait times for all lanes"""
        current_time = time.time()
        for i in range(self.lane_count):
            if self.traffic_states[i] != 'green':
                self.lane_wait_times[i] = current_time - self.last_green_time[i]
            else:
                self.lane_wait_times[i] = 0
                self.last_green_time[i] = current_time
    
    def update_priority_history(self):
        """Update priority history for trend analysis"""
        for i in range(self.lane_count):
            if len(self.priority_history[i]) >= self.trend_window:
                self.priority_history[i].pop(0)
            self.priority_history[i].append(self.lane_priorities[i])
    
    def calculate_trend(self, lane_idx):
        """Calculate trend in priority for a lane using linear regression"""
        if len(self.priority_history[lane_idx]) < self.trend_window:
            return 0
        
        # Simple linear regression to detect trend
        x = np.array(range(len(self.priority_history[lane_idx])))
        y = np.array(self.priority_history[lane_idx])
        slope, _ = np.polyfit(x, y, 1)
        
        return slope
    
    def detect_system_congestion(self):
        """Detect if the overall system is congested or imbalanced"""
        avg_priority = sum(self.lane_priorities) / self.lane_count
        max_priority = max(self.lane_priorities)
        
        # System is congested if average priority is high
        system_congested = avg_priority > 20
        
        # Check for severe imbalance
        severe_imbalance = max_priority > 3 * avg_priority
        
        return system_congested, severe_imbalance
    
    def manage_congestion(self):
        """Adjust traffic control based on congestion state"""
        system_congested, severe_imbalance = self.detect_system_congestion()
        
        if system_congested:
            # Reduce green times slightly to cycle through lanes faster
            self.green_time_factor = 0.8
        else:
            self.green_time_factor = 1.0
        
        if severe_imbalance:
            # Prioritize the most congested lane with longer green time
            self.green_time_factor = 1.2
    
    def get_time_of_day_pattern(self):
        """Determine traffic pattern based on time of day"""
        current_hour = datetime.datetime.now().hour
        
        if 7 <= current_hour < 10:  # Morning rush
            return 'morning_rush'
        elif 16 <= current_hour < 19:  # Evening rush
            return 'evening_rush'
        elif 22 <= current_hour or current_hour < 5:  # Night
            return 'night'
        else:  # Normal daytime
            return 'normal'
    
    def apply_time_pattern(self):
        """Apply time-of-day adjustments to lane priorities"""
        pattern = self.get_time_of_day_pattern()
        
        if pattern == 'morning_rush':
            # Adjust weights for incoming lanes to city
            for i in range(self.lane_count):
                if i in [0, 3]:  # Assuming lanes 0 and 3 are incoming
                    self.lane_priorities[i] *= 1.3
        elif pattern == 'evening_rush':
            # Adjust weights for outgoing lanes from city
            for i in range(self.lane_count):
                if i in [1, 2]:  # Assuming lanes 1 and 2 are outgoing
                    self.lane_priorities[i] *= 1.3
        elif pattern == 'night':
            # Equal priority, shorter cycles at night
            self.green_time = min(self.green_time, 15)
    
    def control_traffic_lights(self, system):
        """Control traffic lights based on improved logic"""
        while system.is_running:
            # Update wait times
            self.update_wait_times()
            
            # Update priority history for trend analysis
            self.update_priority_history()
            
            # Apply time-of-day adjustments
            self.apply_time_pattern()
            
            # Manage congestion
            self.manage_congestion()
            
            # Calculate adjusted priorities including wait times and trends
            adjusted_priorities = []
            for i in range(self.lane_count):
                base_priority = self.lane_priorities[i]
                wait_factor = min(3, self.lane_wait_times[i] / 30) 
                trend = self.calculate_trend(i)
                
                # Combined priority score
                adjusted_priority = base_priority * (1 + wait_factor) + (trend * 5)
                adjusted_priorities.append(adjusted_priority)
            
            # Find lane with highest adjusted priority
            highest_priority_lane = np.argmax(adjusted_priorities)
            highest_priority = adjusted_priorities[highest_priority_lane]
            
            # Check for emergency vehicles in any lane
            emergency_detected = False
            emergency_lane = None
            for i in range(self.lane_count):
                if 'emergency' in self.lane_vehicle_counts[i] and self.lane_vehicle_counts[i]['emergency'] > 0:
                    emergency_detected = True
                    emergency_lane = i
                    print(f"⚠️ EMERGENCY VEHICLE DETECTED in Lane {i+1}! Switching traffic priority immediately")

                    break
            
            if emergency_detected:
                highest_priority_lane = emergency_lane
            
            # Switch to new lane if needed
            if self.current_green_lane != highest_priority_lane:
                # Yellow transition for current green lane
                if self.current_green_lane is not None:
                    self.traffic_states[self.current_green_lane] = 'yellow'
                    
                    # Update in GUI if available
                    if hasattr(system, 'gui'):
                        system.gui.update_traffic_indicator(self.current_green_lane)
                    
                    time.sleep(self.yellow_time)
                
                # All red safety period
                self.traffic_states = ['red'] * self.lane_count
                for i in range(self.lane_count):
                    if hasattr(system, 'gui'):
                        system.gui.update_traffic_indicator(i)
                time.sleep(1)
                
                # Calculate dynamic green time for new lane
                dynamic_green_time = self.calculate_green_time(highest_priority_lane)
                dynamic_green_time *= self.green_time_factor  # Apply congestion factor
                
                # Make new lane green
                self.traffic_states[highest_priority_lane] = 'green'
                self.current_green_lane = highest_priority_lane
                
                if hasattr(system, 'gui'):
                    system.gui.update_traffic_indicator(highest_priority_lane)
                
                # Log lane switch with dynamic time
                print(f"Switching to Lane {highest_priority_lane + 1} with priority {highest_priority:.2f}")
                print(f"Dynamic green time: {dynamic_green_time:.1f} seconds")
                
                # Wait for calculated green time
                time.sleep(dynamic_green_time)
            else:
                # Continue with same green lane
                time.sleep(1)
    
    def set_green_time(self, time_value):
        """Set green light duration"""
        self.green_time = time_value
    
    def set_yellow_time(self, time_value):
        """Set yellow light duration"""
        self.yellow_time = time_value