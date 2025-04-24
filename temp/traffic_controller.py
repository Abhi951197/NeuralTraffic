import time
import threading
from enum import Enum
import numpy as np
from collections import deque

class LightState(Enum):
    """Traffic light states"""
    RED = 0
    YELLOW = 1
    GREEN = 2

class TrafficController:
    def __init__(self, num_lanes=4):
        """
        Initialize the traffic controller
        
        Args:
            num_lanes: Number of lanes to control
        """
        self.num_lanes = num_lanes
        self.states = [LightState.RED] * num_lanes
        self.states[0] = LightState.GREEN  # Start with lane 0 having green
        
        # Lane priority and timing settings
        self.min_green_time = 10  # Minimum green light duration in seconds
        self.max_green_time = 60  # Maximum green light duration in seconds
        self.yellow_time = 3      # Yellow light duration in seconds
        self.all_red_time = 1     # All-red clearance interval in seconds
        
        # Priority weight factors for different vehicles
        self.vehicle_weights = {
            "car": 1.0,
            "truck": 1.5,
            "bus": 2.0,
            "motorcycle": 0.8,
            "emergency": 10.0
        }
        
        # Maximum wait time threshold
        self.max_wait_threshold = 120  # seconds
        
        # Current active lane and timing info
        self.active_lane = 0
        self.active_since = time.time()
        self.phase_start_time = time.time()
        self.yellow_active = False
        
        # Store lane priorities and wait times
        self.lane_priorities = [0] * num_lanes
        self.lane_wait_times = [0] * num_lanes
        
        # Time of day pattern tracking
        self.time_of_day_patterns = {
            "morning_rush": False,    # 7am-9am
            "midday": False,          # 10am-3pm
            "evening_rush": False,    # 4pm-7pm
            "night": False            # 8pm-6am
        }
        
        # For traffic trend analysis
        self.historical_counts = [deque(maxlen=60) for _ in range(num_lanes)]  # Keep last 60 counts
        
        # Control thread
        self.running = False
        self.control_thread = None
        
        # For congestion detection
        self.congestion_threshold = 20  # Number of vehicles to consider congestion
        self.congestion_state = [False] * num_lanes
        
        # For system status
        self.system_status = "Ready"
        self.operation_mode = "Automatic"  # Automatic vs. Manual mode
    
    def start(self):
        """Start the traffic control system"""
        if self.running:
            return
            
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        self.system_status = "Running"
        return True
    
    def stop(self):
        """Stop the traffic control system"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        self.system_status = "Stopped"
    
    def update_vehicle_counts(self, lane_id, counts):
        """
        Update traffic counts for a specific lane
        
        Args:
            lane_id: Lane identifier
            counts: Dictionary of vehicle counts by type
        """
        if lane_id < 0 or lane_id >= self.num_lanes:
            return
            
        # Calculate lane priority based on vehicle types and weights
        priority = sum(counts[vehicle_type] * self.vehicle_weights[vehicle_type] 
                      for vehicle_type in counts)
                      
        # Store historical data for trend analysis
        total_count = sum(counts.values())
        self.historical_counts[lane_id].append(total_count)
        
        # Update lane priority
        self.lane_priorities[lane_id] = priority
        
        # Check for emergency vehicles (immediate priority)
        if counts["emergency"] > 0 and self.states[lane_id] == LightState.RED:
            self._handle_emergency_vehicle(lane_id)
            
        # Detect congestion
        self.congestion_state[lane_id] = (total_count >= self.congestion_threshold)
    
    def update_wait_time(self, lane_id, wait_time):
        """Update wait time for a specific lane"""
        if lane_id < 0 or lane_id >= self.num_lanes:
            return
            
        self.lane_wait_times[lane_id] = wait_time
    
    def _handle_emergency_vehicle(self, lane_id):
        """Handle emergency vehicle detection by prioritizing the lane"""
        if self.operation_mode == "Automatic" and lane_id != self.active_lane:
            # Only switch if we've been in current state for at least minimum time
            current_duration = time.time() - self.active_since
            if current_duration >= self.min_green_time:
                # Transition current green to yellow
                if self.states[self.active_lane] == LightState.GREEN:
                    self.states[self.active_lane] = LightState.YELLOW
                    self.yellow_active = True
                    self.phase_start_time = time.time()
                    # Next active lane will be the emergency lane
                    self.next_active_lane = lane_id
    
    def _control_loop(self):
        """Main traffic control loop"""
        while self.running:
            current_time = time.time()
            
            # Update time of day patterns
            self._update_time_of_day_patterns()
            
            # Handle automatic mode logic
            if self.operation_mode == "Automatic":
                # Check if yellow phase is complete
                if self.yellow_active:
                    if (current_time - self.phase_start_time) >= self.yellow_time:
                        # Transition to all-red phase
                        self.states[self.active_lane] = LightState.RED
                        self.yellow_active = False
                        self.phase_start_time = current_time
                        
                        # After brief all-red, activate next lane
                        if (current_time - self.phase_start_time) >= self.all_red_time:
                            if hasattr(self, 'next_active_lane'):
                                self.active_lane = self.next_active_lane
                                delattr(self, 'next_active_lane')
                            else:
                                self.active_lane = self._select_next_lane()
                            
                            self.states[self.active_lane] = LightState.GREEN
                            self.active_since = current_time
                            self.phase_start_time = current_time
                
                # Check if green phase should end
                elif self.states[self.active_lane] == LightState.GREEN:
                    green_duration = current_time - self.active_since
                    
                    # Check if we should end green phase based on timing and priorities
                    if self._should_end_green_phase(green_duration):
                        self.states[self.active_lane] = LightState.YELLOW
                        self.yellow_active = True
                        self.phase_start_time = current_time
            
            time.sleep(0.1)  # Small delay to prevent high CPU usage
    
    def _should_end_green_phase(self, duration):
        """Determine if the current green phase should end"""
        # Always respect minimum green time
        if duration < self.min_green_time:
            return False
            
        # Always respect maximum green time
        if duration >= self.max_green_time:
            return True
            
        # Check if other lanes have higher priority now
        current_priority = self.lane_priorities[self.active_lane]
        max_priority = max(self.lane_priorities)
        max_priority_lane = self.lane_priorities.index(max_priority)
        
        # Check for excessive wait times in other lanes
        max_wait = max(self.lane_wait_times)
        max_wait_lane = self.lane_wait_times.index(max_wait)
        
        # End green phase if another lane has significantly higher priority
        # or if another lane has waited too long
        return (max_priority > current_priority * 2 and max_priority_lane != self.active_lane) or \
               (max_wait > self.max_wait_threshold and max_wait_lane != self.active_lane)
    
    def _select_next_lane(self):
        """Select the next lane to get green light"""
        # Start with a simple round-robin approach
        next_lane = (self.active_lane + 1) % self.num_lanes
        
        # Check for lanes with high priority or wait time
        highest_priority = -1
        selected_lane = next_lane
        
        for lane in range(self.num_lanes):
            if lane == self.active_lane:
                continue
                
            # Calculate combined score based on priority and wait time
            wait_factor = min(self.lane_wait_times[lane] / 30.0, 3.0)  # Cap at 3x multiplier
            combined_score = self.lane_priorities[lane] * (1.0 + wait_factor)
            
            if combined_score > highest_priority:
                highest_priority = combined_score
                selected_lane = lane
        
        return selected_lane
    
    def _update_time_of_day_patterns(self):
        """Update time of day patterns based on current time"""
        current_hour = time.localtime().tm_hour
        
        self.time_of_day_patterns["morning_rush"] = (7 <= current_hour < 9)
        self.time_of_day_patterns["midday"] = (10 <= current_hour < 15)
        self.time_of_day_patterns["evening_rush"] = (16 <= current_hour < 19)
        self.time_of_day_patterns["night"] = (20 <= current_hour or current_hour < 6)
        
        # Adjust timing parameters based on time of day
        if self.time_of_day_patterns["morning_rush"] or self.time_of_day_patterns["evening_rush"]:
            self.min_green_time = 15
            self.max_green_time = 90
        elif self.time_of_day_patterns["night"]:
            self.min_green_time = 5
            self.max_green_time = 30
        else:  # midday
            self.min_green_time = 10
            self.max_green_time = 60
    
    def get_light_states(self):
        """Return current light states for all lanes"""
        return self.states
    
    def get_priorities(self):
        """Return current priority scores for all lanes"""
        return self.lane_priorities
    
    def get_wait_times(self):
        """Return current wait times for all lanes"""
        return self.lane_wait_times
    
    def get_congestion_states(self):
        """Return congestion states for all lanes"""
        return self.congestion_state
    
    def get_system_status(self):
        """Return current system status"""
        return {
            "status": self.system_status,
            "mode": self.operation_mode,
            "active_lane": self.active_lane,
            "time_patterns": self.time_of_day_patterns
        }
    
    def set_operation_mode(self, mode):
        """Set operation mode (Automatic or Manual)"""
        if mode in ["Automatic", "Manual"]:
            self.operation_mode = mode
            return True
        return False
    
    def manual_set_green(self, lane_id):
        """Manually set a lane to green (only in Manual mode)"""
        if self.operation_mode != "Manual" or lane_id < 0 or lane_id >= self.num_lanes:
            return False
            
        # Set all lanes to red
        self.states = [LightState.RED] * self.num_lanes
        # Set selected lane to green
        self.states[lane_id] = LightState.GREEN
        self.active_lane = lane_id
        self.active_since = time.time()
        return True