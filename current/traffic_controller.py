import time
import math
import threading
from enum import Enum
from collections import deque


class LightState(Enum):
    RED    = 0
    YELLOW = 1
    GREEN  = 2


class TrafficController:
    """
    Controls traffic lights for N lanes.

    Normal operation
    ────────────────
    Priority-based: highest combined (priority × wait-factor) lane gets green.
    Green phase ends when another lane has 2× priority, max wait exceeded, or
    the assigned green time expires.

    Emergency (ambulance) operation
    ────────────────────────────────
    When lane B detects an ambulance and is NOT already green:
      1. Current green lane → YELLOW (3 s)
      2. All-red clearance (1 s)
      3. Lane B → GREEN for up to EMERGENCY_GREEN_MAX seconds
      4. If ambulance leaves B before the max:
           Lane B → YELLOW (3 s) → RED
           Interrupted lane resumes for its remaining assigned time
           (if remaining < min_green_time, give min_green_time)
      5. If max time elapses before ambulance leaves:
           Lane B → YELLOW (3 s) → RED
           Normal priority selection resumes

    Assigned green time
    ───────────────────
    Calculated per-lane using log-scaled priority when the lane turns green.
    Exposed via get_assigned_green_times() / get_green_elapsed() for the UI.

    Wait time
    ─────────
    Supplied by each VideoProcessor: seconds since the lane was last empty.
    It is a measure of queue pressure, not time-at-red.
    """

    EMERGENCY_GREEN_MAX = 15.0  # seconds

    def __init__(self, num_lanes=4):
        self.num_lanes = num_lanes

        # Light states — lane 0 starts green
        self.states = [LightState.RED] * num_lanes
        self.states[0] = LightState.GREEN

        # Timing parameters (adjusted by time-of-day logic)
        self.min_green_time = 10
        self.max_green_time = 60
        self.yellow_time    = 3
        self.all_red_time   = 1

        # Vehicle priority weights
        self.vehicle_weights = {
            "car":              1.0,
            "truck":            1.5,
            "bus":              2.0,
            "motorcycle":       0.8,
            "autorickshaw":     1.2,
            "bicycle":          0.5,
            "vehicle_fallback": 1.0,
            "emergency":        0.0,  # handled separately; don't inflate normal priority
        }

        self.max_wait_threshold = 120

        # ── Phase tracking ────────────────────────────────────────────────
        self.active_lane      = 0
        self.active_since     = time.time()
        self.phase_start_time = time.time()

        self.yellow_active    = False
        self.all_red_active   = False
        self.next_active_lane = None   # overrides _select_next_lane when set

        # Assigned green time for the currently active lane
        # Lane 0 starts green, give it min_green_time so the UI shows a value immediately
        self.assigned_green_times = [0.0] * num_lanes
        self.assigned_green_times[0] = self.min_green_time

        # ── Per-lane data ─────────────────────────────────────────────────
        self.lane_priorities      = [0.0] * num_lanes
        self.lane_wait_times      = [0.0] * num_lanes
        self._lane_emg_counts     = [0]   * num_lanes   # ambulance counts per lane
        self.historical_counts    = [deque(maxlen=60) for _ in range(num_lanes)]
        self.congestion_state     = [False] * num_lanes
        self.congestion_threshold = 20

        # ── Emergency state machine ───────────────────────────────────────
        self._emg_active           = False  # True while handling an emergency
        self._emg_lane             = None   # lane that has the ambulance
        self._emg_green_start      = None   # when the emergency lane went green
        self._pre_emg_lane         = None   # lane that was green before emergency
        self._pre_emg_elapsed      = 0.0    # seconds it had already been green
        self._pre_emg_assigned     = 0.0    # its originally assigned green time
        self._resuming_pre_emg     = False  # True when returning to interrupted lane

        # ── Time-of-day ───────────────────────────────────────────────────
        self.time_of_day_patterns = {
            "morning_rush": False,
            "midday":       False,
            "evening_rush": False,
            "night":        False,
        }

        # ── Thread ────────────────────────────────────────────────────────
        self.running        = False
        self.control_thread = None
        self.operation_mode = "Automatic"
        self.system_status  = "Ready"

    # ══════════════════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════════════════

    def start(self):
        if self.running:
            return True
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        self.system_status = "Running"
        return True

    def stop(self):
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        self.system_status = "Stopped"

    # ══════════════════════════════════════════════════════════════════════
    # Data updates  (called from GUI thread, ~30 fps)
    # ══════════════════════════════════════════════════════════════════════

    def update_vehicle_counts(self, lane_id, counts):
        if not (0 <= lane_id < self.num_lanes):
            return

        # Store emergency count separately; exclude from normal priority
        self._lane_emg_counts[lane_id] = counts.get("emergency", 0)

        priority = sum(
            counts.get(vtype, 0) * weight
            for vtype, weight in self.vehicle_weights.items()
        )
        total = sum(counts.get(k, 0) for k in self.vehicle_weights if k != "emergency")
        self.historical_counts[lane_id].append(total)
        self.lane_priorities[lane_id]  = priority
        self.congestion_state[lane_id] = (total >= self.congestion_threshold)

    def update_wait_time(self, lane_id, wait_time):
        if 0 <= lane_id < self.num_lanes:
            self.lane_wait_times[lane_id] = wait_time

    # ══════════════════════════════════════════════════════════════════════
    # Public getters
    # ══════════════════════════════════════════════════════════════════════

    def get_light_states(self):
        return list(self.states)

    def get_priorities(self):
        return list(self.lane_priorities)

    def get_wait_times(self):
        return list(self.lane_wait_times)

    def get_congestion_states(self):
        return list(self.congestion_state)

    def get_active_lane(self):
        return self.active_lane

    def get_assigned_green_times(self):
        """Assigned green time per lane.  Non-active lanes return 0."""
        return list(self.assigned_green_times)

    def get_green_elapsed(self):
        """Seconds the current active lane has been green (0 if not green)."""
        if self.states[self.active_lane] == LightState.GREEN:
            return time.time() - self.active_since
        return 0.0

    def get_system_status(self):
        return {
            "status":        self.system_status,
            "mode":          self.operation_mode,
            "active_lane":   self.active_lane,
            "emergency":     self._emg_active,
            "time_patterns": self.time_of_day_patterns,
        }

    # ══════════════════════════════════════════════════════════════════════
    # Manual / mode control
    # ══════════════════════════════════════════════════════════════════════

    def set_operation_mode(self, mode):
        if mode in ("Automatic", "Manual"):
            self.operation_mode = mode
            return True
        return False

    def manual_set_green(self, lane_id):
        if self.operation_mode != "Manual" or not (0 <= lane_id < self.num_lanes):
            return False
        self.states = [LightState.RED] * self.num_lanes
        self.states[lane_id] = LightState.GREEN
        self.active_lane      = lane_id
        self.active_since     = time.time()
        self.yellow_active    = False
        self.all_red_active   = False
        self._cancel_emergency()
        return True

    # ══════════════════════════════════════════════════════════════════════
    # Control loop
    # ══════════════════════════════════════════════════════════════════════

    def _control_loop(self):
        while self.running:
            now = time.time()
            self._update_time_of_day_patterns()

            if self.operation_mode != "Automatic":
                time.sleep(0.1)
                continue

            # Check for new ambulance detections (runs every tick)
            self._check_emergency(now)

            # ── Phase 1: Yellow ───────────────────────────────────────
            if self.yellow_active:
                if (now - self.phase_start_time) >= self.yellow_time:
                    self.states[self.active_lane] = LightState.RED
                    self.yellow_active    = False
                    self.all_red_active   = True
                    self.phase_start_time = now

            # ── Phase 2: All-red clearance ────────────────────────────
            elif self.all_red_active:
                if (now - self.phase_start_time) >= self.all_red_time:
                    self.all_red_active = False
                    self._activate_next_lane(now)

            # ── Phase 3: Green ────────────────────────────────────────
            elif self.states[self.active_lane] == LightState.GREEN:
                if self._should_end_green(now):
                    self.states[self.active_lane] = LightState.YELLOW
                    self.yellow_active    = True
                    self.phase_start_time = now
                    self.assigned_green_times[self.active_lane] = 0.0

            time.sleep(0.1)

    # ══════════════════════════════════════════════════════════════════════
    # Phase helpers
    # ══════════════════════════════════════════════════════════════════════

    def _activate_next_lane(self, now):
        """Turn the next lane green after the all-red clearance."""
        if self.next_active_lane is not None:
            self.active_lane      = self.next_active_lane
            self.next_active_lane = None
        else:
            self.active_lane = self._select_next_lane()

        self.states[self.active_lane] = LightState.GREEN

        # When resuming the pre-emergency lane, continue from where it was interrupted.
        if self._resuming_pre_emg and self.active_lane == self._pre_emg_lane:
            remaining = self._pre_emg_assigned - self._pre_emg_elapsed
            if remaining < self.min_green_time:
                # Give a fresh min_green_time if barely any time remains
                self.active_since = now
                self.assigned_green_times[self.active_lane] = self.min_green_time
            else:
                self.active_since = now - self._pre_emg_elapsed
                self.assigned_green_times[self.active_lane] = self._pre_emg_assigned
            self._resuming_pre_emg = False
            self._pre_emg_lane     = None
            self._pre_emg_elapsed  = 0.0
            self._pre_emg_assigned = 0.0
        else:
            self.active_since = now
            assigned = self._calc_green_time(self.active_lane)
            self.assigned_green_times[self.active_lane] = assigned

        self.phase_start_time = now

        # Mark when emergency lane actually goes green
        if self._emg_active and self.active_lane == self._emg_lane:
            self._emg_green_start = now
            # Emergency gets a fixed 15 s slot
            self.assigned_green_times[self.active_lane] = self.EMERGENCY_GREEN_MAX

    def _should_end_green(self, now):
        """Decide whether the current green phase should end."""
        duration = now - self.active_since

        # ── Emergency lane: fixed 15 s, then resume interrupted lane ──────
        if self._emg_active and self.active_lane == self._emg_lane:
            if (self._emg_green_start is not None and
                    now - self._emg_green_start >= self.EMERGENCY_GREEN_MAX):
                self._finish_emergency()
                return True
            return False

        # ── Normal lane: respect assigned time and priority logic ────────
        assigned = self.assigned_green_times[self.active_lane]
        if assigned > 0 and duration >= assigned:
            return True
        if duration < self.min_green_time:
            return False
        if duration >= self.max_green_time:
            return True

        current_p   = self.lane_priorities[self.active_lane]
        max_p       = max(self.lane_priorities)
        max_p_lane  = self.lane_priorities.index(max_p)
        max_wait    = max(self.lane_wait_times)
        max_w_lane  = self.lane_wait_times.index(max_wait)

        return (
            (max_p > current_p * 2 and max_p_lane != self.active_lane) or
            (max_wait > self.max_wait_threshold and max_w_lane != self.active_lane)
        )

    def _select_next_lane(self):
        best_lane  = (self.active_lane + 1) % self.num_lanes
        best_score = -1.0
        for lane in range(self.num_lanes):
            if lane == self.active_lane:
                continue
            wait_factor    = min(self.lane_wait_times[lane] / 30.0, 3.0)
            combined_score = self.lane_priorities[lane] * (1.0 + wait_factor)
            if combined_score > best_score:
                best_score = combined_score
                best_lane  = lane
        return best_lane

    def _calc_green_time(self, lane_id):
        """Log-scaled green time based on current lane priority."""
        priority = self.lane_priorities[lane_id]
        extra    = min(
            self.max_green_time - self.min_green_time,
            5 * math.log2(priority + 1),
        )
        return self.min_green_time + extra

    # ══════════════════════════════════════════════════════════════════════
    # Emergency logic
    # ══════════════════════════════════════════════════════════════════════

    def _check_emergency(self, now):
        """Called every control-loop tick to react to ambulance detections."""
        # Find the first lane with an ambulance that isn't already handled
        emg_lane = next(
            (i for i in range(self.num_lanes) if self._lane_emg_counts[i] > 0),
            None,
        )

        if emg_lane is None:
            # No ambulance anywhere — nothing to trigger
            return

        if emg_lane == self.active_lane and self.states[self.active_lane] == LightState.GREEN:
            # Ambulance is already in the green lane — nothing to do
            return

        if self._emg_active:
            # Already handling an emergency (possibly a different lane)
            return

        # ── New emergency detected ────────────────────────────────────────
        self._emg_active = True
        self._emg_lane   = emg_lane

        if self.states[self.active_lane] == LightState.GREEN and not self.yellow_active:
            # Record the interrupted lane so we can resume it later
            self._pre_emg_lane     = self.active_lane
            self._pre_emg_elapsed  = now - self.active_since
            self._pre_emg_assigned = self.assigned_green_times[self.active_lane]

            # Immediately start yellow on the current green lane
            self.states[self.active_lane] = LightState.YELLOW
            self.yellow_active    = True
            self.phase_start_time = now
            self.assigned_green_times[self.active_lane] = 0.0

        # After yellow + all-red, _activate_next_lane will see this
        self.next_active_lane = emg_lane
        print(f"[Emergency] Ambulance in Lane {emg_lane + 1} — switching in {self.yellow_time}s")

    def _finish_emergency(self):
        """Called when the 15 s emergency green phase expires. Always resumes the interrupted lane."""
        print(f"[Emergency] Lane {self._emg_lane + 1} 15s expired — resuming interrupted lane")
        if self._pre_emg_lane is not None:
            self.next_active_lane  = self._pre_emg_lane
            self._resuming_pre_emg = True
        self._emg_active      = False
        self._emg_green_start = None

    def _cancel_emergency(self):
        """Reset all emergency state (used by manual mode override)."""
        self._emg_active      = False
        self._emg_lane        = None
        self._emg_green_start = None
        self._clear_pre_emg_state()
        self.next_active_lane  = None

    def _clear_pre_emg_state(self):
        self._pre_emg_lane     = None
        self._pre_emg_elapsed  = 0.0
        self._pre_emg_assigned = 0.0
        self._resuming_pre_emg = False

    # ══════════════════════════════════════════════════════════════════════
    # Time-of-day
    # ══════════════════════════════════════════════════════════════════════

    def _update_time_of_day_patterns(self):
        hour = time.localtime().tm_hour
        self.time_of_day_patterns["morning_rush"] = (7  <= hour <  9)
        self.time_of_day_patterns["midday"]        = (10 <= hour < 15)
        self.time_of_day_patterns["evening_rush"]  = (16 <= hour < 19)
        self.time_of_day_patterns["night"]         = (hour >= 20 or hour < 6)

        if self.time_of_day_patterns["morning_rush"] or self.time_of_day_patterns["evening_rush"]:
            self.min_green_time = 15
            self.max_green_time = 90
        elif self.time_of_day_patterns["night"]:
            self.min_green_time = 5
            self.max_green_time = 30
        else:
            self.min_green_time = 10
            self.max_green_time = 60
