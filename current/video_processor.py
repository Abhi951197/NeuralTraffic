import cv2
import time
import threading
import numpy as np
from queue import Queue
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
import os

# Resolve best.pt relative to this file's parent directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YOLO_MODEL_PATH = os.path.join(_BASE_DIR, "best.pt")


class VideoProcessor:
    """
    Per-lane video processor.
    Primary detection: best.pt (YOLO) for all vehicle classes.
    Ambulance detection: Roboflow model, runs in parallel at lower frequency.
    If Roboflow detects an ambulance, that count overrides the emergency field —
    all other vehicle counts always come from best.pt.
    """

    # best.pt class index → vehicle type (person=6, rider=8 are intentionally skipped)
    YOLO_CLASS_MAP = {
        0: "car",
        1: "truck",
        2: "bus",
        3: "motorcycle",
        4: "autorickshaw",
        5: "bicycle",
        7: "vehicle_fallback",
    }
    SKIP_CLASSES = {6, 8}  # person, rider

    BOX_COLORS = {
        "car":              (0, 255, 0),
        "truck":            (255, 165, 0),
        "bus":              (255, 0, 255),
        "motorcycle":       (0, 255, 255),
        "autorickshaw":     (255, 255, 0),
        "bicycle":          (0, 128, 255),
        "vehicle_fallback": (128, 128, 128),
        "emergency":        (0, 0, 255),   # bright red for ambulance
    }

    def __init__(self, source_id, lane_id, api_key="rNZWSRJ2xVRaq8AWduv2"):
        self.source_id = source_id
        self.lane_id = lane_id
        self.cap = None
        self.running = False

        # Frame queues
        self.frame_queue = Queue(maxsize=10)
        self.ambulance_frame_queue = Queue(maxsize=3)

        # Latest frame for display
        self.current_frame = None

        # Detection results (protected by lock)
        self._lock = threading.Lock()
        self._yolo_detections = []        # list of detection dicts from best.pt
        self._ambulance_detections = []   # list of detection dicts from Roboflow

        # Timing
        self.detection_interval = 1.0         # YOLO: once per second
        self.ambulance_check_interval = 2.0   # Roboflow: once every 2 seconds
        self.last_detection_time = 0.0
        self.last_ambulance_check = 0.0

        # Vehicle counts (always kept in sync with detections)
        self.vehicle_counts = {
            "car": 0, "truck": 0, "bus": 0, "motorcycle": 0,
            "autorickshaw": 0, "bicycle": 0, "vehicle_fallback": 0,
            "emergency": 0,
        }
        self.total_vehicles = 0

        # Wait time tracking
        self.last_empty_time = time.time()
        self.current_wait_time = 0.0

        # Load YOLO model once per lane
        print(f"[Lane {lane_id}] Loading YOLO model from {_YOLO_MODEL_PATH}")
        self.yolo_model = YOLO(_YOLO_MODEL_PATH)
        self.confidence_threshold = 0.5

        # Roboflow client (ambulance only)
        self.roboflow_client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )

        # Thread handles
        self.processor_thread = None
        self.detector_thread = None
        self.ambulance_thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Open the video source and start processing threads."""
        if self.running:
            return True

        self.cap = cv2.VideoCapture(self.source_id)
        if not self.cap.isOpened():
            print(f"[Lane {self.lane_id}] Error: cannot open source {self.source_id}")
            return False

        self.running = True
        self.processor_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.detector_thread   = threading.Thread(target=self._detect_vehicles, daemon=True)
        self.ambulance_thread  = threading.Thread(target=self._detect_ambulance, daemon=True)

        self.processor_thread.start()
        self.detector_thread.start()
        self.ambulance_thread.start()
        return True

    def stop(self):
        """Stop all threads and release the video capture."""
        self.running = False
        for t in (self.processor_thread, self.detector_thread, self.ambulance_thread):
            if t:
                t.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    @staticmethod
    def _iou(a, b):
        """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    def get_annotated_frame(self):
        """Return the current frame annotated with all bounding boxes."""
        if self.current_frame is None:
            return None

        annotated = self.current_frame.copy()

        with self._lock:
            yolo_dets      = list(self._yolo_detections)
            ambulance_dets = list(self._ambulance_detections)

        # Suppress YOLO boxes that overlap with a Roboflow ambulance box (IoU > 0.3)
        # so only the ambulance label shows in that region.
        amb_boxes = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in ambulance_dets]
        filtered_yolo = [
            d for d in yolo_dets
            if not any(self._iou((d["x1"], d["y1"], d["x2"], d["y2"]), ab) > 0.3
                       for ab in amb_boxes)
        ]

        for det in filtered_yolo + ambulance_dets:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            color = self.BOX_COLORS.get(det["class"], (200, 200, 200))
            label = f"{det['class']}: {det['confidence']:.2f}"
            if det.get("source") == "roboflow":
                label = f"[AMB] {label}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return annotated

    def get_vehicle_counts(self):
        with self._lock:
            return dict(self.vehicle_counts)

    def get_wait_time(self):
        # Compute live so the GUI ticks every frame, not just on YOLO runs
        if self.total_vehicles > 0:
            return time.time() - self.last_empty_time
        return 0.0

    # ------------------------------------------------------------------
    # Internal threads
    # ------------------------------------------------------------------

    def _process_frames(self):
        """Read frames from the video source and keep queues fresh."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self.current_frame = frame.copy()

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            time.sleep(0.01)

    def _detect_vehicles(self):
        """Primary vehicle detection using best.pt YOLO model."""
        while self.running:
            now = time.time()
            if self.frame_queue.empty() or (now - self.last_detection_time) < self.detection_interval:
                time.sleep(0.05)
                continue

            frame = self.frame_queue.get()
            self.last_detection_time = time.time()

            try:
                results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)
                detections = []

                for box in results[0].boxes.cpu().numpy():
                    cls = int(box.cls[0])
                    if cls in self.SKIP_CLASSES or cls not in self.YOLO_CLASS_MAP:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "class":      self.YOLO_CLASS_MAP[cls],
                        "confidence": float(box.conf[0]),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "source":     "yolo",
                    })

                # Also push a frame to the ambulance checker
                if not self.ambulance_frame_queue.full():
                    self.ambulance_frame_queue.put(frame)

                with self._lock:
                    self._yolo_detections = detections
                    self._recompute_counts()

            except Exception as e:
                print(f"[Lane {self.lane_id}] YOLO error: {e}")

    def _detect_ambulance(self):
        """Ambulance-only detection using Roboflow (lower frequency)."""
        while self.running:
            now = time.time()
            if self.ambulance_frame_queue.empty() or (now - self.last_ambulance_check) < self.ambulance_check_interval:
                time.sleep(0.1)
                continue

            frame = self.ambulance_frame_queue.get()
            self.last_ambulance_check = time.time()

            try:
                _, buffer = cv2.imencode(".jpg", frame)
                pil_image = Image.open(io.BytesIO(buffer.tobytes()))
                response = self.roboflow_client.infer(pil_image, model_id="vehicle-detection-eckrb/4")

                all_preds = response.get("predictions", [])
                print(f"[Lane {self.lane_id}][Roboflow] {len(all_preds)} prediction(s):", flush=True)
                for p in all_preds:
                    print(f"  class={p['class']!r:20s}  conf={p['confidence']:.3f}", flush=True)
                if not all_preds:
                    print(f"  (no detections)", flush=True)

                ambulance_dets = []
                for pred in all_preds:
                    if pred["confidence"] < self.confidence_threshold:
                        continue
                    cls_name = pred["class"].lower().replace(" ", "_")
                    if cls_name not in {"ambulance", "police", "fire_truck", "firetruck"}:
                        continue

                    cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                    print(f"[Lane {self.lane_id}][Roboflow] *** AMBULANCE ACCEPTED *** class={pred['class']!r}  conf={pred['confidence']:.3f}", flush=True)
                    ambulance_dets.append({
                        "class":      "emergency",
                        "confidence": pred["confidence"],
                        "x1": int(cx - w / 2), "y1": int(cy - h / 2),
                        "x2": int(cx + w / 2), "y2": int(cy + h / 2),
                        "source":     "roboflow",
                    })

                with self._lock:
                    self._ambulance_detections = ambulance_dets
                    self._recompute_counts()

            except Exception as e:
                print(f"[Lane {self.lane_id}] Roboflow error: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _recompute_counts(self):
        """
        Rebuild vehicle_counts from current YOLO + Roboflow results.
        Must be called while holding self._lock.
        """
        counts = {k: 0 for k in self.vehicle_counts}

        for det in self._yolo_detections:
            vtype = det["class"]
            if vtype in counts:
                counts[vtype] += 1

        # Roboflow only contributes to the emergency count
        counts["emergency"] = len(self._ambulance_detections)

        self.vehicle_counts = counts
        self.total_vehicles = sum(counts.values())

        if self.total_vehicles == 0:
            self.last_empty_time = time.time()
        else:
            self.current_wait_time = time.time() - self.last_empty_time
