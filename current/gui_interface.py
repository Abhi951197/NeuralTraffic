import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import time
import traceback
import numpy as np

# Fixed video render size — every frame is resized to exactly this before
# being shown.  Chosen so 2×2 grid fits comfortably in an 1280×800 window:
#   2 rows × (220 video + ~115 card chrome) ≈ 670 px  ✓
#   2 cols × (440 video + ~20 padding)      ≈ 920 px  ✓
VIDEO_W = 440
VIDEO_H = 180


class TrafficSystemGUI:
    def __init__(self, video_processors, traffic_controller):
        self.video_processors   = video_processors
        self.traffic_controller = traffic_controller
        self.num_lanes          = len(video_processors)

        self.root = tk.Tk()
        self.root.title("NeuralTraffic")
        self.root.geometry("1280x800")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")

        self._setup_gui()

        self.running = False

        # FPS tracking
        self._last_gui_time = time.time()
        self._fps_samples   = []

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _setup_gui(self):
        # ── Header ──────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#2c3e50", height=60)
        header.pack(fill=tk.X, padx=10, pady=(10, 5))
        header.pack_propagate(False)

        tk.Label(header, text="NEURALTRAFFIC",
                 font=("Arial", 18, "bold"), bg="#2c3e50", fg="white"
                 ).pack(side=tk.LEFT, padx=10, pady=10)

        self.status_label = tk.Label(header, text="Status: Ready",
                                     font=("Arial", 12), bg="#2c3e50", fg="#2ecc71")
        self.status_label.pack(side=tk.RIGHT, padx=10, pady=10)

        # ── Footer (controls) — packed BEFORE main_frame so it's never pushed out ──
        footer = tk.Frame(self.root, bg="#2c3e50", height=45)
        footer.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))
        footer.pack_propagate(False)

        self.mode_var = tk.StringVar(value="Automatic")
        tk.Label(footer, text="Mode:", font=("Arial", 10),
                 bg="#2c3e50", fg="white").pack(side=tk.LEFT, padx=5, pady=5)

        for mode in ("Automatic", "Manual"):
            tk.Radiobutton(footer, text=mode, variable=self.mode_var, value=mode,
                           command=self._mode_changed, bg="#2c3e50", fg="white",
                           selectcolor="#2c3e50", activebackground="#2c3e50",
                           activeforeground="white",
                           ).pack(side=tk.LEFT, padx=5, pady=5)

        self.manual_buttons = []
        for i in range(self.num_lanes):
            btn = tk.Button(footer, text=f"Lane {i + 1} Green",
                            command=lambda lane=i: self._manual_set_green(lane),
                            state=tk.DISABLED)
            btn.pack(side=tk.LEFT, padx=5, pady=5)
            self.manual_buttons.append(btn)

        self.fps_label = tk.Label(footer, text="FPS: 0",
                                  font=("Arial", 10), bg="#2c3e50", fg="white")
        self.fps_label.pack(side=tk.RIGHT, padx=5, pady=5)

        self.start_button = tk.Button(footer, text="Start System",
                                      command=self._toggle_system,
                                      bg="#27ae60", fg="white",
                                      font=("Arial", 10, "bold"))
        self.start_button.pack(side=tk.RIGHT, padx=15, pady=5)

        # ── Main area (video grid) — fills remaining space ───────────────
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Build a shared black placeholder so every video label starts at the
        # correct pixel size before the first real frame arrives.
        _blank_arr   = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        self._blank  = ImageTk.PhotoImage(Image.fromarray(_blank_arr))

        self.video_labels      = []
        self.status_indicators = []
        self.count_labels      = []
        self.weight_labels     = []   # numeric accumulated weight score
        self.priority_bars     = []
        self.wait_time_labels  = []
        self.green_time_labels = []

        rows, cols = 2, 2
        for i in range(self.num_lanes):
            row, col = divmod(i, cols)

            lane_frame = tk.Frame(self.main_frame, bg="white", bd=2, relief=tk.RAISED)
            lane_frame.grid(row=row, column=col, padx=5, pady=5)
            # No sticky="nsew" — let the card be its natural size so it never
            # expands beyond what the video + chrome require.

            # Lane header bar
            lane_header = tk.Frame(lane_frame, bg="#34495e")
            lane_header.pack(fill=tk.X)

            tk.Label(lane_header, text=f"Lane {i + 1}",
                     font=("Arial", 11, "bold"), bg="#34495e", fg="white"
                     ).pack(side=tk.LEFT, padx=8, pady=4)

            light_canvas = tk.Canvas(lane_header, width=20, height=20,
                                     bg="#34495e", highlightthickness=0)
            light_canvas.pack(side=tk.RIGHT, padx=8, pady=4)
            light_canvas.create_oval(2, 2, 18, 18, fill="red", tags="light")

            # Video label — initialised with a black placeholder image so tkinter
            # always treats width/height as pixels, keeping the card stable.
            video_label = tk.Label(lane_frame, bg="black",
                                   image=self._blank)
            video_label.image = self._blank   # prevent GC
            video_label.pack()

            # Card footer
            lane_footer = tk.Frame(lane_frame, bg="#ecf0f1")
            lane_footer.pack(fill=tk.X)

            count_label = tk.Label(
                lane_footer,
                text="Car:0  Truck:0  Bus:0  MC:0  Auto:0  Emg:0",
                font=("Arial", 9), bg="#ecf0f1", anchor="w",
            )
            count_label.pack(side=tk.TOP, padx=6, pady=(4, 0), fill=tk.X)

            stats_row = tk.Frame(lane_footer, bg="#ecf0f1")
            stats_row.pack(fill=tk.X, padx=6, pady=2)

            # Numeric weight score (accumulated vehicle-type weights)
            tk.Label(stats_row, text="Wt:", font=("Arial", 9),
                     bg="#ecf0f1").pack(side=tk.LEFT, padx=(0, 2))
            weight_label = tk.Label(stats_row, text="0.0",
                                    font=("Arial", 9, "bold"),
                                    bg="#ecf0f1", fg="#2980b9", width=5, anchor="w")
            weight_label.pack(side=tk.LEFT, padx=(0, 4))

            # Relative bar — shows this lane vs the current max
            priority_bar = ttk.Progressbar(stats_row, orient=tk.HORIZONTAL,
                                           length=90, mode="determinate")
            priority_bar.pack(side=tk.LEFT, padx=2)

            # Wait time — live, ticks every GUI frame
            wait_label = tk.Label(stats_row, text="Wait: 0s",
                                  font=("Arial", 9), bg="#ecf0f1")
            wait_label.pack(side=tk.RIGHT, padx=4)

            # Green time row
            green_row = tk.Frame(lane_footer, bg="#ecf0f1")
            green_row.pack(fill=tk.X, padx=6, pady=(0, 4))

            green_time_label = tk.Label(
                green_row,
                text="Green: --",
                font=("Arial", 9, "bold"), bg="#ecf0f1", fg="#27ae60", anchor="w",
            )
            green_time_label.pack(side=tk.LEFT)

            self.video_labels.append(video_label)
            self.status_indicators.append(light_canvas)
            self.count_labels.append(count_label)
            self.weight_labels.append(weight_label)
            self.priority_bars.append(priority_bar)
            self.wait_time_labels.append(wait_label)
            self.green_time_labels.append(green_time_label)

        # Centre the 2×2 grid inside main_frame
        for r in range(rows):
            self.main_frame.grid_rowconfigure(r, weight=1)
        for c in range(cols):
            self.main_frame.grid_columnconfigure(c, weight=1)

    # ------------------------------------------------------------------
    # System start / stop
    # ------------------------------------------------------------------

    def _toggle_system(self):
        if not self.running:
            ok = all(p.start() for p in self.video_processors)
            ok = ok and self.traffic_controller.start()
            if ok:
                self.running = True
                self.start_button.config(text="Stop System", bg="#c0392b")
                self.status_label.config(text="Status: Running", fg="#2ecc71")
                self.root.after(33, self._scheduled_update)
        else:
            self.running = False
            for p in self.video_processors:
                p.stop()
            self.traffic_controller.stop()
            self.start_button.config(text="Start System", bg="#27ae60")
            self.status_label.config(text="Status: Stopped", fg="#e74c3c")

    # ------------------------------------------------------------------
    # GUI update — main thread only via root.after()
    # ------------------------------------------------------------------

    def _scheduled_update(self):
        if not self.running:
            return
        try:
            self._update_gui()
        except Exception:
            print("[GUI ERROR] Exception in _update_gui — full traceback:")
            traceback.print_exc()
        self.root.after(33, self._scheduled_update)

    def _update_gui(self):
        # FPS
        now = time.time()
        elapsed = now - self._last_gui_time
        self._last_gui_time = now
        self._fps_samples.append(elapsed)
        if len(self._fps_samples) > 30:
            self._fps_samples.pop(0)
        avg = sum(self._fps_samples) / len(self._fps_samples) if self._fps_samples else 1
        self.fps_label.config(text=f"FPS: {1.0 / avg:.1f}")

        light_states    = self.traffic_controller.get_light_states()
        priorities      = self.traffic_controller.get_priorities()
        wait_times      = self.traffic_controller.get_wait_times()
        assigned_greens = self.traffic_controller.get_assigned_green_times()
        green_elapsed   = self.traffic_controller.get_green_elapsed()
        active_lane     = self.traffic_controller.get_active_lane()
        max_priority    = max(priorities) if any(p > 0 for p in priorities) else 1.0

        for i, processor in enumerate(self.video_processors):
            # Always render at exactly VIDEO_W × VIDEO_H — the label size
            # never changes, so no card ever jumps or reflows.
            frame = processor.get_annotated_frame()
            if frame is not None:
                fh, fw = frame.shape[:2]
                scale  = min(VIDEO_W / fw, VIDEO_H / fh)
                new_w, new_h = int(fw * scale), int(fh * scale)
                frame  = cv2.resize(frame, (new_w, new_h))

                # Letterbox onto a black canvas of exactly VIDEO_W × VIDEO_H
                canvas = cv2.copyMakeBorder(
                    frame,
                    (VIDEO_H - new_h) // 2,
                    VIDEO_H - new_h - (VIDEO_H - new_h) // 2,
                    (VIDEO_W - new_w) // 2,
                    VIDEO_W - new_w - (VIDEO_W - new_w) // 2,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0),
                )
                img_tk = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                )
                self.video_labels[i].config(image=img_tk)
                self.video_labels[i].image = img_tk  # prevent GC

            # Traffic light
            color = {"RED": "red", "YELLOW": "yellow", "GREEN": "green"}.get(
                light_states[i].name, "red"
            )
            self.status_indicators[i].itemconfig("light", fill=color)

            # Vehicle counts
            counts = processor.get_vehicle_counts()
            self.count_labels[i].config(
                text=(f"Car:{counts['car']}  Truck:{counts['truck']}  "
                      f"Bus:{counts['bus']}  MC:{counts['motorcycle']}  "
                      f"Auto:{counts['autorickshaw']}  Emg:{counts['emergency']}")
            )

            # Numeric accumulated weight + relative bar
            self.weight_labels[i].config(text=f"{priorities[i]:.1f}")
            pct = (priorities[i] / max_priority) * 100
            self.priority_bars[i].config(value=pct)

            # Wait time — live value straight from the processor
            live_wait = processor.get_wait_time()
            self.wait_time_labels[i].config(text=f"Wait: {int(live_wait)}s")

            # Green time — show elapsed / assigned for the active lane only
            if i == active_lane and assigned_greens[i] > 0:
                elapsed  = min(green_elapsed, assigned_greens[i])
                assigned = assigned_greens[i]
                self.green_time_labels[i].config(
                    text=f"Green: {elapsed:.0f}s / {assigned:.0f}s",
                    fg="#27ae60",
                )
            else:
                self.green_time_labels[i].config(text="Green: --", fg="#95a5a6")

            # Push data back to controller
            self.traffic_controller.update_vehicle_counts(i, counts)
            self.traffic_controller.update_wait_time(i, processor.get_wait_time())

    # ------------------------------------------------------------------
    # Mode / manual controls
    # ------------------------------------------------------------------

    def _mode_changed(self):
        mode = self.mode_var.get()
        self.traffic_controller.set_operation_mode(mode)
        state = tk.NORMAL if mode == "Manual" else tk.DISABLED
        for btn in self.manual_buttons:
            btn.config(state=state)

    def _manual_set_green(self, lane):
        self.traffic_controller.manual_set_green(lane)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def start(self):
        self.root.mainloop()
