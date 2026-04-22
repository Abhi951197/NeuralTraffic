import os
import sys
sys.dont_write_bytecode = True  # never create __pycache__ files
import argparse

# Ensure imports resolve relative to this file's directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_processor import VideoProcessor
from traffic_controller import TrafficController
from gui_interface import TrafficSystemGUI
from analytics_engine import AnalyticsEngine
from config import ROBOFLOW_API_KEY, VIDEO_SOURCES, NUM_LANES


def main():
    parser = argparse.ArgumentParser(description="NeuralTraffic - Smart Traffic Management System")
    parser.add_argument("--video_sources", nargs="+", default=VIDEO_SOURCES)
    parser.add_argument("--api_key",       default=ROBOFLOW_API_KEY)
    parser.add_argument("--num_lanes",     type=int, default=NUM_LANES)
    args = parser.parse_args()

    # Pad / trim sources to match num_lanes
    sources = args.video_sources
    if len(sources) < args.num_lanes:
        sources = (sources * ((args.num_lanes // len(sources)) + 1))[:args.num_lanes]
    sources = sources[:args.num_lanes]

    # Convert numeric strings to int (camera index)
    for i, src in enumerate(sources):
        try:
            sources[i] = int(src)
        except (ValueError, TypeError):
            if isinstance(src, str):
                sources[i] = src.strip(",")

    print("Initialising Smart Traffic Management System...")
    print(f"  Lanes        : {args.num_lanes}")
    print(f"  Video sources: {sources}")

    video_processors = [
        VideoProcessor(sources[i], i, api_key=args.api_key)
        for i in range(args.num_lanes)
    ]

    traffic_controller = TrafficController(num_lanes=args.num_lanes)
    analytics_engine   = AnalyticsEngine(traffic_controller)
    gui                = TrafficSystemGUI(video_processors, traffic_controller)

    analytics_engine.start()

    print("System ready — starting GUI.")
    gui.start()  # blocks until window is closed

    # ── Cleanup ──────────────────────────────────────────────────────────────
    print("Shutting down...")
    for p in video_processors:
        p.stop()
    traffic_controller.stop()
    analytics_engine.stop()

    report_path = analytics_engine.generate_report()
    print(f"Traffic report saved to: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
