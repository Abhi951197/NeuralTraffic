import os
import argparse
from video_processor import VideoProcessor
from traffic_controller import TrafficController, LightState
from gui_interface import TrafficSystemGUI
from analytics_engine import AnalyticsEngine
from config import ROBOFLOW_API_KEY, VIDEO_SOURCES, NUM_LANES

def main():
    """Main entry point for the Smart Traffic Management System"""
    parser = argparse.ArgumentParser(description='Smart Traffic Management System')
    # Optional: Allow overrides via CLI
    parser.add_argument('--video_sources', type=str, nargs='+', default=VIDEO_SOURCES,
                        help='Video sources for each lane (can be camera ids or video files)')
    parser.add_argument('--api_key', type=str, default=ROBOFLOW_API_KEY,
                        help='Roboflow API key')
    parser.add_argument('--num_lanes', type=int, default=NUM_LANES,
                        help='Number of traffic lanes to monitor')
    args = parser.parse_args()
    
    # Ensure correct number of video sources
    if len(args.video_sources) < args.num_lanes:
        print(f"Warning: Only {len(args.video_sources)} video sources provided for {args.num_lanes} lanes.")
        print("Using available sources in rotation to fill all lanes.")
        args.video_sources = args.video_sources * ((args.num_lanes // len(args.video_sources)) + 1)
        args.video_sources = args.video_sources[:args.num_lanes]
    
    # Convert camera indexes (e.g., "0") to int, keep strings for file paths
    for i, source in enumerate(args.video_sources):
        try:
            args.video_sources[i] = int(source)
        except ValueError:
            args.video_sources[i] = source.strip(',')  # Clean trailing commas if any
    
    print("Initializing Smart Traffic Management System...")
    print(f"Number of lanes: {args.num_lanes}")
    print(f"Video sources: {args.video_sources}")
    
    # Initialize components
    print("Creating video processors...")
    video_processors = [
        VideoProcessor(args.video_sources[i], i, api_key=args.api_key)
        for i in range(args.num_lanes)
    ]
    
    print("Creating traffic controller...")
    traffic_controller = TrafficController(num_lanes=args.num_lanes)
    
    print("Creating analytics engine...")
    analytics_engine = AnalyticsEngine(traffic_controller) 
    
    print("Creating GUI...")
    gui = TrafficSystemGUI(video_processors, traffic_controller)
    
    # Start analytics
    analytics_engine.start()
    
    print("System initialized. Starting GUI...")
    gui.start()
    
    # Shutdown and cleanup
    print("Shutting down...")
    for processor in video_processors:
        processor.stop()
    
    traffic_controller.stop()
    analytics_engine.stop()
    
    print("Generating final traffic report...")
    report_path = analytics_engine.generate_report()
    print(f"Report saved to: {report_path}")
    print("System shutdown complete.")

if __name__ == "__main__":
    main()