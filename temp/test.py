import cv2
import os
import random
from inference_sdk import InferenceHTTPClient
from PIL import Image

# === Roboflow Client Setup ===
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="rNZWSRJ2xVRaq8AWduv2"
)
MODEL_ID = "vehicle-detection-eckrb/4"

# === Function to extract one random frame every 5 seconds ===
def extract_random_frame_every_5s(video_path, output_dir, default_fps=30):
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file doesn't exist at path: {video_path}")
        return False
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Check if FPS is valid, use default if not
    if fps <= 0:
        print(f"[WARNING] Invalid FPS detected ({fps}), using default value: {default_fps}")
        fps = default_fps
    
    # Calculate duration and number of chunks
    duration = total_frames / fps
    chunks = max(1, int(duration / 5))
    
    print(f"[INFO] Video properties: {total_frames} frames, {fps} FPS")
    print(f"[INFO] Total video duration: {duration:.2f} seconds, extracting {chunks} frames...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract frames
    frames_saved = 0
    for i in range(chunks):
        start_frame = int(i * 5 * fps)
        end_frame = int((i + 1) * 5 * fps) - 1
        
        if end_frame >= total_frames:
            end_frame = total_frames - 1
        
        if start_frame >= total_frames:
            break
            
        # Choose a random frame in this chunk
        random_frame = random.randint(start_frame, end_frame)
        
        # Set frame position and read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        
        if ret:
            frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_saved += 1
            print(f"[INFO] Saved frame {random_frame} at {frame_path}")
        else:
            print(f"[WARNING] Failed to read frame {random_frame} at chunk {i}")
    
    cap.release()
    print(f"[INFO] Extraction complete. Saved {frames_saved} frames.")
    return True

# === Function to run Roboflow prediction on all frames ===
def predict_on_frames(frame_dir):
    if not os.path.exists(frame_dir):
        print(f"[ERROR] Frame directory doesn't exist: {frame_dir}")
        return
    
    frames = [f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not frames:
        print(f"[ERROR] No image frames found in directory: {frame_dir}")
        return
    
    print(f"[INFO] Running predictions on {len(frames)} frames...")
    
    for frame_name in frames:
        frame_path = os.path.join(frame_dir, frame_name)
        try:
            image = Image.open(frame_path)
            results = client.infer(image, model_id=MODEL_ID)
            
            print(f"\nPredictions for {frame_name}:")
            if "predictions" in results and results["predictions"]:
                for prediction in results["predictions"]:
                    print(f"  - Class: {prediction['class']}, Confidence: {prediction['confidence']:.2f}")
            else:
                print("  - No objects detected")
                
        except Exception as e:
            print(f"[ERROR] Failed to process {frame_name}: {str(e)}")

# === Main Execution ===
if __name__ == "__main__":
    video_path = "static/videos/WhatsApp Video 2025-04-24 at 8.27.31 PM.mp4"  # <-- Replace with your video path
    frame_output_dir = "extracted_frames"
    
    # Extract frames with a default FPS if the video metadata is unavailable
    success = extract_random_frame_every_5s(video_path, frame_output_dir, default_fps=30)
    
    if success:
        predict_on_frames(frame_output_dir)
    else:
        print("[ERROR] Frame extraction failed. Cannot proceed with predictions.")