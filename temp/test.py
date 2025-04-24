# from inference_sdk import InferenceHTTPClient
# from PIL import Image

# # Initialize Roboflow client
# client = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key="rNZWSRJ2xVRaq8AWduv2"
# )

# # Image path
# image_path = "ambulance.jpg"  # <- Change this to your image path

# # Open image with PIL
# image = Image.open(image_path)

# # Run inference (update with your model version)
# results = client.infer(image, model_id="vehicle-detection-eckrb/4")

# # Print results
# for prediction in results["predictions"]:
#     print(f"Class: {prediction['class']}, Confidence: {prediction['confidence']:.2f}, "
#           f"X: {prediction['x']}, Y: {prediction['y']}, Width: {prediction['width']}, Height: {prediction['height']}")





#------------------------------------------------------------------------------------------------------


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
def extract_random_frame_every_5s(video_path, output_dir, fps):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    chunks = int(duration / 5)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[INFO] Total video duration: {duration:.2f} seconds, extracting {chunks} frames...")

    for i in range(chunks):
        start_frame = int(i * 5 * fps)
        end_frame = int((i + 1) * 5 * fps)
        if end_frame > total_frames:
            break
        random_frame = random.randint(start_frame, end_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"[INFO] Saved frame at {frame_path}")
        else:
            print(f"[WARNING] Failed to read frame at chunk {i}")
    cap.release()

# === Function to run Roboflow prediction on all frames ===
def predict_on_frames(frame_dir):
    for frame_name in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, frame_name)
        image = Image.open(frame_path)
        results = client.infer(image, model_id=MODEL_ID)
        print(f"\nPredictions for {frame_name}:")
        for prediction in results["predictions"]:
            print(f"  - Class: {prediction['class']}, Confidence: {prediction['confidence']:.2f}")

# === Main Execution ===
if __name__ == "__main__":
    video_path = "static/videos/WhatsApp Video 2025-04-24 at 19.25.42_2b0f7658.mp4"  # <-- Replace with your video path
    frame_output_dir = "extracted_frames"
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    extract_random_frame_every_5s(video_path, frame_output_dir, fps)
    predict_on_frames(frame_output_dir)

