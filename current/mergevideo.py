from moviepy import VideoFileClip, concatenate_videoclips

# Load videos
clip1 = VideoFileClip("static\\videos\\amb.mp4")
clip2 = VideoFileClip("static\\videos\\traffic-congestion-and-street-life-in-the-city-of-jaipur-pink-gate-city-walls--SBV-300214180-preview.mp4")

# Merge videos
final_clip = concatenate_videoclips([clip1, clip2])

# Export result
final_clip.write_videofile("static\\videos\\output.mp4")