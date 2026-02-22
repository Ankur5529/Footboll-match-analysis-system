import pickle
import cv2
import numpy as np
import os

stub_path = r"c:\Users\Asus\football_analysis\stubs\track_stubs_My Video.pkl"
video_path = r"c:\Users\Asus\football_analysis\input_videos\My Video.mp4"

print(f"Checking stub: {stub_path}")
if not os.path.exists(stub_path):
    print("Stub file does not exist!")
    exit()

with open(stub_path, 'rb') as f:
    tracks = pickle.load(f)

# Inspect frame 0
if len(tracks['players']) == 0:
    print("No players in tracks!")
    exit()
    
frame_0_players = tracks['players'][0]
print(f"Frame 0 has {len(frame_0_players)} players detected.")

# Check video frame 0
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read frame 0")
    exit()

print(f"Frame 0 shape: {frame.shape}")

# Check extracts
for pid, info in list(frame_0_players.items())[:5]: # limit to 5
    bbox = info['bbox']
    print(f"Player {pid}: {bbox}")
    
    # Extract image
    y1, y2 = int(bbox[1]), int(bbox[3])
    x1, x2 = int(bbox[0]), int(bbox[2])
    
    # Check boundaries
    if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
        print(f"  WARNING: Bbox out of bounds!")
    
    image = frame[y1:y2, x1:x2]
    print(f"  Player {pid} image shape: {image.shape}")
    
    if image.size == 0:
        print("  Empty image!")
        continue
        
    # Top half
    top_half = image[0:int(image.shape[0]/2), :]
    print(f"  Player {pid} top half shape: {top_half.shape}")
    
    # Check size valid for kmeans
    image_2d = top_half.reshape(-1, 3)
    print(f"  Player {pid} pixels: {len(image_2d)}")
