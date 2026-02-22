import pickle
import cv2
import numpy as np
import os

video_path = r"c:\Users\Asus\football_analysis\input_videos\My Video.mp4"

print(f"Checking video path: {video_path}")
if not os.path.exists(video_path):
    print("Video file does not exist!")
    # Try alternate path just in case
    video_path = os.path.abspath("input_videos/My Video.mp4")
    print(f"Trying alternate path: {video_path}")
    if not os.path.exists(video_path):
        print("Still does not exist.")
        exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read frame 0")
else:
    print(f"Read frame 0 successfully. Shape: {frame.shape}")
