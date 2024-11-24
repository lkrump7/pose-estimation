# Updating the provided script with the suggested fixes and implementing the changes.

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import os
import signal

# Function to gracefully close OpenCV windows on interrupt
def cleanup(signum, frame):
    cv2.destroyAllWindows()
    exit()

signal.signal(signal.SIGINT, cleanup)

# Check for file existence
if not os.path.exists('pose1.png'):
    print("Error: Reference pose image 'pose1.png' not found.")
    exit()

if not os.path.exists('user_video.mp4'):
    print("Error: Video file 'user_video.mp4' not found.")
    exit()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Function to extract pose landmarks from an image/frame
def extract_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    if result.pose_landmarks:
        # Normalize landmarks by image dimensions
        height, width, _ = image.shape
        landmarks = np.array([[lm.x * width, lm.y * height] for lm in result.pose_landmarks.landmark])
        return landmarks
    else:
        return None

# Function to calculate similarity score between two sets of landmarks
def calculate_similarity(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return float('inf')  # If any landmark is missing, return infinite distance
    return np.linalg.norm(landmarks1 - landmarks2)

# Load reference pose image
ref_pose_image = cv2.imread('pose1.png')
if ref_pose_image is None:
    print("Error: Unable to read 'pose1.png'.")
    exit()

# Extract reference pose landmarks
ref_landmarks = extract_pose_landmarks(ref_pose_image)
if ref_landmarks is None:
    print("Error: Could not extract landmarks from the reference image.")
    exit()

# Open the video
cap = cv2.VideoCapture('user_video.mp4')
if not cap.isOpened():
    print("Error: Unable to open 'user_video.mp4'.")
    exit()

best_frame = None
best_similarity = float('inf')
best_match_frame = 0
current_frame = 0
frame_skip = 5  # Process every 5th frame for efficiency

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if current_frame % frame_skip != 0:
        current_frame += 1
        continue

    # Extract landmarks from the current frame
    user_landmarks = extract_pose_landmarks(frame)

    # Calculate similarity score
    similarity = calculate_similarity(ref_landmarks, user_landmarks)

    # If this frame has the best match, save it
    if similarity < best_similarity:
        best_similarity = similarity
        best_frame = frame
        best_match_frame = current_frame

    print(f"Processing frame {current_frame}, Similarity: {similarity:.2f}")
    current_frame += 1

cap.release()

# Convert similarity to a percentage match (assuming similarity=0 is perfect match)
max_distance = np.max([np.linalg.norm(ref_landmarks)])  # Dynamic max distance based on reference
match_percentage = max(0, 100 - (best_similarity / max_distance * 100))

# Display the best matching frame
if best_frame is not None:
    cv2.imshow(f'Best Matching Frame (Match: {match_percentage:.2f}%)', best_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matching frame found.")

# Output the best match and the match percentage
print(f'Best matching frame: {best_match_frame}, Match Percentage: {match_percentage:.2f}%')
