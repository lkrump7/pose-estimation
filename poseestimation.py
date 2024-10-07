import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Function to extract pose landmarks from an image/frame
def extract_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        # Extract the landmarks and normalize them by width and height of the image
        landmarks = np.array([[lm.x, lm.y] for lm in result.pose_landmarks.landmark])
        return landmarks
    else:
        return None

# Function to calculate similarity score between two sets of landmarks
def calculate_similarity(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return float('inf')  # If any landmark is missing, return infinite distance

    # Calculate Euclidean distance between corresponding landmarks
    return np.linalg.norm(landmarks1 - landmarks2)

# Load reference pose image
ref_pose_image = cv2.imread('pose1.png')

# Extract reference pose landmarks
ref_landmarks = extract_pose_landmarks(ref_pose_image)

# Open the video
cap = cv2.VideoCapture('user_video.mp4')

best_frame = None
best_similarity = float('inf')
best_match_frame = 0
current_frame = 0

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract landmarks from the current frame
    user_landmarks = extract_pose_landmarks(frame)

    # Calculate similarity score
    similarity = calculate_similarity(ref_landmarks, user_landmarks)

    # If this frame has the best match, save it
    if similarity < best_similarity:
        best_similarity = similarity
        best_frame = frame
        best_match_frame = current_frame

    current_frame += 1

cap.release()

# Convert similarity to a percentage match (assuming similarity=0 is perfect match)
max_distance = 5  # Arbitrary threshold for "maximum distance"
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