import cv2
import mediapipe as mp
import numpy as np
from scipy.stats import linregress
from collections import deque



video = cv2.VideoCapture(1)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



possible_actions = {
    "PUNCH_RIGHT": 0,
    "PUNCH_LEFT": 1,
}

frames_per_action = 5
# Distance history buffers for each hand
hand_histories = {
    "RIGHT": deque(maxlen=frames_per_action),
    "LEFT": deque(maxlen=frames_per_action)
}


def detect_punch(landmarks, hand_side):
    """
    Detects punch motion for LEFT or RIGHT hand using distance change over time.
    Args:
        landmarks: pose landmarks from mediapipe
        hand_side: "RIGHT" or "LEFT"
    """
    wrist_idx = getattr(mp_pose.PoseLandmark, f"{hand_side}_WRIST").value
    shoulder_idx = getattr(mp_pose.PoseLandmark, f"{hand_side}_SHOULDER").value

    wrist = landmarks[wrist_idx]
    shoulder = landmarks[shoulder_idx]

    distance = np.linalg.norm(
        np.array([wrist.x, wrist.y, wrist.z]) -
        np.array([shoulder.x, shoulder.y, shoulder.z])
    )

    # Automatically removes oldest if > maxlen
    hand_histories[hand_side].append(distance)

    # Detect punch when buffer is full
    if len(hand_histories[hand_side]) == frames_per_action:
        x = list(range(frames_per_action))
        y = list(hand_histories[hand_side])
        slope, _, _, _, _ = linregress(x, y)
        print(f"Slope for {hand_side}: {slope}")

        if slope > 0.08:
            print(f"💥 PUNCH {hand_side} detected!")
            return possible_actions[f"PUNCH_{hand_side}"]






#  Get pose estimation model.
with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as pose:




    while video.isOpened():
        rect, frame  = video.read()

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To improve performance, mark the image as not writeable 
        image.flags.writeable = False

        # Detect pose landmarks.
        results = pose.process(image)

        # Make it writeable again and convert it back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks.
        try:
            landmarks = results.pose_landmarks.landmark
            detect_punch(landmarks, "RIGHT")
            detect_punch(landmarks, "LEFT")
        except:
            pass

        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing.DrawingSpec(color=(102, 178, 255), thickness=4, circle_radius=2),  
            connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 140, 0), thickness=4, circle_radius=2) 
        )

        cv2.imshow("Ayo's Video", image)

        # Quit webcam. 
        waitKey = cv2.waitKey(1)
        if waitKey == 27 or waitKey == ord("q"): # ESC key
            break

    video.release()
    cv2.destroyAllWindows()
    