import cv2
import mediapipe as mp
import numpy as np
from scipy.stats import linregress
from collections import deque
import time
from pynput.keyboard import Controller


video = cv2.VideoCapture(1)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

possible_actions = {
    "PUNCH_RIGHT": 0,
    "PUNCH_LEFT": 1,
    "LEAN_FWD": 2,
    "LEAN_BACK": 3,
}

# Map each action ID to a keyboard key
ACTION_KEY_MAP = {
    possible_actions["PUNCH_RIGHT"]: "R",  
    possible_actions["PUNCH_LEFT"]: "T",   
    possible_actions["LEAN_FWD"]: "A",     
    possible_actions["LEAN_BACK"]: "D",    
}

frames_per_action = 5         
COOLDOWN = 0.5                


hand_histories = {side: deque(maxlen=frames_per_action) for side in ["RIGHT", "LEFT"]}
lean_hist = deque(maxlen=frames_per_action)
last_triggered = {aid: 0 for aid in possible_actions.values()}

keyboard_controller = Controller()


def trigger_key(action_id: int):
    """Send the mapped key for the action if the cooldown has expired."""
    now = time.time()
    if now - last_triggered[action_id] >= COOLDOWN:
        key = ACTION_KEY_MAP[action_id]
        keyboard_controller.press(key)
        keyboard_controller.release(key)
        last_triggered[action_id] = now
        print(f"[KEY] Sent '{key}' for action {action_id}")


def detect_lean(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    metric = nose.z  # z‑value is depth relative to the camera
    lean_hist.append(metric)

    if len(lean_hist) == frames_per_action:
        x = range(frames_per_action)
        slope, *_ = linregress(x, lean_hist)
        if slope < -0.05:
            return possible_actions["LEAN_FWD"]
        elif slope > 0.05:
            return possible_actions["LEAN_BACK"]
    return None


def detect_punch(landmarks, hand_side: str):
    wrist_idx = getattr(mp_pose.PoseLandmark, f"{hand_side}_WRIST").value
    shoulder_idx = getattr(mp_pose.PoseLandmark, f"{hand_side}_SHOULDER").value

    wrist = landmarks[wrist_idx]
    shoulder = landmarks[shoulder_idx]

    distance = np.linalg.norm(
        np.array([wrist.x, wrist.y, wrist.z]) -
        np.array([shoulder.x, shoulder.y, shoulder.z])
    )

    hand_histories[hand_side].append(distance)

    if len(hand_histories[hand_side]) == frames_per_action:
        x = range(frames_per_action)
        slope, *_ = linregress(x, hand_histories[hand_side])
        if slope > 0.08:
            return possible_actions[f"PUNCH_{hand_side}"]
    return None

with mp_pose.Pose(min_detection_confidence=0.7,
                  min_tracking_confidence=0.7) as pose:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Detect actions
            actions = [
                detect_punch(landmarks, "RIGHT"),
                detect_punch(landmarks, "LEFT"),
                detect_lean(landmarks),
            ]

            # Send key for each detected action
            for action in actions:
                if action is not None:
                    trigger_key(action)

        except Exception:
            pass

        # Draw landmarks for visual feedback
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(102, 178, 255), thickness=4, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 140, 0), thickness=4, circle_radius=2),
        )

        cv2.imshow("Ayo's Video", image)
        if cv2.waitKey(1) in (27, ord("q")):
            break

video.release()
cv2.destroyAllWindows()
