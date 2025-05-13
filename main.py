import cv2
import mediapipe as mp
import numpy as np
from scipy.stats import linregress
from collections import deque
import time
from pynput.keyboard import Controller
from pynput.keyboard import Key


# Initialize video capture and Mediapipe
video = cv2.VideoCapture(1)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

possible_actions = {
    "PUNCH_RIGHT": 0,
    "PUNCH_LEFT": 1,
    "LEAN_FWD": 2,
    "LEAN_BACK": 3,
}


ACTION_KEY_MAP = {
    possible_actions["PUNCH_RIGHT"]: Key.space,   # jump
    possible_actions["PUNCH_LEFT"]: Key.space,   # jump
    # possible_actions["LEAN_BACK"]:  Key.space,   # duck 
}
# Map each action ID to a keyboard key
# ACTION_KEY_MAP = {
#     possible_actions["PUNCH_RIGHT"]: "R",  
#     possible_actions["PUNCH_LEFT"]: "T",   
#     possible_actions["LEAN_FWD"]: "A",     
#     possible_actions["LEAN_BACK"]: "D",    
# }

frames_per_action = 7         
COOLDOWN = 0.5                
SLOPE_THRESH = 0.02  
VIS_THRESH  = 0.6   # ignore fuzzy frames
CALIBRATION_FRAMES = 20 # number of frames to calibrate lean
LEAN_FWD_THR  = -0.06   
LEAN_BACK_THR =  0.06  # more positive  → farther from camera
RECENTER_RATE = 0.003  # rate of recentering

calib_depths = deque(maxlen=CALIBRATION_FRAMES)
neutral_depth = None  # will hold the mid‑point once locked in



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


def detect_lean(landmarks, neutral_depth: float):
    """Return LEAN_FWD / LEAN_BACK or None, based on difference from neutral."""
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    if nose.visibility < VIS_THRESH or neutral_depth is None:
        return None

    diff = nose.z - neutral_depth 
    neutral_depth += RECENTER_RATE * diff

    if diff < LEAN_FWD_THR:
        return possible_actions["LEAN_FWD"]
    elif diff > LEAN_BACK_THR:
        return possible_actions["LEAN_BACK"]
    return None



def detect_punch(landmarks, hand_side: str):
    wrist_idx = getattr(mp_pose.PoseLandmark, f"{hand_side}_WRIST").value
    shoulder_idx = getattr(mp_pose.PoseLandmark, f"{hand_side}_SHOULDER").value

    wrist = landmarks[wrist_idx]
    shoulder = landmarks[shoulder_idx]

    # bail out if Mediapipe thinks the landmark is unreliable
    if wrist.visibility   < VIS_THRESH or shoulder.visibility < VIS_THRESH:
        return None


    dist = np.hypot(wrist.x - shoulder.x, wrist.y - shoulder.y)

    hand_histories[hand_side].append(dist)

    # not enough history yet
    if len(hand_histories[hand_side]) < frames_per_action:
        return None

    smoothed = np.median(hand_histories[hand_side])

    # linear trend on the smoothed series
    x = np.arange(frames_per_action)
    slope, *_ = linregress(x, list(hand_histories[hand_side]))
    # print(f"[DEBUG] {hand_side} slope: {slope:.3f}, smoothed: {smoothed:.3f}")

    if slope > SLOPE_THRESH and smoothed > 0.1: 
        return possible_actions[f"PUNCH_{hand_side}"]
    return None


with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
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

            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            if neutral_depth is None:
                if nose.visibility > VIS_THRESH:
                    calib_depths.append(nose.z)
                if len(calib_depths) == CALIBRATION_FRAMES:
                    neutral_depth = float(np.median(calib_depths))
                    print(f"[INFO] Neutral depth locked at {neutral_depth:.3f}")
                # Skip action detection until we have a midpoint
                continue


            # Detect actions
            actions = [
                detect_punch(landmarks, "RIGHT"),
                detect_punch(landmarks, "LEFT"),
                detect_lean(landmarks, neutral_depth),
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
