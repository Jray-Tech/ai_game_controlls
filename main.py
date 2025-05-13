import cv2
import mediapipe as mp
import numpy as np




video = cv2.VideoCapture(1)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Refactoring will mak this a finite state machine
# Define a function to detect actions based on landmarks
def detect_actions(landmarks):
    actions = []
    
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    


    if right_wrist.y > right_shoulder.y:
        print("Punch Right")
        actions.append("PUNCH_RIGHT")
    if left_wrist.y > left_shoulder.y:
        print("Punch Left")
        actions.append("PUNCH_LEFT")

    return actions



#  Get pose estimation model.
with mp_pose.Pose(
    # static_image_mode=False,
    # model_complexity=2,
    # enable_segmentation=True,
    # smooth_landmarks=True,
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
            detect_actions(landmarks)
            # print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
            # print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility)
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
    