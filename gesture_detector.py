# gesture_detector.py - UPDATED FOR GAZE/HEAD POSE TRACKING

import cv2
import mediapipe as mp
import time
import numpy as np
import threading 

# --- GLOBAL STATE (Thread-Shared) ---
fidget_score_global = 0.0 # Renamed but still used for focus input
latest_jpeg_frame = None 
is_active = False        
data_lock = threading.Lock() 

# Initialize MediaPipe outside the loop for efficiency
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Increased min_detection_confidence for better head tracking robustness
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) 

# --- Helper Function for Head Pose (Gaze Proxy) ---
# This function approximates gaze based on the angle of the head's rotation (yaw)
def get_head_angles(landmarks, image_w, image_h):
    # Select key landmarks for the head: Nose, Left/Right Eye, Left/Right Mouth
    # We will use Nose (0) and the ear landmarks (7, 8) as proxies for head pose.
    points = [
        landmarks[mp_pose.PoseLandmark.NOSE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    ]
    
    # Convert landmarks to pixel coordinates
    image_points = np.array([
        (int(p.x * image_w), int(p.y * image_h)) for p in points
    ], dtype="double")
    
    # Simple calculation: Check if the nose is significantly off-center from the ears.
    # We check if the nose is far left or far right of the line connecting the ears.
    if len(image_points) < 3:
        return 0, 0, 0 # Return 0s if detection is poor

    # Find the midpoint between the left and right ear
    ear_midpoint_x = (image_points[1][0] + image_points[2][0]) / 2
    
    # Horizontal deviation of the nose from the ear midpoint (yaw)
    yaw_deviation = image_points[0][0] - ear_midpoint_x 
    
    # Normalize deviation by image width to get a score (0.0 to 1.0)
    # Max deviation is considered when the nose is close to the edge of the frame
    # A max deviation of 0.2*image_w is considered fully looking away.
    normalized_yaw = min(abs(yaw_deviation) / (image_w * 0.2), 1.0)
    
    return normalized_yaw, ear_midpoint_x, image_points[0][0]

# --- PUBLIC ACCESSORS ---
def get_fidget_score():
    """Returns the current gaze/distraction score (0.0 to 1.0)."""
    with data_lock:
        return fidget_score_global

def get_jpeg_frame():
    """Returns the latest JPEG frame bytes for web streaming."""
    with data_lock:
        return latest_jpeg_frame

# --- MAIN CAMERA LOOP ---
def start_camera_stream():
    """
    Real-time computer vision process to detect gaze/head posture.
    """
    global fidget_score_global, latest_jpeg_frame
    
    # NOTE: In a multi-threaded flask app, ensure only ONE thread accesses cap.
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("ERROR: Cannot open webcam for gesture detection.")
        return

    print("Gesture Detector: Camera stream starting...")
    
    while is_active:
        success, image = cap.read()
        if not success:
            time.sleep(0.1)
            continue
            
        image_h, image_w, _ = image.shape

        # Processing the image...
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # --- Gaze Detection and Score Update ---
        current_gaze_score = 0.0
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Get normalized yaw (gaze) deviation
            normalized_yaw, _, _ = get_head_angles(results.pose_landmarks.landmark, image_w, image_h)
            
            current_gaze_score = normalized_yaw
            
            # Update the global score using the lock
            with data_lock:
                # Use current_gaze_score directly as the primary distraction input
                # Simple smoothing/decay to make the score persist briefly
                fidget_score_global = 0.8 * fidget_score_global + 0.2 * current_gaze_score
        
        # --- Draw Score Overlay ---
        cv2.putText(image, f"Gaze Score: {fidget_score_global:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # --- Encode Frame for Web Streaming ---
        ret, buffer = cv2.imencode('.jpg', image)
        
        with data_lock:
            latest_jpeg_frame = buffer.tobytes()

        # Control loop speed (approx. 20 FPS)
        time.sleep(0.05) 
        
    cap.release()
    print("Gesture Detector: Camera stream stopped.")