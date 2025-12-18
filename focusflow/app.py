import os
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

from flask import Flask, render_template, jsonify, Response, request
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import subprocess
import platform
import random
import datetime

app = Flask(__name__)

# --- CORE INTELLIGENCE MODULE ---
class FocusPredictor:
    def __init__(self):
        # Adjusted thresholds for higher sensitivity to looking away
        self.FIDGET_LOW = 0.10  # Very focused
        self.FIDGET_HIGH = 0.35 # Looking away significantly

    def calculate_focus(self, fidget_score: float) -> float:
        """Converts raw head deviation into a 0.1 to 1.0 focus ratio."""
        if fidget_score <= self.FIDGET_LOW:
            return 1.0
        elif fidget_score >= self.FIDGET_HIGH:
            return 0.1
        else:
            # Linear drop between the low and high thresholds
            scale = (fidget_score - self.FIDGET_LOW) / (self.FIDGET_HIGH - self.FIDGET_LOW)
            return 1.0 - (0.9 * scale)

# --- STATE MANAGEMENT ---
class FocusFlowState:
    def __init__(self):
        self.camera_active = False
        self.microphone_active = False
        self.focus_score = 100
        self.fidget_score_global = 0.0
        self.predictor = FocusPredictor()
        self.data_lock = threading.Lock()
        self.latest_frame = None
        self.total_sessions = 0
        self.total_focus_accumulated = 0
        self.distraction_count = 0
        self.start_time = time.time()
        
        # Historical focus for bio-rhythm
        self.focus_history = {}  # hour: list of scores
        
        # UI Data
        self.tasks = []

        # Real-time analytics structure
        self.analytics = {
            "deep_focus_score": 100,
            "total_focus_time": "0h 0m",
            "consistency": 100,
            "distractions": 0,
            "distraction_sources": {"Phone": 0, "Noise": 0, "Daydream": 0, "Fidgeting": 0},
            "peak_insight_time": "10:00 AM - 11:30 AM"
        }
        self.settings = {
            "proactive_interventions": True,
            "dyslexia_font": False,
            "reduced_motion": False,
            "color_contrast": "Standard",
            "buddy_persona": "Friendly Peer",
            "language": "English"
        }
        self.buddy_messages_queue = []

    def update_live_analytics(self):
        """Calculates analytics based on the current session performance."""
        # Note: self.data_lock is already held when this is called from the camera loop
        
        # 1. Update Distractions if focus is very low (ADHD threshold)
        if self.focus_score < 30:
            self.distraction_count += 1
            sources = ["Phone", "Noise", "Daydream", "Fidgeting"]
            source = random.choice(sources)
            self.analytics["distraction_sources"][source] += 1
        
        # 2. Update Total Elapsed Time
        elapsed_minutes = int((time.time() - self.start_time) / 60)
        hours = elapsed_minutes // 60
        mins = elapsed_minutes % 60
        self.analytics["total_focus_time"] = f"{hours}h {mins}m"
        
        # 3. Dynamic Deep Focus Score (Running average of the session)
        self.total_sessions += 1
        self.total_focus_accumulated += self.focus_score
        self.analytics["deep_focus_score"] = int(self.total_focus_accumulated / self.total_sessions)
        
        # 4. Total Distraction Counter
        self.analytics["distractions"] = self.distraction_count

        # 5. Update consistency (simple std dev approximation)
        self.analytics["consistency"] = max(0, 100 - (self.distraction_count * 5))

state = FocusFlowState()

# MediaPipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Using Pose tracker for stable ear and nose detection
pose_tracker = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# --- COMPUTER VISION LOGIC ---

def get_head_angles(landmarks, image_w, image_h):
    """
    Detects if the user is looking away by checking the nose position 
    relative to the distance between the ears.
    """
    try:
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

        # 1. Baseline: Distance between ears represents head width in 3D space
        head_width = abs(l_ear.x - r_ear.x)
        
        # 2. Midpoint: The center line of the face
        ear_mid_x = (l_ear.x + r_ear.x) / 2

        # 3. Deviation: How far the nose has turned away from the center line
        yaw_deviation = abs(nose.x - ear_mid_x) / (head_width / 2)

        return min(yaw_deviation, 1.0)
    except Exception:
        # If landmarks are lost (head turned too far), assume distracted
        return 0.8 

def camera_loop():
    """Background thread for continuous processing."""
    camera = cv2.VideoCapture(0)
    while True:
        with state.data_lock:
            active = state.camera_active
        if not active:
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success:
            continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 1. Calculate raw gaze deviation outside the lock for performance
            current_raw_val = get_head_angles(results.pose_landmarks.landmark, w, h)
            
            with state.data_lock:
                # 2. Update global score with Smoothing
                state.fidget_score_global = 0.7 * state.fidget_score_global + 0.3 * current_raw_val
                
                # 3. Predict focus percentage
                focus_ratio = state.predictor.calculate_focus(state.fidget_score_global)
                state.focus_score = int(focus_ratio * 100)

                # 4. Record to history
                current_hour = datetime.datetime.now().hour
                if current_hour not in state.focus_history:
                    state.focus_history[current_hour] = []
                state.focus_history[current_hour].append(state.focus_score)

                # 5. Trigger Live Analytics update
                state.update_live_analytics()

                # Update peak insight based on history
                hour_averages = {h: sum(scores)/len(scores) if scores else 0 for h, scores in state.focus_history.items()}
                if hour_averages:
                    peak_hour = max(hour_averages, key=hour_averages.get)
                    peak_start = f"{peak_hour:02d}:00"
                    peak_end = f"{(peak_hour + 1) % 24:02d}:00"
                    state.analytics["peak_insight_time"] = f"{peak_start} AM - {peak_end} AM" if peak_hour < 12 else f"{peak_start} PM - {peak_end} PM"

                # Overlay for the video feed
                color = (0, 255, 0) if state.focus_score > 50 else (0, 0, 255)
                cv2.putText(frame, f"Focus: {state.focus_score}%", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Intervention logic
                if state.focus_score < 40 and state.settings['proactive_interventions']:
                    trigger_buddy_intervention()

        with state.data_lock:
            state.latest_frame = frame.copy()

    camera.release()

# --- BUDDY & API ROUTES ---
def trigger_buddy_intervention():
    personas = {
        "Friendly Peer": ["Hey, focus is slipping!", "Come on, eyes on the screen!", "Let's get back to it."],
        "Gentle Mentor": ["I notice your attention wandering.", "Let's gently return to the task.", "Take a breath and refocus."],
        "Motivational Coach": ["You got this! Snap back!", "Don't let distractions win!", "Push through! Focus now!"]
    }
    msgs = personas.get(state.settings.get('buddy_persona', 'Friendly Peer'), ["Focus!"])
    if not state.buddy_messages_queue:
        msg = random.choice(msgs)
        state.buddy_messages_queue.append(msg)
        threading.Thread(target=speak_message, args=(msg,), daemon=True).start()

def speak_message(message):
    try:
        if platform.system() == "Darwin": subprocess.run(['say', message])
        else: print(f"Buddy: {message}")
    except: pass

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/state')
def get_state():
    with state.data_lock:
        return jsonify({
            'focus_score': state.focus_score,
            'camera_active': state.camera_active,
            'microphone_active': state.microphone_active,
            'tasks': state.tasks,
            'analytics': state.analytics,
            'settings': state.settings
        })

@app.route('/api/schedule')
def get_schedule():
    date_param = request.args.get('date', datetime.date.today().isoformat())
    try:
        sched_date = datetime.date.fromisoformat(date_param)
    except:
        sched_date = datetime.date.today()
    is_future = sched_date > datetime.date.today()
    
    with state.data_lock:
        hour_averages = {h: sum(scores)/len(scores) if scores else 0 for h, scores in state.focus_history.items()}
        if hour_averages:
            peak_hour = max(hour_averages, key=hour_averages.get)
            peak_start = f"{peak_hour:02d}:00"
            peak_end = f"{(peak_hour + 2) % 24:02d}:00"
        else:
            peak_start = "09:00"
            peak_end = "11:00"
    
        # Get unfinished tasks
        unfinished_tasks = [t for t in state.tasks if t['completed'] < t['total']]
        schedule = []
        current_time = datetime.datetime.strptime(peak_start, "%H:%M")
        for task in unfinished_tasks:
            remaining_duration = sum(st['duration'] for st in task['subtasks'] if not st['completed'])
            if remaining_duration > 0:
                schedule.append({
                    'time': current_time.strftime("%H:%M"),
                    'duration': remaining_duration,
                    'title': task['title'],
                    'energy': 'High',
                    'type': 'Deep Focus'
                })
                current_time += datetime.timedelta(minutes=remaining_duration)
                # Add break
                break_duration = 15
                schedule.append({
                    'time': current_time.strftime("%H:%M"),
                    'duration': break_duration,
                    'title': 'Short Break',
                    'energy': 'Low'
                })
                current_time += datetime.timedelta(minutes=break_duration)
        
        # Calculate daily stats
        total_scheduled = sum(item['duration'] for item in schedule if 'Break' not in item['title'])
        total_breaks = sum(item['duration'] for item in schedule if 'Break' in item['title'])
        scheduled_str = f"{total_scheduled // 60}h {total_scheduled % 60}m"
        breaks_str = f"{total_breaks // 60}h {total_breaks % 60}m"
        daily_stats = {'scheduled': scheduled_str, 'breaks': breaks_str}
    
    bio_rhythm = {'peak_start': peak_start, 'peak_end': peak_end}
    return jsonify({'schedule': schedule, 'bio_rhythm': bio_rhythm, 'daily_stats': daily_stats})

@app.route('/api/toggle_camera', methods=['POST'])
def toggle_camera():
    with state.data_lock:
        state.camera_active = not state.camera_active
    return jsonify({'camera_active': state.camera_active})

@app.route('/api/toggle_microphone', methods=['POST'])
def toggle_microphone():
    with state.data_lock:
        state.microphone_active = not state.microphone_active
    return jsonify({'microphone_active': state.microphone_active})

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with state.data_lock:
                frame = state.latest_frame
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/buddy_messages')
def get_buddy_messages():
    with state.data_lock:
        msgs = list(state.buddy_messages_queue)
        state.buddy_messages_queue.clear()
        return jsonify(msgs)

@app.route('/api/add_task', methods=['POST'])
def add_task():
    data = request.json
    title = data['title']
    title_lower = title.lower()
    if "essay" in title_lower:
        subtasks = [
            {"id": 1, "text": "Brainstorm ideas", "completed": False, "duration": 20},
            {"id": 2, "text": "Outline structure", "completed": False, "duration": 15},
            {"id": 3, "text": "Write draft", "completed": False, "duration": 60},
            {"id": 4, "text": "Edit and conclude", "completed": False, "duration": 30}
        ]
    elif "proposal" in title_lower:
        subtasks = [
            {"id": 1, "text": "Research topic", "completed": False, "duration": 30},
            {"id": 2, "text": "Outline proposal", "completed": False, "duration": 20},
            {"id": 3, "text": "Write sections", "completed": False, "duration": 45},
            {"id": 4, "text": "Review and refine", "completed": False, "duration": 25}
        ]
    elif "report" in title_lower:
        subtasks = [
            {"id": 1, "text": "Gather data", "completed": False, "duration": 40},
            {"id": 2, "text": "Analyze findings", "completed": False, "duration": 30},
            {"id": 3, "text": "Write report", "completed": False, "duration": 50},
            {"id": 4, "text": "Proofread", "completed": False, "duration": 20}
        ]
    elif "presentation" in title_lower:
        subtasks = [
            {"id": 1, "text": "Research content", "completed": False, "duration": 25},
            {"id": 2, "text": "Create slides", "completed": False, "duration": 35},
            {"id": 3, "text": "Practice delivery", "completed": False, "duration": 30},
            {"id": 4, "text": "Finalize", "completed": False, "duration": 15}
        ]
    elif "homework" in title_lower:
        subtasks = [
            {"id": 1, "text": "Understand assignment", "completed": False, "duration": 10},
            {"id": 2, "text": "Complete exercises", "completed": False, "duration": 40},
            {"id": 3, "text": "Check answers", "completed": False, "duration": 15}
        ]
    else:
        subtasks = [
            {"id": 1, "text": f"Prepare for {title}", "completed": False, "duration": 15},
            {"id": 2, "text": "Execute main task", "completed": False, "duration": 30},
            {"id": 3, "text": "Review and finish", "completed": False, "duration": 15}
        ]
    new_task = {
        "id": len(state.tasks) + 1,
        "title": title,
        "completed": 0,
        "total": len(subtasks),
        "subtasks": subtasks
    }
    with state.data_lock:
        state.tasks.append(new_task)
    return jsonify(new_task)

@app.route('/api/toggle_subtask', methods=['POST'])
def toggle_subtask():
    data = request.json
    with state.data_lock:
        task = next((t for t in state.tasks if t['id'] == data['task_id']), None)
        if task:
            sub = next((s for s in task['subtasks'] if s['id'] == data['subtask_id']), None)
            if sub:
                sub['completed'] = not sub['completed']
                task['completed'] = sum(s['completed'] for s in task['subtasks'])
    return jsonify({'success': True})

@app.route('/api/analytics')
def get_analytics():
    with state.data_lock:
        return jsonify(state.analytics)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    with state.data_lock:
        state.settings.update(data)
    return jsonify({'success': True})

# Start background thread
threading.Thread(target=camera_loop, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)
