import os
# Allow macOS to show the camera permission prompt
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '0'
import webbrowser
import threading

from flask import Flask, render_template, jsonify, Response, request
import threading
import time
import subprocess
import platform
import random
import datetime
import json
import pathlib
import re

# Optional native dependencies (cv2 / mediapipe) — make robust if missing
HAVE_CV = True
try:
    import cv2
    import mediapipe as mp
    import numpy as np
except Exception:
    HAVE_CV = False
    cv2 = None
    mp = None
    np = None

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
        self.wake_word_active = False
        self.focus_score = 100
        self.fidget_score_global = 0.0
        self.predictor = FocusPredictor()
        self.data_lock = threading.Lock()
        self.latest_frame = None
        self.last_frame_ts = 0.0
        self.camera_error = None
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
            "language": "English",
            "dark_mode": False,
            "color_blind_mode": False,
            "hide_sidebar": False,
            "camera_access": True
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
mp_pose = mp.solutions.pose # Using Pose tracker for stable ear and nose detection
pose_tracker = None 
if HAVE_CV and mp is not None: 
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

def _open_camera():
    """Try opening a camera device with AVFoundation (macOS) and fallbacks."""
    indices = [0, 1, 2]
    for idx in indices:
        try:
            cam = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if cam is not None and cam.isOpened():
                return cam
        except Exception:
            pass
    for idx in indices:
        try:
            cam = cv2.VideoCapture(idx)
            if cam is not None and cam.isOpened():
                return cam
        except Exception:
            pass
    return None

def camera_loop():
    """Background thread for continuous processing."""
    if not HAVE_CV or cv2 is None:
        # Camera processing not available on this system
        print('Camera loop disabled — OpenCV/MediaPipe not available')
        return
    camera = None
    while True:
        with state.data_lock:
            active = state.camera_active
        if not active:
            time.sleep(0.1)
            continue

        if camera is None or not camera.isOpened():
            camera = _open_camera()
            if camera is None:
                with state.data_lock:
                    state.camera_error = 'Camera not available. Check permissions or close other apps.'
                time.sleep(0.2)
                continue

        success, frame = camera.read()
        if not success or frame is None:
            with state.data_lock:
                if time.time() - state.last_frame_ts > 2.0:
                    state.camera_error = 'No camera frames received. Check permissions or close other apps.'
            try:
                camera.release()
            except Exception:
                pass
            camera = None
            time.sleep(0.1)
            continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 1. Calculate raw gaze deviation outside the lock for performance
            current_raw_val = get_head_angles(results.pose_landmarks.landmark, w, h)
            
            with state.data_lock:
                state.camera_error = None
                state.last_frame_ts = time.time()
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

    if camera is not None:
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
        # Convert focus_history dict to hourly averages for bio-rhythm
        hour_averages = {h: sum(scores)/len(scores) if scores else 0 for h, scores in state.focus_history.items()}
        return jsonify({
            'focus_score': state.focus_score,
            'camera_active': state.camera_active,
            'microphone_active': state.microphone_active,
            'wake_word_active': state.wake_word_active,
            'tasks': state.tasks,
            'analytics': state.analytics,
            'settings': state.settings,
            'focus_history': state.focus_history,
            'hour_averages': hour_averages,
            'camera_error': state.camera_error,
            'last_frame_ts': state.last_frame_ts
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

@app.route('/api/toggle_wake_word', methods=['POST'])
def toggle_wake_word():
    with state.data_lock:
        state.wake_word_active = not state.wake_word_active
    return jsonify({'wake_word_active': state.wake_word_active})

@app.route('/video_feed')
def video_feed():
    if not HAVE_CV or cv2 is None:
        return Response(status=503)
    def generate():
        while True:
            with state.data_lock:
                frame = state.latest_frame
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)
    response = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route('/video_frame')
def video_frame():
    if not HAVE_CV or cv2 is None:
        return Response(status=503)
    with state.data_lock:
        frame = state.latest_frame
    if frame is None:
        return Response(status=204)
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return Response(status=500)
    response = Response(buffer.tobytes(), mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Pragma'] = 'no-cache'
    return response

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
    # optional difficulty and duration
    difficulty = data.get('difficulty')
    duration_minutes = data.get('duration_minutes')
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
        "subtasks": subtasks,
        "difficulty": difficulty or 'medium',
        "duration_minutes": duration_minutes or sum(st['duration'] for st in subtasks)
    }
    with state.data_lock:
        state.tasks.append(new_task)
    try:
        save_state_to_disk()
    except Exception:
        pass
    return jsonify(new_task)


def start_task_timer(task_id, minutes):
    def _timer():
        try:
            time.sleep(max(0, int(minutes) * 60))
            with state.data_lock:
                task = next((t for t in state.tasks if t['id'] == task_id), None)
                if not task:
                    return
                # If task still incomplete, enqueue reminder with tips
                if task.get('completed', 0) < task.get('total', 1):
                    tip = generate_tip_for_task(task)
                    msg = f"Reminder: Your task '{task['title']}' is not completed. {tip}"
                    state.buddy_messages_queue.append(msg)
        except Exception:
            pass
    threading.Thread(target=_timer, daemon=True).start()


def generate_tip_for_task(task):
    # Simple tip generator based on difficulty
    diff = (task.get('difficulty') or 'medium').lower()
    tips = {
        'low': ["Nice start — keep going for a few more minutes.", "Break it into two 10-minute chunks."],
        'medium': ["Try a 10-minute focused sprint, then a short break.", "Remove distractions: silence phone and close tabs."],
        'high': ["Try a short 5-minute plan: list 3 micro-steps and begin.", "Use a reward after 15 minutes of progress (small treat)."]
    }
    import random
    return random.choice(tips.get(diff, tips['medium']))

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
    try:
        save_state_to_disk()
    except Exception:
        pass
    return jsonify({'success': True})

@app.route('/api/analytics')
def get_analytics():
    with state.data_lock:
        return jsonify(state.analytics)


@app.route('/api/ai_insight')
def ai_insight():
    with state.data_lock:
        a = dict(state.analytics)
        # Compute a readable peak window from history if available
        peak_window = a.get('peak_insight_time')
        try:
            hour_averages = {h: (sum(scores) / len(scores)) if scores else 0 for h, scores in state.focus_history.items()}
            if hour_averages:
                peak_hour = max(hour_averages, key=hour_averages.get)
                peak_start = f"{peak_hour:02d}:00"
                peak_end = f"{(peak_hour + 1) % 24:02d}:00"
                if peak_hour < 12:
                    peak_window = f"{peak_start} AM - {peak_end} AM"
                else:
                    peak_window = f"{peak_start} PM - {peak_end} PM"
        except Exception:
            peak_window = peak_window or ''

        deep = a.get('deep_focus_score')
        consistency = a.get('consistency')
        distractions = a.get('distractions', 0)

        # Suggest a task
        suggested_task = None
        for t in state.tasks:
            if t.get('completed', 0) < t.get('total', 1):
                suggested_task = t.get('title')
                break
        if not suggested_task:
            for t in state.tasks:
                if 'math' in t.get('title', '').lower():
                    suggested_task = t.get('title')
                    break
        if not suggested_task:
            suggested_task = 'challenging subjects (e.g., Math)'

        # Session advice
        session_advice = 'Try 20–25 minute Pomodoro sessions to build momentum.'
        if deep is not None:
            try:
                if int(deep) >= 75:
                    session_advice = 'Schedule longer, uninterrupted sessions (60–90 minutes).'
                elif int(deep) >= 50:
                    session_advice = 'Aim for 45–60 minute focused sessions.'
            except Exception:
                pass

        # Compose descriptive HTML
        desc_parts = []
        if peak_window:
            desc_parts.append(f"Based on recent performance, your focus tends to peak around <strong>{peak_window}</strong>.")
        else:
            desc_parts.append('Based on recent performance, we detected a recurring peak focus window.')

        desc_parts.append(f"We recommend scheduling your most challenging tasks — like <em>{suggested_task}</em> — during that period. {session_advice}")

        if consistency is not None and consistency < 60:
            desc_parts.append('Your routine consistency is low; try a regular daily start time and limit context switching.')
        if distractions is not None and distractions > 3:
            desc_parts.append('Reduce notifications and external interruptions during focused blocks to lower distractions.')

        title = 'AI Insight: Peak Focus'
        desc = ' '.join(desc_parts)
        return jsonify({'title': title, 'desc': desc})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    with state.data_lock:
        state.settings.update(data)
    # persist
    try:
        save_state_to_disk()
    except Exception:
        pass
    return jsonify({'success': True})


@app.route('/api/set_microphone', methods=['POST'])
def set_microphone():
    data = request.json or {}
    value = data.get('value')
    if value is None:
        return jsonify({'success': False, 'error': 'missing value'}), 400
    with state.data_lock:
        state.microphone_active = bool(value)
    try:
        save_state_to_disk()
    except Exception:
        pass
    return jsonify({'microphone_active': state.microphone_active})


@app.route('/api/voice_command', methods=['POST'])
def voice_command():
    data = request.json or {}
    transcript = (data.get('transcript') or '').strip()
    print(f"\n=== VOICE COMMAND DEBUG ===")
    print(f"Received data: {data}")
    print(f"Transcript: '{transcript}'")
    print(f"===========================\n")
    reply = "Sorry, I didn't understand. Could you repeat that?"
    action = {}
    
    print(f"Voice command received: '{transcript}'")

    if not transcript:
        print("Empty transcript received")
        return jsonify({'reply': reply, 'action': action})

    t = transcript.lower()
    print(f"Processing lowercase: '{t}'")
    with state.data_lock:
        # Camera commands - more flexible regex
        if re.search(r"\b(turn on|enable|start|open).{0,10}camera\b", t) or re.search(r"\bcamera.{0,10}(on|enable|start|open)\b", t):
            print("Matched: Camera ON command")
            state.camera_active = True
            reply = "Camera turned on."
            action = {'type': 'camera', 'value': True}
        elif re.search(r"\b(turn off|disable|stop|close).{0,10}camera\b", t) or re.search(r"\bcamera.{0,10}(off|disable|stop|close)\b", t):
            print("Matched: Camera OFF command")
            state.camera_active = False
            reply = "Camera turned off."
            action = {'type': 'camera', 'value': False}
        
        # Remove/delete task - more flexible patterns
        elif re.search(r"\b(remove|delete|clear|get rid of).{0,40}(task|todo|item)\b", t):
            # Try to extract numeric id
            id_match = re.search(r"(?:task|todo|item).{0,5}(\d+)", t)
            title_match = re.search(r"\b(remove|delete|clear|get rid of).{0,5}(?:the\s+)?(?:task|todo|item).{0,5}(?:called|named|titled)?\s*['\"]?(?P<title>[^'\"]+)['\"]?", t, re.I)
            removed = None
            
            if id_match:
                tid = int(id_match.group(1))
                for i, tsk in enumerate(state.tasks):
                    if tsk.get('id') == tid:
                        removed = state.tasks.pop(i)
                        break
            elif title_match:
                title = title_match.group('title').strip()
                for i, tsk in enumerate(state.tasks):
                    if title.lower() in tsk.get('title', '').lower():
                        removed = state.tasks.pop(i)
                        break
            else:
                # Try to extract title from after the command
                words = t.split()
                cmd_words = ['remove', 'delete', 'clear', 'get rid of']
                start_idx = None
                for i, word in enumerate(words):
                    if word in cmd_words:
                        start_idx = i + 1
                        break
                if start_idx and start_idx < len(words):
                    potential_title = ' '.join(words[start_idx:])
                    if 'task' not in potential_title and 'todo' not in potential_title:
                        for i, tsk in enumerate(state.tasks):
                            if potential_title.lower() in tsk.get('title', '').lower():
                                removed = state.tasks.pop(i)
                                break
            
            if removed:
                save_state_to_disk()
                reply = f"Removed task: {removed.get('title')}"
                action = {'type': 'removed_task', 'task': removed}
            else:
                reply = "Could not find the task to remove. Please specify the task id or exact title."
        
        # Add task - more flexible patterns
        elif re.search(r"\b(add|create|make|new).{0,40}(task|todo|item)\b", t):
            # Extract difficulty
            diff_match = re.search(r"\b(difficulty|level).{0,5}(low|medium|easy|hard|high)\b", t, re.I)
            # Extract time
            time_match = re.search(r"\b(for|duration|time).{0,5}(\d{1,3})\s*(minutes|minute|mins|min|hours|hour|hrs)\b", t, re.I)
            
            # Extract title - look for content after keywords up to difficulty/time indicators
            title = ""
            add_keywords = ['add task', 'create task', 'make task', 'new task', 'add todo', 'create todo']
            for kw in add_keywords:
                if kw in t:
                    idx = t.find(kw)
                    # Extract everything after keyword
                    candidate = transcript[idx+len(kw):].strip()
                    if candidate:
                        # Remove difficulty and time phrases
                        if diff_match:
                            candidate = re.sub(diff_match.group(0), '', candidate, flags=re.I).strip()
                        if time_match:
                            candidate = re.sub(time_match.group(0), '', candidate, flags=re.I).strip()
                        # Remove common filler words
                        candidate = re.sub(r'\b(please|could you|would you|can you|I need to)\b', '', candidate, flags=re.I).strip()
                        if candidate:
                            title = candidate
                            break
            
            if not title:
                # Fallback: try to get text after the last keyword
                for word in ['called', 'named', 'titled', 'about', 'for']:
                    if word in t:
                        parts = t.split(word, 1)
                        if len(parts) > 1:
                            title = parts[1].strip()
                            # Clean up title
                            for phrase in ['with difficulty', 'for minutes', 'difficulty', 'duration']:
                                title = title.split(phrase)[0].strip()
                            break
            
            if not title:
                title = 'New task'
            
            difficulty = diff_match.group(2).lower() if diff_match else 'medium'
            # Normalize difficulty levels
            if difficulty in ['easy', 'low']:
                difficulty = 'low'
            elif difficulty in ['hard', 'high']:
                difficulty = 'high'
            else:
                difficulty = 'medium'
            
            duration_minutes = None
            if time_match:
                v = int(time_match.group(2))
                unit = time_match.group(3)
                if 'hour' in unit:
                    duration_minutes = v * 60
                else:
                    duration_minutes = v
            
            # Create subtasks based on difficulty
            if difficulty == 'high':
                subtasks = [
                    {"id": 1, "text": "Plan micro-steps", "completed": False, "duration": 10},
                    {"id": 2, "text": "Work focused sprint", "completed": False, "duration": 30},
                    {"id": 3, "text": "Review & checkpoint", "completed": False, "duration": 20}
                ]
            elif difficulty == 'low':
                subtasks = [
                    {"id": 1, "text": "Quick prep", "completed": False, "duration": 5},
                    {"id": 2, "text": "Finish small step", "completed": False, "duration": 10}
                ]
            else:
                subtasks = [
                    {"id": 1, "text": "Prepare", "completed": False, "duration": 10},
                    {"id": 2, "text": "Focused work", "completed": False, "duration": 20},
                    {"id": 3, "text": "Wrap up", "completed": False, "duration": 10}
                ]
            
            new_task = {
                "id": len(state.tasks) + 1,
                "title": title,
                "completed": 0,
                "total": len(subtasks),
                "subtasks": subtasks,
                "difficulty": difficulty,
                "duration_minutes": duration_minutes or sum(st['duration'] for st in subtasks)
            }
            state.tasks.append(new_task)
            reply = f"Added task: {title} (difficulty: {difficulty})"
            action = {'type': 'add_task', 'task': new_task}
            if new_task.get('duration_minutes'):
                start_task_timer(new_task['id'], new_task['duration_minutes'])
        
        # Add schedule - more flexible patterns
        elif re.search(r"\b(schedule|plan|calendar).{0,40}(add|create|make|set up)\b", t) or re.search(r"\b(add|create|make|set up).{0,40}(schedule|plan|appointment|event)\b", t):
            time_match = re.search(r"\b(at|around|by)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m|p\.m)?\b", t, re.I)
            dur_match = re.search(r"\b(for|duration|length).{0,5}(\d{1,3})\s*(minutes|minute|mins|min|hours|hour|hrs)\b", t, re.I)
            date_match = re.search(r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t, re.I)
            
            # Extract title
            title = transcript
            schedule_keywords = ['add schedule', 'create schedule', 'make schedule', 'set up schedule', 
                               'schedule', 'plan', 'calendar event']
            for kw in schedule_keywords:
                if kw in t:
                    idx = t.find(kw)
                    candidate = transcript[idx+len(kw):].strip()
                    if candidate:
                        title = candidate
                        break
            
            # Clean up title
            if time_match:
                title = re.sub(time_match.group(0), '', title, flags=re.I).strip()
            if dur_match:
                title = re.sub(dur_match.group(0), '', title, flags=re.I).strip()
            if date_match:
                title = re.sub(date_match.group(0), '', title, flags=re.I).strip()
            
            if not title or len(title) < 2:
                title = "Scheduled activity"
            
            # Parse time
            sched_time = None
            if time_match:
                hour = int(time_match.group(2))
                minute = int(time_match.group(3) or 0)
                ampm = (time_match.group(4) or '').lower()
                if 'pm' in ampm and hour < 12:
                    hour += 12
                if 'am' in ampm and hour == 12:
                    hour = 0
                sched_time = f"{hour:02d}:{minute:02d}"
            
            # Parse duration
            duration = None
            if dur_match:
                v = int(dur_match.group(2))
                unit = dur_match.group(3)
                if 'hour' in unit:
                    duration = v * 60
                else:
                    duration = v
            
            # Add to tasks as a scheduled todo
            new_task = {
                "id": len(state.tasks) + 1,
                "title": title,
                "completed": 0,
                "total": 1,
                "subtasks": [{"id":1, "text": title, "completed": False, "duration": duration or 30}],
                "scheduled_time": sched_time,
                "duration_minutes": duration or 30,
                "is_scheduled": True
            }
            state.tasks.append(new_task)
            reply = f"Scheduled '{title}' at {sched_time or 'ASAP'} for {new_task['duration_minutes']} minutes."
            action = {'type': 'add_schedule', 'entry': {"time": sched_time, "title": title, "duration": duration or 30}}
            start_task_timer(new_task['id'], new_task['duration_minutes'])
        
        # Dopamine/break suggestions
        elif re.search(r"\b(suggest|recommend|give me|I need).{0,40}(break|pause|rest|dopamine|motivation|energy)\b", t) or re.search(r"\b(break|pause|rest).{0,40}(suggestion|idea|recommendation)\b", t):
            suggestions = [
                "Take a 2-minute dance break with your favorite song!",
                "Do 5 jumping jacks for a quick energy boost.",
                "Drink a glass of water and stretch your arms.",
                "Stand up and walk to the window for 30 seconds.",
                "Take 3 deep breaths to reset your focus."
            ]
            import random
            s = random.choice(suggestions)
            reply = f"Suggestion: {s}"
            action = {'type': 'suggest_dopamine', 'suggestion': s}
        
        # Query focus - more flexible patterns
        elif re.search(r"\b(how.{0,5}focus|what.{0,5}focus|current focus|focus level|focus score)\b", t) or re.search(r"\b(tell me|show me|what is).{0,20}focus\b", t):
            reply = f"Your current focus is {state.focus_score} percent."
            action = {'type': 'report_focus', 'value': state.focus_score}
        
        # Toggle dyslexia font
        elif re.search(r"\b(dyslexic|dyslexia).{0,20}(font|text|type)\b", t):
            if re.search(r"\b(enable|turn on|activate|use|on)\b", t):
                state.settings['dyslexia_font'] = True
                reply = 'Dyslexia-friendly font enabled.'
            elif re.search(r"\b(disable|turn off|deactivate|stop|off)\b", t):
                state.settings['dyslexia_font'] = False
                reply = 'Dyslexia-friendly font disabled.'
            else:
                # Toggle if not specified
                state.settings['dyslexia_font'] = not state.settings['dyslexia_font']
                status = "enabled" if state.settings['dyslexia_font'] else "disabled"
                reply = f'Dyslexia-friendly font {status}.'
            action = {'type': 'setting', 'name': 'dyslexia_font', 'value': state.settings['dyslexia_font']}
        
        # Toggle dark mode
        elif re.search(r"\b(dark|night).{0,10}mode\b", t):
            if re.search(r"\b(enable|turn on|activate|use|on|switch to)\b", t):
                state.settings['dark_mode'] = True
                reply = 'Dark mode enabled.'
            elif re.search(r"\b(disable|turn off|deactivate|stop|off|switch from)\b", t):
                state.settings['dark_mode'] = False
                reply = 'Dark mode disabled.'
            else:
                state.settings['dark_mode'] = not state.settings['dark_mode']
                status = "enabled" if state.settings['dark_mode'] else "disabled"
                reply = f'Dark mode {status}.'
            action = {'type': 'setting', 'name': 'dark_mode', 'value': state.settings['dark_mode']}
        
        # Change buddy persona
        elif re.search(r"\b(buddy|assistant|companion|helper).{0,20}(persona|character|type|style)\b", t):
            personas = ["Friendly Peer", "Gentle Mentor", "Motivational Coach"]
            for persona in personas:
                if persona.lower() in t:
                    state.settings['buddy_persona'] = persona
                    reply = f'Buddy persona changed to {persona}.'
                    action = {'type': 'setting', 'name': 'buddy_persona', 'value': persona}
                    break
            else:
                # If no specific persona mentioned, list options
                reply = f"I can set buddy persona to: {', '.join(personas)}. Please specify which one."
        
        # Change primary language
        elif re.search(r"\b(language|speak|talk).{0,20}(change|set|switch|use)\b", t) or re.search(r"\b(speak|talk).{0,10}(english|urdu|mixed)\b", t):
            languages = ["English", "Urdu", "Mixed"]
            for lang in languages:
                if lang.lower() in t:
                    state.settings['language'] = lang
                    reply = f'Primary language changed to {lang}.'
                    action = {'type': 'setting', 'name': 'language', 'value': lang}
                    break
            else:
                # If no specific language mentioned, list options
                reply = f"I can set language to: {', '.join(languages)}. Please specify which one."
        
        # Toggle microphone access
        elif re.search(r"\b(microphone|mic).{0,20}(access|permission)\b", t):
            if re.search(r"\b(enable|turn on|allow|grant|on)\b", t):
                state.settings['camera_access'] = True
                reply = 'Microphone access enabled.'
            elif re.search(r"\b(disable|turn off|deny|revoke|off)\b", t):
                state.settings['camera_access'] = False
                reply = 'Microphone access disabled.'
            else:
                state.settings['camera_access'] = not state.settings['camera_access']
                status = "enabled" if state.settings['camera_access'] else "disabled"
                reply = f'Microphone access {status}.'
            action = {'type': 'setting', 'name': 'camera_access', 'value': state.settings['camera_access']}
        
        # Toggle wake word
        elif re.search(r"\b(wake word|wake up|voice activation|always listen)\b", t):
            if re.search(r"\b(enable|turn on|activate|start|on)\b", t):
                state.wake_word_active = True
                reply = 'Wake word activated. Say "Hey Focus" to give commands without pressing the mic button.'
                action = {'type': 'wake_word', 'value': True}
            elif re.search(r"\b(disable|turn off|deactivate|stop|off)\b", t):
                state.wake_word_active = False
                reply = 'Wake word deactivated. Press the mic button to give commands.'
                action = {'type': 'wake_word', 'value': False}
            else:
                state.wake_word_active = not state.wake_word_active
                status = "activated" if state.wake_word_active else "deactivated"
                reply = f'Wake word {status}.'
                if state.wake_word_active:
                    reply += ' Say "Hey Focus" to give commands.'
                action = {'type': 'wake_word', 'value': state.wake_word_active}
        
        # Toggle proactive interventions
        elif re.search(r"\b(proactive|intervention|help|assistance).{0,20}(on|off|enable|disable)\b", t) or re.search(r"\b(remind|notify|alert).{0,20}me\b", t):
            if re.search(r"\b(enable|turn on|activate|yes|on)\b", t):
                state.settings['proactive_interventions'] = True
                reply = 'Proactive interventions enabled. I will remind you when focus drops.'
            elif re.search(r"\b(disable|turn off|deactivate|no|off)\b", t):
                state.settings['proactive_interventions'] = False
                reply = 'Proactive interventions disabled.'
            else:
                state.settings['proactive_interventions'] = not state.settings['proactive_interventions']
                status = "enabled" if state.settings['proactive_interventions'] else "disabled"
                reply = f'Proactive interventions {status}.'
            action = {'type': 'setting', 'name': 'proactive_interventions', 'value': state.settings['proactive_interventions']}
        
        # Change color contrast
        elif re.search(r"\b(color|contrast|brightness).{0,20}(change|set|adjust|increase|decrease)\b", t):
            contrasts = ["Standard", "High", "Maximum"]
            matched = False
            
            # First check for exact contrast names
            for contrast in contrasts:
                if contrast.lower() in t:
                    state.settings['color_contrast'] = contrast
                    reply = f'Color contrast set to {contrast}.'
                    action = {'type': 'setting', 'name': 'color_contrast', 'value': contrast}
                    matched = True
                    break
            
            # If no exact match, check for keywords
            if not matched:
                if re.search(r"\b(high|higher|increase|more)\b", t):
                    state.settings['color_contrast'] = "High"
                    reply = 'Color contrast set to High.'
                    action = {'type': 'setting', 'name': 'color_contrast', 'value': "High"}
                elif re.search(r"\b(maximum|max|highest|extreme)\b", t):
                    state.settings['color_contrast'] = "Maximum"
                    reply = 'Color contrast set to Maximum.'
                    action = {'type': 'setting', 'name': 'color_contrast', 'value': "Maximum"}
                elif re.search(r"\b(standard|normal|default|medium)\b", t):
                    state.settings['color_contrast'] = "Standard"
                    reply = 'Color contrast set to Standard.'
                    action = {'type': 'setting', 'name': 'color_contrast', 'value': "Standard"}
                else:
                    reply = f"I can set color contrast to: Standard, High, or Maximum. Please specify which level."
        
        # Toggle sidebar
        elif re.search(r"\b(sidebar|navigation|menu).{0,20}(show|hide|toggle|visible)\b", t):
            if re.search(r"\b(hide|close|remove|off)\b", t):
                state.settings['hide_sidebar'] = True
                reply = 'Sidebar hidden.'
            elif re.search(r"\b(show|open|display|on)\b", t):
                state.settings['hide_sidebar'] = False
                reply = 'Sidebar shown.'
            else:
                state.settings['hide_sidebar'] = not state.settings['hide_sidebar']
                status = "hidden" if state.settings['hide_sidebar'] else "shown"
                reply = f'Sidebar {status}.'
            action = {'type': 'setting', 'name': 'hide_sidebar', 'value': state.settings['hide_sidebar']}
        
        # Toggle color blind mode
        elif re.search(r"\b(color.?blind|colorblind|color vision).{0,20}(mode|friendly|accessible)\b", t):
            if re.search(r"\b(enable|turn on|activate|use|on)\b", t):
                state.settings['color_blind_mode'] = True
                reply = 'Color blind mode enabled.'
            elif re.search(r"\b(disable|turn off|deactivate|stop|off)\b", t):
                state.settings['color_blind_mode'] = False
                reply = 'Color blind mode disabled.'
            else:
                state.settings['color_blind_mode'] = not state.settings['color_blind_mode']
                status = "enabled" if state.settings['color_blind_mode'] else "disabled"
                reply = f'Color blind mode {status}.'
            action = {'type': 'setting', 'name': 'color_blind_mode', 'value': state.settings['color_blind_mode']}
        
        # Open different tabs/sections
        elif re.search(r"\b(open|go to|show|view|navigate to).{0,15}(dashboard|home|main)\b", t):
            reply = "Opening dashboard."
            action = {'type': 'navigate', 'view': 'dashboard'}
        elif re.search(r"\b(open|go to|show|view|navigate to).{0,15}(settings|preferences|options)\b", t):
            reply = "Opening settings."
            action = {'type': 'navigate', 'view': 'settings'}
        elif re.search(r"\b(open|go to|show|view|navigate to).{0,15}(analytics|insights|stats|statistics|reports)\b", t):
            reply = "Opening analytics."
            action = {'type': 'navigate', 'view': 'analytics'}
        elif re.search(r"\b(open|go to|show|view|navigate to).{0,15}(schedule|calendar|planner|timeline)\b", t):
            reply = "Opening schedule."
            action = {'type': 'navigate', 'view': 'schedule'}
        
        # General help or unknown command
        elif re.search(r"\b(help|what can you do|commands|options|capabilities)\b", t):
            reply = "I can help you with: turning camera on/off, adding/removing tasks, scheduling, changing settings (dark mode, language, etc.), checking focus, and navigating between dashboard, settings, analytics, and schedule tabs. Just tell me what you need!"
        
        else:
            # Generic echo with encouragement
            replies = [
                f"I heard: '{transcript}'. You can ask me to control camera, manage tasks, change settings, or check your focus.",
                f"Got it: '{transcript}'. Need help with camera, tasks, or settings?",
                f"'{transcript}' - I'm here to help with focus tracking and task management!"
            ]
            import random
            reply = random.choice(replies)

    # Save settings if they were changed
    if action.get('type') == 'setting':
        try:
            save_state_to_disk()
        except Exception:
            pass

    return jsonify({'reply': reply, 'action': action})

@app.route('/api/remove_task', methods=['POST'])
def remove_task():
    data = request.json or {}
    task_id = data.get('id') or data.get('task_id')
    title = data.get('title')
    removed = None
    with state.data_lock:
        if task_id is not None:
            try:
                tid = int(task_id)
            except Exception:
                tid = None
            if tid is not None:
                for i, t in enumerate(state.tasks):
                    if t.get('id') == tid:
                        removed = state.tasks.pop(i)
                        break
        elif title:
            for i, t in enumerate(state.tasks):
                if title.lower() == t.get('title','').lower():
                    removed = state.tasks.pop(i)
                    break

    if removed:
        try:
            save_state_to_disk()
        except Exception:
            pass
        return jsonify({'success': True, 'removed': removed})
    return jsonify({'success': False, 'error': 'task not found'}), 404


@app.route('/api/task/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    removed = None
    with state.data_lock:
        for i, t in enumerate(state.tasks):
            if t.get('id') == task_id:
                removed = state.tasks.pop(i)
                break
    if removed:
        try:
            save_state_to_disk()
        except Exception:
            pass
        return jsonify({'success': True, 'removed': removed})
    return jsonify({'success': False, 'error': 'task not found'}), 404


@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    # Accept a single file in 'file' field and save it for inspection
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'no file provided'}), 400
    f = request.files['file']
    uploads = pathlib.Path(__file__).parent / 'uploads'
    uploads.mkdir(exist_ok=True)
    filename = f.filename or f'audio_{int(time.time())}.webm'
    dest = uploads / filename
    f.save(str(dest))
    return jsonify({'success': True, 'path': str(dest), 'size': dest.stat().st_size})

def _state_file_path():
    return pathlib.Path(__file__).parent / 'focusflow_state.json'

def load_state_from_disk():
    p = _state_file_path()
    if not p.exists():
        return
    try:
        with p.open('r', encoding='utf-8') as f:
            data = json.load(f)
        with state.data_lock:
            state.settings.update(data.get('settings', {}))
            # restore tasks if present
            # Do NOT restore tasks on startup — reset tasks each run per user preference
            # state.tasks = data.get('tasks', state.tasks)
            state.analytics.update(data.get('analytics', {}))
            fh = data.get('focus_history')
            if isinstance(fh, dict):
                state.focus_history = {int(k): v for k, v in fh.items()}
    except Exception as e:
        print('Failed to load state from disk:', e)

def save_state_to_disk():
    p = _state_file_path()
    try:
        data = {
            # Persist settings and analytics, but do NOT persist tasks so app resets each run
            'settings': state.settings,
            'analytics': state.analytics,
            'focus_history': state.focus_history
        }
        with p.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print('Failed to save state to disk:', e)

# Load persisted state if present
load_state_from_disk()

# Start camera thread only if CV libs available
if HAVE_CV:
    threading.Thread(target=camera_loop, daemon=True).start()


# Periodic dopamine suggestion thread for ADHD-friendly boosts
def dopamine_suggester():
    suggestions = [
        "Take a 2-minute dance break.",
        "Do 5 jumping jacks — quick energy boost.",
        "Drink a glass of water and stretch.",
        "Stand up and walk to the window for 30 seconds.",
        "Give your pet or plant some attention for a minute."
    ]
    import random
    last_suggest = 0
    while True:
        try:
            time.sleep(300)  # check every 5 minutes
            with state.data_lock:
                now = time.time()
                if now - last_suggest < 300:
                    continue
                # Suggest if focus is low or many distractions
                if state.focus_score < 60 or state.analytics.get('distractions', 0) > 2:
                    msg = random.choice(suggestions)
                    state.buddy_messages_queue.append(msg)
                    last_suggest = now
        except Exception:
            pass

threading.Thread(target=dopamine_suggester, daemon=True).start()

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)

