import os
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

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
mp_pose = mp.solutions.pose
# Using Pose tracker for stable ear and nose detection
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

def camera_loop():
    """Background thread for continuous processing."""
    if not HAVE_CV or cv2 is None:
        # Camera processing not available on this system
        print('Camera loop disabled — OpenCV/MediaPipe not available')
        return
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
    reply = "Sorry, I didn't understand. Could you repeat that?"
    action = {}

    if not transcript:
        return jsonify({'reply': reply, 'action': action})

    t = transcript.lower()
    with state.data_lock:
        # Camera commands
        if any(k in t for k in ['turn on camera', 'camera on', 'open camera', 'start camera']):
            state.camera_active = True
            reply = "Camera turned on."
            action = {'type': 'camera', 'value': True}
        elif any(k in t for k in ['turn off camera', 'camera off', 'close camera', 'stop camera']):
            state.camera_active = False
            reply = "Camera turned off."
            action = {'type': 'camera', 'value': False}
        # Remove / delete task via chat: e.g. "remove task 3" or "delete task homework" or "remove task called Paint" 
        elif re.search(r"\b(remove|delete)\b.*\b(task|todo)\b", t):
            # try to extract numeric id first
            id_match = re.search(r"(?:task\s*(?:number|#)?\s*(\d+)|#(\d+))", t)
            removed = None
            if id_match:
                tid = int(id_match.group(1) or id_match.group(2))
                for i,tsk in enumerate(state.tasks):
                    if tsk.get('id') == tid:
                        removed = state.tasks.pop(i)
                        break
            else:
                # try to parse title after the phrase
                m = re.search(r"(?:remove|delete)\s+(?:the\s+)?(?:task|todo)(?:\s+(?:called|named)\s+)?['\"]?(?P<title>[^'\"]+)['\"]?", transcript, re.I)
                title = None
                if m:
                    title = m.group('title').strip()
                else:
                    # fallback: take words after 'remove task'
                    parts = re.split(r"remove task|delete task", t, maxsplit=1)
                    if len(parts) > 1:
                        title = parts[1].strip()
                if title:
                    for i,tsk in enumerate(state.tasks):
                        if title.lower() in tsk.get('title','').lower():
                            removed = state.tasks.pop(i)
                            break
            if removed:
                save_state_to_disk()
                reply = f"Removed task: {removed.get('title')}"
                action = {'type': 'removed_task', 'task': removed}
            else:
                reply = "Could not find the task to remove. Please specify the task id or exact title."
        # Add task with optional difficulty/time: "add task <title> difficulty high time 20"
        elif 'add task' in t or t.startswith('create task') or t.startswith('new task'):
            # Try to parse difficulty and time (minutes)
            diff_match = re.search(r"difficulty\s+(low|medium|high)", t)
            time_match = re.search(r"(time|for)\s+(\d{1,3})\s*(minutes|minute|mins|min)", t)
            title = transcript
            for kw in ['add task', 'create task', 'new task']:
                if kw in t:
                    idx = t.find(kw)
                    candidate = transcript[idx+len(kw):].strip()
                    if candidate:
                        title = candidate
                    break
            # remove parsed phrases from title
            if diff_match:
                title = re.sub(diff_match.group(0), '', title, flags=re.I).strip()
            if time_match:
                title = re.sub(time_match.group(0), '', title, flags=re.I).strip()

            if not title:
                title = 'Quick task'

            difficulty = diff_match.group(1) if diff_match else 'medium'
            duration_minutes = int(time_match.group(2)) if time_match else None

            # create basic subtasks depending on difficulty
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
            # If user provided a time, start a reminder timer
            if new_task.get('duration_minutes'):
                start_task_timer(new_task['id'], new_task['duration_minutes'])
        # Add schedule: "add schedule <title> at 3pm for 30 minutes"
        elif 'add schedule' in t or t.startswith('create schedule'):
            time_match = re.search(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
            dur_match = re.search(r"for\s+(\d{1,3})\s*(minutes|minute|mins|min|hours|hour|hrs)", t)
            title = transcript
            for kw in ['add schedule', 'create schedule']:
                if kw in t:
                    idx = t.find(kw)
                    candidate = transcript[idx+len(kw):].strip()
                    if candidate:
                        title = candidate
                    break
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                ampm = (time_match.group(3) or '').lower()
                if ampm == 'pm' and hour < 12: hour += 12
                if ampm == 'am' and hour == 12: hour = 0
                sched_time = f"{hour:02d}:{minute:02d}"
            else:
                sched_time = None
            duration = None
            if dur_match:
                v = int(dur_match.group(1))
                unit = dur_match.group(2)
                if 'hour' in unit:
                    duration = v * 60
                else:
                    duration = v
            # Add to a simple schedule list (re-using tasks structure)
            sched_entry = {"time": sched_time or 'ASAP', "title": title, "duration": duration or 30}
            # For simplicity, append to tasks as a scheduled todo
            new_task = {
                "id": len(state.tasks) + 1,
                "title": title,
                "completed": 0,
                "total": 1,
                "subtasks": [{"id":1, "text": title, "completed": False, "duration": duration or 30}],
                "scheduled_time": sched_time,
                "duration_minutes": duration or 30
            }
            state.tasks.append(new_task)
            reply = f"Scheduled {title} at {sched_time or 'ASAP'} for {new_task['duration_minutes']} minutes."
            action = {'type': 'add_schedule', 'entry': sched_entry}
            start_task_timer(new_task['id'], new_task['duration_minutes'])

        # Suggest dopamine activity (require suggest/recommend near break/dopamine)
        elif re.search(r"\b(suggest|recommend|give me)\b.{0,40}\b(break|dopamine|pause)\b", t) or re.search(r"\bgive me a break\b", t):
            suggestions = ["Take a 2-minute dance break.", "Have a glass of water.", "Do 5 jumping jacks.", "Give your pet some love."]
            s = random.choice(suggestions)
            reply = f"Suggestion: {s}"
            action = {'type': 'suggest_dopamine', 'suggestion': s}

        # Query focus — require interrogative or 'tell' near 'focus' to avoid accidental triggers
        elif re.search(r"\b(tell me|what(?:'s| is)|how am i|how's|current|show me|do i have)\b.{0,40}\bfocus\b", t) or re.search(r"\bfocus\b.{0,40}\b(tell me|what(?:'s| is)|how am i|how's|current|show me)\b", t):
            reply = f"Your current focus is {state.focus_score} percent."
            action = {'type': 'report_focus', 'value': state.focus_score}
        # Toggle dyslexic font
        elif 'dyslexic' in t or 'dyslexia' in t:
            if 'enable' in t or 'turn on' in t or 'on' in t:
                state.settings['dyslexia_font'] = True
                reply = 'Dyslexic friendly font enabled.'
            else:
                state.settings['dyslexia_font'] = False
                reply = 'Dyslexic friendly font disabled.'
            action = {'type': 'setting', 'name': 'dyslexia_font', 'value': state.settings['dyslexia_font']}
        else:
            # generic echo
            reply = "I heard: '" + transcript + "'. I can open/close camera, add tasks, or report focus."

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

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)
