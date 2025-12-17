import os
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
from flask import Flask, render_template, jsonify, request, Response
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
import speech_recognition as sr
from collections import deque
import subprocess
import platform

app = Flask(__name__)

# Global state
class FocusFlowState:
    def __init__(self):
        self.camera_active = False
        self.microphone_active = False
        self.focus_score = 82
        self.tasks = [
            {
                "id": 1,
                "title": "Review Math Chapter 4",
                "subtasks": [
                    {"id": 1, "text": "Read introduction", "completed": False, "duration": 10},
                    {"id": 2, "text": "Solve practice problems 1-5", "completed": False, "duration": 30}
                ],
                "completed": 0,
                "total": 2
            },
            {
                "id": 2,
                "title": "Complete HCI Project Proposal",
                "subtasks": [
                    {"id": 1, "text": "Write Abstract", "completed": True, "duration": 15},
                    {"id": 2, "text": "Define User Personas", "completed": False, "duration": 20},
                    {"id": 3, "text": "Draft Methodology", "completed": False, "duration": 25}
                ],
                "completed": 1,
                "total": 3
            }
        ]
        self.schedule = []
        self.bio_rhythm = {"peak_start": "10:00 AM", "peak_end": "01:00 PM"}
        self.analytics = {
            "deep_focus_score": 78,
            "total_focus_time": "4h 12m",
            "consistency": 85,
            "distractions": 12,
            "focus_rhythm": [],
            "distraction_sources": {"Phone": 45, "Noise": 25, "Daydream": 60, "Fidgeting": 30}
        }
        self.settings = {
            "buddy_persona": "Friendly Peer",
            "language": "English",
            "proactive_interventions": True,
            "dyslexia_font": False,
            "reduced_motion": False,
            "color_contrast": "Standard"
        }
        self.gaze_data = deque(maxlen=100)
        self.posture_data = deque(maxlen=100)
        self.distraction_count = 0
        self.current_session_start = None
        self.focus_history = []
        
state = FocusFlowState()

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Speech setup
recognizer = sr.Recognizer()

# TTS setup - Use system TTS on macOS, fallback for other systems
def init_tts():
    """Initialize text-to-speech engine based on platform"""
    system = platform.system()
    if system == "Darwin":  # macOS
        return None  # Will use 'say' command
    else:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            return engine
        except:
            print("Warning: TTS not available")
            return None

tts_engine = init_tts()

# Voice commands queue
voice_commands_queue = []
buddy_messages_queue = []

def calculate_gaze_focus(landmarks):
    """Calculate focus score based on eye gaze direction"""
    if not landmarks:
        return 50
    
    # Simple heuristic: check if eyes are looking forward
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]
    
    # Calculate if gaze is centered (simplified)
    eye_center_x = (left_eye.x + right_eye.x) / 2
    deviation = abs(eye_center_x - nose.x)
    
    focus_score = max(0, 100 - (deviation * 1000))
    return min(100, focus_score)

def calculate_posture_score(landmarks):
    """Calculate posture score"""
    if not landmarks:
        return 50
    
    # Check shoulder and neck alignment
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    nose = landmarks[0]
    
    # Calculate shoulder balance
    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
    posture_score = max(0, 100 - (shoulder_diff * 500))
    
    return min(100, posture_score)

def generate_frames():
    """Generate video frames with face/pose detection"""
    camera = cv2.VideoCapture(0)
    
    while state.camera_active:
        success, frame = camera.read()
        if not success:
            break
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection for gaze tracking
        face_results = face_mesh.process(rgb_frame)
        pose_results = pose.process(rgb_frame)
        
        focus_score = 50
        posture_score = 50
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                focus_score = calculate_gaze_focus(face_landmarks.landmark)
                state.gaze_data.append(focus_score)
        
        if pose_results.pose_landmarks:
            posture_score = calculate_posture_score(pose_results.pose_landmarks.landmark)
            state.posture_data.append(posture_score)
        
        # Update overall focus score
        if len(state.gaze_data) > 0:
            avg_gaze = sum(state.gaze_data) / len(state.gaze_data)
            avg_posture = sum(state.posture_data) / len(state.posture_data) if len(state.posture_data) > 0 else 50
            state.focus_score = int((avg_gaze * 0.7 + avg_posture * 0.3))
            
            # Trigger intervention if focus drops
            if state.focus_score < 60 and state.settings['proactive_interventions']:
                trigger_buddy_intervention()
        
        # Draw on frame
        cv2.putText(frame, f"Focus: {state.focus_score}%", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

def trigger_buddy_intervention():
    """Trigger AI buddy to speak when focus drops"""
    messages = [
        "Hi Abdullah! I noticed you've been staring at the wall. Let's finish this paragraph together?",
        "Abdullah, you seem a bit distracted. Should we take a quick 2-minute break?",
        "Hey! Your focus is slipping. Let's tackle the next small task.",
        "I see you're struggling. Want to switch to an easier task for now?"
    ]
    
    if len(buddy_messages_queue) == 0:
        import random
        message = random.choice(messages)
        buddy_messages_queue.append(message)
        
        # Speak the message
        threading.Thread(target=speak_message, args=(message,)).start()

def speak_message(message):
    """Text-to-speech for buddy messages - macOS compatible"""
    try:
        if platform.system() == "Darwin":  # macOS
            # Use native macOS 'say' command
            subprocess.run(['say', message], check=False)
        elif tts_engine:
            tts_engine.say(message)
            tts_engine.runAndWait()
        else:
            print(f"TTS: {message}")
    except Exception as e:
        print(f"TTS error: {e}")

def listen_for_voice_commands():
    """Continuously listen for voice commands"""
    while state.microphone_active:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                
                text = recognizer.recognize_google(audio).lower()
                voice_commands_queue.append(text)
                
                # Process commands
                process_voice_command(text)
                
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except Exception as e:
            print(f"Voice recognition error: {e}")
            time.sleep(1)

def process_voice_command(command):
    """Process voice commands"""
    if "take a break" in command or "break" in command:
        buddy_messages_queue.append("Sure! Taking a 5-minute break. Stretch and hydrate!")
        speak_message("Sure! Taking a 5-minute break.")
    
    elif "how am i doing" in command or "progress" in command:
        msg = f"You're doing great! Your focus score is {state.focus_score}%."
        buddy_messages_queue.append(msg)
        speak_message(msg)
    
    elif "complete task" in command or "finish task" in command:
        msg = "Task marked as complete! Great job!"
        buddy_messages_queue.append(msg)
        speak_message(msg)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    """Get current application state"""
    return jsonify({
        'focus_score': state.focus_score,
        'camera_active': state.camera_active,
        'microphone_active': state.microphone_active,
        'tasks': state.tasks,
        'analytics': state.analytics,
        'settings': state.settings,
        'bio_rhythm': state.bio_rhythm
    })

@app.route('/api/tasks', methods=['GET', 'POST', 'PUT'])
def tasks():
    if request.method == 'GET':
        return jsonify(state.tasks)
    
    elif request.method == 'POST':
        new_task = request.json
        new_task['id'] = len(state.tasks) + 1
        state.tasks.append(new_task)
        return jsonify(new_task)
    
    elif request.method == 'PUT':
        task_id = request.json.get('id')
        for task in state.tasks:
            if task['id'] == task_id:
                task.update(request.json)
                return jsonify(task)
        return jsonify({'error': 'Task not found'}), 404

@app.route('/api/toggle_camera', methods=['POST'])
def toggle_camera():
    state.camera_active = not state.camera_active
    if state.camera_active:
        state.current_session_start = datetime.now()
    return jsonify({'camera_active': state.camera_active})

@app.route('/api/toggle_microphone', methods=['POST'])
def toggle_microphone():
    state.microphone_active = not state.microphone_active
    
    if state.microphone_active:
        # Start voice recognition thread
        threading.Thread(target=listen_for_voice_commands, daemon=True).start()
    
    return jsonify({'microphone_active': state.microphone_active})

@app.route('/api/buddy_messages')
def buddy_messages():
    """Get recent buddy messages"""
    messages = buddy_messages_queue.copy()
    buddy_messages_queue.clear()
    return jsonify(messages)

@app.route('/api/schedule')
def get_schedule():
    """Generate optimized schedule based on bio-rhythm"""
    schedule = []
    current_time = datetime.now().replace(hour=9, minute=0, second=0)
    
    # Sort tasks by difficulty/duration
    all_subtasks = []
    for task in state.tasks:
        for subtask in task['subtasks']:
            if not subtask['completed']:
                all_subtasks.append({
                    'task_title': task['title'],
                    'subtask': subtask['text'],
                    'duration': subtask['duration']
                })
    
    # Schedule hard tasks during peak hours
    for i, subtask in enumerate(all_subtasks):
        schedule.append({
            'time': current_time.strftime('%I:%M %p'),
            'duration': subtask['duration'],
            'title': subtask['subtask'],
            'energy': 'High' if 10 <= current_time.hour <= 13 else 'Medium',
            'type': 'Deep Focus' if subtask['duration'] > 20 else 'Task'
        })
        
        current_time += timedelta(minutes=subtask['duration'])
        
        # Add breaks
        if i < len(all_subtasks) - 1:
            schedule.append({
                'time': current_time.strftime('%I:%M %p'),
                'duration': 15,
                'title': 'Break: Stretch & Hydrate',
                'energy': 'Low',
                'type': 'Break'
            })
            current_time += timedelta(minutes=15)
    
    return jsonify(schedule)

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data"""
    # Generate focus rhythm data
    focus_rhythm = []
    for hour in range(9, 16):
        focus_rhythm.append({
            'time': f"{hour}am" if hour < 12 else f"{hour-12}pm",
            'value': np.random.randint(50, 100)
        })
    
    state.analytics['focus_rhythm'] = focus_rhythm
    return jsonify(state.analytics)

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'GET':
        return jsonify(state.settings)
    else:
        state.settings.update(request.json)
        return jsonify(state.settings)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)