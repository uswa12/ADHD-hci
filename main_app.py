from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import time, threading, subprocess, sys

import gesture_detector
import voice_listener
from core_intelligence.adaptive_manager import AdaptiveManager
from core_intelligence.focus_predictor import FocusPredictor
from core_intelligence.task_manager import TaskManager

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

USER_LANGUAGE = "english"

current_focus_level = 1.0
previous_focus_level = 1.0
focus_loss_count = 0
is_study_session_active = False
SESSION_START_TIME = time.time()
LAST_INTERVENTION_TIME = time.time()

break_active = False
break_end_time = None
break_duration = 0

manager = None
predictor = FocusPredictor()
task_manager = TaskManager()
current_task_info = ""

def speak_message_windows(message):
    try:
        safe = message.replace("'", "").replace('"', "")
        subprocess.Popen([
            "powershell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"
        ])
    except Exception as e:
        print("TTS ERROR:", e, file=sys.stderr)

def update_focus_level():
    global current_focus_level, previous_focus_level, focus_loss_count
    fidget = gesture_detector.get_fidget_score()
    predicted = predictor.calculate_focus(fidget)
    previous_focus_level = current_focus_level
    current_focus_level = 0.8 * current_focus_level + 0.2 * predicted

    if previous_focus_level >= 0.5 and current_focus_level < 0.5:
        focus_loss_count += 1

    if current_focus_level < 0.28 and previous_focus_level >= 0.28:
        msg = ("You are losing focus extremely! Take a short break now."
               if USER_LANGUAGE == "english" else
               "Apki tawajjo bohat zyada kam ho gayi hai! Filhal chhoti break lein.")
        speak_message_windows(msg)
        print("[ALERT]", msg)

    return current_focus_level, fidget

def check_for_intervention():
    global LAST_INTERVENTION_TIME
    if not is_study_session_active or manager is None:
        return
    if time.time() - LAST_INTERVENTION_TIME < manager.MIN_INTERVAL:
        return
    if current_focus_level < manager.LOAD_THRESHOLD:
        msg = manager.get_intervene_message("low_focus", current_focus_level)
        speak_message_windows(msg)
        LAST_INTERVENTION_TIME = time.time()
    command = voice_listener.get_last_command()
    if command == "break":
        msg = manager.get_intervene_message("manual_break", current_focus_level)
        speak_message_windows(msg)
        voice_listener.clear_last_command()
        LAST_INTERVENTION_TIME = time.time()

@socketio.on("request_tasks")
def handle_request_tasks(data):
    title = data.get("title", "Study Session")
    tasks = task_manager.set_task(title)
    emit("task_breakdown", {"tasks": tasks})

@socketio.on("start_session")
def handle_start_session(data):
    global is_study_session_active, USER_LANGUAGE, manager
    global SESSION_START_TIME, focus_loss_count, current_task_info
    if is_study_session_active: return

    USER_LANGUAGE = data.get("language", "english")
    task_id = data.get("task_id", 1)

    focus_loss_count = 0
    task_manager.task_pointer = task_id - 1
    current_task_info = task_manager.get_current_task_info()

    manager = AdaptiveManager(socketio_ref=socketio, language=USER_LANGUAGE)
    start_msg = (f"Session started. Your task is {current_task_info}" 
                 if USER_LANGUAGE=="english" else 
                 f"Session shuru ho raha hai. Task hai {current_task_info}")
    speak_message_windows(start_msg)

    gesture_detector.is_active = True
    voice_listener.is_active = True
    threading.Thread(target=gesture_detector.start_camera_stream, daemon=True).start()
    threading.Thread(target=voice_listener.start_listening, daemon=True).start()

    is_study_session_active = True
    SESSION_START_TIME = time.time()
    socketio.start_background_task(main_loop)

@socketio.on("stop_session")
def handle_stop_session():
    global is_study_session_active
    if not is_study_session_active: return

    final_score = f"{current_focus_level:.2f}"
    stop_msg = (f"Session ended. Final score is {final_score}" 
                if USER_LANGUAGE=="english" else 
                f"Session khatam ho gaya. Score hai {final_score}")
    speak_message_windows(stop_msg)

    is_study_session_active = False
    gesture_detector.is_active = False
    voice_listener.is_active = False

    emit("score_update", {
        "focus": final_score,
        "fidget": "0.00",
        "elapsed_time": "00:00",
        "status": "Session Ended",
        "task": "",
        "focus_lost_count": focus_loss_count
    })

@socketio.on("start_break")
def handle_start_break(data):
    global break_active, break_end_time, break_duration, SESSION_START_TIME
    if not is_study_session_active or break_active: return

    duration = int(data.get("seconds", 60))
    break_duration = duration
    break_end_time = time.time() + duration
    break_active = True

    msg = (f"Your break started. Take {duration} seconds." 
           if USER_LANGUAGE=="english" else
           f"Aap ka break shuru ho gaya hai. {duration} seconds ka break lein.")
    speak_message_windows(msg)
    print(f"[BREAK START] {duration}s")

def main_loop():
    global break_active, SESSION_START_TIME
    while is_study_session_active:
        try:
            if break_active:
                remaining = int(break_end_time - time.time())
                if remaining <= 0:
                    break_active = False
                    SESSION_START_TIME += break_duration
                    msg = ("Break is over. Session resumed." 
                           if USER_LANGUAGE=="english" else
                           "Break khatam ho gaya hai. Ab session wapas shuru.")
                    speak_message_windows(msg)
                    print("[BREAK END] Session resumed")
                else:
                    socketio.emit("score_update", {
                        "focus": f"{current_focus_level:.2f}",
                        "fidget": "0.00",
                        "elapsed_time": f"Break: {remaining}s",
                        "status": "On Break",
                        "task": current_task_info
                    })
                    socketio.sleep(1)
                    continue

            focus, fidget = update_focus_level()
            check_for_intervention()
            elapsed = int(time.time() - SESSION_START_TIME)
            time_str = f"{elapsed//60:02}:{elapsed%60:02}"
            status = "Focused" if focus>=0.7 else "High Risk" if focus<0.4 else "Losing Focus"

            socketio.emit("score_update", {
                "focus": f"{focus:.2f}",
                "fidget": f"{fidget:.2f}",
                "elapsed_time": time_str,
                "status": status,
                "task": current_task_info
            })
            socketio.sleep(1)
        except Exception as e:
            print("MAIN LOOP ERROR:", e)
            break

@app.route("/")
def index():
    return render_template("study_dashboard.html")

@app.route("/video_feed")
def video_feed():
    def gen():
        while is_study_session_active:
            frame = gesture_detector.get_jpeg_frame()
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__=="__main__":
    print("🚀 ADHD Study Buddy Running")
    socketio.run(app, debug=False)
