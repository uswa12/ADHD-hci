# voice_listener.py
import speech_recognition as sr
import time

# --- GLOBAL STATE ---
last_command_global = None
is_active = False # Controlled by main_app

def get_last_command():
    return last_command_global

def clear_last_command():
    global last_command_global
    last_command_global = None

def start_listening():
    """Listens for manual override commands ("Break," "Pause") in a loop."""
    global last_command_global
    r = sr.Recognizer()
    
    # Use index 0 for default microphone (adjust if needed)
    try:
        mic = sr.Microphone()
        print("Voice Listener: Microphone initialized.")
    except Exception as e:
        print(f"Voice Listener Error: Could not initialize microphone. Check PyAudio/system settings. {e}")
        return

    while is_active:
        try:
            with mic as source:
                # Adjust for ambient noise to improve accuracy
                r.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for up to 5 seconds for a command
                audio = r.listen(source, timeout=5, phrase_time_limit=3) 
            
            # Recognition happens here
            command = r.recognize_google(audio).lower() 
            
            if "break" in command or "pause" in command or "stop" in command:
                last_command_global = "break"
                
        except sr.WaitTimeoutError:
            # This is normal; no one spoke during the listening window
            pass
        except sr.UnknownValueError:
            # User spoke, but recognition failed (muffled, non-speech)
            pass
        except Exception as e:
            # Other errors (network, recognition server issue)
            print(f"Voice listener runtime error: {e}") 
        
        time.sleep(0.5) # Prevent excessive CPU usage