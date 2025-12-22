FocusFlow: Assistant AI Study CompanionFocusFlow is an assistive AI study companion specifically designed to help neurodivergent students, particularly those with ADHD, manage their focus and academic tasks1111. By utilizing computer vision and multimodal interaction, the system provides a "sense of presence" and immediate feedback to keep students engaged

🚀 Key Features
🧠 Intelligent Monitoring
Gaze & Posture Tracking: Uses OpenCV and MediaPipe to detect head tilt, eye movement, and focus loss
Real-time Focus Scoring: Provides a live "Focus Bar" (0-100%) indicating the current level of attention.
Proactive Interventions: The AI Buddy "intervenes" with voice nudges before a student becomes fully distracted.


🗣️ Multimodal Interaction
3 Inputs:
Visual: Computer vision tracking gaze and posture.
Aural: Voice commands and wake-word activation ("Hey Focus").
Textual: Manual task entry and chat-based communication with the Buddy.
3 Outputs:
Voice Interactions: Chatbot responses and verbal confirmations of commands.
Acoustic Notifications: Ear-friendly sine-wave sound effects for alerts.
Visual Analytics: Dynamic charts, bio-rhythm timelines, and focus rhythm graphs.


🛠️ ADHD-Specific Tools
Dynamic Task Chunking: Automatically breaks large tasks (like "Write Essay") into smaller, manageable micro-tasks.
Dopamine Menu: Suggests quick physical or sensory activities (e.g., jumping jacks, stretching) when focus is low.
Bio-Rhythm Scheduling: Identifies "Peak Focus Hours" and recommends scheduling the hardest tasks during those windows.


🛠️ Tech Stack
Backend: Python (Flask)
Computer Vision: OpenCV & MediaPipe
Frontend: HTML5, CSS3, JavaScript (Chart.js)
Voice Engine: Web Speech API
Persistence: Local JSON State (Privacy-First)


🛡️ Privacy & Accessibility
Local Edge Processing: Camera and microphone feeds are processed on-device and never sent to the cloud.
Accessibility Modes: Includes an OpenDyslexic font toggle, high-contrast settings, and color-blind friendly palettes (Deuteranopia, Protanopia, Tritanopia).


💻 Installation & Setup
Make virtual Environment:
python -m venv activity

Activate Virtual Environment:
.\venv\Scripts\Activate

Install Dependencies:
pip install flask opencv-python mediapipe numpy
pip install -r requirements.txt

Run the Application:
python app.py

Access the Dashboard: Open your browser to http://127.0.0.1:5000.

🎮 Voice Commands Reference
Camera: "Turn on camera" / "Disable camera".
Tasks: "Add task [name] with high difficulty".
Navigation: "Go to analytics" / "Open schedule".
Settings: "Enable dark mode" / "Turn on dyslexia font".

👥 TeamCourse: HCI Project 
Instructor: Dr. Farzana Jabeen 
Members: Muhammad Abdullah Waqar & Uswa Khan 
