# Architecture

This document describes the current FocusFlow layout and runtime flow.

## High-Level View
- Single Flask app (`app.py`) serving a one-page UI (`templates/index.html`).
- Optional computer-vision stack (OpenCV + MediaPipe) estimates head pose and feeds focus scores to the UI.
- Background threads handle camera capture and periodic dopamine nudges; REST endpoints handle UI/voice interactions.
- Lightweight persistence via JSON file (`focusflow_state.json`) for settings/analytics.

## Core Modules (all in `app.py`)
- **FocusPredictor**: converts MediaPipe head-yaw deviation into a normalized focus ratio for ADHD-friendly thresholds.
- **FocusFlowState**: central in-memory store (focus scores, tasks, analytics, settings, buddy queue, history); guarded by a lock.
- **Camera Loop**: `camera_loop()` reads webcam frames, runs MediaPipe Pose, updates focus metrics/analytics, and publishes MJPEG frames to `/video_feed`.
- **Buddy & Voice Engine**: `trigger_buddy_intervention()` + `/api/voice_command` parse natural commands to mutate state (camera/mic toggles, tasks, schedules, accessibility settings, navigation) and enqueue buddy messages.
- **Tasks & Scheduling**: `/api/add_task`, `/api/toggle_subtask`, `/api/remove_task`, `/api/schedule` manage tasks, subtasks, timers, and generate daily schedules.
- **Analytics & Insights**: `/api/analytics` and `/api/ai_insight` expose live stats (deep focus, distractions, consistency, peak window) derived from session history.
- **Settings & Persistence**: `/api/settings` plus helpers `load_state_from_disk()` / `save_state_to_disk()` persist settings/analytics to `focusflow_state.json`; tasks reset each run by design.
- **Wellness Thread**: `dopamine_suggester()` runs in the background to push break suggestions when focus drops.

## Frontend (`templates/index.html`)
- Single-page layout with dashboard, tasks/schedule, analytics, settings, and buddy/chat panel.
- Fetches JSON from the APIs above, renders charts/components, and streams the MJPEG feed from `/video_feed` when camera is active.
- Uses Web Speech API (client-side) for speech-to-text; sends transcripts to `/api/voice_command` for intent handling.

## Data Flow
1) Browser loads `index.html` → polls `/api/state` and streams `/video_feed` when camera is on.
2) Background `camera_loop()` updates `FocusFlowState` with focus scores and analytics; state is serialized in responses.
3) Voice or UI actions hit REST endpoints to mutate state (tasks, settings, schedules); buddy actions are returned for UI to display.
4) `save_state_to_disk()` writes settings/analytics to `focusflow_state.json` periodically; reloaded on startup.

## Deployment Notes
- Default port is 5000 (`app.run(..., port=5000)`); change in code if needed.
- If OpenCV/MediaPipe are unavailable, the app skips the camera loop but the UI and APIs remain usable.
