class AdaptiveManager:
    LOAD_THRESHOLD = 0.50

    def __init__(self, socketio_ref, language="english"):
        self.MIN_INTERVAL = 120
        self.socketio = socketio_ref
        self.language = language.lower()

        self.prompts = {
            "english": {
                "low_focus": "Your focus is dropping. Please take a short break and reset.",
                "manual_break": "Starting a break now. I will remind you shortly.",
                "extreme_focus": "Warning! You are losing focus extremely. Please stop and refocus immediately."
            },
            "urdu": {
                "low_focus": "Aap ki tawajjo kam ho rahi hai. Thora sa break lein.",
                "manual_break": "Break shuru kiya ja raha hai.",
                "extreme_focus": "Khabarrdaaar! Aap ki tawajjoo bohat zyada kam ho chuki hai. Foran tawajjoo wapas le aayen."
            }
        }

    def get_intervene_message(self, reason: str, score: float) -> str:
        return self.prompts.get(
            self.language,
            self.prompts["english"]
        ).get(reason, "")
