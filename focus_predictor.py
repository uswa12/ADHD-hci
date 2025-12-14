# core_intelligence/focus_predictor.py

class FocusPredictor:
    """
    Predictive Intelligence: Converts multimodal input (fidgeting) into a focus score (0.0 to 1.0).
    A high fidget score PREDICTS low focus/distraction.
    """
    
    def __init__(self):
        # These thresholds define when fidgeting is considered low vs high
        self.FIDGET_LOW = 0.15 
        self.FIDGET_HIGH = 0.50

    def calculate_focus(self, fidget_score: float) -> float:
        """Calculates predicted focus level."""
        
        # Fully focused state
        if fidget_score <= self.FIDGET_LOW:
            return 1.0
            
        # Highly distracted/overstimulated state
        elif fidget_score >= self.FIDGET_HIGH:
            return 0.1
            
        else:
            # Linear scale: Focus drops as fidgeting increases between thresholds
            scale = (fidget_score - self.FIDGET_LOW) / (self.FIDGET_HIGH - self.FIDGET_LOW)
            return 1.0 - (0.9 * scale)