from tracking_frame import VehicleTrackingFrame

class Vehicle:
    def __init__(self, track_id, type):
        self.track_id = track_id
        self.type = type
        self.track_history : list[VehicleTrackingFrame] = []

    def add_track_history(self, tracking_frame: VehicleTrackingFrame):
        self.track_history.append(tracking_frame)

    def find_best_license_plate(self):
        return self._get_track_history_with_confidence_threshold(0.5)

    def _get_track_history_with_confidence_threshold(self, threshold):
        return [frame for frame in self.track_history if frame.confidence >= threshold]
    

    