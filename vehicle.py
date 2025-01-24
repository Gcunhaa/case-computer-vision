from tracking_frame import VehicleTrackingFrame

class Vehicle:
    def __init__(self, track_id, type):
        self.track_id = track_id
        self.type = type
        self.track_history : list[VehicleTrackingFrame] = []

    def add_track_history(self, tracking_frame: VehicleTrackingFrame):
        self.track_history.append(tracking_frame)

    def find_best_license_plate_track_history(self):
        best_track_histories = [frame for frame in self.track_history if float(frame.confidence) >= 0.5 and frame.license_plate is not None and float(frame.license_plate.confidence) >= 0.4]
        best_track_history = max(best_track_histories, key=lambda x: x.license_plate.get_box_size()) if len(best_track_histories) > 0 else None
        
        return best_track_history
    
    def get_license_plate_text(self):
        best_license_plate = self.find_best_license_plate()
        if best_license_plate is not None:
            return best_license_plate.extract_text()
        return None
