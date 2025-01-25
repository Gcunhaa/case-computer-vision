from .tracking_frame import VehicleTrackingFrame

class Vehicle:
    def __init__(self, track_id, type):
        self.track_id = track_id
        self.type = type
        self.track_history : list[VehicleTrackingFrame] = []

    def add_track_history(self, tracking_frame: VehicleTrackingFrame):
        self.track_history.append(tracking_frame)

    def find_best_license_plate_track_history(self):
        # Filter valid tracks first
        valid_tracks = [
            frame for frame in self.track_history 
            if frame.license_plate is not None 
        ]
        
        if not valid_tracks:
            return None
            
        # Get all box sizes for normalization
        box_sizes = [frame.license_plate.get_box_size() for frame in valid_tracks]
        min_size = min(box_sizes)
        size_range = max(box_sizes) - min_size
        
        # Calculate weighted scores (60% box size, 40% confidence)
        def calculate_score(frame):
            normalized_size = (frame.license_plate.get_box_size() - min_size) / size_range if size_range > 0 else 1
            return 0.6 * normalized_size + 0.4 * float(frame.license_plate.confidence)
        return max(valid_tracks, key=calculate_score)
    
    def get_license_plate_text(self):
        best_license_plate = self.find_best_license_plate()
        if best_license_plate is not None:
            return best_license_plate.extract_text()
        return None
