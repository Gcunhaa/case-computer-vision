class VehicleTrackingFrame:
    def __init__(self, frame_number, confidence, bounding_box):
        self.frame_number = frame_number
        self.confidence = confidence
        self.bounding_box = bounding_box

    def detect_license_plate(self, frame):
        return self.bounding_box
