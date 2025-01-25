from .license_plate import LicencePlate


class VehicleTrackingFrame:
    def __init__(self, frame_number, confidence, bounding_box, license_plate : LicencePlate = None):
        self.frame_number = frame_number
        self.confidence = confidence
        self.bounding_box = bounding_box
        self.license_plate = license_plate

    def detect_license_plate(self, frame):
        return self.bounding_box
