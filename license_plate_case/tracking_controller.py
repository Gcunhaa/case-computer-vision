from collections import defaultdict
from ultralytics import YOLO
import cv2
import math
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import os

from .license_plate import LicencePlate
from .tracking_frame import VehicleTrackingFrame
from .vehicle import Vehicle

class TrackingController:
    def __init__(self):
        # Vehicle classes in COCO dataset that YOLO uses:
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7]
        
        # Get the absolute path to the models directory
        package_dir = Path(__file__).parent.absolute()
        models_dir = package_dir / "models"
        
        # Load models using absolute paths
        self.tracking_model = YOLO(str(models_dir / "yolo11n.pt"))
        self.tracking_model.fuse()
        self.license_plate_model = YOLO(str(models_dir / "best.pt"))
        self.tracked_vehicles : list[Vehicle] = []
        
    def process_frame(self, frame, frame_number):
        
        tracking_results = self.tracking_model.track(frame, tracker="bytetrack.yaml", verbose=False)
        if tracking_results is None and tracking_results[0] is None and tracking_results[0].boxes is None and tracking_results[0].boxes.id is None:
            return
        
        # Obtain bounding boxes (xywh format) of detected objects
        boxes = tracking_results[0].boxes.xywh.cpu()
        # Extract confidence scores for each detected object
        conf_list = tracking_results[0].boxes.conf.cpu()
        # Get unique IDs assigned to each tracked object
        track_ids = tracking_results[0].boxes.id.int().cpu().tolist()
        # Obtain the class labels (e.g., 'car', 'truck') for detected objects
        clss = tracking_results[0].boxes.cls.cpu().tolist()
        # Retrieve the names of the detected objects based on class labels
        names = tracking_results[0].names
        license_plate_results = self.license_plate_model(frame, verbose=False)
        license_plate_results = license_plate_results[0]
        license_plate_boxes = license_plate_results.boxes.xyxy.cpu()
        
        for box, track_id, cls, conf in zip(boxes, track_ids, clss, conf_list):
            if cls not in self.vehicle_classes:
                continue
            
            vehicle = self._get_vehicle_by_track_id(track_id)
            if vehicle is None:
                vehicle = self._create_vehicle(track_id, names[cls])
            
            license_plate = None
            # Convert xywh format to xyxy for easier comparison
            x, y, w, h = box
            vehicle_box_xyxy = [x - w/2, y - h/2, x + w/2, y + h/2]

            #TODO: Refactor this to be more efficient
            # Check each license plate box
            license_plate_box = [license_plate_box for license_plate_box in license_plate_boxes if self._is_box_inside(license_plate_box, vehicle_box_xyxy)]
            if len(license_plate_box) > 0:
                license_plate_coords = license_plate_box[0].tolist()
                license_plate_index = license_plate_results.boxes.xyxy.tolist().index(license_plate_coords)
                license_plate_conf = license_plate_results.boxes.conf.tolist()[license_plate_index]
                license_plate = LicencePlate(license_plate_coords, license_plate_conf)
            vehicle.add_track_history(VehicleTrackingFrame(frame_number, conf, vehicle_box_xyxy, license_plate))
        return tracking_results
    

    def process_video(self, video_path, frame_breakpoint:int=None):
        """Process a video file to detect and recognize license plates.

        Args:
            input_path (str): Path to the input file. Supports common video formats (mp4, avi) 
                and image formats (jpg, png).
            frame_breakpoint (int): Maximum number of frames to process. 
                If None, processes the entire video. Defaults to None.

        Returns:
            None: Prints detected license plates for each tracked vehicle.

        Example:
            To process a video:
            $ license-plate-case path/to/video.mp4

            To process only first 100 frames:
            $ license-plate-case path/to/video.mp4 --frame_breakpoint=100
        """
        # Open the video file
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        
        # Get total frames for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if frame_breakpoint is None else frame_breakpoint
        
        # Create progress bar
        with tqdm(total=total_frames, desc="Processing video", unit="frames") as pbar:
            frame_number = 0
            while cap.isOpened():
                success, frame = cap.read()
                
                if success:
                    self.process_frame(frame, frame_number)
                    frame_number += 1
                    pbar.update(1)  # Update progress bar
                    
                    if frame_breakpoint is not None and frame_number == frame_breakpoint:
                        break
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    break
            
        self.get_license_plates(cap)
        
        cap.release()
        cv2.destroyAllWindows()

    def _get_vehicle_by_track_id(self, track_id):
        for vehicle in self.tracked_vehicles:
            if vehicle.track_id == track_id:
                return vehicle
        return None
    
    def _create_vehicle(self, track_id, class_name):
        vehicle = Vehicle(track_id, class_name)
        self.tracked_vehicles.append(vehicle)
        return vehicle
    
    def _is_box_inside(self, inner_box, outer_box):
        """
        Check if inner_box is inside outer_box
        Both boxes should be in [x1, y1, x2, y2] format
        """
        return (outer_box[0] <= inner_box[0] and
                outer_box[1] <= inner_box[1] and
                outer_box[2] >= inner_box[2] and
                outer_box[3] >= inner_box[3])

    def get_license_plates(self, video_cap):
        """
        Loop through all tracked vehicles and find their best license plates
        """
        results = []
        # Create progress bar for processing license plates
        with tqdm(total=len(self.tracked_vehicles), desc="Processing license plates", unit="vehicles") as pbar:
            for vehicle in self.tracked_vehicles:
                track_history = vehicle.find_best_license_plate_track_history()
                
                if track_history is None:
                    results.append(f"No license plate found for vehicle {vehicle.type} {vehicle.track_id}")
                else:
                    is_image = self.video_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
                    success, frame = (None, None)
                    if is_image:
                        success, frame = (True, cv2.imread(self.video_path))
                    else:
                        video_cap.set(cv2.CAP_PROP_POS_FRAMES, track_history.frame_number)
                        success, frame = video_cap.read()
                    
                    if success:
                        plate_text = track_history.license_plate.process_license_plate(frame)
                        results.append(f"License plate found for vehicle {vehicle.type} {vehicle.track_id}: {plate_text}")
                
                pbar.update(1)
        
        # Print all results at the end
        print("\nLicense Plate Detection Results:")
        for result in results:
            print(result)

if __name__ == '__main__':
    tracking_controller = TrackingController()
    tracking_controller.process_video('datasets/mini.jpeg', 2000)