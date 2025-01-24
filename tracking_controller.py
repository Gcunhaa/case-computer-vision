from collections import defaultdict
from ultralytics import YOLO
import cv2
import math
import numpy as np
import json

from tracking_frame import VehicleTrackingFrame
from vehicle import Vehicle

class TrackingController:
    def __init__(self):
        # Vehicle classes in COCO dataset that YOLO uses:
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7]
        self.tracking_model = YOLO("yolo11n.pt")
        self.tracking_model.fuse()
        self.license_plate_model = YOLO("best.pt")
        self.tracked_vehicles : list[Vehicle] = []
        
    def process_frame(self, frame, frame_number):
        
        tracking_results = self.tracking_model.track(frame, show=True, tracker="bytetrack.yaml")
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
        license_plate_results = self.license_plate_model(frame)
        for box, track_id, cls, conf in zip(boxes, track_ids, clss, conf_list):
            if cls not in self.vehicle_classes:
                continue
            vehicle = self._get_vehicle_by_track_id(track_id)
            if vehicle is None:
                vehicle = self._create_vehicle(track_id, names[cls])
            vehicle.add_track_history(VehicleTrackingFrame(frame_number, conf, box))
        return tracking_results
    

    def process_video(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_number = 0
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            
            if success:
                # Run YOLOv8 inference on the frame
                self.process_frame(frame, frame_number)
                frame_number += 1
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the video is finished
                break
    
        # Release the video capture object and close the display window
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
    
    
    


tc = TrackingController()
tc.process_video('datasets/video-india.mp4')