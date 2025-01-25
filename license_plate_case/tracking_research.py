from ultralytics import YOLO

model = YOLO("yolo/yolov8x.pt")
model.fuse()

CLASS_NAMES_DICT = model.model.names
CLASS_ID = [2, 3, 5, 7]
