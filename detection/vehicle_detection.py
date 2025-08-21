import cv2
import numpy as np
import os
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'models', 'yolov8n.pt')
model = YOLO(model_path)

def detect_vehicles(frame, conf_threshold=0.5):
    """
    Runs YOLOv8 detection for vehicles on a single frame.
    """
    vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck
    results = model(frame, conf=conf_threshold, classes=vehicle_classes, verbose=False)

    annotated_frame = frame.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 2)
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"{class_name}: {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return annotated_frame