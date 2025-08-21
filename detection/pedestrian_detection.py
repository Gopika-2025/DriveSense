import cv2
import numpy as np
import os
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'models', 'yolov8n.pt')
model = YOLO(model_path)

def detect_pedestrians(frame, conf_threshold=0.5):
    """
    Runs YOLOv8 detection on a single frame and draws bounding boxes.
    """
    results = model(frame, conf=conf_threshold, classes=[0], verbose=False)

    annotated_frame = frame.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 2)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Person: {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return annotated_frame