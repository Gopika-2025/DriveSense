import cv2
import numpy as np
import os
from ultralytics import YOLO

# Get the absolute path to the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the YOLO model file
model_path = os.path.join(script_dir, '..', 'models', 'yolov8n.pt')
# Load the pre-trained YOLOv8 model
model = YOLO(model_path)

def detect_road_signs(frame, conf_threshold=0.5):
    """
    Runs YOLOv8 detection for road signs (specifically stop signs) on a single frame.
    
    Args:
        frame (np.array): The input image frame.
        conf_threshold (float): The minimum confidence score to display a detection.
        
    Returns:
        np.array: The frame with detected road signs and bounding boxes.
    """
    # The COCO class ID for a stop sign is 11. Other signs may not be in this dataset.
    road_sign_classes = [11]
    results = model(frame, conf=conf_threshold, classes=road_sign_classes, verbose=False)

    annotated_frame = frame.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates and explicitly cast to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 2)
            class_name = "stop sign"
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(annotated_frame, f"{class_name}: {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            
    return annotated_frame