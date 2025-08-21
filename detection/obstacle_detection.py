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

def detect_obstacles(frame, conf_threshold=0.5):
    """
    Runs YOLOv8 detection for multiple classes of obstacles on a single frame.
    
    Args:
        frame (np.array): The input image frame.
        conf_threshold (float): The minimum confidence score to display a detection.
        
    Returns:
        np.array: The frame with detected obstacles and bounding boxes.
    """
    # A list of COCO class IDs that represent potential obstacles
    # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
    # 9: traffic light, 11: stop sign, 13: bench, 14: bird, etc.
    obstacle_classes = [0, 2, 3, 5, 7, 9, 11]
    
    results = model(frame, conf=conf_threshold, classes=obstacle_classes, verbose=False)

    annotated_frame = frame.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates and explicitly cast to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 2)
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Use a red color for obstacles
            cv2.putText(annotated_frame, f"{class_name}: {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
    return annotated_frame