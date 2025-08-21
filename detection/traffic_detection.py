import cv2
import numpy as np
import os
from ultralytics import YOLO

# Get the absolute path to the directory containing the current script (detection/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the YOLO model file.
# It assumes 'yolov8n.pt' is in the 'models' directory,
# which is one level up from the 'detection' directory.
model_path = os.path.join(script_dir, '..', 'models', 'yolov8n.pt')

# Load the pre-trained YOLOv8 model.
# This model will be loaded once when the module is imported.
try:
    model = YOLO(model_path)
except Exception as e:
    # Handle cases where the model file might not be found or is corrupted.
    # It's important to provide a clear message and set the model to None
    # so the detection function can gracefully handle the failure.
    print(f"Error loading YOLO model from {model_path}: {e}")
    model = None

def detect_traffic_lights(frame, conf_threshold=0.5):
    """
    Runs YOLOv8 detection specifically for traffic lights on a single frame.

    Args:
        frame (np.array): The input image frame (BGR format).
        conf_threshold (float): The minimum confidence score to display a detection.

    Returns:
        np.array: The frame with detected traffic lights and bounding boxes drawn.
                  Returns the original frame if the model failed to load or
                  if no detections are made.
    """
    # If the model failed to load during module import, return the original frame.
    if model is None:
        print("YOLO model not loaded. Cannot perform traffic light detection.")
        return frame

    # The COCO dataset class ID for a traffic light is 9.
    # We filter the detection results to only include this class.
    traffic_light_class_id = [9] 
    
    # Perform inference (detection) on the input frame.
    # 'conf' sets the confidence threshold for bounding box predictions.
    # 'classes' filters the results to only include specified class IDs.
    # 'verbose=False' suppresses logging messages from YOLO during inference.
    results = model(frame, conf=conf_threshold, classes=traffic_light_class_id, verbose=False)

    # Create a copy of the frame to draw on. This prevents modifying the original
    # input frame directly, which can be important for subsequent processing steps.
    annotated_frame = frame.copy()

    # Iterate through each set of detection results (usually one per inference call).
    for r in results:
        # 'r.boxes' contains all the bounding box detections for the current result.
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates (x1, y1, x2, y2) and cast them to integers.
            # box.xyxy[0] returns a tensor/array of [x1, y1, x2, y2] for the current box.
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Extract the confidence score and round it to two decimal places for display.
            conf = round(float(box.conf[0]), 2)
            
            # Define the class name for the label.
            class_name = "Traffic Light"
            
            # Draw the bounding box rectangle on the annotated frame.
            # Arguments: image, top-left corner (x1,y1), bottom-right corner (x2,y2),
            # color (BGR format, here Yellow), thickness of the line.
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow color

            # Prepare the text label to be displayed.
            label = f"{class_name}: {conf}"
            
            # Put the text label above the bounding box.
            # Arguments: image, text string, starting point (x,y), font type,
            # font scale, color (BGR), thickness of the text line.
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text
            
    return annotated_frame
