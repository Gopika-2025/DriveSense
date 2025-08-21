import streamlit as st
import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO

# --- Python Path Debugging Info (will appear in Streamlit sidebar) ---
# This section helps in diagnosing ModuleNotFoundError by showing Python's
# search paths. You can remove it once your app runs successfully.
st.sidebar.subheader("Python Environment Debug Info")
st.sidebar.write(f"**Current Working Directory:** `{os.getcwd()}`")
st.sidebar.write("**Python Sys Path (where Python looks for modules):**")
for i, path in enumerate(sys.path):
    st.sidebar.markdown(f"- `{i}: {path}`")
st.sidebar.markdown("---")
# --- End Debugging Info ---

# --- Global YOLO Model Loading ---
# This model is loaded once at the start of the application.
# Get the absolute path to the directory containing this script.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the YOLO model file.
# It assumes 'yolov8n.pt' is in the 'models' directory,
# which is a sibling to the 'app.py' file.
model_path = os.path.join(script_dir, 'models', 'yolov8n.pt')

try:
    global_yolo_model = YOLO(model_path)
    st.sidebar.success(f"YOLO model loaded from: {model_path.replace(os.getcwd(), '.')}")
except Exception as e:
    st.sidebar.error(f"Error loading YOLO model from {model_path}: {e}")
    global_yolo_model = None # Set to None to prevent further errors if loading fails


# --- Lane Detection Functions (from lane_detection.py) ---
def draw_averaged_lines(img, lines, color):
    """Averages the lines and draws a single, smooth line."""
    if not lines:
        return

    lines = [line[0] for line in lines]
    x_coords = np.array([p for line in lines for p in (line[0], line[2])])
    y_coords = np.array([p for line in lines for p in (line[1], line[3])])
    
    # Check if there are enough points to fit a line
    if len(x_coords) > 1:
        # Fit a first-degree polynomial (a line) to the x and y coordinates.
        # We fit x as a function of y (x = ay + b) because lines might be vertical.
        poly_fit = np.polyfit(y_coords, x_coords, 1)
        fit_a = poly_fit[0] # Slope
        fit_b = poly_fit[1] # Y-intercept (actually X-intercept if x=ay+b)

        # Determine the y-range for drawing the line
        y_min = int(min(y_coords))
        y_max = int(img.shape[0]) # Draw to the bottom of the image
        
        # Calculate corresponding x-coordinates using the fitted line equation
        x_min = int(fit_a * y_min + fit_b)
        x_max = int(fit_a * y_max + fit_b)
        
        # Draw the averaged line on the image
        # Arguments: image, start_point(x,y), end_point(x,y), color(BGR), thickness
        cv2.line(img, (x_min, y_min), (x_max, y_max), color, 10)

def process_frame_lanes(frame):
    """
    Processes a single video frame to detect and draw lane lines.
    This uses Canny edge detection, ROI masking, Hough Line Transform, and line averaging.
    """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth the image and reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 150) # Low and high thresholds

    # Define a region of interest (ROI) to focus on the road ahead
    height, width = frame.shape[:2]
    # Vertices for a triangular ROI (bottom-left, top-center, bottom-right)
    roi_vertices = [
        (0, height),           # Bottom-left
        (width / 2, height / 2 + 100), # Approximately mid-point above horizon
        (width, height)        # Bottom-right
    ]
    # Create a black mask image of the same size as edges
    mask = np.zeros_like(edges)
    # Fill the ROI polygon with white (255) on the mask
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
    # Apply the mask to the edges image
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Use Hough Line Transform to detect lines in the masked edge image
    # rho: Distance resolution of the accumulator in pixels.
    # theta: Angle resolution of the accumulator in radians.
    # threshold: Minimum number of votes (intersections in accumulator cell) to consider a line.
    # minLineLength: Minimum length of line. Line segments shorter than this are rejected.
    # maxLineGap: Maximum allowed gap between line segments to treat them as a single line.
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
    
    # Create a blank image to draw the detected lines on
    line_image = np.zeros_like(frame)

    if lines is not None:
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Avoid division by zero for vertical lines
            if x1 != x2:
                slope = (y2 - y1) / (x2 - x1)
                # Classify lines based on their slope
                # Negative slope for left lane, positive for right lane (approximate ranges)
                if -0.8 < slope < -0.2: # Adjust these ranges based on typical road angles
                    left_lines.append(line)
                elif 0.2 < slope < 0.8: # Adjust these ranges
                    right_lines.append(line)
        
        # Draw averaged left and right lane lines in different colors
        draw_averaged_lines(line_image, left_lines, (255, 0, 0))  # Blue color for left lane
        draw_averaged_lines(line_image, right_lines, (0, 0, 255)) # Red color for right lane

    # Combine the original frame with the detected lines.
    # alpha (0.8) for original frame, beta (1) for line image.
    # This overlays the lines onto the original video feed.
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    return result


# --- Object Detection Functions (using global_yolo_model) ---

def detect_vehicles(frame, conf_threshold=0.5):
    """
    Detects vehicles (car, truck, bus, motorcycle) in a frame using YOLOv8.
    """
    if global_yolo_model is None:
        print("YOLO model not loaded. Cannot perform vehicle detection.")
        return frame
    
    # COCO class IDs for vehicles: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
    vehicle_class_ids = [2, 3, 5, 7]
    results = global_yolo_model(frame, conf=conf_threshold, classes=vehicle_class_ids, verbose=False)

    annotated_frame = frame.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 2)
            # You might want to get the actual class name from model.names[int(box.cls[0])]
            class_name = global_yolo_model.names[int(box.cls[0])] # Get actual class name
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green
            cv2.putText(annotated_frame, f"{class_name}: {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    return annotated_frame

def detect_pedestrians(frame, conf_threshold=0.5):
    """
    Detects pedestrians (person) in a frame using YOLOv8.
    """
    if global_yolo_model is None:
        print("YOLO model not loaded. Cannot perform pedestrian detection.")
        return frame

    # COCO class ID for 'person' is 0
    person_class_id = [0]
    results = global_yolo_model(frame, conf=conf_threshold, classes=person_class_id, verbose=False)
    
    annotated_frame = frame.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 2)
            class_name = "Pedestrian" # Or global_yolo_model.names[int(box.cls[0])]
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue
            cv2.putText(annotated_frame, f"{class_name}: {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
    return annotated_frame

def detect_traffic_lights(frame, conf_threshold=0.5):
    """
    Detects traffic lights in a frame using YOLOv8.
    (This is the exact code you provided earlier for traffic_detection.py)
    """
    if global_yolo_model is None:
        print("YOLO model not loaded. Cannot perform traffic light detection.")
        return frame

    # The COCO dataset class ID for a traffic light is 9.
    traffic_light_class_id = [9] 
    
    results = global_yolo_model(frame, conf=conf_threshold, classes=traffic_light_class_id, verbose=False)

    annotated_frame = frame.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 2)
            class_name = "Traffic Light"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow color
            cv2.putText(annotated_frame, f"{class_name}: {conf}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
    return annotated_frame

def detect_road_signs(frame, conf_threshold=0.5):
    """
    Detects road signs in a frame using YOLOv8.
    Note: Standard YOLOv8 (COCO) has limited specific road sign classes.
    For comprehensive road sign detection, a custom-trained model is usually needed.
    Here, we'll try to detect 'stop sign' (class 11) and 'traffic light' (class 9) as examples.
    """
    if global_yolo_model is None:
        print("YOLO model not loaded. Cannot perform road sign detection.")
        return frame

    # COCO class IDs that *might* represent some road signs:
    # 9 (traffic light), 11 (stop sign), 12 (fire hydrant - can sometimes be near signs),
    # 13 (parking meter), 25 (backpack), 26 (umbrella), 27 (handbag) -- these are loose examples.
    # For robust road sign detection, a dedicated model or dataset is best.
    road_sign_class_ids = [9, 11] # Example: traffic light and stop sign
    results = global_yolo_model(frame, conf=conf_threshold, classes=road_sign_class_ids, verbose=False)
    
    annotated_frame = frame.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 2)
            class_name = global_yolo_model.names[int(box.cls[0])] # Get actual class name
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 165, 0), 2) # Orange
            cv2.putText(annotated_frame, f"{class_name}: {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
    return annotated_frame

def detect_obstacles(frame, conf_threshold=0.5):
    """
    Detects various potential obstacles in a frame using YOLOv8.
    This will pick up general objects that could be obstacles.
    """
    if global_yolo_model is None:
        print("YOLO model not loaded. Cannot perform obstacle detection.")
        return frame

    # COCO class IDs for common objects that could be obstacles (examples):
    # 0 (person), 2 (car), 3 (motorcycle), 5 (bus), 7 (truck), 8 (boat),
    # 9 (traffic light), 10 (fire hydrant), 11 (stop sign), 14 (bench),
    # 15 (bird), 16 (cat), 17 (dog), 18 (horse), 19 (sheep), 20 (cow),
    # 21 (elephant), 22 (bear), 23 (zebra), 24 (giraffe), 28 (tie), 29 (suitcase),
    # 30 (frisbee), 31 (skis), 32 (snowboard), 33 (sports ball), 34 (kite),
    # 35 (baseball bat), 36 (baseball glove), 37 (skateboard), 38 (surfboard),
    # 39 (tennis racket), 40 (bottle), 41 (wine glass), 42 (cup), 43 (fork),
    # 44 (knife), 45 (spoon), 46 (bowl), 47 (banana), 48 (apple), 49 (sandwich),
    # 50 (orange), 51 (broccoli), 52 (carrot), 53 (hot dog), 54 (pizza),
    # 55 (donut), 56 (cake), 57 (chair), 58 (couch), 59 (potted plant),
    # 60 (bed), 61 (dining table), 62 (toilet), 63 (tv), 64 (laptop), 65 (mouse),
    # 66 (remote), 67 (keyboard), 68 (cell phone), 69 (microwave), 70 (oven),
    # 71 (toaster), 72 (sink), 73 (refrigerator), 74 (book), 75 (clock),
    # 76 (vase), 77 (scissors), 78 (teddy bear), 79 (hair drier), 80 (toothbrush)
    obstacle_class_ids = list(range(80)) # Detect almost all common COCO objects
    
    results = global_yolo_model(frame, conf=conf_threshold, classes=obstacle_class_ids, verbose=False)
    
    annotated_frame = frame.copy()
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 2)
            class_name = global_yolo_model.names[int(box.cls[0])] # Get actual class name
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red
            cv2.putText(annotated_frame, f"{class_name}: {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    return annotated_frame


# --- Combined Detection Logic (from full_analysis.py) ---
def perform_detection(frame, detection_task, confidence_threshold):
    """
    Performs the selected detection task on the input frame.

    This function acts as a central dispatcher, calling the appropriate
    detection function based on the 'detection_task' string.

    Args:
        frame (np.array): The input image frame (BGR format).
        detection_task (str): The name of the detection task to perform
                              (e.g., "Lane Detection", "Vehicle Detection").
        confidence_threshold (float): The minimum confidence score for object detections.
                                      This parameter is passed to object detection functions.

    Returns:
        np.array: The processed frame with detections drawn.
                  Returns the original frame if the task is unknown or
                  if an error occurs within a specific detection function.
    """
    processed_frame = frame.copy() # Start with a copy of the original frame

    # Use if/elif/else to route to the correct detection function
    if detection_task == "Lane Detection":
        try:
            processed_frame = process_frame_lanes(frame)
        except Exception as e:
            st.error(f"Error in Lane Detection: {e}")
            cv2.putText(processed_frame, "Lane Detection Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
    elif detection_task == "Vehicle Detection":
        try:
            processed_frame = detect_vehicles(frame, conf_threshold=confidence_threshold)
        except Exception as e:
            st.error(f"Error in Vehicle Detection: {e}")
            cv2.putText(processed_frame, "Vehicle Detection Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif detection_task == "Pedestrian Detection":
        try:
            processed_frame = detect_pedestrians(frame, conf_threshold=confidence_threshold)
        except Exception as e:
            st.error(f"Error in Pedestrian Detection: {e}")
            cv2.putText(processed_frame, "Pedestrian Detection Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif detection_task == "Traffic Detection":
        try:
            processed_frame = detect_traffic_lights(frame, conf_threshold=confidence_threshold)
        except Exception as e:
            st.error(f"Error in Traffic Detection: {e}")
            cv2.putText(processed_frame, "Traffic Detection Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif detection_task == "Road Sign Detection":
        try:
            processed_frame = detect_road_signs(frame, conf_threshold=confidence_threshold)
        except Exception as e:
            st.error(f"Error in Road Sign Detection: {e}")
            cv2.putText(processed_frame, "Road Sign Detection Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif detection_task == "Obstacle Detection":
        try:
            processed_frame = detect_obstacles(frame, conf_threshold=confidence_threshold)
        except Exception as e:
            st.error(f"Error in Obstacle Detection: {e}")
            cv2.putText(processed_frame, "Obstacle Detection Error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        # If an unrecognized task is selected, print a warning and return the original frame.
        st.warning(f"Unknown detection task '{detection_task}'. Displaying original frame without processing.")
        
    return processed_frame


# --- Streamlit UI and Main Application Logic ---
st.set_page_config(layout="wide", page_title="Self-Driving Car: Multi-Task Detection")

st.title("Self-Driving Car: Multi-Task Detection 🚗")

# Sidebar for user input and task selection
st.sidebar.header("User Input & Model Settings")
detection_task = st.sidebar.radio("Select a task:", [
    "Lane Detection",
    "Vehicle Detection",
    "Pedestrian Detection",
    "Traffic Detection",
    "Road Sign Detection",
    "Obstacle Detection"
])

confidence_threshold = 0.5
# Only show confidence slider for object detection tasks
if detection_task in ["Vehicle Detection", "Pedestrian Detection", "Traffic Detection", "Road Sign Detection", "Obstacle Detection"]:
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
else:
    # Reset confidence_threshold if not applicable to avoid confusion
    confidence_threshold = 0.5


st.sidebar.markdown("""
---
**Instructions:**
1. Select a detection task from the options above.
2. Adjust the confidence threshold if applicable.
3. Upload an image or video file.
4. The processed output will appear below.
""")

# Main content area
st.header(f"Live {detection_task}")
st.markdown("Upload a video or image to see the detection in action.")

# File uploader
uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "png", "mp4", "avi"])

if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower() # Convert to lowercase for consistent checking
    
    if file_extension in [".jpg", ".png"]:
        # --- Image Processing ---
        st.subheader("Original Image")
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True) # Display original
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        st.info("Processing image...")
        
        # Call the single combined detection function
        processed_frame = perform_detection(frame, detection_task, confidence_threshold)

        if processed_frame is not None:
            st.subheader(f"Processed Image: {detection_task}")
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption=f"Detection: {detection_task}", use_column_width=True)
        else:
            st.warning("No processing performed for the selected task.")
    
    elif file_extension in [".mp4", ".avi"]:
        # --- Video Processing ---
        st.subheader("Original Video")
        st.video(uploaded_file)
        
        # Save the uploaded video to a temporary file for OpenCV processing
        temp_file_path = "temp_video.mp4"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture(temp_file_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            st.subheader(f"Processed Video: {detection_task}")
            st_placeholder = st.empty() # Placeholder for live video feed
            
            st.info("Processing video frames... This may take some time depending on video length and task complexity.")
            
            frame_count = 0
            # Optional: Get total_frames for a more accurate progress bar
            # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break # End of video
                
                # Call the single combined detection function
                processed_frame = perform_detection(frame, detection_task, confidence_threshold)

                if processed_frame is not None:
                    # Display the processed frame
                    st_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                
                frame_count += 1
                # Optional: Update progress bar
                # if total_frames > 0:
                #     st.progress(min(1.0, frame_count / total_frames))

            cap.release()
            os.remove(temp_file_path) # Clean up the temporary file
            st.success("Video processing complete!")
    else:
        st.error("Unsupported file type. Please upload a JPG, PNG, MP4, or AVI file.")

