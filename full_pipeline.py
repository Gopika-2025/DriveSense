# full_pipeline.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils.lane_detection import detect_lanes

# ----------------------------------------------------------
# Configuration - change model path if needed
# ----------------------------------------------------------
YOLO_WEIGHTS = os.path.join("models", "yolov8n.pt")  # <- ensure this file exists
CONFIDENCE = 0.25
IMGSZ = 640  # inference size
# ----------------------------------------------------------

# Load YOLO model once (kept global so multiple calls reuse it)
print("Loading YOLO model from:", YOLO_WEIGHTS)
model = YOLO(YOLO_WEIGHTS)

# Category groups (COCO labels). We'll use model.names during inference.
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle"}
PEDESTRIAN_CLASSES = {"person"}
TRAFFIC_LIGHT_CLASSES = {"traffic light"}
# Road signs in COCO are limited; keep as an example
ROAD_SIGN_CLASSES = {"stop sign"}  # COCO has 'stop sign'
# Some example obstacle classes
OBSTACLE_CLASSES = {"bench", "chair", "backpack", "suitcase"}

def process_frame(frame, draw_labels=True):
    """
    Process a single BGR frame:
      - Run YOLOv8 inference once
      - Get annotated image from ultralytics (RGB), convert to BGR
      - Run lane detection (OpenCV) and overlay
    Returns annotated BGR frame and a dict of detections.
    """
    if frame is None:
        return None, {}

    # Run model: returns Results object
    results = model(frame, imgsz=IMGSZ, conf=CONFIDENCE, verbose=False)

    # annotated image from ultralytics is RGB numpy array
    annotated_rgb = results[0].plot()
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    # collect detections in dict
    detections = {"vehicles": [], "pedestrians": [], "traffic_lights": [], "road_signs": [], "obstacles": []}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        # coordinates: xyxy tensor
        xyxy = box.xyxy[0].tolist()
        label = model.names[cls_id]

        entry = {"label": label, "conf": conf, "xyxy": xyxy}

        if label in VEHICLE_CLASSES:
            detections["vehicles"].append(entry)
        elif label in PEDESTRIAN_CLASSES:
            detections["pedestrians"].append(entry)
        elif label in TRAFFIC_LIGHT_CLASSES:
            detections["traffic_lights"].append(entry)
        elif label in ROAD_SIGN_CLASSES:
            detections["road_signs"].append(entry)
        elif label in OBSTACLE_CLASSES:
            detections["obstacles"].append(entry)
        else:
            # other classes ignored or could be treated as obstacles
            pass

    # Overlay lane detection on top of annotated image
    final_frame = detect_lanes(annotated_bgr)

    return final_frame, detections


def process_image_file(input_path):
    """
    Convenience wrapper to process a single image file path.
    Returns annotated BGR image.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    annotated, detections = process_frame(img)
    return annotated, detections


def process_video(input_path, output_path=None, show_live=False, max_frames=None):
    """
    Process video file frame-by-frame, run pipeline, save annotated video.
    - input_path: path to input video
    - output_path: if None, save next to input with suffix
    - show_live: if True, display cv2 window while processing
    - max_frames: optional int to limit frames for quick tests
    Returns path to saved output video.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_annotated{ext}"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print("Processing video:", input_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, detections = process_frame(frame)
        out.write(annotated_frame)

        if show_live:
            cv2.imshow("Full Pipeline", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        if (max_frames is not None) and (frame_count >= max_frames):
            break

    cap.release()
    out.release()
    if show_live:
        cv2.destroyAllWindows()
    print("Saved annotated video to:", output_path)
    return output_path


# If run directly, simple CLI for quick tests
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python full_pipeline.py <image_or_video_path> [show_live]")
        sys.exit(1)
    path = sys.argv[1]
    show = (len(sys.argv) > 2 and sys.argv[2].lower() == "true")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        out_img, det = process_image_file(path)
        print("Detections:", det)
        cv2.imshow("Annotated Image", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        process_video(path, show_live=show)
    else:
        print("Unsupported file type.")
