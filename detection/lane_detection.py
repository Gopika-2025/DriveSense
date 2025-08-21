import cv2
import numpy as np

def process_frame_lanes(frame):
    """
    Processes a single video frame to detect and draw lane lines.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    height, width = frame.shape[:2]
    roi_vertices = [
        (0, height),
        (width / 2, height / 2 + 100),
        (width, height)
    ]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
    
    line_image = np.zeros_like(frame)

    if lines is not None:
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 != x2:
                slope = (y2 - y1) / (x2 - x1)
                if -0.8 < slope < -0.2:
                    left_lines.append(line)
                elif 0.2 < slope < 0.8:
                    right_lines.append(line)
        
        draw_averaged_lines(line_image, left_lines, (255, 0, 0))
        draw_averaged_lines(line_image, right_lines, (0, 0, 255))

    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    return result

def draw_averaged_lines(img, lines, color):
    """Averages the lines and draws a single, smooth line."""
    if not lines:
        return

    lines = [line[0] for line in lines]
    x_coords = np.array([p for line in lines for p in (line[0], line[2])])
    y_coords = np.array([p for line in lines for p in (line[1], line[3])])
    
    if len(x_coords) > 1:
        poly_fit = np.polyfit(y_coords, x_coords, 1)
        fit_a = poly_fit[0]
        fit_b = poly_fit[1]

        y_min = int(min(y_coords))
        y_max = int(img.shape[0])
        
        x_min = int(fit_a * y_min + fit_b)
        x_max = int(fit_a * y_max + fit_b)
        
        cv2.line(img, (x_min, y_min), (x_max, y_max), color, 10)