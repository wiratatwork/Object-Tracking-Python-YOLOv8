import cv2
import numpy as np
from sort import Sort
from ultralytics import YOLO

# Set video path
video_path = r"D:\Computer Vision\Computer Vision  Basic Source Code\Object Tracking Algorithm\Video\video0.mp4"  # Change this to your video file path
MAX_AGE = 500
VIDEO_FPS = 29.97
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# Pre-processing Config
ENABLE_ADAPTIVE_BRIGHTNESS = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
ENABLE_ADAPTIVE_SHARPEN = False
SHARPEN_AMOUNT = 1.5
SHARPEN_RADIUS = 3

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load class names
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

def adaptive_brightness(frame):
    """Adjusts brightness using CLAHE in LAB color space."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def adaptive_sharpen(frame):
    """Applies sharpening selectively to edges using a Laplacian mask."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Create a mask for high-gradient areas
    mask = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = cv2.GaussianBlur(mask, (SHARPEN_RADIUS * 2 + 1, SHARPEN_RADIUS * 2 + 1), 0)
    mask = mask.astype(float) / 255.0
    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(float) / 255.0

    # Unsharp mask implementation
    blurred = cv2.GaussianBlur(frame, (SHARPEN_RADIUS, SHARPEN_RADIUS), 0)
    sharpened = cv2.addWeighted(frame, SHARPEN_AMOUNT, blurred, -(SHARPEN_AMOUNT - 1), 0)
    
    # Blend original and sharpened based on the edge mask
    result = (frame * (1.0 - mask) + sharpened * mask).astype(np.uint8)
    return result

# Initialize SORT tracker
tracker = Sort(max_age=MAX_AGE)

# Start processing frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-processing
    processed_frame = frame.copy()
    if ENABLE_ADAPTIVE_BRIGHTNESS:
        processed_frame = adaptive_brightness(processed_frame)
    if ENABLE_ADAPTIVE_SHARPEN:
        processed_frame = adaptive_sharpen(processed_frame)

    # Detect objects using YOLOv8
    detections = np.empty((0, 6))  # Store valid detections
    results = model(processed_frame, imgsz=IMAGE_SIZE, conf=0.4, iou=0.45)

    # Extract bounding boxes for "person" class
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # Convert to float
            classindex = int(box.cls[0])  # Convert to int
            object_detected = classnames[classindex]

            if object_detected and  conf > CONFIDENCE_THRESHOLD:
                new_detection = np.array([x1, y1, x2, y2, conf, classindex])
                detections = np.vstack((detections, new_detection))

    # Update tracker with new detections
    track_results = tracker.update(detections)

    # Draw tracking results
    for result in track_results:
        x1, y1, x2, y2, obj_id, class_id = map(int, result)
        class_name = classnames[class_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name}_{obj_id}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Show frame
    cv2.imshow("Object Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
