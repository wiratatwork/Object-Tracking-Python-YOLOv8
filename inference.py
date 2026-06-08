import cv2
import numpy as np
from sort import Sort, adaptive_brightness, adaptive_sharpen
from ultralytics import YOLO

# Set video path
video_path = r"D:\Computer Vision\Object Tracking Algorithm\Video\video1.mp4"  # Change this to your video file path
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
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit(1)

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load class names
classnames = []
try:
    with open('classes.txt', 'r') as f:
        classnames = f.read().splitlines()
except FileNotFoundError:
    print("Error: 'classes.txt' file not found.")
    exit(1)
except Exception as e:
    print(f"Error reading 'classes.txt': {e}")
    exit(1)

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
        processed_frame = adaptive_brightness(processed_frame, CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE)
    if ENABLE_ADAPTIVE_SHARPEN:
        processed_frame = adaptive_sharpen(processed_frame, SHARPEN_AMOUNT, SHARPEN_RADIUS)

    # Detect objects using YOLOv8
    detections_list = []
    results = model(processed_frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD, iou=0.45)

    # Extract bounding boxes for "car" class
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # Convert to float
            classindex = int(box.cls[0])  # Convert to int
            object_detected = classnames[classindex]

            if object_detected == 'car' and conf > CONFIDENCE_THRESHOLD:
                detections_list.append([x1, y1, x2, y2, conf, classindex])

    # Convert list to numpy array
    detections = np.array(detections_list) if detections_list else np.empty((0, 6))

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
