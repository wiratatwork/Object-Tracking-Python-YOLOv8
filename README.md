# Object Tracking with YOLOv8 and SORT

This project implements an object tracking system using the YOLOv8 object detection model and the SORT (Simple Online and Realtime Tracking) algorithm.

## Features
- Object detection using YOLOv8
- Real-time tracking using SORT
- Optional image pre-processing (Adaptive Brightness and Sharpening)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/wiratatwork/Object-Tracking-Python-YOLOv8.git
   cd Object-Tracking-Python-YOLOv8
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Update the `video_path` in `inference.py` to point to your video file.
2. Run the inference script:
   ```bash
   python inference.py
   ```

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- Ultralytics (YOLOv8)
- Filterpy
- Scikit-image
- SciPy
- Lap (optional, for better linear assignment)
