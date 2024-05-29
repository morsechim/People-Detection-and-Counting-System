# People Detection and Counting System

This Python script employs YOLO (You Only Look Once) object detection model to detect people in a given video, tracks their movements, and counts the number of people passing through designated lines.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
The script processes input from a video file, applies YOLO object detection to detect people within the frame, tracks the detected people using the SORT (Simple Online and Realtime Tracking) algorithm, and counts the number of people crossing predefined lines. It annotates the video feed with bounding boxes around the detected people and counts the total number of people crossing the lines.

## Features
- Real-time people detection and tracking.
- Counting the number of people crossing designated lines.
- Annotating the video feed with bounding boxes and counters.
- Customizable parameters for thresholding, line positions, and video paths.

## Prerequisites
Ensure you have the following installed:
- Python 3.6 or later
- `pip` package installer
- GPU with CUDA support (optional for faster processing)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/morsechim/People-Detection-and-Counting-System.git
    cd People-Detection-and-Counting-System
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your environment:
    - Place the YOLOv9e weights file in the `weights` directory.
    - Place your input video file in the `videos` directory.
    - Ensure the `people_mask.png` and `counter_bg.png` images are in the `images` directory.

2. Run the script:
    ```bash
    python main.py
    ```

3. View the output:
    - The processed video with detections will be saved as `output.mp4` in the `videos` directory.
    - During processing, the script will display the video with annotated detections and count overlays.

## Configuration
- **Model and Device:**
    - The YOLO model weights are expected to be located at `./weights/yolov9e.pt`.
    - The script automatically selects the processing device (`mps` if available, otherwise `cpu`).

- **Video Input/Output:**
    - Input video path: `./videos/people.mp4`
    - Output video path: `./videos/output.mp4`

- **Counter Lines:**
    - The coordinates of the lines for counting people crossing:
        - Upward line: `[1016, 351, 1181, 518]`
        - Downward line: `[472, 498, 568, 661]`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **[YOLOv8 by Ultralytics](https://github.com/autogyro/yolo-V8)**
- **[SORT: Simple Online and Realtime Tracking](https://github.com/abewley/sort)**
- **[cvzone](https://github.com/cvzone/cvzone)**

This project is inspired by the need for efficient crowd monitoring systems utilizing computer vision techniques.

---

*Note: Customize the repository URL, paths, and any other project-specific details as needed.*
