from ultralytics import YOLO
from sort import *
import numpy as np
import cv2
import torch
import math
import time
import random
import os
import cvzone

# define processing device mode
device: str = "mps" if torch.backends.mps.is_available() else "cpu"

# define dispaly parameters
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 2
font_thickness = 2
rect_thickness = 2
obj_fill = cv2.FILLED
white_color = (255, 255, 255)
red_color = (0, 0, 255)
green_color = (0, 255, 0)
rect_color = (255, 0, 255)
background_color = (255, 0, 255)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]

# define yolo weight path
model_path = os.path.join(".", "weights", "yolov9e.pt")

# initiial yolo weight
model = YOLO(model_path)
model.to(device)

# define video path
video_path = os.path.join(".", "videos", "people.mp4")
output_path = os.path.join(".", "videos", "output.mp4")

# define video input instance
cap = cv2.VideoCapture(video_path)
# get the width and height of the frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# set resolution
cap.set(3, frame_width)
cap.set(4, frame_height)

# define video output instance
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MP4V"), cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

# mouse callback function
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: x={x}, y={y}")
# define window name
cv2.namedWindow("Frame")
# set the mouse callback function
cv2.setMouseCallback("Frame", show_coordinates)

# define time to calculate FPS
previous_time = time.time()

# define ROI mask
mask_path = os.path.join(".", "images", "people_mask.png")
mask = cv2.imread(mask_path)

# define tracking instance
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# define limits line
limits_up = [1016, 351, 1181, 518]
limits_down = [472, 498, 568, 661]
# define total count list
total_count_up = []
total_count_down = []

# couter background
counter_overlay_path = os.path.join(".", "images", "counter_bg.png")
counter_overlay = cv2.imread(counter_overlay_path, cv2.IMREAD_UNCHANGED)
counter_overlay = cv2.resize(counter_overlay, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

while True: 
    
    success, frame = cap.read()
    
    # interupt if not have frame
    if not success:
        break
    
    cvzone.overlayPNG(frame, counter_overlay, (100, 100))
    
    # define detection roi
    detection_roi = cv2.bitwise_and(frame, mask)
    # get results with original frame
    results = model(detection_roi, stream=True)
    
    # define empty detection
    detections = np.empty((0, 5))
    
    # get current time and calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    
    # display processing mode
    # cv2.putText(frame, f"[Processing Mode: {device} ] [FPS: {int(fps)}]", (20 , frame.shape[0] - 20), font, font_scale, green_color, font_thickness, lineType=cv2.LINE_AA)
    
    # get detection results
    for r in results:
        # get all class name 
        class_names = r.names   
        boxes = r.boxes
        # print(r)
        # get all bbox
        for bbox in boxes:
            # get bbox corrdinate
            x1, y1, x2, y2 = bbox.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # get current class name
            cls = int(bbox.cls[0])
            current_class = class_names[cls]
            
            # get confident scores
            conf = math.ceil(bbox.conf[0] * 100) / 100
            
            # bbox and tracking threshold
            if current_class == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                
        # SORT results
        tracker_results = tracker.update(detections)
        
        # counter up
        cv2.line(frame, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 0, 255), 4)
        
        # counter down
        cv2.line(frame, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 0, 255), 4)
        
        for result in tracker_results:
            
            # get object bbox 
            x1, y1, x2, y2, Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # calcalate object bbox width, height
            w, h = x2 - x1, y2 - y1
    
            # calculate center object
            cx, cy = x1 + w // 2, y1 + h // 2
            
            # draw confident score and class name
            current_id = int(Id)
            
            # counter tracker up
            if limits_up[0] < cx < limits_up[2] and limits_up[1] - 5 < cy < limits_up[3] + 5:
                # check id not exites 
                if total_count_up.count(current_id) == 0:
                    total_count_up.append(current_id)
                    # holding line
                    cv2.line(frame, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 255, 0), 5)
            
            # counter tracker down
            if limits_down[0] < cx < limits_down[2] and limits_down[1] - 5 < cy < limits_down[3] + 5:
                # check id not exites 
                if total_count_down.count(current_id) == 0:
                    total_count_down.append(current_id)
                    # holding line
                    cv2.line(frame, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (0, 255, 0), 5)
            
            # draw count up text
            cv2.putText(frame, f"{len(total_count_up)}", (274, 220), font, (font_scale + 1), (255,0,255), (font_thickness + 1), lineType=cv2.LINE_AA)
            # draw count down text
            cv2.putText(frame, f"{len(total_count_down)}", (274, 310), font, (font_scale + 1), (255,0,255), (font_thickness + 1), lineType=cv2.LINE_AA)
            
            # draw object bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[current_id % len(colors)]), rect_thickness)
            
            text = f"ID: {current_id}"
             
            # calculate text size for dynamically words
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            x2, y2 = x1 + text_width, y1 - text_height - baseline

            # draw background label
            cv2.rectangle(frame, (max(0, x1), max(35, y1)), (x2, y2), (colors[current_id % len(colors)]), obj_fill)
            # draw confident score and class name
            cv2.putText(frame, text, (max(0, x1), max(35, y1)), font, font_scale, white_color, font_thickness, lineType=cv2.LINE_AA)
            # draw center object 
            cv2.circle(frame, (cx, cy), 3, (colors[current_id % len(colors)]), obj_fill)
            
        cv2.imshow("Frame", frame)
        cap_out.write(frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
        
cap.release()
cap_out.release()
cv2.destroyAllWindows()
