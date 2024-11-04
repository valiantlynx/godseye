import cv2 
from ultralytics import YOLO
import os

# Get the current working directory
current_working_directory = os.getcwd()

print("Current working directory:", current_working_directory)

pose_model_path = current_working_directory + "/models/yolov8n-pose.pt"

# load the model
model = YOLO(pose_model_path)

# open the video file path
# video_path = "./resources/pose-img.jpg"

video_path = 0
cap = cv2.VideoCapture(video_path)
def stream():
    # loop through the video frames
    while cap.isOpened():
        #read a frame
        success, frame = cap.read()
        
        if success:
            results = model(frame, save=True)
            
            #visualize the results
            annotated_frame = results[0].plot()
            
            cv2.imshow("Output", annotated_frame)
            
            #break the loop on key press q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        else:
            #the video end is reached
            break

    cap.release()
    
stream()