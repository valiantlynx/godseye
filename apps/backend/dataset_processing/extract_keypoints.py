import os
import json
import cv2
from ultralytics import YOLO
import numpy as np

# Initialize YOLO model
model = YOLO("yolov8_model_path.pt")  # Replace with your YOLO model path

# Define dataset paths
input_root_dir = 'archive/RWF-2000'  # Original dataset directory
output_root_dir = 'archive/processed_RWF-2000'  # Directory to save processed dataset

# Define categories and subcategories
sets = ['train', 'val']
categories = ['Fight', 'NonFight']

# Loop through each set and category
for set_name in sets:
    for category in categories:
        input_dir = os.path.join(input_root_dir, set_name, category)
        output_dir = os.path.join(output_root_dir, set_name, category)
        os.makedirs(output_dir, exist_ok=True)

        # Loop through videos in the current category
        for video_file in os.listdir(input_dir):
            if video_file.endswith('.avi'):  # Process only .avi files
                video_name = os.path.splitext(video_file)[0]
                video_input_path = os.path.join(input_dir, video_file)

                # Create a subfolder for each video in the output directory
                video_output_dir = os.path.join(output_dir, video_name)
                os.makedirs(video_output_dir, exist_ok=True)

                # Paths for saving original and processed videos, and keypoints
                original_video_path = os.path.join(video_output_dir, 'original.avi')
                processed_video_path = os.path.join(video_output_dir, 'processed.avi')
                keypoints_path = os.path.join(video_output_dir, 'keypoints.jsonl')

                # Open the input video
                cap = cv2.VideoCapture(video_input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Set up video writers for original and processed videos
                out_original = cv2.VideoWriter(original_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
                out_processed = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

                # Open keypoints file for writing
                with open(keypoints_path, 'w') as kp_file:
                    frame_idx = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Save the original frame
                        out_original.write(frame)

                        # Run YOLO model to get keypoints and process the frame
                        results = model(frame)
                        keypoints = results[0].keypoints.cpu().numpy().tolist()  # Convert keypoints to list

                        # Save keypoints for this frame to JSONL file
                        json.dump({"frame": frame_idx, "keypoints": keypoints}, kp_file)
                        kp_file.write('\n')

                        # Draw keypoints on frame and save to processed video
                        frame_with_keypoints = results[0].plot()
                        out_processed.write(frame_with_keypoints)

                        frame_idx += 1

                # Release resources
                cap.release()
                out_original.release()
                out_processed.release()

print(f"Processed dataset saved in '{output_root_dir}' directory.")
